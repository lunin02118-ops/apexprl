// SPDX-License-Identifier: ISC
// ApexPRL v0.2 "Hyper-L2 Direct" CUDA/PTX Compute Engine for NVIDIA RTX 40 (sm_89) & RTX 50 (sm_120a).
// Key innovations:
//   1. Completely removes dynamic basis-gather bank conflicts, IADD3, PRMT, STS.128 from the O(M*N*K) inner loop.
//   2. Pre-materializes L2-resident swizzled K_STAGE=64 panels via SWAR (0x40404040u) + warp-shuffle transpose.
//   3. Zero-MOV contiguous B-fragment lane mapping for `ldmatrix.sync.aligned.m8n8.x4.shared.b16`.
//   4. 2 CTAs/SM occupancy (`128x128` CTA, 4 warps/CTA = 8 warps/SM, <= 208 regs, 0 register spills!)
//      hiding barrier, TMA/cp.async, and BLAKE3 epilogue latency across the two CTAs sharing every SM.
//   5. `elect.sync` single-issue `cp.async.bulk` TMA on Blackwell (`sm_120a`) with zero peel loops.
#pragma once
#include <cstdint>
#include "blake3_regs.cuh"
#include "prep_kernels.cuh"
#include "direct_l2_params.cuh"

namespace apx_dl2 {

// Non-interleaved 16-row x 128-comp packed basis generator (universal for both Ada & Blackwell Direct-L2).
extern "C" __global__ __launch_bounds__(64) void apx_gen_basis_linear(
    const uint32_t *__restrict__ noise_key,
    uint8_t *__restrict__ out_packed,
    int total_rows,
    int is_b) {
  apx_prep::gen_basis_body<false>(noise_key, out_packed, total_rows, is_b);
}

// Generates compact uint16_t pairs (plus | (minus << 8)) for A and B independently in registers.
extern "C" __global__ void apx_gen_u16_pairs(
    const uint32_t *__restrict__ key_a,
    const uint32_t *__restrict__ key_b,
    uint16_t *__restrict__ out_pairs_a,
    uint16_t *__restrict__ out_pairs_b,
    int k) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h * 8 >= k) return;

  uint32_t ka[8], kb[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ka[i] = key_a[i];
    kb[i] = key_b[i];
  }

  uint32_t msg_a[16] = {0}, msg_b[16] = {0};
  msg_a[1] = uint32_t(1 + h);
  msg_a[8] = 0x65745f41u; // "A_te"
  msg_a[9] = 0x726f736eu; // "nsor"

  msg_b[1] = uint32_t(1 + h);
  msg_b[8] = 0x65745f42u; // "B_te"
  msg_b[9] = 0x726f736eu; // "nsor"

  uint32_t da[8], db[8];
  apx_b3::keyed_hash64(ka, msg_a, da);
  apx_b3::keyed_hash64(kb, msg_b, db);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int col = h * 8 + j;
    if (col >= k) break;
    uint32_t ua = da[j];
    uint32_t pa0 = ua & 127u;
    uint32_t pa1 = pa0 ^ (1u + __umulhi(127u, ua));
    out_pairs_a[col] = uint16_t(pa0 | (pa1 << 8));

    uint32_t ub = db[j];
    uint32_t pb0 = ub & 127u;
    uint32_t pb1 = pb0 ^ (1u + __umulhi(127u, ub));
    out_pairs_b[col] = uint16_t(pb0 | (pb1 << 8));
  }
}

// SWAR + 4-lane warp-shuffle transpose materializer:
//   Input `bas_t`:  [total_frags, 128, 16] (16-row packed basis, values in 0..63)
//   Input `pairs`:  [K] uint16_t = plus | (minus << 8)
//   Output `panel`: [num_tiles, K/64, 128, 64] pre-swizzled for 0-bank-conflict ldmatrix.m8n8.x4.b16!
extern "C" __global__ __launch_bounds__(256) void apx_materialize_panel_128(
    const uint8_t *__restrict__ bas_t,
    const uint16_t *__restrict__ pairs,
    uint8_t *__restrict__ out_panel,
    int start_frag,
    int K) {
  constexpr int TILE_R = 128;
  constexpr int FRAGS = TILE_R / 16; // 8 (1 warp per 16-row fragment!)
  __shared__ __align__(16) uint8_t s_bas[TILE_R * 128]; // 16 KiB
  __shared__ __align__(16) uint32_t s_tr[8][16][9];     // 4.5 KiB (stride 9 coprime to 32 banks!)

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int tile = blockIdx.x;
  const int k_part = blockIdx.y;
  const int k_parts = gridDim.y;
  const int num_ks = K >> 6;
  const int ks_per_part = (num_ks + k_parts - 1) / k_parts;
  const int ks_begin = k_part * ks_per_part;
  const int ks_end = (ks_begin + ks_per_part < num_ks) ? (ks_begin + ks_per_part) : num_ks;

  const uint4 *src_b = reinterpret_cast<const uint4 *>(bas_t + size_t(start_frag + tile * FRAGS) * 2048);
  uint4 *dst_s = reinterpret_cast<uint4 *>(s_bas);
  constexpr int VEC_CNT = (TILE_R * 128) / 16; // 1024
#pragma unroll
  for (int i = tid; i < VEC_CNT; i += 256) dst_s[i] = src_b[i];
  __syncthreads();

  const int sub_grp = lane & 3;
  const int k_word = lane >> 2;
  const int out_r = lane >> 1;
  const int out_c = lane & 1;
  const uint4 *f_bas = reinterpret_cast<const uint4 *>(s_bas + warp * 2048);

  for (int ks = ks_begin; ks < ks_end; ++ks) {
    uint8_t *dst_tile = out_panel + (size_t(tile) * num_ks + ks) * (TILE_R * 64);

#pragma unroll
    for (int half32 = 0; half32 < 2; ++half32) {
      uint16_t pr = __ldg(pairs + ks * 64 + half32 * 32 + lane);
      uint4 vp = f_bas[pr & 127];
      uint4 vm = f_bas[pr >> 8];
      uint32_t d[4];
      d[0] = vp.x + 0x40404040u - vm.x;
      d[1] = vp.y + 0x40404040u - vm.y;
      d[2] = vp.z + 0x40404040u - vm.z;
      d[3] = vp.w + 0x40404040u - vm.w;

#pragma unroll
      for (int g = 0; g < 4; ++g) {
        uint32_t x0 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 0);
        uint32_t x1 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 1);
        uint32_t x2 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 2);
        uint32_t x3 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 3);
        uint32_t t01 = (sub_grp == 0) ? __byte_perm(x0, x1, 0x4400)
                     : (sub_grp == 1) ? __byte_perm(x0, x1, 0x5511)
                     : (sub_grp == 2) ? __byte_perm(x0, x1, 0x6622)
                                      : __byte_perm(x0, x1, 0x7733);
        uint32_t t23 = (sub_grp == 0) ? __byte_perm(x2, x3, 0x4400)
                     : (sub_grp == 1) ? __byte_perm(x2, x3, 0x5511)
                     : (sub_grp == 2) ? __byte_perm(x2, x3, 0x6622)
                                      : __byte_perm(x2, x3, 0x7733);
        s_tr[warp][g * 4 + sub_grp][k_word] = __byte_perm(t01, t23, 0x6420);
      }
      __syncwarp();

      int global_r = warp * 16 + out_r;
      int c_logical = half32 * 2 + out_c;
      int c_swiz = c_logical ^ ((global_r >> 1) & 3);
      uint32_t w0 = s_tr[warp][out_r][out_c * 4 + 0];
      uint32_t w1 = s_tr[warp][out_r][out_c * 4 + 1];
      uint32_t w2 = s_tr[warp][out_r][out_c * 4 + 2];
      uint32_t w3 = s_tr[warp][out_r][out_c * 4 + 3];
      *reinterpret_cast<uint4 *>(dst_tile + global_r * 64 + c_swiz * 16) = make_uint4(w0, w1, w2, w3);
      __syncwarp();
    }
  }
}

__device__ __forceinline__ unsigned sm_u32(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4_a(uint32_t (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_b2(uint32_t (&b0)[2], uint32_t (&b1)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}

__device__ __forceinline__ void imma_u8(int32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

extern __shared__ __align__(1024) uint8_t apx_dl2_sm[];

// ---------------------------------------------------------------------------
// Variant 0 & 1: 2-CTA/SM Hyper-L2 Direct (`128x128` CTA, 4 warps/CTA = 8 warps/SM)
//   DELAY_FOLD = false (`apx_direct_l2_v0`): ultra-compact 201-206 regs, 0 spills (`-O3` sm_89, `-O1` sm_120a)
//   DELAY_FOLD = true  (`apx_direct_l2_v1`): zero-RAW delayed fold on sm_120a (`208 regs, 0 spills`)
// ---------------------------------------------------------------------------
template <bool DELAY_FOLD, int LDSM_SHIFT = 0>
__device__ __forceinline__ void run_direct_128x128(const ApxDirectL2Params &p) {
  constexpr int BM = 128, BN = 128, NT = 128, W_ROWS = 2, W_COLS = 2;
  constexpr int WM = 64, WN = 64, MI = 4, NJ = 8, K_STAGE = 64;
  constexpr int SA_BYTES = BM * K_STAGE;         // 8,192 B
  constexpr int SB_BYTES = BN * K_STAGE;         // 8,192 B
  constexpr int STG_BYTES = SA_BYTES + SB_BYTES; // 16,384 B
  constexpr int NUM_STAGES = 2;
  constexpr int OFF_TR = NUM_STAGES * STG_BYTES; // 32,768 B
  constexpr int TRACE_BYTES = NT * 16 * 4;       // 8,192 B
  constexpr int OFF_BAR = OFF_TR + TRACE_BYTES;  // 40,960 B (Total 41,088 B x 2 CTAs = 82,176 B <= 99 KiB!)

  uint8_t *sm = apx_dl2_sm;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wr = warp >> 1;
  const int wc = warp & 1;

  const unsigned sbase = sm_u32(sm);
  const unsigned tr_base = sbase + OFF_TR + tid * 4u;

#if APX_IS_SM120
  const unsigned bar0 = sbase + OFF_BAR;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0 + 0u) : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0 + 8u) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  unsigned phase0 = 0, phase1 = 0;
#else
  (void)OFF_BAR;
#endif

  const int a_row = lane & 15;
  const int a_c0 = lane >> 4;
  const unsigned a_off0 = (wr * WM + a_row) * 64u + ((a_c0 ^ ((a_row >> 1) & 3)) * 16u);

  const int b_row = (lane & 7) | ((lane >> 1) & 8);
  const int b_c0 = (lane >> 3) & 1;
  const unsigned b_off0 = SA_BYTES + (wc * WN + b_row) * 64u + ((b_c0 ^ ((b_row >> 1) & 3)) * 16u);

  const int tiles = p.m_tiles * p.n_tiles;
  const int num_ks = p.num_ks;
  const int tile_idx = blockIdx.x;
  if (tile_idx >= tiles) return;

  // L2-friendly 2D swizzle (4 M-tiles grouped along N to maximize B cache-line broadcast!)
  constexpr int GROUP_M = 4;
  const int group_size = GROUP_M * p.n_tiles;
  const int grp_id = tile_idx / group_size;
  const int first_m = grp_id * GROUP_M;
  const int cur_m_cnt = (first_m + GROUP_M <= p.m_tiles) ? GROUP_M : (p.m_tiles - first_m);
  const int in_grp = tile_idx - grp_id * group_size;
  const int bm = first_m + (in_grp % cur_m_cnt);
  const int bn = in_grp / cur_m_cnt;

  const uint8_t *gA = p.panelA + size_t(bm) * size_t(num_ks) * SA_BYTES;
  const uint8_t *gB = p.panelB + size_t(bn) * size_t(num_ks) * SB_BYTES;

  auto issue_stage = [&]<int ST>(int ks) {
#if APX_IS_SM120
    if (warp == 0) {
      unsigned elected;
      asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }" : "=r"(elected));
      if (elected) {
        unsigned dst_a = sbase + ST * STG_BYTES;
        unsigned dst_b = dst_a + SA_BYTES;
        unsigned b = bar0 + ST * 8u;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(STG_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dst_a), "l"(gA + size_t(ks) * SA_BYTES), "r"(SA_BYTES), "r"(b) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dst_b), "l"(gB + size_t(ks) * SB_BYTES), "r"(SB_BYTES), "r"(b) : "memory");
      }
    }
#else
    unsigned dst_a = sbase + ST * STG_BYTES + tid * 16u;
    unsigned dst_b = dst_a + SA_BYTES;
    const uint8_t *src_a = gA + size_t(ks) * SA_BYTES + tid * 16u;
    const uint8_t *src_b = gB + size_t(ks) * SB_BYTES + tid * 16u;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_a + it * 2048u), "l"(src_a + it * 2048));
    }
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_b + it * 2048u), "l"(src_b + it * 2048));
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
  };

  auto wait_stage = [&]<int ST>() {
#if APX_IS_SM120
    unsigned b = bar0 + ST * 8u;
    unsigned &ph = (ST == 0) ? phase0 : phase1;
    asm volatile("{ .reg .pred q; L_W%=: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L_W%=; }"
                 :: "r"(b), "r"(ph) : "memory");
    ph ^= 1u;
    __syncthreads();
#else
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#endif
  };

  int32_t acc[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0;

  issue_stage.template operator()<0>(0);
  wait_stage.template operator()<0>();

  uint32_t af[2][MI][4];
  uint32_t bf[2][NJ][2];

  {
    unsigned ab = sbase + a_off0;
    unsigned bb = sbase + b_off0;
#pragma unroll
    for (int i = 0; i < MI; ++i) ldmatrix_x4_a(af[0][i], ab + i * 1024u);
#pragma unroll
    for (int jp = 0; jp < NJ / 2; ++jp) ldmatrix_x4_b2(bf[0][2 * jp], bf[0][2 * jp + 1], bb + jp * 1024u);
  }

  auto mma_step = [&]<int CUR_BUF, int NXT_BUF, bool DO_LOAD, bool DO_FOLD, bool PRE_FOLD>(
      unsigned nxt_base, unsigned nxt_xor, int fold_stp) {
    unsigned ab = nxt_base + (a_off0 ^ nxt_xor);
    unsigned bb = nxt_base + (b_off0 ^ nxt_xor);
    uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        if constexpr (DO_FOLD && PRE_FOLD) {
          if ((j & 1) == 0) {
            h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          } else {
            h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          }
        }
        imma_u8(acc[i][j], af[CUR_BUF][i], bf[CUR_BUF][j]);
        if constexpr (DO_LOAD) {
          int sid = j * MI + i;
          if (sid == (1 + LDSM_SHIFT))  ldmatrix_x4_a(af[NXT_BUF][0], ab + 0u);
          if (sid == (4 + LDSM_SHIFT))  ldmatrix_x4_a(af[NXT_BUF][1], ab + 1024u);
          if (sid == (7 + LDSM_SHIFT))  ldmatrix_x4_a(af[NXT_BUF][2], ab + 2048u);
          if (sid == (10 + LDSM_SHIFT)) ldmatrix_x4_a(af[NXT_BUF][3], ab + 3072u);
          if (sid == (14 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][0], bf[NXT_BUF][1], bb + 0u);
          if (sid == (18 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][2], bf[NXT_BUF][3], bb + 1024u);
          if (sid == (22 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][4], bf[NXT_BUF][5], bb + 2048u);
          if (sid == (26 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][6], bf[NXT_BUF][7], bb + 3072u);
        }
        if constexpr (DO_FOLD && !PRE_FOLD) {
          if ((j & 1) == 0) {
            h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          } else {
            h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          }
        }
      }
    }
    if constexpr (DO_FOLD) {
      uint32_t h = apx_b3::xor3_lop3(h0, h1, h2) ^ h3;
      unsigned taddr = tr_base + unsigned((fold_stp & 15) * NT * 4);
      uint32_t out_w = h;
      if (fold_stp >= 16) {
        uint32_t prev_w;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
        out_w = apx_b3::rotl_xor<13>(prev_w, h);
      }
      asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
    }
  };

  if constexpr (!DELAY_FOLD) {
    // Variant 0 (ultra-compact single loop, 200-206 regs, 0 spills)
#pragma unroll 1
    for (int ks = 0; ks < num_ks - 2; ks += 2) {
      issue_stage.template operator()<1>(ks + 1);
      mma_step.template operator()<0, 1, true, false, false>(sbase + 0u, 32u, 0);
      wait_stage.template operator()<1>();
      mma_step.template operator()<1, 0, true, false, false>(sbase + STG_BYTES, 0u, 0);

      issue_stage.template operator()<0>(ks + 2);
      mma_step.template operator()<0, 1, true, false, false>(sbase + STG_BYTES, 32u, 0);
      wait_stage.template operator()<0>();
      mma_step.template operator()<1, 0, true, true, false>(sbase + 0u, 0u, ks >> 1);
    }
    {
      int ks = num_ks - 2;
      issue_stage.template operator()<1>(ks + 1);
      mma_step.template operator()<0, 1, true, false, false>(sbase + 0u, 32u, 0);
      wait_stage.template operator()<1>();
      mma_step.template operator()<1, 0, true, false, false>(sbase + STG_BYTES, 0u, 0);
      mma_step.template operator()<0, 1, true, false, false>(sbase + STG_BYTES, 32u, 0);
      mma_step.template operator()<1, 0, false, true, false>(sbase + 0u, 0u, ks >> 1);
    }
  } else {
    // Variant 1 (zero-RAW delayed fold: folds previous pair BEFORE ks's first IMMA!)
    issue_stage.template operator()<1>(1);
    mma_step.template operator()<0, 1, true, false, true>(sbase + 0u, 32u, 0);
    wait_stage.template operator()<1>();
    mma_step.template operator()<1, 0, true, false, true>(sbase + STG_BYTES, 0u, 0);
    issue_stage.template operator()<0>(2);
    mma_step.template operator()<0, 1, true, false, true>(sbase + STG_BYTES, 32u, 0);
    wait_stage.template operator()<0>();
    mma_step.template operator()<1, 0, true, false, true>(sbase + 0u, 0u, 0);

#pragma unroll 1
    for (int ks = 2; ks < num_ks - 2; ks += 2) {
      issue_stage.template operator()<1>(ks + 1);
      mma_step.template operator()<0, 1, true, true, true>(sbase + 0u, 32u, (ks >> 1) - 1);
      wait_stage.template operator()<1>();
      mma_step.template operator()<1, 0, true, false, true>(sbase + STG_BYTES, 0u, 0);

      issue_stage.template operator()<0>(ks + 2);
      mma_step.template operator()<0, 1, true, false, true>(sbase + STG_BYTES, 32u, 0);
      wait_stage.template operator()<0>();
      mma_step.template operator()<1, 0, true, false, true>(sbase + 0u, 0u, 0);
    }
    {
      int ks = num_ks - 2;
      issue_stage.template operator()<1>(ks + 1);
      mma_step.template operator()<0, 1, true, true, true>(sbase + 0u, 32u, (ks >> 1) - 1);
      wait_stage.template operator()<1>();
      mma_step.template operator()<1, 0, true, false, true>(sbase + STG_BYTES, 0u, 0);
      mma_step.template operator()<0, 1, true, false, true>(sbase + STG_BYTES, 32u, 0);
      mma_step.template operator()<1, 0, false, false, true>(sbase + 0u, 0u, 0);
    }
    // Final fold of last pair
    {
      uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
#pragma unroll
      for (int i = 0; i < MI; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; j += 2) {
          h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
          h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j + 1][0]), uint32_t(acc[i][j + 1][1]));
          h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j + 1][2]), uint32_t(acc[i][j + 1][3]));
        }
      }
      uint32_t h = apx_b3::xor3_lop3(h0, h1, h2) ^ h3;
      int last_stp = (num_ks >> 1) - 1;
      unsigned taddr = tr_base + unsigned((last_stp & 15) * NT * 4);
      uint32_t out_w = h;
      if (last_stp >= 16) {
        uint32_t prev_w;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
        out_w = apx_b3::rotl_xor<13>(prev_w, h);
      }
      asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
    }
  }

  uint32_t msg[16], key[8], digest[8];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(msg[j]) : "r"(tr_base + unsigned(j * NT * 4)));
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) key[j] = p.key_a[j];
  apx_b3::keyed_hash64(key, msg, digest);

  bool good = true;
#pragma unroll
  for (int j = 7; j >= 0; --j) {
    uint32_t tw = p.target_le[j];
    if (digest[j] > tw) { good = false; break; }
    if (digest[j] < tw) break;
  }
  if (good) {
    uint32_t g = uint32_t((((p.base_bm + bm) * W_ROWS + wr) * p.total_n_warps + bn * W_COLS + wc) * 32 + lane);
    uint32_t s = atomicAdd(p.hit_count, 1u);
    if (s < uint32_t(p.hit_cap)) p.hit_groups[s] = g;
  }
}

// ---------------------------------------------------------------------------
// Variant 2: 1-CTA/SM 3-Stage Hyper-L2 Direct (`128x256` CTA, 8 warps/CTA)
// Reads the EXACT SAME `128`-col `panelB` layout (`bn_left = 2*bn2`, `bn_right = 2*bn2 + 1`)!
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 1) void apx_direct_l2_v2(const __grid_constant__ ApxDirectL2Params p) {
  constexpr int BM = 128, BN = 256, NT = 256, W_ROWS = 2, W_COLS = 4;
  constexpr int WM = 64, WN = 64, MI = 4, NJ = 8, K_STAGE = 64;
  constexpr int SA_BYTES = BM * K_STAGE;         // 8,192 B
  constexpr int SB_HALF  = 128 * K_STAGE;        // 8,192 B
  constexpr int SB_BYTES = BN * K_STAGE;         // 16,384 B
  constexpr int STG_BYTES = SA_BYTES + SB_BYTES; // 24,576 B
  constexpr int NUM_STAGES = 3;
  constexpr int OFF_TR = NUM_STAGES * STG_BYTES; // 73,728 B
  constexpr int TRACE_BYTES = NT * 16 * 4;       // 16,384 B
  constexpr int OFF_BAR = OFF_TR + TRACE_BYTES;  // 90,112 B (<= 99 KiB!)

  uint8_t *sm = apx_dl2_sm;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wr = warp >> 2;
  const int wc = warp & 3;

  const unsigned sbase = sm_u32(sm);
  const unsigned tr_base = sbase + OFF_TR + tid * 4u;

#if APX_IS_SM120
  const unsigned bar0 = sbase + OFF_BAR;
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NUM_STAGES; ++st) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0 + st * 8u) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  unsigned phase[NUM_STAGES] = {0, 0, 0};
#else
  (void)OFF_BAR;
#endif

  const int a_row = lane & 15;
  const int a_c0 = lane >> 4;
  const unsigned a_off0 = (wr * WM + a_row) * 64u + ((a_c0 ^ ((a_row >> 1) & 3)) * 16u);

  const int b_row = (lane & 7) | ((lane >> 1) & 8);
  const int b_c0 = (lane >> 3) & 1;
  const unsigned b_off0 = SA_BYTES + (wc * WN + b_row) * 64u + ((b_c0 ^ ((b_row >> 1) & 3)) * 16u);

  const int n_tiles2 = p.n_tiles >> 1;
  const int tiles = p.m_tiles * n_tiles2;
  const int num_ks = p.num_ks;
  const int tile_idx = blockIdx.x;
  if (tile_idx >= tiles) return;

  constexpr int GROUP_M = 4;
  const int group_size = GROUP_M * n_tiles2;
  const int grp_id = tile_idx / group_size;
  const int first_m = grp_id * GROUP_M;
  const int cur_m_cnt = (first_m + GROUP_M <= p.m_tiles) ? GROUP_M : (p.m_tiles - first_m);
  const int in_grp = tile_idx - grp_id * group_size;
  const int bm = first_m + (in_grp % cur_m_cnt);
  const int bn2 = in_grp / cur_m_cnt;

  const uint8_t *gA  = p.panelA + size_t(bm) * size_t(num_ks) * SA_BYTES;
  const uint8_t *gB0 = p.panelB + size_t(bn2 * 2 + 0) * size_t(num_ks) * SB_HALF;
  const uint8_t *gB1 = p.panelB + size_t(bn2 * 2 + 1) * size_t(num_ks) * SB_HALF;

  auto issue_stage = [&]<int ST>(int ks) {
#if APX_IS_SM120
    if (warp == 0) {
      unsigned elected;
      asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }" : "=r"(elected));
      if (elected) {
        unsigned dst_a  = sbase + ST * STG_BYTES;
        unsigned dst_b0 = dst_a + SA_BYTES;
        unsigned dst_b1 = dst_b0 + SB_HALF;
        unsigned b = bar0 + ST * 8u;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(STG_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dst_a), "l"(gA + size_t(ks) * SA_BYTES), "r"(SA_BYTES), "r"(b) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dst_b0), "l"(gB0 + size_t(ks) * SB_HALF), "r"(SB_HALF), "r"(b) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dst_b1), "l"(gB1 + size_t(ks) * SB_HALF), "r"(SB_HALF), "r"(b) : "memory");
      }
    }
#else
    unsigned dst_a  = sbase + ST * STG_BYTES + tid * 16u;
    unsigned dst_b0 = dst_a + SA_BYTES;
    unsigned dst_b1 = dst_b0 + SB_HALF;
    const uint8_t *src_a  = gA  + size_t(ks) * SA_BYTES + tid * 16u;
    const uint8_t *src_b0 = gB0 + size_t(ks) * SB_HALF  + tid * 16u;
    const uint8_t *src_b1 = gB1 + size_t(ks) * SB_HALF  + tid * 16u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_a),         "l"(src_a));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_a + 4096u), "l"(src_a + 4096));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_b0),         "l"(src_b0));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_b0 + 4096u), "l"(src_b0 + 4096));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_b1),         "l"(src_b1));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_b1 + 4096u), "l"(src_b1 + 4096));
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
  };

  auto wait_stage = [&]<int ST, bool HAS_ISS>() {
#if APX_IS_SM120
    unsigned b = bar0 + ST * 8u;
    asm volatile("{ .reg .pred q; L_W%=: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L_W%=; }"
                 :: "r"(b), "r"(phase[ST]) : "memory");
    phase[ST] ^= 1u;
    __syncthreads();
#else
    if constexpr (HAS_ISS) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else                   asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#endif
  };

  int32_t acc[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0;

  issue_stage.template operator()<0>(0);
  issue_stage.template operator()<1>(1);
  wait_stage.template operator()<0, true>();

  uint32_t af[2][MI][4];
  uint32_t bf[2][NJ][2];
  {
    unsigned ab = sbase + a_off0;
    unsigned bb = sbase + b_off0;
#pragma unroll
    for (int i = 0; i < MI; ++i) ldmatrix_x4_a(af[0][i], ab + i * 1024u);
#pragma unroll
    for (int jp = 0; jp < NJ / 2; ++jp) ldmatrix_x4_b2(bf[0][2 * jp], bf[0][2 * jp + 1], bb + jp * 1024u);
  }

  auto mma_step = [&]<int CUR_BUF, int NXT_BUF, bool DO_LOAD, bool DO_FOLD>(
      unsigned nxt_base, unsigned nxt_xor, int fold_stp) {
    unsigned ab = nxt_base + (a_off0 ^ nxt_xor);
    unsigned bb = nxt_base + (b_off0 ^ nxt_xor);
    uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        imma_u8(acc[i][j], af[CUR_BUF][i], bf[CUR_BUF][j]);
        if constexpr (DO_LOAD) {
          constexpr int LDSM_SHIFT = 0;
          int sid = j * MI + i;
          if (sid == (1 + LDSM_SHIFT))  ldmatrix_x4_a(af[NXT_BUF][0], ab + 0u);
          if (sid == (4 + LDSM_SHIFT))  ldmatrix_x4_a(af[NXT_BUF][1], ab + 1024u);
          if (sid == (7 + LDSM_SHIFT))  ldmatrix_x4_a(af[NXT_BUF][2], ab + 2048u);
          if (sid == (10 + LDSM_SHIFT)) ldmatrix_x4_a(af[NXT_BUF][3], ab + 3072u);
          if (sid == (14 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][0], bf[NXT_BUF][1], bb + 0u);
          if (sid == (18 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][2], bf[NXT_BUF][3], bb + 1024u);
          if (sid == (22 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][4], bf[NXT_BUF][5], bb + 2048u);
          if (sid == (26 + LDSM_SHIFT)) ldmatrix_x4_b2(bf[NXT_BUF][6], bf[NXT_BUF][7], bb + 3072u);
        }
        if constexpr (DO_FOLD) {
          if ((j & 1) == 0) {
            h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          } else {
            h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          }
        }
      }
    }
    if constexpr (DO_FOLD) {
      uint32_t h = apx_b3::xor3_lop3(h0, h1, h2) ^ h3;
      unsigned taddr = tr_base + unsigned((fold_stp & 15) * NT * 4);
      uint32_t out_w = h;
      if (fold_stp >= 16) {
        uint32_t prev_w;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
        out_w = apx_b3::rotl_xor<13>(prev_w, h);
      }
      asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
    }
  };

  auto run3 = [&]<int ST, bool DO_FOLD, bool HAS_NXT, bool HAS_ISS>(int ks) {
    constexpr int NXT_ST = (ST + 1) % 3;
    constexpr int ISS_ST = (ST + 2) % 3;
    if constexpr (HAS_ISS) issue_stage.template operator()<ISS_ST>(ks + 2);
    mma_step.template operator()<0, 1, true, false>(sbase + ST * STG_BYTES, 32u, 0);
    if constexpr (HAS_NXT) wait_stage.template operator()<NXT_ST, HAS_ISS>();
    mma_step.template operator()<1, 0, HAS_NXT, DO_FOLD>(sbase + NXT_ST * STG_BYTES, 0u, ks >> 1);
  };

  int ks = 0;
#pragma unroll 1
  for (; ks + 6 <= num_ks - 2; ks += 6) {
    run3.template operator()<0, false, true, true>(ks + 0);
    run3.template operator()<1, true,  true, true>(ks + 1);
    run3.template operator()<2, false, true, true>(ks + 2);
    run3.template operator()<0, true,  true, true>(ks + 3);
    run3.template operator()<1, false, true, true>(ks + 4);
    run3.template operator()<2, true,  true, true>(ks + 5);
  }
  while (ks < num_ks) {
    bool hn = (ks + 1 < num_ks), hi = (ks + 2 < num_ks);
    int st = ks % 3;
    auto d = [&]<int ST, bool DF>() {
      if (hi)      run3.template operator()<ST, DF, true, true>(ks);
      else if (hn) run3.template operator()<ST, DF, true, false>(ks);
      else         run3.template operator()<ST, DF, false, false>(ks);
    };
    if (st == 0) { if ((ks & 1) == 0) d.template operator()<0, false>(); else d.template operator()<0, true>(); }
    else if (st == 1) { if ((ks & 1) == 0) d.template operator()<1, false>(); else d.template operator()<1, true>(); }
    else { if ((ks & 1) == 0) d.template operator()<2, false>(); else d.template operator()<2, true>(); }
    ++ks;
  }

  uint32_t msg[16], key[8], digest[8];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(msg[j]) : "r"(tr_base + unsigned(j * NT * 4)));
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) key[j] = p.key_a[j];
  apx_b3::keyed_hash64(key, msg, digest);

  bool good = true;
#pragma unroll
  for (int j = 7; j >= 0; --j) {
    uint32_t tw = p.target_le[j];
    if (digest[j] > tw) { good = false; break; }
    if (digest[j] < tw) break;
  }
  if (good) {
    uint32_t g = uint32_t((((p.base_bm + bm) * W_ROWS + wr) * p.total_n_warps + bn2 * W_COLS + wc) * 32 + lane);
    uint32_t s = atomicAdd(p.hit_count, 1u);
    if (s < uint32_t(p.hit_cap)) p.hit_groups[s] = g;
  }
}

extern "C" __global__ __launch_bounds__(128, 2) void apx_direct_l2_v0(const __grid_constant__ ApxDirectL2Params p) {
  run_direct_128x128<false, 0>(p);
}

extern "C" __global__ __launch_bounds__(128, 2) void apx_direct_l2_v1(const __grid_constant__ ApxDirectL2Params p) {
#if APX_IS_SM120
  run_direct_128x128<true, 0>(p);
#else
  run_direct_128x128<false, 2>(p);
#endif
}

} // namespace apx_dl2
