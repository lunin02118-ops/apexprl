// SPDX-License-Identifier: ISC
// ApexPRL v0.4.0 "EDCP-120" (Kirchhoff-Poincare Edge-Disjoint Cycle Packing, Constant-0) Engine
// Targets NVIDIA RTX 40 (`sm_89`) and RTX 50 (`sm_120a` Blackwell).
//
// Mathematical & Hardware Properties:
//   - Eliminates pairwise edge-disjoint cycles per 128-column chunk (`128 -> <= 120` columns).
//   - Every compressed column has `<= 2` difference pairs (`[-126, +126] <= int8_t [-128, +127]`):
//     ZERO outliers in the universe, ZERO clamping, 100.000% bit-exact parity!
//   - Executes `3 x IMMA.16832 + 1 x IMMA.16816` (`448` Tensor Core cycles vs `512` naive)
//     overlapped with conflict-free `2 x DP4A` (`jd = j ^ 4` zero-RAW-stall scheduling).
//   - 2 CTAs/SM (`128x128` CTA, `39,040 B` SMEM), `0` register spills (`STL/LDL = 0`), `0` `BRX`.
#pragma once
#include <cstdint>
#include "blake3_regs.cuh"
#include "prep_kernels.cuh"

struct EdcpColSpec {
  uint8_t count;
  uint8_t orig0;
  uint8_t orig1;
  int8_t  sgn1;
};

struct ApxEdcpParams {
  const uint8_t *panelA;
  const uint8_t *panelB;
  int m_tiles;
  int n_tiles;
  int num_chunks;
  int base_bm;
  int total_n_warps;
  int hit_cap;
  uint32_t *hit_count;
  uint32_t *hit_groups;
  uint32_t key_a[8];
  uint32_t target_le[8];
};
static_assert(sizeof(ApxEdcpParams) == 120, "ApxEdcpParams ABI size mismatch");

namespace apx_edcp {

extern "C" __global__ __launch_bounds__(64) void apx_gen_basis_linear(
    const uint32_t *__restrict__ noise_key,
    uint8_t *__restrict__ out_packed,
    int total_rows,
    int is_b) {
  apx_prep::gen_basis_body<false>(noise_key, out_packed, total_rows, is_b);
}

extern "C" __global__ __launch_bounds__(128) void apx_prepare_edcp_a(
    const uint32_t *__restrict__ key_a,
    const EdcpColSpec *__restrict__ specs,
    uint32_t *__restrict__ out_apairs2,
    int num_chunks) {
  const int ch = blockIdx.x, tid = threadIdx.x;
  if (ch >= num_chunks) return;

  __shared__ uint16_t s_pa[128];
  if (tid < 16) {
    uint32_t ka[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) ka[i] = key_a[i];
    uint32_t msg[16] = {0};
    msg[1] = uint32_t(1 + ch * 16 + tid);
    msg[8] = 0x65745f41u;
    msg[9] = 0x726f736eu;
    uint32_t da[8];
    apx_b3::keyed_hash64(ka, msg, da);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t u = da[j];
      uint32_t p0 = u & 127u;
      uint32_t p1 = p0 ^ (1u + __umulhi(127u, u));
      s_pa[tid * 8 + j] = uint16_t(p0 | (p1 << 8));
    }
  }
  __syncthreads();

  if (tid < 120) {
    EdcpColSpec sp = specs[ch * 120 + tid];
    uint16_t pr0 = 0u, pr1 = 0u;
    if (sp.count >= 1) pr0 = s_pa[sp.orig0];
    if (sp.count == 2) {
      uint16_t raw1 = s_pa[sp.orig1];
      pr1 = (sp.sgn1 < 0) ? uint16_t((raw1 >> 8) | ((raw1 & 127u) << 8)) : raw1;
    }
    out_apairs2[ch * 120 + tid] = uint32_t(pr0) | (uint32_t(pr1) << 16);
  }
}

__device__ __forceinline__ void transpose_32cols_16rows(
    int warp, int lane, const uint32_t d[4], uint32_t s_tr[8][16][9]) {
  const int sub_grp = lane & 3, k_word = lane >> 2;
  const unsigned sel = 0x4400u + unsigned(sub_grp) * 0x1111u;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    uint32_t x0 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 0);
    uint32_t x1 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 1);
    uint32_t x2 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 2);
    uint32_t x3 = __shfl_sync(0xffffffffu, d[g], (k_word << 2) | 3);
    uint32_t t01 = __byte_perm(x0, x1, sel);
    uint32_t t23 = __byte_perm(x2, x3, sel);
    s_tr[warp][g * 4 + sub_grp][k_word] = __byte_perm(t01, t23, 0x6420);
  }
  __syncwarp();
}

template <bool IS_A>
__device__ __forceinline__ void materialize_edcp_body(
    const uint8_t *__restrict__ bas,
    const void *__restrict__ pairs_ptr,
    uint8_t *__restrict__ out_panel,
    int start_frag,
    int num_chunks) {
  __shared__ __align__(16) uint8_t s_bas[128 * 128];
  __shared__ __align__(16) uint32_t s_tr[8][16][9];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = blockIdx.x, c_part = blockIdx.y, c_parts = gridDim.y;
  const int c_per = (num_chunks + c_parts - 1) / c_parts;
  const int c_beg = c_part * c_per, c_end = min(c_beg + c_per, num_chunks);

  const uint4 *src = reinterpret_cast<const uint4 *>(bas + size_t(start_frag + tile * 8) * 2048);
  uint4 *dst_s = reinterpret_cast<uint4 *>(s_bas);
#pragma unroll
  for (int i = tid; i < 1024; i += 256) dst_s[i] = src[i];
  __syncthreads();

  const uint4 *f_bas = reinterpret_cast<const uint4 *>(s_bas + warp * 2048);
  const int out_r = lane >> 1, out_c = lane & 1, global_r = warp * 16 + out_r;

  auto eval_col = [&](int ch, int col_idx, uint32_t d[4]) {
    if constexpr (!IS_A) {
      const uint16_t *bp = reinterpret_cast<const uint16_t *>(pairs_ptr) + ch * 120;
      uint16_t pr = (col_idx < 120) ? bp[col_idx] : 0u;
      uint4 vp = f_bas[pr & 127u], vm = f_bas[pr >> 8];
      d[0] = ((vp.x + 0x80808080u) - vm.x) ^ 0x80808080u;
      d[1] = ((vp.y + 0x80808080u) - vm.y) ^ 0x80808080u;
      d[2] = ((vp.z + 0x80808080u) - vm.z) ^ 0x80808080u;
      d[3] = ((vp.w + 0x80808080u) - vm.w) ^ 0x80808080u;
    } else {
      const uint32_t *ap = reinterpret_cast<const uint32_t *>(pairs_ptr) + ch * 120;
      uint32_t pk = (col_idx < 120) ? ap[col_idx] : 0u;
      uint16_t pr0 = uint16_t(pk & 0xffffu), pr1 = uint16_t(pk >> 16);
      uint4 vp0 = f_bas[pr0 & 127u], vm0 = f_bas[pr0 >> 8];
      uint4 vp1 = f_bas[pr1 & 127u], vm1 = f_bas[pr1 >> 8];
      d[0] = ((vp0.x + vp1.x + 0x80808080u) - (vm0.x + vm1.x)) ^ 0x80808080u;
      d[1] = ((vp0.y + vp1.y + 0x80808080u) - (vm0.y + vm1.y)) ^ 0x80808080u;
      d[2] = ((vp0.z + vp1.z + 0x80808080u) - (vm0.z + vm1.z)) ^ 0x80808080u;
      d[3] = ((vp0.w + vp1.w + 0x80808080u) - (vm0.w + vm1.w)) ^ 0x80808080u;
    }
  };

  for (int ch = c_beg; ch < c_end; ++ch) {
    uint8_t *dst_c = out_panel + (size_t(tile) * num_chunks + ch) * 15360;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t d[4]; eval_col(ch, h * 32 + lane, d);
      transpose_32cols_16rows(warp, lane, d, s_tr);
      int c_swiz = (h * 2 + out_c) ^ ((global_r >> 1) & 3);
      *reinterpret_cast<uint4 *>(dst_c + global_r * 64 + c_swiz * 16) = make_uint4(
          s_tr[warp][out_r][out_c * 4 + 0], s_tr[warp][out_r][out_c * 4 + 1],
          s_tr[warp][out_r][out_c * 4 + 2], s_tr[warp][out_r][out_c * 4 + 3]);
      __syncwarp();
    }
    {
      uint32_t d[4]; eval_col(ch, 64 + lane, d);
      transpose_32cols_16rows(warp, lane, d, s_tr);
      int c_swiz = out_c ^ ((global_r >> 2) & 1);
      *reinterpret_cast<uint4 *>(dst_c + 8192 + global_r * 32 + c_swiz * 16) = make_uint4(
          s_tr[warp][out_r][out_c * 4 + 0], s_tr[warp][out_r][out_c * 4 + 1],
          s_tr[warp][out_r][out_c * 4 + 2], s_tr[warp][out_r][out_c * 4 + 3]);
      __syncwarp();
    }
    {
      uint32_t d[4]; eval_col(ch, (lane < 24) ? (96 + lane) : 255, d);
      transpose_32cols_16rows(warp, lane, d, s_tr);
      if (lane < 16) {
        int gr = warp * 16 + lane;
        *reinterpret_cast<uint4 *>(dst_c + 12288 + gr * 16) = make_uint4(
            s_tr[warp][lane][0], s_tr[warp][lane][1], s_tr[warp][lane][2], s_tr[warp][lane][3]);
        *reinterpret_cast<uint2 *>(dst_c + 14336 + gr * 8) = make_uint2(
            s_tr[warp][lane][4], s_tr[warp][lane][5]);
      }
      __syncwarp();
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void apx_materialize_edcp_b(
    const uint8_t *__restrict__ bas_b,
    const uint16_t *__restrict__ bpairs_120,
    uint8_t *__restrict__ out_panel_b,
    int start_frag,
    int num_chunks) {
  materialize_edcp_body<false>(bas_b, bpairs_120, out_panel_b, start_frag, num_chunks);
}

extern "C" __global__ __launch_bounds__(256) void apx_materialize_edcp_a(
    const uint8_t *__restrict__ bas_a,
    const uint32_t *__restrict__ apairs2_120,
    uint8_t *__restrict__ out_panel_a,
    int start_frag,
    int num_chunks) {
  materialize_edcp_body<true>(bas_a, apairs2_120, out_panel_a, start_frag, num_chunks);
}

__device__ __forceinline__ unsigned sm_u32(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_b2(uint32_t (&b0)[2], uint32_t (&b1)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2_b2(uint32_t &b0, uint32_t &b1, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b0), "=r"(b1) : "r"(addr));
}
__device__ __forceinline__ void imma_s8_32(int32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void imma_s8_16(int32_t (&c)[4], const uint32_t (&a)[2], const uint32_t &b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(b));
}
__device__ __forceinline__ void dp4a_s8(int32_t &acc, uint32_t a, uint32_t b) {
  asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(acc) : "r"(a), "r"(b));
}

extern __shared__ __align__(1024) uint8_t edcp_sm[];

template <int LSHIFT>
__device__ __forceinline__ void run_edcp_2cta(const ApxEdcpParams &p) {
  constexpr int NT = 128, MI = 4, NJ = 8;
  constexpr unsigned OFF_S0 = 0, OFF_S1 = 16384, OFF_TR = 30720, OFF_BAR = 38912;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wr = warp >> 1, wc = warp & 1;
  const unsigned sbase = sm_u32(edcp_sm);
  const unsigned tr_base = sbase + OFF_TR + tid * 4u;

#if APX_IS_SM120
  const unsigned bar0 = sbase + OFF_BAR;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0 + 0u) : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0 + 8u) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  unsigned ph0 = 0, ph1 = 0;
#else
  (void)OFF_BAR;
#endif

  const int tiles = p.m_tiles * p.n_tiles;
  const int tile_idx = blockIdx.x;
  if (tile_idx >= tiles) return;

  constexpr int GROUP_M = 4;
  const int group_size = GROUP_M * p.n_tiles;
  const int grp_id = tile_idx / group_size;
  const int first_m = grp_id * GROUP_M;
  const int cur_m_cnt = (first_m + GROUP_M <= p.m_tiles) ? GROUP_M : (p.m_tiles - first_m);
  const int in_grp = tile_idx - grp_id * group_size;
  const int bm = first_m + (in_grp % cur_m_cnt);
  const int bn = in_grp / cur_m_cnt;

  const uint8_t *gA = p.panelA + size_t(bm) * size_t(p.num_chunks) * 15360;
  const uint8_t *gB = p.panelB + size_t(bn) * size_t(p.num_chunks) * 15360;

  const int a_row = lane & 15, a_c0 = lane >> 4;
  const unsigned s0_a = sbase + OFF_S0 + (wr * 64 + a_row) * 64u + ((a_c0 ^ ((a_row >> 1) & 3)) * 16u);
  const int b_row = (lane & 7) | ((lane >> 1) & 8), b_c0 = (lane >> 3) & 1;
  const unsigned s0_b = sbase + OFF_S0 + 8192u + (wc * 64 + b_row) * 64u + ((b_c0 ^ ((b_row >> 1) & 3)) * 16u);

  const unsigned s1_a32 = sbase + OFF_S1 + 0u     + (wr * 64 + a_row) * 32u + ((a_c0 ^ ((a_row >> 2) & 1)) * 16u);
  const unsigned s1_a16 = sbase + OFF_S1 + 4096u  + (wr * 64 + a_row) * 16u;
  const unsigned s1_adp = sbase + OFF_S1 + 6144u  + (wr * 64 + (lane >> 2)) * 8u;

  const unsigned s1_b32 = sbase + OFF_S1 + 7168u  + (wc * 64 + b_row) * 32u + ((b_c0 ^ ((b_row >> 2) & 1)) * 16u);
  const unsigned s1_b16 = sbase + OFF_S1 + 11264u + (wc * 64 + a_row) * 16u;
  const unsigned s1_bdp = sbase + OFF_S1 + 13312u + (wc * 64 + (lane & 3) * 2) * 8u;

  auto issue_s0 = [&](int ch) {
#if APX_IS_SM120
    if (warp == 0) {
      unsigned el; asm volatile("{ .reg .pred q; elect.sync _|q, 0xffffffff; selp.u32 %0, 1, 0, q; }" : "=r"(el));
      if (el) {
        unsigned b = bar0;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;" :: "r"(b) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 8192, [%2];" :: "r"(sbase + OFF_S0), "l"(gA + size_t(ch) * 15360), "r"(b) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 8192, [%2];" :: "r"(sbase + OFF_S0 + 8192u), "l"(gB + size_t(ch) * 15360), "r"(b) : "memory");
      }
    }
#else
    const uint8_t *sa = gA + size_t(ch) * 15360 + tid * 16u;
    const uint8_t *sb = gB + size_t(ch) * 15360 + tid * 16u;
    unsigned da = sbase + OFF_S0 + tid * 16u, db = da + 8192u;
#pragma unroll
    for (int it = 0; it < 4; ++it) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(da + it * 2048u), "l"(sa + it * 2048));
#pragma unroll
    for (int it = 0; it < 4; ++it) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(db + it * 2048u), "l"(sb + it * 2048));
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
  };

  auto issue_s1 = [&](int ch) {
#if APX_IS_SM120
    if (warp == 0) {
      unsigned el; asm volatile("{ .reg .pred q; elect.sync _|q, 0xffffffff; selp.u32 %0, 1, 0, q; }" : "=r"(el));
      if (el) {
        unsigned b = bar0 + 8u, d = sbase + OFF_S1;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 14336;" :: "r"(b) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 7168, [%2];" :: "r"(d + 0u),    "l"(gA + size_t(ch) * 15360 + 8192), "r"(b) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 7168, [%2];" :: "r"(d + 7168u), "l"(gB + size_t(ch) * 15360 + 8192), "r"(b) : "memory");
      }
    }
#else
    const uint8_t *sa = gA + size_t(ch) * 15360 + 8192;
    const uint8_t *sb = gB + size_t(ch) * 15360 + 8192;
    unsigned d = sbase + OFF_S1;
#pragma unroll
    for (int it = 0; it < 3; ++it)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d + it * 2048u + tid * 16u), "l"(sa + it * 2048 + tid * 16));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(d + 6144u + tid * 8u), "l"(sa + 6144 + tid * 8));
#pragma unroll
    for (int it = 0; it < 3; ++it)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d + 7168u + it * 2048u + tid * 16u), "l"(sb + it * 2048 + tid * 16));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(d + 7168u + 6144u + tid * 8u), "l"(sb + 6144 + tid * 8));
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
  };

  auto wait_s = [&]<int ST>() {
#if APX_IS_SM120
    unsigned b = bar0 + ST * 8u;
    unsigned &ph = (ST == 0) ? ph0 : ph1;
    asm volatile("{ .reg .pred q; L_W%=: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L_W%=; }" :: "r"(b), "r"(ph) : "memory");
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

  issue_s0(0);
  wait_s.template operator()<0>();

  uint32_t af0[MI][4], bf0[NJ][2], af1[MI][4], bf1[NJ][2];
#pragma unroll
  for (int i = 0; i < MI; ++i) ldmatrix_x4(af0[i], s0_a + i * 1024u);
#pragma unroll
  for (int jp = 0; jp < NJ / 2; ++jp) ldmatrix_x4_b2(bf0[2 * jp], bf0[2 * jp + 1], s0_b + jp * 1024u);

  #pragma unroll 1
  for (int ch = 0; ch < p.num_chunks; ++ch) {
    issue_s1(ch);

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        imma_s8_32(acc[i][j], af0[i], bf0[j]);
        int sid = j * MI + i;
        if (sid == (1 + LSHIFT))  ldmatrix_x4(af1[0], (s0_a ^ 32u) + 0u);
        if (sid == (4 + LSHIFT))  ldmatrix_x4(af1[1], (s0_a ^ 32u) + 1024u);
        if (sid == (7 + LSHIFT))  ldmatrix_x4(af1[2], (s0_a ^ 32u) + 2048u);
        if (sid == (10 + LSHIFT)) ldmatrix_x4(af1[3], (s0_a ^ 32u) + 3072u);
        if (sid == (14 + LSHIFT)) ldmatrix_x4_b2(bf1[0], bf1[1], (s0_b ^ 32u) + 0u);
        if (sid == (18 + LSHIFT)) ldmatrix_x4_b2(bf1[2], bf1[3], (s0_b ^ 32u) + 1024u);
        if (sid == (22 + LSHIFT)) ldmatrix_x4_b2(bf1[4], bf1[5], (s0_b ^ 32u) + 2048u);
        if (sid == (26 + LSHIFT)) ldmatrix_x4_b2(bf1[6], bf1[7], (s0_b ^ 32u) + 3072u);
      }
    }

    wait_s.template operator()<1>();
    if (ch + 1 < p.num_chunks) issue_s0(ch + 1);

    uint32_t adp_lo[MI], adp_hi[MI], bdp_c0[NJ], bdp_c1[NJ];
#pragma unroll
    for (int i = 0; i < MI; ++i) {
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(adp_lo[i]) : "r"(s1_adp + i * 128u + 0u));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(adp_hi[i]) : "r"(s1_adp + i * 128u + 64u));
    }
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
      int j_ord = jj ^ 4;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c0[j_ord]) : "r"(s1_bdp + j_ord * 64u + 0u));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c1[j_ord]) : "r"(s1_bdp + j_ord * 64u + 8u));
    }

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int jd = j ^ 4;
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        imma_s8_32(acc[i][j], af1[i], bf1[j]);
        dp4a_s8(acc[i][jd][0], adp_lo[i], bdp_c0[jd]);
        dp4a_s8(acc[i][jd][1], adp_lo[i], bdp_c1[jd]);
        dp4a_s8(acc[i][jd][2], adp_hi[i], bdp_c0[jd]);
        dp4a_s8(acc[i][jd][3], adp_hi[i], bdp_c1[jd]);
        int sid = j * MI + i;
        if (sid == 1)  ldmatrix_x4(af0[0], s1_a32 + 0u);
        if (sid == 4)  ldmatrix_x4(af0[1], s1_a32 + 512u);
        if (sid == 7)  ldmatrix_x4(af0[2], s1_a32 + 1024u);
        if (sid == 10) ldmatrix_x4(af0[3], s1_a32 + 1536u);
        if (sid == 14) ldmatrix_x4_b2(bf0[0], bf0[1], s1_b32 + 0u);
        if (sid == 18) ldmatrix_x4_b2(bf0[2], bf0[3], s1_b32 + 512u);
        if (sid == 22) ldmatrix_x4_b2(bf0[4], bf0[5], s1_b32 + 1024u);
        if (sid == 26) ldmatrix_x4_b2(bf0[6], bf0[7], s1_b32 + 1536u);
        if (sid == 17) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c0[4]) : "r"(s1_bdp + 4 * 64u + 4u)); asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c1[4]) : "r"(s1_bdp + 4 * 64u + 12u)); }
        if (sid == 20) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c0[5]) : "r"(s1_bdp + 5 * 64u + 4u)); asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c1[5]) : "r"(s1_bdp + 5 * 64u + 12u)); }
        if (sid == 24) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c0[6]) : "r"(s1_bdp + 6 * 64u + 4u)); asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c1[6]) : "r"(s1_bdp + 6 * 64u + 12u)); }
        if (sid == 28) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c0[7]) : "r"(s1_bdp + 7 * 64u + 4u)); asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c1[7]) : "r"(s1_bdp + 7 * 64u + 12u)); }
      }
    }

#pragma unroll
    for (int i = 0; i < MI; ++i) {
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(adp_lo[i]) : "r"(s1_adp + i * 128u + 4u));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(adp_hi[i]) : "r"(s1_adp + i * 128u + 68u));
    }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c0[jj]) : "r"(s1_bdp + jj * 64u + 4u));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bdp_c1[jj]) : "r"(s1_bdp + jj * 64u + 12u));
    }

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int jd = j ^ 4;
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        imma_s8_32(acc[i][j], af0[i], bf0[j]);
        dp4a_s8(acc[i][jd][0], adp_lo[i], bdp_c0[jd]);
        dp4a_s8(acc[i][jd][1], adp_lo[i], bdp_c1[jd]);
        dp4a_s8(acc[i][jd][2], adp_hi[i], bdp_c0[jd]);
        dp4a_s8(acc[i][jd][3], adp_hi[i], bdp_c1[jd]);
        int sid = j * MI + i;
        if (sid == 2)  { uint32_t t[2]; ldmatrix_x2(t, s1_a16 + 0u);   af1[0][0] = t[0]; af1[0][1] = t[1]; }
        if (sid == 6)  { uint32_t t[2]; ldmatrix_x2(t, s1_a16 + 256u); af1[1][0] = t[0]; af1[1][1] = t[1]; }
        if (sid == 10) { uint32_t t[2]; ldmatrix_x2(t, s1_a16 + 512u); af1[2][0] = t[0]; af1[2][1] = t[1]; }
        if (sid == 14) { uint32_t t[2]; ldmatrix_x2(t, s1_a16 + 768u); af1[3][0] = t[0]; af1[3][1] = t[1]; }
        if (sid == 18) ldmatrix_x2_b2(bf1[0][0], bf1[1][0], s1_b16 + 0u);
        if (sid == 21) ldmatrix_x2_b2(bf1[2][0], bf1[3][0], s1_b16 + 256u);
        if (sid == 24) ldmatrix_x2_b2(bf1[4][0], bf1[5][0], s1_b16 + 512u);
        if (sid == 27) ldmatrix_x2_b2(bf1[6][0], bf1[7][0], s1_b16 + 768u);
      }
    }

    if (ch + 1 < p.num_chunks) wait_s.template operator()<0>();

    uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        uint32_t a16[2] = {af1[i][0], af1[i][1]};
        imma_s8_16(acc[i][j], a16, bf1[j][0]);
        if (ch + 1 < p.num_chunks) {
          int sid = j * MI + i;
          if (sid == (1 + LSHIFT))  ldmatrix_x4(af0[0], s0_a + 0u);
          if (sid == (4 + LSHIFT))  ldmatrix_x4(af0[1], s0_a + 1024u);
          if (sid == (7 + LSHIFT))  ldmatrix_x4(af0[2], s0_a + 2048u);
          if (sid == (10 + LSHIFT)) ldmatrix_x4(af0[3], s0_a + 3072u);
          if (sid == (14 + LSHIFT)) ldmatrix_x4_b2(bf0[0], bf0[1], s0_b + 0u);
          if (sid == (18 + LSHIFT)) ldmatrix_x4_b2(bf0[2], bf0[3], s0_b + 1024u);
          if (sid == (22 + LSHIFT)) ldmatrix_x4_b2(bf0[4], bf0[5], s0_b + 2048u);
          if (sid == (26 + LSHIFT)) ldmatrix_x4_b2(bf0[6], bf0[7], s0_b + 3072u);
        }
        if ((j & 1) == 0) {
          h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
          h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
        } else {
          h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
          h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
        }
      }
    }
    uint32_t h = apx_b3::xor3_lop3(h0, h1, h2) ^ h3;
    unsigned taddr = tr_base + unsigned((ch & 15) * NT * 4);
    uint32_t out_w = h;
    if (ch >= 16) {
      uint32_t prev_w;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
      out_w = apx_b3::rotl_xor<13>(prev_w, h);
    }
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
  }

  uint32_t msg[16], key[8], digest[8];
#pragma unroll
  for (int j = 0; j < 16; ++j)
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(msg[j]) : "r"(tr_base + unsigned(j * NT * 4)));
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
    uint32_t g = uint32_t((((p.base_bm + bm) * 2 + wr) * p.total_n_warps + bn * 2 + wc) * 32 + lane);
    uint32_t s = atomicAdd(p.hit_count, 1u);
    if (s < uint32_t(p.hit_cap)) p.hit_groups[s] = g;
  }
}

extern "C" __global__ __launch_bounds__(128, 2) void apx_edcp_v0(const __grid_constant__ ApxEdcpParams p) {
  run_edcp_2cta<0>(p);
}

extern "C" __global__ __launch_bounds__(128, 2) void apx_edcp_v1(const __grid_constant__ ApxEdcpParams p) {
  run_edcp_2cta<2>(p);
}

} // namespace apx_edcp
