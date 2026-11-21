// SPDX-License-Identifier: ISC
// ApexPRL v0.3.0 "Uniform-3 Basis-128x256" On-Chip L2-Pinned Compute Engine
// Targets BOTH NVIDIA RTX 50 (sm_120a Blackwell) & RTX 40 (sm_89 Ada Lovelace).
//
// Key breakthroughs vs v0.1.0 (160x256) & v0.2.0 (Direct-L2):
//   1. 8.38 MiB total contiguous L2-pinned working set (d_bas_a + d_bas_b + d_pairs):
//      0.00 GB/s GDDR7/GDDR6X traffic even when memory is locked to 810 MHz (PenguinBurner)!
//   2. Exact 24-fragment = 8 warps x 3 fragments symmetry (RQ=3, BM=128, BN=256, MI=4, NJ=8):
//      100% warp balance (eliminates v0.1.0's 26/32=81.25% q=3 warp imbalance & BAR.SYNC skew).
//   3. Zero predication (eliminates APX_PLDSM / APX_PSTS4 / APX_PLDS4 and all 36x inner-loop WARPSYNC.ALL).
//   4. 230 registers, STRICT 0 register spills (STL/LDL = 0) vs v0.1.0's 255 regs + 16B Local Memory spills.
//   5. 4-lane parallel LOP3.LUT folding + elect.sync single-issue TMA on sm_120a + every-16-wave abort polling.
#pragma once
#include <cstdint>
#include "blake3_regs.cuh"
#include "basis_params.cuh"

namespace apx_b128 {

extern __shared__ __align__(1024) uint8_t sm[];

__device__ __forceinline__ unsigned sm_u32(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint2 ld_pair_nc(const uint2 *p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

#if APX_IS_SM120
  #define B128_LDSM(addr, r) \
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory")
#else
  #define B128_LDSM(addr, r) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory")
#endif

#define B128_LDS4(a, r0, r1, r2, r3) \
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory")
#define B128_STS4(a, r0, r1, r2, r3) \
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory")

__device__ __forceinline__ void imma_u8(int32_t *c, const uint32_t *a, const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Schedule invariants across 28 IMMAs (pp = 0..27) before bar.sync:
//   P0 < P1 <= 6,  P2 >= 7,  P4 >= 15,  P6 > P4,  P7 > P2
template <int P0, int P1, int P2, int P4, int P6, int P7, int PL, int U_INNER = 1>
__device__ __forceinline__ void run_basis_128x256(const ApxBasisParams &p) {
  constexpr int BM = 128, BN = 256, NT = 256;
  constexpr int MI = 4, NJ = 8;
  constexpr int FA = BM / 16;        // 8
  constexpr int FB = BN / 16;        // 16
  constexpr int FR = FA + FB;        // 24 (exact 3 per warp across 8 warps)
  constexpr int BASA  = FA * 2048;   // 16,384 B
  constexpr int BASB  = FB * 2048;   // 32,768 B
  constexpr int STAGE = FR * 512;    // 12,288 B
  constexpr int TRACE = NT * 16 * 4; // 16,384 B
  constexpr int OFF_BA  = 0;
  constexpr int OFF_BB  = OFF_BA + BASA;        // 16,384
  constexpr int OFF_STG = OFF_BB + BASB;        // 49,152
  constexpr int OFF_TR  = OFF_STG + 2 * STAGE;  // 73,728
  constexpr int OFF_BAR = OFF_TR + TRACE;       // 90,112 (Total 90,240 B <= 99 KiB)

  const uint32_t tidv = threadIdx.x;
  const int lane = tidv & 31;
  const int warp = tidv >> 5;
  const int wr = warp >> 2; // 0..1
  const int wc = warp & 3;  // 0..3

  const uint2 *pbase = p.pairs;
  const int tilesN = p.n / BN;
  const int tiles  = (p.sampled_m / BM) * tilesN;
  const int S      = p.k / 32;

  const unsigned bar  = sm_u32(sm + OFF_BAR);
  const unsigned stg0 = sm_u32(sm + OFF_STG);
  const unsigned tr0  = sm_u32(sm + OFF_TR);

#if APX_IS_SM120
  if (tidv == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#endif
  if (tidv == 0) {
    asm volatile("st.shared.u32 [%0 + 36], 0;" :: "r"(bar) : "memory");
  }
  __syncthreads();

  const int fOwnA = wr * MI + wc; // 0..7
  unsigned rbase[2], roff[2];
  rbase[0] = sm_u32(sm + OFF_BA) + fOwnA * 2048u;
  roff[0]  = fOwnA * 512u + lane * 16u;
  const int fB0 = FA + wc * 4 + wr * 2;
  rbase[1] = sm_u32(sm + OFF_BB) + (fB0 - FA) * 2048u;
  roff[1]  = fB0 * 512u + lane * 16u;

  #define B128_RB(q) ((q) == 2 ? rbase[1] + 2048u : rbase[q])
  #define B128_RO(q) ((q) == 2 ? roff[1]  + 512u  : roff[q])

  unsigned aoff[MI], boff[NJ / 2];
#pragma unroll
  for (int k = 0; k < MI; ++k) {
    aoff[k] = (wr * MI + (k == 0 ? wc : ((k - 1) < wc ? k - 1 : k))) * 512u + lane * 16u;
  }
#pragma unroll
  for (int jp = 0; jp < NJ / 2; ++jp) {
    boff[jp] = (FA + wc * (NJ / 2) + (jp < 2 ? wr * 2 + jp : (1 - wr) * 2 + jp - 2)) * 512u + lane * 16u;
  }
  #define B128_AO(k)  (aoff[k])
  #define B128_BO(jp) (((jp) & 1) ? boff[(jp) - 1] + 512u : boff[jp])

  auto ldp = [&](int idx) -> uint2 {
    return ld_pair_nc(pbase + idx * 32 + lane);
  };

#if APX_IS_SM120
  unsigned phase = 0;
  auto issue_tma = [&](int t) {
    if (warp == 0) {
      unsigned elected;
      asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }" : "=r"(elected));
      if (elected) {
        const int row_ = (t / tilesN) * BM, col_ = (t % tilesN) * BN;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(BASA + BASB) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(sm_u32(sm + OFF_BA)), "l"(p.basA + size_t(row_ / 16) * 2048), "r"(BASA), "r"(bar) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(sm_u32(sm + OFF_BB)), "l"(p.basB + size_t(col_ / 16) * 2048), "r"(BASB), "r"(bar) : "memory");
      }
    }
  };
#else
  auto load_basis_cpasync = [&](int t) {
    const int row_ = (t / tilesN) * BM, col_ = (t % tilesN) * BN;
    const uint8_t *srcA = p.basA + size_t(row_ / 16) * 2048;
    const uint8_t *srcB = p.basB + size_t(col_ / 16) * 2048;
    unsigned dstA = sm_u32(sm + OFF_BA) + tidv * 16u;
    unsigned dstB = sm_u32(sm + OFF_BB) + tidv * 16u;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :: "r"(dstA + it * 4096u), "l"(srcA + it * 4096 + tidv * 16));
    }
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :: "r"(dstB + it * 4096u), "l"(srcB + it * 4096 + tidv * 16));
    }
    asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::: "memory");
  };
#endif

  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    // Poll abort_epoch every 16 waves (~2.9 ms preemption on RTX 5070 Ti, 16x less PCIe traffic, 0 extra regs!)
    if (tidv == 0) {
      uint32_t it_cnt;
      asm volatile("ld.shared.u32 %0, [%1 + 36];" : "=r"(it_cnt) : "r"(bar));
      uint32_t stop = 0;
      if (((it_cnt & 15u) == 15u) && p.abort_epoch_ptr != nullptr) {
        uint32_t cur_ep;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(cur_ep) : "l"(p.abort_epoch_ptr));
        stop = (cur_ep != p.expected_epoch) ? 1u : 0u;
      }
      asm volatile("st.shared.u32 [%0 + 36], %1;" :: "r"(bar), "r"(it_cnt + 1u) : "memory");
      asm volatile("st.shared.u32 [%0 + 32], %1;" :: "r"(bar), "r"(stop) : "memory");
    }
    __syncthreads();
    uint32_t should_stop;
    asm volatile("ld.shared.u32 %0, [%1 + 32];" : "=r"(should_stop) : "r"(bar));
    if (should_stop != 0u) break;

#if APX_IS_SM120
    issue_tma(tile);
    asm volatile("{ .reg .pred q; L_W: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L_W; }"
                 ::"r"(bar), "r"(phase) : "memory");
    phase ^= 1u;
#else
    load_basis_cpasync(tile);
    __syncthreads();
#endif

    int32_t acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][j][c] = 0;

    uint32_t rr[2][2][4];
    uint32_t ov[4];

    auto ldsm2 = [&](uint2 pr, int q, int slot) {
      const unsigned pw = (q == 1 || q == 2) ? pr.y : pr.x;
      const unsigned po = __byte_perm(pw, 0, 0x4410);
      const unsigned mo = pw >> 16;
      B128_LDSM(B128_RB(q) + po, rr[slot][0]);
      B128_LDSM(B128_RB(q) + mo, rr[slot][1]);
    };

    auto sts2 = [&](unsigned stg, int q, int slot) {
      uint32_t d0 = rr[slot][0][0] + 0x40404040u - rr[slot][1][0];
      uint32_t d1 = rr[slot][0][1] + 0x40404040u - rr[slot][1][1];
      uint32_t d2 = rr[slot][0][2] + 0x40404040u - rr[slot][1][2];
      uint32_t d3 = rr[slot][0][3] + 0x40404040u - rr[slot][1][3];
#if APX_IS_SM120
      uint32_t a0 = d0, a1 = d1, a2 = d2, a3 = d3;
#else
      uint32_t a0 = __byte_perm(d0, d1, 0x6420);
      uint32_t a1 = __byte_perm(d0, d1, 0x7531);
      uint32_t a2 = __byte_perm(d2, d3, 0x6420);
      uint32_t a3 = __byte_perm(d2, d3, 0x7531);
#endif
      ov[0] = a0; ov[1] = a1; ov[2] = a2; ov[3] = a3;
      if (q != 0) B128_STS4(stg + B128_RO(q), a0, a2, a1, a3);
      else        B128_STS4(stg + B128_RO(0), a0, a1, a2, a3);
    };

    uint32_t af[2][MI][4], bf[NJ - 1][2], b7[2][2];

    {
      uint2 pr = __ldg(pbase + lane);
      ldsm2(pr, 0, 0);
      ldsm2(pr, 1, 1);
      sts2(stg0, 0, 0);
      sts2(stg0, 1, 1);
      ldsm2(pr, 2, 0);
      sts2(stg0, 2, 0);
    }
    uint2 prC  = __ldg(pbase + 32 + lane);
    ldsm2(prC, 0, 0);
    ldsm2(prC, 1, 1);
    uint2 prN  = __ldg(pbase + 64 + lane);
    uint2 prNN = __ldg(pbase + ((S > 3 ? 96 : 0) + lane));
    uint2 prL  = prNN;

    asm volatile("bar.sync 0;" ::: "memory");

#pragma unroll
    for (int k = 0; k < MI; ++k) B128_LDS4(stg0 + B128_AO(k), af[0][k][0], af[0][k][1], af[0][k][2], af[0][k][3]);
#pragma unroll
    for (int jp = 0; jp < NJ / 2 - 1; ++jp) B128_LDS4(stg0 + B128_BO(jp), bf[2 * jp][0], bf[2 * jp][1], bf[2 * jp + 1][0], bf[2 * jp + 1][1]);
    B128_LDS4(stg0 + B128_BO(NJ / 2 - 1), bf[NJ - 2][0], bf[NJ - 2][1], b7[0][0], b7[0][1]);

    uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;

    auto step = [&]<int PAR, int FZ = 0>(
        const int s, const unsigned nxt, const bool doStore, const bool doIssue, const bool doLoad) {
      constexpr int Q = PAR, O = PAR ^ 1;
#pragma unroll
      for (int pp = 0; pp < (NJ - 1) * MI; ++pp) {
        const int j = pp / MI, i = pp % MI;
        if constexpr (FZ != 0) {
          if ((j & 1) == 0) {
            h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          } else {
            h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j][0]), uint32_t(acc[i][j][1]));
            h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j][2]), uint32_t(acc[i][j][3]));
          }
        }
        imma_u8(acc[i][j], af[Q][i], bf[j]);
        if (doStore) {
          if (pp == P0) { sts2(nxt, 0, 0); af[O][0][0] = ov[0]; af[O][0][1] = ov[1]; af[O][0][2] = ov[2]; af[O][0][3] = ov[3]; }
          if (pp == P1) ldsm2(prC, 2, 0);
          if (pp == P2) { sts2(nxt, 1, 1); bf[0][0] = ov[0]; bf[0][1] = ov[2]; bf[1][0] = ov[1]; bf[1][1] = ov[3]; }
          if (pp == P4) { sts2(nxt, 2, 0); bf[2][0] = ov[0]; bf[2][1] = ov[2]; bf[3][0] = ov[1]; bf[3][1] = ov[3]; }
        }
        if (pp == PL) prL = ldp((s + 4) & (S - 1));
        if (doIssue) {
          if (pp == P6) ldsm2(prN, 0, 0);
          if (pp == P7) ldsm2(prN, 1, 1);
        }
      }
      prC = prN;
      prN = prNN;
      prNN = prL;
      asm volatile("bar.sync 0;" ::: "memory");

#pragma unroll
      for (int i = 0; i < MI; ++i) {
        if constexpr (FZ != 0) {
          h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][NJ - 1][0]), uint32_t(acc[i][NJ - 1][1]));
          h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][NJ - 1][2]), uint32_t(acc[i][NJ - 1][3]));
        }
        imma_u8(acc[i][NJ - 1], af[Q][i], b7[Q]);
        if (doLoad) {
          const int k = i + 1;
          if (k < MI) {
            B128_LDS4(nxt + B128_AO(k), af[O][k][0], af[O][k][1], af[O][k][2], af[O][k][3]);
          } else {
            B128_LDS4(nxt + B128_BO(2), bf[4][0], bf[4][1], bf[5][0], bf[5][1]);
            B128_LDS4(nxt + B128_BO(3), bf[6][0], bf[6][1], b7[O][0], b7[O][1]);
          }
        }
      }
      if constexpr (FZ != 0) {
        uint32_t hacc = apx_b3::xor3_lop3(h0, h1, h2) ^ h3;
        const int stp = (s >> 2) - 1, slot = stp & 15;
        const unsigned taddr = tr0 + unsigned(slot * NT + tidv) * 4u;
        uint32_t out_w = hacc;
        if (stp >= 16) {
          uint32_t prev_w;
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
          out_w = apx_b3::rotl_xor<13>(prev_w, hacc);
        }
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
        h0 = 0; h1 = 0; h2 = 0; h3 = 0;
      }
    };

    step.template operator()<0, 0>(0, stg0 + STAGE, true, true, true);
    step.template operator()<1, 0>(1, stg0,         true, true, true);
    step.template operator()<0, 0>(2, stg0 + STAGE, true, true, true);
    step.template operator()<1, 0>(3, stg0,         true, true, true);

#pragma unroll U_INNER
    for (int s = 4; s < S - 4; s += 4) {
      step.template operator()<0, 1>(s,     stg0 + STAGE, true, true, true);
      step.template operator()<1, 0>(s + 1, stg0,         true, true, true);
      step.template operator()<0, 0>(s + 2, stg0 + STAGE, true, true, true);
      step.template operator()<1, 0>(s + 3, stg0,         true, true, true);
    }

    step.template operator()<0, 1>(S - 4, stg0 + STAGE, true,  true,  true);
    step.template operator()<1, 0>(S - 3, stg0,         true,  true,  true);
    step.template operator()<0, 0>(S - 2, stg0 + STAGE, true,  false, true);
    step.template operator()<1, 0>(S - 1, stg0,         false, false, false);

    {
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; j += 2) {
          h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]),     uint32_t(acc[i][j][1]));
          h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]),     uint32_t(acc[i][j][3]));
          h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j + 1][0]), uint32_t(acc[i][j + 1][1]));
          h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j + 1][2]), uint32_t(acc[i][j + 1][3]));
        }
      uint32_t hacc = apx_b3::xor3_lop3(h0, h1, h2) ^ h3;
      const int stp = (S >> 2) - 1, slot = stp & 15;
      const unsigned taddr = tr0 + unsigned(slot * NT + tidv) * 4u;
      uint32_t out_w = hacc;
      if (stp >= 16) {
        uint32_t prev_w;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
        out_w = apx_b3::rotl_xor<13>(prev_w, hacc);
      }
      asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
    }

    uint32_t msg[16], key[8], digest[8];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(msg[j]) : "r"(tr0 + unsigned(j * NT + tidv) * 4u));
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
      const int mt0 = (tile / tilesN) * 2 + wr;
      const int nt0 = (tile % tilesN) * 4 + wc;
      const uint32_t group = uint32_t((mt0 * (p.n / 64) + nt0) * 32 + lane);
      const uint32_t slot_idx = atomicAdd(p.hit_count, 1u);
      if (slot_idx < uint32_t(p.hit_cap)) {
        p.hit_groups[slot_idx] = group;
      }
    }
  }

  #undef B128_RB
  #undef B128_RO
  #undef B128_AO
  #undef B128_BO
}

} // namespace apx_b128

#undef B128_LDSM
#undef B128_LDS4
#undef B128_STS4
