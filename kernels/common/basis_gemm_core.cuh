// SPDX-License-Identifier: ISC
// Multi-instantiation template header (intentionally NO #pragma once: included once per variant namespace!).
// Supports both NVIDIA RTX 50-series Blackwell (`APX_IS_SM120 == 1`, TMA + ldmatrix.m16n16.x2.trans.b8)
// and NVIDIA RTX 40-series Ada Lovelace (`APX_IS_SM120 == 0`, cp.async.cg + ldmatrix.m8n8.x4.trans.b16 + prmt).
#include <cstdint>
#include <utility>
#include "blake3_regs.cuh"
#include "basis_params.cuh"

#ifndef APX_P0
#define APX_P0 5
#endif
#ifndef APX_P1
#define APX_P1 10
#endif
#ifndef APX_P2
#define APX_P2 12
#endif
#ifndef APX_P3
#define APX_P3 14
#endif
#ifndef APX_P4
#define APX_P4 25
#endif
#ifndef APX_P5
#define APX_P5 27
#endif
#ifndef APX_P6
#define APX_P6 29
#endif
#ifndef APX_P7
#define APX_P7 31
#endif
#ifndef APX_PL
#define APX_PL 20
#endif
#ifndef APX_MID
#define APX_MID 1
#endif

namespace APX_VARIANT_NS {

constexpr int P0 = APX_P0, P1 = APX_P1, P2 = APX_P2, P3 = APX_P3;
constexpr int P4 = APX_P4, P5 = APX_P5, P6 = APX_P6, P7 = APX_P7, PL = APX_PL;
constexpr bool MIDX = (APX_MID != 0);

constexpr int BM = 160, BN = 256, WM = 80, WN = 64;
constexpr int MI = WM / 16; // 5
constexpr int NJ = WN / 8;  // 8
constexpr int NT = 256;     // 8 warps
constexpr int FA = BM / 16; // 10
constexpr int FB = BN / 16; // 16
constexpr int FR = FA + FB; // 26
constexpr int RQ = (FR + 7) / 8; // 4

constexpr int BASA  = FA * 2048;      // 20,480 B
constexpr int BASB  = FB * 2048;      // 32,768 B
constexpr int STG   = FR * 512;       // 13,312 B
constexpr int TRACE = NT * 16 * 4;    // 16,384 B
constexpr int OFF_BA  = 0;
constexpr int OFF_BB  = BASA;
constexpr int OFF_STG = BASA + BASB;
constexpr int OFF_TR  = OFF_STG + 2 * STG;
constexpr int OFF_BAR = OFF_TR + TRACE;
constexpr int SMEM_BYTES = OFF_BAR + 64; // 96,320 B <= 100,352 B

static_assert(SMEM_BYTES <= 101376 - 1024, "SMEM exceeds architectural dynamic carveout");
static_assert(P0 < P1 && P2 >= 2 * MI && P4 >= 4 * MI && RQ == 4, "Invalid knob bounds");
static_assert(P1 <= P4 && P4 <= P6 && P2 <= P3 && P3 <= P5 && P5 <= P7 && P0 >= 0 && P7 < (NJ - 1) * MI && PL >= 0 && PL < (NJ - 1) * MI, "Invalid knob ordering");

__device__ __forceinline__ unsigned sm_u32(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint2 ld_pair_nc(const uint2 *p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

#if APX_IS_SM120
  #define APX_LDSM(addr, r) \
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory")
  #define APX_PLDSM(pv, addr, r) \
    asm volatile("{.reg .pred pp; setp.ne.u32 pp, %5, 0; @pp ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];}" \
                 : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]) : "r"(addr), "r"(pv) : "memory")
#else
  #define APX_LDSM(addr, r) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory")
  #define APX_PLDSM(pv, addr, r) \
    asm volatile("{.reg .pred pp; setp.ne.u32 pp, %5, 0; @pp ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];}" \
                 : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]) : "r"(addr), "r"(pv) : "memory")
#endif

#define APX_LDS4(a, r0, r1, r2, r3) \
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory")
#define APX_PLDS4(pv, a, r0, r1, r2, r3) \
  asm volatile("{.reg .pred pp; setp.ne.u32 pp, %5, 0; @pp ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];}" \
               : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3) : "r"(a), "r"(pv) : "memory")
#define APX_STS4(a, r0, r1, r2, r3) \
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory")
#define APX_PSTS4(pv, a, r0, r1, r2, r3) \
  asm volatile("{.reg .pred pp; setp.ne.u32 pp, %5, 0; @pp st.shared.v4.u32 [%0], {%1,%2,%3,%4};}" \
               ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(pv) : "memory")

__device__ __forceinline__ void imma_u8(int32_t *c, const uint32_t *a, const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void run_basis_gemm(const ApxBasisParams &p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t tidv = threadIdx.x;
  const int lane = tidv & 31;
  const int warp = tidv >> 5;
  const int wr = warp / (BN / WN);
  const int wc = warp % (BN / WN);

  const uint2 *pbase = p.pairs;
  const int tilesN = p.n / BN;
  const int tiles  = (p.sampled_m / BM) * tilesN;
  const int S      = p.k / 32;

  const unsigned bar  = sm_u32(sm + OFF_BAR);
  const unsigned stg0 = sm_u32(sm + OFF_STG);
  const unsigned tr0  = sm_u32(sm + OFF_TR);

#if APX_IS_SM120
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
#endif

  const int fOwnA = wr * MI + wc;
  const int fExtra = wr * MI + 4;
  const unsigned xq = (warp == 0 || warp == 5) ? 1u : 0u;
  const unsigned nxq = xq ^ 1u;

  unsigned rbase[RQ], roff[RQ];
#pragma unroll
  for (int q = 0; q < RQ; ++q) {
    const int f = (q == 0) ? fOwnA
                : (q == 1) ? (FA + wc * (NJ / 2) + wr * 2)
                : (q == 2) ? (FA + wc * (NJ / 2) + wr * 2 + 1)
                : ((warp == 0 || warp == 5) ? fExtra : FR);
    rbase[q] = (f < FA) ? (sm_u32(sm + OFF_BA) + f * 2048u)
                        : (sm_u32(sm + OFF_BB) + (f < FR ? (f - FA) : 0) * 2048u);
    roff[q] = f * 512u + lane * 16u;
  }

  auto ldp = [&](int idx) -> uint2 {
    return ld_pair_nc(pbase + idx * 32 + lane);
  };

  unsigned aoff[MI], boff[NJ / 2];
  #define APX_RB(q)  ((q) == 2 ? rbase[1] + 2048u : rbase[q])
  #define APX_RO(q)  ((q) == 2 ? roff[1] + 512u   : roff[q])
  #define APX_AO(k)  (aoff[k])
  #define APX_BO(jp) (((jp) & 1) ? boff[(jp) - 1] + 512u : boff[jp])

#pragma unroll
  for (int k = 0; k < MI; ++k) {
    aoff[k] = (wr * MI + (k == 0 ? wc : ((k - 1) < wc ? k - 1 : k))) * 512u + lane * 16u;
  }
#pragma unroll
  for (int jp = 0; jp < NJ / 2; ++jp) {
    boff[jp] = (FA + wc * (NJ / 2) + (jp < 2 ? wr * 2 + jp : (1 - wr) * 2 + jp - 2)) * 512u + lane * 16u;
  }

#if APX_IS_SM120
  unsigned phase = 0;
  auto issue_tma = [&](int t) {
    const int row_ = (t / tilesN) * BM, col_ = (t % tilesN) * BN;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(BASA + BASB) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(sm_u32(sm + OFF_BA)), "l"(p.basA + size_t(row_ / 16) * 2048), "r"(BASA), "r"(bar) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(sm_u32(sm + OFF_BB)), "l"(p.basB + size_t(col_ / 16) * 2048), "r"(BASB), "r"(bar) : "memory");
  };
#else
  auto load_basis_cpasync = [&](int t) {
    const int row_ = (t / tilesN) * BM, col_ = (t % tilesN) * BN;
    const uint8_t *srcA = p.basA + size_t(row_ / 16) * 2048;
    const uint8_t *srcB = p.basB + size_t(col_ / 16) * 2048;
    unsigned dstA = sm_u32(sm + OFF_BA) + threadIdx.x * 16u;
    unsigned dstB = sm_u32(sm + OFF_BB) + threadIdx.x * 16u;
#pragma unroll
    for (int it = 0; it < 5; ++it) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :: "r"(dstA + it * 4096u), "l"(srcA + it * 4096 + threadIdx.x * 16));
    }
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :: "r"(dstB + it * 4096u), "l"(srcB + it * 4096 + threadIdx.x * 16));
    }
    asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::: "memory");
  };
#endif

  if (threadIdx.x == 0) {
    asm volatile("st.shared.u32 [%0 + 36], 0;" :: "r"(bar) : "memory");
  }
  __syncthreads();

  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    // Poll abort_epoch every 16 waves (~3.5 ms preemption, 16x less PCIe traffic, 0 extra regs!)
    if (threadIdx.x == 0) {
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
    if (threadIdx.x == 0) issue_tma(tile);
    asm volatile("{ .reg .pred q; L_WAIT: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L_WAIT; }"
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
      if (q == 3) {
        APX_PLDSM(xq, APX_RB(q) + po, rr[slot][0]);
        APX_PLDSM(xq, APX_RB(q) + mo, rr[slot][1]);
      } else {
        APX_LDSM(APX_RB(q) + po, rr[slot][0]);
        APX_LDSM(APX_RB(q) + mo, rr[slot][1]);
      }
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
      const bool isA = (q == 0 || q == 3);
      ov[0] = a0; ov[1] = a1; ov[2] = a2; ov[3] = a3;
      if (q == 3) APX_PSTS4(xq, stg + APX_RO(q), a0, a1, a2, a3);
      else if (!isA) APX_STS4(stg + APX_RO(q), a0, a2, a1, a3);
      else APX_STS4(stg + APX_RO(q), a0, a1, a2, a3);
    };

    uint32_t af[2][MI][4], bf[NJ - 1][2], b7[2][2];

    {
      uint2 pr = __ldg(pbase + lane);
#pragma unroll
      for (int q = 0; q < RQ; q += 2) {
        ldsm2(pr, q, 0);
        ldsm2(pr, q + 1, 1);
        sts2(stg0, q, 0);
        sts2(stg0, q + 1, 1);
      }
    }
    uint2 prC = __ldg(pbase + 32 + lane);
    ldsm2(prC, 0, 0);
    ldsm2(prC, 1, 1);
    uint2 prN = __ldg(pbase + 64 + lane);
    uint2 prNN = __ldg(pbase + ((S > 3 ? 96 : 0) + lane));
    uint2 prL = prNN;

    asm volatile("bar.sync 0;" ::: "memory");

#pragma unroll
    for (int k = 0; k < MI; ++k) APX_LDS4(stg0 + APX_AO(k), af[0][k][0], af[0][k][1], af[0][k][2], af[0][k][3]);
#pragma unroll
    for (int jp = 0; jp < NJ / 2 - 1; ++jp) APX_LDS4(stg0 + APX_BO(jp), bf[2 * jp][0], bf[2 * jp][1], bf[2 * jp + 1][0], bf[2 * jp + 1][1]);
    APX_LDS4(stg0 + APX_BO(NJ / 2 - 1), bf[NJ - 2][0], bf[NJ - 2][1], b7[0][0], b7[0][1]);

    uint32_t hacc = 0;

    auto step = [&]<int PAR, int FOLD = 2, int FZ = 0>(
        const int s, const unsigned nxt, const bool doStore, const bool doIssue, const bool doLoad) {
      constexpr int Q = PAR, O = PAR ^ 1;
#pragma unroll
      for (int pp = 0; pp < (NJ - 1) * MI; ++pp) {
        const int j = pp / MI, i = pp % MI;
        if (FZ) hacc ^= uint32_t(acc[i][j][0]) ^ uint32_t(acc[i][j][1]) ^ uint32_t(acc[i][j][2]) ^ uint32_t(acc[i][j][3]);
        imma_u8(acc[i][j], af[Q][i], bf[j]);
        if (doStore) {
          if (pp == P0) { sts2(nxt, 0, 0); af[O][0][0] = ov[0]; af[O][0][1] = ov[1]; af[O][0][2] = ov[2]; af[O][0][3] = ov[3]; }
          if (pp == P1) ldsm2(prC, 2, 0);
          if (pp == P2) { sts2(nxt, 1, 1); bf[0][0] = ov[0]; bf[0][1] = ov[2]; bf[1][0] = ov[1]; bf[1][1] = ov[3]; }
          if (pp == P3) ldsm2(prC, 3, 1);
          if (pp == P4) { sts2(nxt, 2, 0); bf[2][0] = ov[0]; bf[2][1] = ov[2]; bf[3][0] = ov[1]; bf[3][1] = ov[3]; }
          if (pp == P5) { sts2(nxt, 3, 1); af[O][MI - 1][0] = ov[0]; af[O][MI - 1][1] = ov[1]; af[O][MI - 1][2] = ov[2]; af[O][MI - 1][3] = ov[3]; }
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
        if (FZ) hacc ^= uint32_t(acc[i][NJ - 1][0]) ^ uint32_t(acc[i][NJ - 1][1]) ^ uint32_t(acc[i][NJ - 1][2]) ^ uint32_t(acc[i][NJ - 1][3]);
        imma_u8(acc[i][NJ - 1], af[Q][i], b7[Q]);
        if (doLoad) {
          const int k = i + 1;
          if (k < MI) {
            if (k == MI - 1) APX_PLDS4(nxq, nxt + APX_AO(k), af[O][k][0], af[O][k][1], af[O][k][2], af[O][k][3]);
            else APX_LDS4(nxt + APX_AO(k), af[O][k][0], af[O][k][1], af[O][k][2], af[O][k][3]);
          } else {
            APX_LDS4(nxt + APX_BO(2), bf[4][0], bf[4][1], bf[5][0], bf[5][1]);
            APX_LDS4(nxt + APX_BO(3), bf[6][0], bf[6][1], b7[O][0], b7[O][1]);
          }
        }
      }
      if (FZ) {
        const int stp = (s >> 2) - 1, slot = stp & 15;
        const unsigned taddr = tr0 + unsigned(slot * NT + threadIdx.x) * 4u;
        uint32_t out_w = hacc;
        if (stp >= 16) {
          uint32_t prev_w;
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
          out_w = apx_b3::rotl_xor<13>(prev_w, hacc);
        }
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
        hacc = 0;
      }
      if (FOLD == 2 ? ((s & 3) == 3) : (FOLD == 1 && s == S - 1)) {
        uint32_t hx[NJ];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          uint32_t v[MI * 4];
#pragma unroll
          for (int i = 0; i < MI; ++i) {
            v[i * 4 + 0] = uint32_t(acc[i][j][0]);
            v[i * 4 + 1] = uint32_t(acc[i][j][1]);
            v[i * 4 + 2] = uint32_t(acc[i][j][2]);
            v[i * 4 + 3] = uint32_t(acc[i][j][3]);
          }
          uint32_t t = apx_b3::xor3_lop3(v[0], v[1], v[2]);
#pragma unroll
          for (int k = 3; k + 1 < MI * 4; k += 2) t = apx_b3::xor3_lop3(t, v[k], v[k + 1]);
          if ((MI * 4 - 3) % 2) t ^= v[MI * 4 - 1];
          hx[j] = t;
        }
        uint32_t h01 = hx[0] ^ hx[1];
        uint32_t h23 = hx[2] ^ hx[3];
        uint32_t h45 = hx[4] ^ hx[5];
        uint32_t h67 = hx[6] ^ hx[7];
        uint32_t h = apx_b3::xor3_lop3(h01, h23, h45) ^ h67;
        const int stp = s >> 2, slot = stp & 15;
        const unsigned taddr = tr0 + unsigned(slot * NT + threadIdx.x) * 4u;
        uint32_t out_w = h;
        if (stp >= 16) {
          uint32_t prev_w;
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
          out_w = apx_b3::rotl_xor<13>(prev_w, h);
        }
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
      }
    };

    const unsigned b0 = stg0, b1 = stg0 + STG;
    auto grp = [&]<bool FIRST, bool MID, int... k>(std::integer_sequence<int, k...>, const int s0, const int mode) {
      (step.template operator()<(k & 1), ((k & 3) == 3 && !(MID && MIDX)) ? 1 : 0, ((k & 3) == 0 && !(FIRST && k == 0)) ? 1 : 0>(
           s0 + k, (k & 1) ? b0 : b1, mode != 2 || s0 + k < S - 1, mode != 2 || s0 + k < S - 2, mode != 2 || s0 + k < S - 1),
       ...);
    };

    grp.template operator()<true, false>(std::make_integer_sequence<int, 4>{}, 0, 0);
#pragma unroll 1
    for (int s = 4; s < S - 4; s += 4) {
      grp.template operator()<false, true>(std::make_integer_sequence<int, 4>{}, s, 1);
    }
    grp.template operator()<false, false>(std::make_integer_sequence<int, 4>{}, S - 4, 2);

    uint32_t msg[16], key[8], digest[8];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(msg[j]) : "r"(tr0 + unsigned(j * NT + threadIdx.x) * 4u));
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) key[j] = p.key_a[j];
    apx_b3::keyed_hash64(key, msg, digest);

    const int bm = tile / tilesN;
    const int bn = tile % tilesN;
    const uint32_t group = uint32_t(((bm * 2 + wr) * (p.n / 64) + bn * 4 + wc) * 32 + lane);
    bool good = true;
#pragma unroll
    for (int j = 7; j >= 0; --j) {
      uint32_t tw = p.target_le[j];
      if (digest[j] > tw) { good = false; break; }
      if (digest[j] < tw) break;
    }
    if (good) {
      uint32_t slot_idx = atomicAdd(p.hit_count, 1u);
      if (slot_idx < uint32_t(p.hit_cap)) {
        p.hit_groups[slot_idx] = group;
      }
    }
  }

  #undef APX_RB
  #undef APX_RO
  #undef APX_AO
  #undef APX_BO
}

extern "C" __global__ __launch_bounds__(NT, 1) void APX_KERNEL_ENTRY(const __grid_constant__ ApxBasisParams p) {
  run_basis_gemm(p);
}

} // namespace APX_VARIANT_NS

#undef APX_LDSM
#undef APX_PLDSM
#undef APX_LDS4
#undef APX_PLDS4
#undef APX_STS4
#undef APX_PSTS4
#undef APX_P0
#undef APX_P1
#undef APX_P2
#undef APX_P3
#undef APX_P4
#undef APX_P5
#undef APX_P6
#undef APX_P7
#undef APX_PL
#undef APX_MID
