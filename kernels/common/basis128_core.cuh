// SPDX-License-Identifier: ISC
// ApexPRL v0.5.0 "Uniform-3 Persistent-A Unpredicated Basis-128x256" On-Chip L2-Pinned Engine
// Targets BOTH NVIDIA RTX 50 (sm_120a Blackwell) & RTX 40 (sm_89 Ada Lovelace).
//
// Key breakthroughs in v0.5.0:
//   1. Persistent per-SM `BASA` outer-M / inner-N tile schedule (`0` `BASA` reloads across all `tilesN`!).
//   2. 100% Unpredicated 4-step inner K-loop (`0` `doStore`/`doIssue`/`doLoad` runtime predicates,
//      `-25%` SASS code size, `215` regs on `sm_89` / `224` regs on `sm_120a`, STRICT `STL/LDL = 0`, `PRMT = 0` on `sm_120a`).
//   3. Zero-overhead tile-register abort polling (`(tn & 15) == 15`, eliminating 15/16 per-tile barriers).
//   4. Supports exact `SM_COUNT`-aligned `M = k * SM_COUNT * 128 + 16` (`100%` wave occupancy, `4.9 MiB` L2 working set).
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
  #define B128_LDSM2048(addr, r) \
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4 + 2048];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory")
#else
  #define B128_LDSM(addr, r) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory")
  #define B128_LDSM2048(addr, r) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4 + 2048];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory")
#endif

#define B128_LDS4(a, r0, r1, r2, r3) \
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory")
#define B128_STS4(a, r0, r1, r2, r3) \
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory")

__device__ __forceinline__ void imma_u8(int32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
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
  const int tilesM = p.sampled_m / BM;
  const int tilesN = p.n / BN;
  const int S      = p.k / 32;
  const unsigned sb = sm_u32(sm);

  const int fOwnA = wr * 4 + wc;
  const int fB0   = wc * 4 + wr * 2;
  const int fBo   = wc * 4 + (1 - wr) * 2;
  const unsigned a_rbase0 = sb + unsigned(fOwnA * 2048);
  const unsigned b_rbase1 = sb + unsigned(OFF_BB + fB0 * 2048);
  const unsigned a_wroff0 = unsigned(OFF_STG + fOwnA * 512 + lane * 16);
  const unsigned b_wroff1 = unsigned(OFF_STG + (FA + fB0) * 512 + lane * 16);
  const unsigned rd_bo    = unsigned(OFF_STG + (FA + fBo) * 512 + lane * 16);

  unsigned aoff[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    int src_wc = (k < wc) ? k : (k + 1);
    aoff[k] = unsigned(OFF_STG + (wr * 4 + src_wc) * 512 + lane * 16);
  }

  for (int tm = int(blockIdx.x); tm < tilesM; tm += int(gridDim.x)) {
    const int row_ = tm * BM;
#if APX_IS_SM120
    if (tidv == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0 + 90112], 1;" ::"r"(sb) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == 0) {
      unsigned el;
      asm volatile("{ .reg .pred q; elect.sync _|q, 0xffffffff; selp.u32 %0, 1, 0, q; }" : "=r"(el));
      if (el) {
        unsigned bar_a = sb + OFF_BAR;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_a), "r"(BASA) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(sb + OFF_BA), "l"(p.basA + size_t(row_ / 16) * 2048), "r"(BASA), "r"(bar_a) : "memory");
      }
    }
    asm volatile("{ .reg .pred q; L_BA%=: mbarrier.try_wait.parity.shared::cta.b64 q, [%0 + 90112], 0; @!q bra L_BA%=; }"
                 ::"r"(sb) : "memory");
    __syncthreads();
#else
    {
      const uint8_t *srcA = p.basA + size_t(row_ / 16) * 2048;
      unsigned dstA = sb + OFF_BA + tidv * 16u;
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(dstA + it * 4096u), "l"(srcA + it * 4096 + tidv * 16));
      }
      asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::: "memory");
      __syncthreads();
    }
#endif

    for (int tn = 0; tn < tilesN; ++tn) {
      // Zero-overhead abort check every 16 N-tiles (~20 ms preemption on RTX 4060 / ~11 ms on RTX 5070 Ti!)
      if (((tn & 15) == 15) && p.abort_epoch_ptr != nullptr) {
        if (tidv == 0) {
          uint32_t cur_ep;
          asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(cur_ep) : "l"(p.abort_epoch_ptr));
          asm volatile("st.shared.u32 [%0 + 90144], %1;" :: "r"(sb), "r"((cur_ep != p.expected_epoch) ? 1u : 0u) : "memory");
        }
        __syncthreads();
        uint32_t should_stop;
        asm volatile("ld.shared.u32 %0, [%1 + 90144];" : "=r"(should_stop) : "r"(sb));
        if (should_stop != 0u) return;
      }

      const int col_ = tn * BN;
#if APX_IS_SM120
      if (tidv == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0 + 90112], 1;" ::"r"(sb) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      __syncthreads();
      if (warp == 0) {
        unsigned el;
        asm volatile("{ .reg .pred q; elect.sync _|q, 0xffffffff; selp.u32 %0, 1, 0, q; }" : "=r"(el));
        if (el) {
          unsigned bar_b = sb + OFF_BAR;
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_b), "r"(BASB) : "memory");
          asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       ::"r"(sb + OFF_BB), "l"(p.basB + size_t(col_ / 16) * 2048), "r"(BASB), "r"(bar_b) : "memory");
        }
      }
      asm volatile("{ .reg .pred q; L_BB%=: mbarrier.try_wait.parity.shared::cta.b64 q, [%0 + 90112], 0; @!q bra L_BB%=; }"
                   ::"r"(sb) : "memory");
      __syncthreads();
#else
      {
        const uint8_t *srcB = p.basB + size_t(col_ / 16) * 2048;
        unsigned dstB = sb + OFF_BB + tidv * 16u;
#pragma unroll
        for (int it = 0; it < 8; ++it) {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                       :: "r"(dstB + it * 4096u), "l"(srcB + it * 4096 + tidv * 16));
        }
        asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
      }
#endif

      int32_t acc[MI][NJ][4];
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0;

      uint32_t rr[2][2][4], ov[4];
      uint32_t af[2][MI][4], bf[NJ - 1][2], b7[2][2];

      auto ldsm2 = [&](uint2 pr, int q, int sl) {
        const unsigned pw = (q == 0) ? pr.x : pr.y;
        const unsigned po = pw & 0xffffu, mo = pw >> 16;
        if (q == 0)      { B128_LDSM(a_rbase0 + po, rr[sl][0]);     B128_LDSM(a_rbase0 + mo, rr[sl][1]); }
        else if (q == 1) { B128_LDSM(b_rbase1 + po, rr[sl][0]);     B128_LDSM(b_rbase1 + mo, rr[sl][1]); }
        else             { B128_LDSM2048(b_rbase1 + po, rr[sl][0]); B128_LDSM2048(b_rbase1 + mo, rr[sl][1]); }
      };

      auto sts2 = [&](unsigned stg_bias, int q, int sl) {
        uint32_t d0 = rr[sl][0][0] + 0x40404040u - rr[sl][1][0];
        uint32_t d1 = rr[sl][0][1] + 0x40404040u - rr[sl][1][1];
        uint32_t d2 = rr[sl][0][2] + 0x40404040u - rr[sl][1][2];
        uint32_t d3 = rr[sl][0][3] + 0x40404040u - rr[sl][1][3];
#if APX_IS_SM120
        uint32_t a0 = d0, a1 = d1, a2 = d2, a3 = d3;
#else
        uint32_t a0 = __byte_perm(d0, d1, 0x6420);
        uint32_t a1 = __byte_perm(d0, d1, 0x7531);
        uint32_t a2 = __byte_perm(d2, d3, 0x6420);
        uint32_t a3 = __byte_perm(d2, d3, 0x7531);
#endif
        ov[0] = a0; ov[1] = a1; ov[2] = a2; ov[3] = a3;
        if (q == 0)      B128_STS4(stg_bias + a_wroff0,        a0, a1, a2, a3);
        else if (q == 1) B128_STS4(stg_bias + b_wroff1,        a0, a2, a1, a3);
        else             B128_STS4(stg_bias + b_wroff1 + 512u, a0, a2, a1, a3);
      };

      {
        uint2 pr0 = ld_pair_nc(pbase + lane);
        ldsm2(pr0, 0, 0); ldsm2(pr0, 1, 1);
        sts2(sb, 0, 0); af[0][0][0] = ov[0]; af[0][0][1] = ov[1]; af[0][0][2] = ov[2]; af[0][0][3] = ov[3];
        sts2(sb, 1, 1); bf[0][0] = ov[0]; bf[0][1] = ov[2]; bf[1][0] = ov[1]; bf[1][1] = ov[3];
        ldsm2(pr0, 2, 0);
        sts2(sb, 2, 0); bf[2][0] = ov[0]; bf[2][1] = ov[2]; bf[3][0] = ov[1]; bf[3][1] = ov[3];
      }
      uint2 prC  = ld_pair_nc(pbase + 32 + lane);
      ldsm2(prC, 0, 0); ldsm2(prC, 1, 1);
      uint2 prN  = ld_pair_nc(pbase + 64 + lane);
      uint2 prNN = ld_pair_nc(pbase + ((S > 3 ? 96 : 0) + lane));

      asm volatile("bar.sync 0;" ::: "memory");

#pragma unroll
      for (int k = 0; k < 3; ++k) B128_LDS4(sb + aoff[k], af[0][k + 1][0], af[0][k + 1][1], af[0][k + 1][2], af[0][k + 1][3]);
      B128_LDS4(sb + rd_bo,        bf[4][0], bf[4][1], bf[5][0], bf[5][1]);
      B128_LDS4(sb + rd_bo + 512u, bf[6][0], bf[6][1], b7[0][0], b7[0][1]);

      auto write_tr_slot = [&](int ch_idx, uint32_t hval) {
        const unsigned taddr = sb + OFF_TR + unsigned(((ch_idx & 15) * NT + tidv) * 4);
        uint32_t out_w = hval;
        if (ch_idx >= 16) {
          uint32_t prev_w;
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(prev_w) : "r"(taddr));
          out_w = apx_b3::rotl_xor<13>(prev_w, hval);
        }
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(taddr), "r"(out_w) : "memory");
      };

      // 100% Unpredicated 4-step body (wrapping `(s + 4) & (S - 1)` safely on tail steps!)
      auto step = [&]<int PAR, int FZ>(const int s) -> uint32_t {
        constexpr int Q = PAR, O = PAR ^ 1;
        const unsigned nxt = sb + unsigned(O) * STAGE;
        uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
        uint2 prL = prNN;
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
          if (pp == P0) { sts2(nxt, 0, 0); af[O][0][0] = ov[0]; af[O][0][1] = ov[1]; af[O][0][2] = ov[2]; af[O][0][3] = ov[3]; }
          if (pp == P1) ldsm2(prC, 2, 0);
          if (pp == P2) { sts2(nxt, 1, 1); bf[0][0] = ov[0]; bf[0][1] = ov[2]; bf[1][0] = ov[1]; bf[1][1] = ov[3]; }
          if (pp == P4) { sts2(nxt, 2, 0); bf[2][0] = ov[0]; bf[2][1] = ov[2]; bf[3][0] = ov[1]; bf[3][1] = ov[3]; }
          if (pp == P6) ldsm2(prN, 0, 0);
          if (pp == P7) ldsm2(prN, 1, 1);
          if (pp == PL) prL = ld_pair_nc(pbase + ((s + 4) & (S - 1)) * 32 + lane);
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
          if (i < 3) {
            B128_LDS4(nxt + aoff[i], af[O][i + 1][0], af[O][i + 1][1], af[O][i + 1][2], af[O][i + 1][3]);
          } else {
            B128_LDS4(nxt + rd_bo,        bf[4][0], bf[4][1], bf[5][0], bf[5][1]);
            B128_LDS4(nxt + rd_bo + 512u, bf[6][0], bf[6][1], b7[O][0], b7[O][1]);
          }
        }
        if constexpr (FZ != 0) return apx_b3::xor3_lop3(h0, h1, h2) ^ h3;
        else return 0u;
      };

      const int n_chunks = S >> 2;
#pragma unroll U_INNER
      for (int ch = 0; ch < n_chunks; ++ch) {
        const int s0 = ch << 2;
        uint32_t h = step.template operator()<0, 1>(s0 + 0);
        if (ch > 0) write_tr_slot(ch - 1, h);
        (void)step.template operator()<1, 0>(s0 + 1);
        (void)step.template operator()<0, 0>(s0 + 2);
        (void)step.template operator()<1, 0>(s0 + 3);
      }

      {
        uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
          for (int j = 0; j < NJ; j += 2) {
            h0 = apx_b3::xor3_lop3(h0, uint32_t(acc[i][j][0]),     uint32_t(acc[i][j][1]));
            h1 = apx_b3::xor3_lop3(h1, uint32_t(acc[i][j][2]),     uint32_t(acc[i][j][3]));
            h2 = apx_b3::xor3_lop3(h2, uint32_t(acc[i][j + 1][0]), uint32_t(acc[i][j + 1][1]));
            h3 = apx_b3::xor3_lop3(h3, uint32_t(acc[i][j + 1][2]), uint32_t(acc[i][j + 1][3]));
          }
        write_tr_slot(n_chunks - 1, apx_b3::xor3_lop3(h0, h1, h2) ^ h3);
      }

      uint32_t msg[16], key[8], digest[8];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(msg[j]) : "r"(sb + OFF_TR + unsigned((j * NT + tidv) * 4)));
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
        const int mt0 = tm * 2 + wr;
        const int nt0 = tn * 4 + wc;
        const uint32_t group = uint32_t((mt0 * (p.n / 64) + nt0) * 32 + lane);
        const uint32_t slot_idx = atomicAdd(p.hit_count, 1u);
        if (slot_idx < uint32_t(p.hit_cap)) {
          p.hit_groups[slot_idx] = group;
        }
      }
    }
  }
}

} // namespace apx_b128

#undef B128_LDSM
#undef B128_LDSM2048
#undef B128_LDS4
#undef B128_STS4
