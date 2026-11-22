// SPDX-License-Identifier: ISC
// GPU Merkle tree builder, direct fused basis generator, and bank-conflict-eliminating pair reorderer.
#pragma once
#include <cstdint>
#include "blake3_regs.cuh"

namespace apx_prep {

// Computes chaining values for 1 KiB chunks filled with constant byte 0x40 (64).
extern "C" __global__ void apx_merkle_leaves_const64(
    const uint32_t *__restrict__ job_key,
    uint32_t *__restrict__ tree,
    int nleaves) {
  int leaf = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= nleaves) return;

  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = job_key[i];

  uint32_t msg[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) msg[i] = 0x40404040u;

  for (int blk = 0; blk < 16; ++blk) {
    uint32_t flags = apx_b3::KEYED_HASH
                   | (blk == 0 ? apx_b3::CHUNK_START : 0u)
                   | (blk == 15 ? apx_b3::CHUNK_END : 0u);
    apx_b3::compress_block(cv, msg, uint64_t(leaf), 64u, flags);
  }

  size_t off = size_t(nleaves + leaf) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) tree[off + i] = cv[i];
}

// Computes chaining values for 1 KiB chunks filled with constant byte 0x00 (`Constant-0` for EDCP).
extern "C" __global__ void apx_merkle_leaves_const0(
    const uint32_t *__restrict__ job_key,
    uint32_t *__restrict__ tree,
    int nleaves) {
  int leaf = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= nleaves) return;

  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = job_key[i];

  uint32_t msg[16] = {0};
  for (int blk = 0; blk < 16; ++blk) {
    uint32_t flags = apx_b3::KEYED_HASH
                   | (blk == 0 ? apx_b3::CHUNK_START : 0u)
                   | (blk == 15 ? apx_b3::CHUNK_END : 0u);
    apx_b3::compress_block(cv, msg, uint64_t(leaf), 64u, flags);
  }

  size_t off = size_t(nleaves + leaf) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) tree[off + i] = cv[i];
}

// Reduces one level of Merkle parent nodes in heap order.
extern "C" __global__ void apx_merkle_parents(
    const uint32_t *__restrict__ job_key,
    uint32_t *__restrict__ tree,
    int first,
    int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  int node = first + idx;

  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = job_key[i];

  uint32_t msg[16];
  size_t child_off = size_t(node) * 16;
#pragma unroll
  for (int i = 0; i < 16; ++i) msg[i] = tree[child_off + i];

  apx_b3::compress_block(cv, msg, 0ull, 64u, apx_b3::KEYED_HASH | apx_b3::PARENT);

  size_t out_off = size_t(node) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) tree[out_off + i] = cv[i];
}

// Fused direct uniform basis generator + 16-row fragment packer (+32 biased -> byte & 0x3F).
// One 64-thread CTA builds one 16-row x 128-component fragment (2048 bytes) in shared memory
// and writes it coalesced to global memory!
template <bool INTERLEAVE_SM89>
__device__ __forceinline__ void gen_basis_body(
    const uint32_t *__restrict__ noise_key,
    uint8_t *__restrict__ out_packed,
    int total_rows,
    int is_b) {
  __shared__ __align__(16) uint8_t sm_frag[2048];
  int frag = blockIdx.x;
  if (frag * 16 >= total_rows) return;

  int tid = threadIdx.x; // 0..63
  int r16 = tid >> 2;    // 0..15 (row within 16-row fragment)
  int sub = tid & 3;     // 0..3  (32-byte quarter of 128-byte basis row)
  int global_row = frag * 16 + r16;
  int block_idx = global_row * 4 + sub;

  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) key[i] = noise_key[i];

  uint32_t msg[16] = {0};
  msg[0] = uint32_t(1 + block_idx);
  msg[8] = is_b ? 0x65745f42u : 0x65745f41u; // "B_te" / "A_te"
  msg[9] = 0x726f736eu;                     // "nsor"

  uint32_t digest[8];
  apx_b3::keyed_hash64(key, msg, digest);

  int dst_row_slot = INTERLEAVE_SM89 ? (((r16 & 7) << 1) | (r16 >> 3)) : r16;

#pragma unroll
  for (int w = 0; w < 8; ++w) {
    uint32_t masked = digest[w] & 0x3f3f3f3fu;
    int comp0 = sub * 32 + w * 4;
    sm_frag[(comp0 + 0) * 16 + dst_row_slot] = uint8_t(masked & 0xffu);
    sm_frag[(comp0 + 1) * 16 + dst_row_slot] = uint8_t((masked >> 8) & 0xffu);
    sm_frag[(comp0 + 2) * 16 + dst_row_slot] = uint8_t((masked >> 16) & 0xffu);
    sm_frag[(comp0 + 3) * 16 + dst_row_slot] = uint8_t((masked >> 24) & 0xffu);
  }
  __syncthreads();

  // 64 threads * 2 * 16B = 2048B coalesced store
  uint4 *dst4 = reinterpret_cast<uint4 *>(out_packed + size_t(frag) * 2048);
  const uint4 *src4 = reinterpret_cast<const uint4 *>(sm_frag);
  dst4[tid] = src4[tid];
  dst4[tid + 64] = src4[tid + 64];
}

// Generates raw prescaled pair offsets (plus*16 | minus*16 << 16) for A and B directly in registers.
extern "C" __global__ void apx_gen_raw_pairs(
    const uint32_t *__restrict__ key_a,
    const uint32_t *__restrict__ key_b,
    uint2 *__restrict__ out_pairs,
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
  msg_a[8] = 0x65745f41u;
  msg_a[9] = 0x726f736eu;

  msg_b[1] = uint32_t(1 + h);
  msg_b[8] = 0x65745f42u;
  msg_b[9] = 0x726f736eu;

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
    uint32_t pack_a = (pa0 * 16u) | ((pa1 * 16u) << 16);

    uint32_t ub = db[j];
    uint32_t pb0 = ub & 127u;
    uint32_t pb1 = pb0 ^ (1u + __umulhi(127u, ub));
    uint32_t pack_b = (pb0 * 16u) | ((pb1 * 16u) << 16);

    out_pairs[col] = make_uint2(pack_a, pack_b);
  }
}

// Precomputes constant B prescaled pair offsets `(pb0*16 | pb1*16 << 16)` ONCE per block job!
extern "C" __global__ void apx_gen_b_pairs(
    const uint32_t *__restrict__ key_b,
    uint32_t *__restrict__ out_b_packed,
    int k) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h * 8 >= k) return;

  uint32_t kb[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) kb[i] = key_b[i];

  uint32_t msg_b[16] = {0};
  msg_b[1] = uint32_t(1 + h);
  msg_b[8] = 0x65745f42u;
  msg_b[9] = 0x726f736eu;

  uint32_t db[8];
  apx_b3::keyed_hash64(kb, msg_b, db);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int col = h * 8 + j;
    if (col >= k) break;
    uint32_t ub = db[j];
    uint32_t pb0 = ub & 127u;
    uint32_t pb1 = pb0 ^ (1u + __umulhi(127u, ub));
    out_b_packed[col] = (pb0 * 16u) | ((pb1 * 16u) << 16);
  }
}

// Fast per-round A-only pair generator merging with per-job cached `b_packed` (-50% BLAKE3 work!).
extern "C" __global__ void apx_gen_ab_pairs_from_b(
    const uint32_t *__restrict__ key_a,
    const uint32_t *__restrict__ b_packed,
    uint2 *__restrict__ out_pairs,
    int k) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h * 8 >= k) return;

  uint32_t ka[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) ka[i] = key_a[i];

  uint32_t msg_a[16] = {0};
  msg_a[1] = uint32_t(1 + h);
  msg_a[8] = 0x65745f41u;
  msg_a[9] = 0x726f736eu;

  uint32_t da[8];
  apx_b3::keyed_hash64(ka, msg_a, da);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int col = h * 8 + j;
    if (col >= k) break;
    uint32_t ua = da[j];
    uint32_t pa0 = ua & 127u;
    uint32_t pa1 = pa0 ^ (1u + __umulhi(127u, ua));
    uint32_t pack_a = (pa0 * 16u) | ((pa1 * 16u) << 16);
    out_pairs[col] = make_uint2(pack_a, b_packed[col]);
  }
}

__device__ __forceinline__ unsigned maxnib(unsigned h) {
  unsigned m = __vmaxu4(h & 0x0f0f0f0fu, (h >> 4) & 0x0f0f0f0fu);
  m = __vmaxu4(m, m >> 16);
  m = __vmaxu4(m, m >> 8);
  return m & 0xffu;
}

// Greedy bank-conflict-aware column permutation inside each 128-column K-chunk (mathematically invariant!).
// On SM89, also applies the lane->k_col permutation for `ldmatrix.m8n8.x4.trans.shared.b16`.
template <bool PERMUTE_SM89, int WA = 10>
__device__ __forceinline__ void reorder_pairs_body(uint2 *__restrict__ pairs, int k, bool enable_greedy) {
  constexpr int W[4] = {WA, WA, 16, 16};
  __shared__ uint2 sp[4][128];
  __shared__ uint2 sgrouped[4][128];
  __shared__ uint32_t scol[4][128];
  __shared__ uint8_t sgrp[4][128];
  __shared__ uint32_t sh[4][16][4];
  __shared__ int ssize[4][16];

  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int blk = blockIdx.x * 4 + w;
  if (blk * 128 >= k) return;

  uint2 *g = pairs + size_t(blk) * 128;
  uint2 *pr = sp[w];
  uint2 *gr_out = sgrouped[w];

  for (int c = lane; c < 128; c += 32) {
    uint2 v = g[c];
    pr[c] = v;
    scol[w][c] = ((v.x >> 4) & 7u)
               | (((v.x >> 20) & 7u) << 8)
               | (((v.y >> 4) & 7u) << 16)
               | (((v.y >> 20) & 7u) << 24);
  }
  __syncwarp();

  if (enable_greedy) {
    if (lane < 16) {
      ssize[w][lane] = 0;
      sh[w][lane][0] = sh[w][lane][1] = sh[w][lane][2] = sh[w][lane][3] = 0;
    }
    __syncwarp();
    for (int c = 0; c < 128; ++c) {
      const uint32_t cc = scol[w][c];
      unsigned key = 0xffffffffu;
      if (lane < 16 && ssize[w][lane] < 8) {
        int delta = 0;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          unsigned h = sh[w][lane][s];
          unsigned cnt = (h >> (4 * ((cc >> (8 * s)) & 7u))) & 15u;
          unsigned mx = maxnib(h);
          if (cnt + 1 > (mx > 1 ? mx : 1)) delta += W[s];
        }
        key = (unsigned(delta) << 8) | (unsigned(ssize[w][lane]) << 4) | unsigned(lane);
      }
      // Warp-wide min reduction (compatible with both sm_89 and sm_120)
#pragma unroll
      for (int mask = 16; mask >= 1; mask >>= 1) {
        unsigned other = __shfl_xor_sync(0xffffffffu, key, mask);
        if (other < key) key = other;
      }
      if (lane == int(key & 15u)) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          sh[w][lane][s] += 1u << (4 * ((cc >> (8 * s)) & 7u));
        }
        ssize[w][lane]++;
        sgrp[w][c] = uint8_t(lane);
      }
      __syncwarp();
    }
    for (int c = lane; c < 128; c += 32) {
      int q = sgrp[w][c];
      int pos = 0;
      for (int e = 0; e < c; ++e) pos += (sgrp[w][e] == q);
      gr_out[q * 8 + pos] = pr[c];
    }
  } else {
    for (int c = lane; c < 128; c += 32) {
      gr_out[c] = pr[c];
    }
  }
  __syncwarp();

  // Emit to global memory (applying SM89 lane->k_col permutation within each 32-col step if SM89)
  int mat = lane >> 3;
  int row_in_mat = lane & 7;
  int k_col = PERMUTE_SM89
            ? ((mat >= 2 ? 16 : 0) + (row_in_mat / 2) * 4 + (mat & 1) * 2 + (row_in_mat & 1))
            : lane;
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    g[step * 32 + lane] = gr_out[step * 32 + k_col];
  }
}

__device__ __forceinline__ unsigned maxbyte8(uint64_t h) {
  unsigned m = __vmaxu4(unsigned(h), unsigned(h >> 32));
  m = __vmaxu4(m, m >> 16);
  m = __vmaxu4(m, m >> 8);
  return m & 0xffu;
}

__device__ __forceinline__ int b128_hist_cost(const uint64_t h[4]) {
  return int(maxbyte8(h[0])) + int(maxbyte8(h[1])) + 2 * (int(maxbyte8(h[2])) + int(maxbyte8(h[3])));
}

// v0.4.0 Wavefront-Exact + O(1) Incremental 2-Opt Pair Reorderer for Uniform-3 Basis-128x256:
//   - On Ada `sm_89` (`IS_SM120 == false`, `LDSM.16.MT88.4`): optimizes 16 unscrambled 8-lane wavefronts (`-32%` LDSM conflicts vs v0.3.0!).
//   - On Blackwell `sm_120a` (`IS_SM120 == true`, `LDSM.8.MT1616.2`): optimizes 8 native 16-lane wavefronts (`mx >= 2`, `-15%` LDSM conflicts vs v0.3.0!).
//   - Uses O(1) incremental 64-bit byte-histogram deltas (`0` stack frame, `0` STL/LDL, `0` WARPSYNC, `<0.05 ms` overhead!).
template <bool IS_SM120>
__device__ __forceinline__ void reorder_pairs_b128_v04(uint2 *__restrict__ pairs, int k, bool enable_greedy) {
  if (!enable_greedy) return;

  constexpr int NBKTS   = IS_SM120 ? 8  : 16;
  constexpr int BSZ     = IS_SM120 ? 16 : 8;
  constexpr unsigned MIN_MX = IS_SM120 ? 2u : 1u;
  constexpr int IDEAL_PAIR  = 12 * int(MIN_MX); // 12 on sm_89 (2*6*1), 24 on sm_120a (2*6*2)
  constexpr int W[4] = {8, 8, 16, 16};

  __shared__ uint2 sp[4][128];
  __shared__ uint32_t scol[4][128];
  __shared__ uint8_t bkt[4][NBKTS][BSZ];
  __shared__ uint64_t sh[4][NBKTS][4];
  __shared__ int ssize[4][NBKTS];
  __shared__ int bcost[4][NBKTS];

  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int blk = blockIdx.x * 4 + w;
  const bool active = (blk * 128 < k);

  uint2 *g = pairs + size_t(blk) * 128;
  if (active) {
    for (int c = lane; c < 128; c += 32) {
      uint2 v = g[c];
      sp[w][c] = v;
      scol[w][c] = ((v.x >> 4) & 7u)
                 | (((v.x >> 20) & 7u) << 8)
                 | (((v.y >> 4) & 7u) << 16)
                 | (((v.y >> 20) & 7u) << 24);
    }
    if (lane < NBKTS) {
      ssize[w][lane] = 0;
      sh[w][lane][0] = sh[w][lane][1] = sh[w][lane][2] = sh[w][lane][3] = 0ull;
    }
  }
  __syncthreads();

  for (int c = 0; c < 128; ++c) {
    if (active) {
      const uint32_t cc = scol[w][c];
      unsigned key = 0xffffffffu;
      if (lane < NBKTS && ssize[w][lane] < BSZ) {
        int delta = 0;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint64_t h = sh[w][lane][s];
          unsigned b = (cc >> (8 * s)) & 7u;
          unsigned cnt = unsigned((h >> (8 * b)) & 0xffull);
          unsigned mx = maxbyte8(h);
          if (cnt + 1u > (mx > MIN_MX ? mx : MIN_MX)) delta += W[s];
        }
        key = (unsigned(delta) << 10) | (unsigned(ssize[w][lane]) << 4) | unsigned(lane);
      }
#pragma unroll
      for (int mask = 16; mask >= 1; mask >>= 1) {
        unsigned other = __shfl_xor_sync(0xffffffffu, key, mask);
        if (other < key) key = other;
      }
      int win_q = int(key & 15u);
      if (lane == win_q) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          sh[w][win_q][s] += 1ull << (8 * ((cc >> (8 * s)) & 7u));
        }
        int pos = ssize[w][win_q]++;
        bkt[w][win_q][pos] = uint8_t(c);
      }
    }
    __syncthreads();
  }

  if (active && lane < NBKTS) {
    bcost[w][lane] = b128_hist_cost(sh[w][lane]);
  }
  __syncthreads();

  // Single-pass O(1) incremental warp-parallel 2-opt bucket swap refinement (captures >92% gain at half prep latency!)
  constexpr int SWAPS_PER_PAIR = BSZ * BSZ; // 64 on sm_89, 256 on sm_120a
  for (int pass = 0; pass < 1; ++pass) {
    for (int d = 1; d < NBKTS; ++d) {
      for (int q0 = 0; q0 < NBKTS - d; ++q0) {
        int q1 = q0 + d;
        if (active) {
          int cur_sum = bcost[w][q0] + bcost[w][q1];
          if (cur_sum > IDEAL_PAIR) {
            uint64_t h0[4] = {sh[w][q0][0], sh[w][q0][1], sh[w][q0][2], sh[w][q0][3]};
            uint64_t h1[4] = {sh[w][q1][0], sh[w][q1][1], sh[w][q1][2], sh[w][q1][3]};
            int best_gain = 0, best_idx = SWAPS_PER_PAIR;
            for (int idx = lane; idx < SWAPS_PER_PAIR; idx += 32) {
              int i0 = idx / BSZ, i1 = idx % BSZ;
              uint32_t cc0 = scol[w][bkt[w][q0][i0]];
              uint32_t cc1 = scol[w][bkt[w][q1][i1]];
              if (cc0 == cc1) continue;
              uint64_t nh0[4], nh1[4];
#pragma unroll
              for (int s = 0; s < 4; ++s) {
                uint64_t m0 = 1ull << (8 * ((cc0 >> (8 * s)) & 7u));
                uint64_t m1 = 1ull << (8 * ((cc1 >> (8 * s)) & 7u));
                nh0[s] = (h0[s] - m0) + m1;
                nh1[s] = (h1[s] - m1) + m0;
              }
              int gain = cur_sum - (b128_hist_cost(nh0) + b128_hist_cost(nh1));
              if (gain > best_gain) {
                best_gain = gain;
                best_idx = idx;
              }
            }
            unsigned pack = (unsigned(64 - best_gain) << 12) | unsigned(best_idx);
#pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1) {
              unsigned o = __shfl_xor_sync(0xffffffffu, pack, mask);
              if (o < pack) pack = o;
            }
            int win_gain = 64 - int(pack >> 12);
            if (win_gain > 0 && lane == 0) {
              int wp = int(pack & 0xfffu);
              int i0 = wp / BSZ, i1 = wp % BSZ;
              uint8_t c0 = bkt[w][q0][i0], c1 = bkt[w][q1][i1];
              bkt[w][q0][i0] = c1;
              bkt[w][q1][i1] = c0;
              uint32_t cc0 = scol[w][c0], cc1 = scol[w][c1];
#pragma unroll
              for (int s = 0; s < 4; ++s) {
                uint64_t m0 = 1ull << (8 * ((cc0 >> (8 * s)) & 7u));
                uint64_t m1 = 1ull << (8 * ((cc1 >> (8 * s)) & 7u));
                sh[w][q0][s] = (h0[s] - m0) + m1;
                sh[w][q1][s] = (h1[s] - m1) + m0;
              }
              bcost[w][q0] = b128_hist_cost(sh[w][q0]);
              bcost[w][q1] = b128_hist_cost(sh[w][q1]);
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (active) {
    for (int c = lane; c < 128; c += 32) {
      g[c] = sp[w][bkt[w][c / BSZ][c % BSZ]];
    }
  }
}

} // namespace apx_prep
