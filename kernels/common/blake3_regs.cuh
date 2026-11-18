// SPDX-License-Identifier: ISC
// Pure in-register BLAKE3 primitives for ApexPRL (zero shared/local memory spills).
#pragma once
#include <cstdint>

namespace apx_b3 {

static constexpr uint32_t IV0 = 0x6A09E667u;
static constexpr uint32_t IV1 = 0xBB67AE85u;
static constexpr uint32_t IV2 = 0x3C6EF372u;
static constexpr uint32_t IV3 = 0xA54FF53Au;
static constexpr uint32_t IV4 = 0x510E527Fu;
static constexpr uint32_t IV5 = 0x9B05688Cu;
static constexpr uint32_t IV6 = 0x1F83D9ABu;
static constexpr uint32_t IV7 = 0x5BE0CD19u;

static constexpr uint32_t CHUNK_START = 1u << 0;
static constexpr uint32_t CHUNK_END   = 1u << 1;
static constexpr uint32_t PARENT      = 1u << 2;
static constexpr uint32_t ROOT        = 1u << 3;
static constexpr uint32_t KEYED_HASH  = 1u << 4;

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
  return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
  return __funnelshift_l(x, x, n);
}

__device__ __forceinline__ uint32_t xor3_lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

template <int SHIFT>
__device__ __forceinline__ uint32_t rotl_xor(uint32_t x, uint32_t y) {
  uint32_t r;
  asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(x), "n"(SHIFT));
  return r ^ y;
}

__device__ __forceinline__ void g_quarter(
    uint32_t &va, uint32_t &vb, uint32_t &vc, uint32_t &vd,
    uint32_t mx, uint32_t my) {
  va = va + vb + mx;
  vd = rotr32(vd ^ va, 16);
  vc = vc + vd;
  vb = rotr32(vb ^ vc, 12);
  va = va + vb + my;
  vd = rotr32(vd ^ va, 8);
  vc = vc + vd;
  vb = rotr32(vb ^ vc, 7);
}

#define APX_B3_ROUND(v, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15) \
  do { \
    g_quarter(v[0], v[4], v[8],  v[12], m0,  m1);  \
    g_quarter(v[1], v[5], v[9],  v[13], m2,  m3);  \
    g_quarter(v[2], v[6], v[10], v[14], m4,  m5);  \
    g_quarter(v[3], v[7], v[11], v[15], m6,  m7);  \
    g_quarter(v[0], v[5], v[10], v[15], m8,  m9);  \
    g_quarter(v[1], v[6], v[11], v[12], m10, m11); \
    g_quarter(v[2], v[7], v[8],  v[13], m12, m13); \
    g_quarter(v[3], v[4], v[9],  v[14], m14, m15); \
  } while (0)

// Full 7-round BLAKE3 block compression entirely in registers (compile-time schedule).
__device__ __forceinline__ void compress_block(
    uint32_t cv[8],
    const uint32_t m[16],
    uint64_t counter,
    uint32_t block_len,
    uint32_t flags) {
  uint32_t v[16] = {
      cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
      IV0, IV1, IV2, IV3,
      uint32_t(counter), uint32_t(counter >> 32), block_len, flags
  };

  // Round 0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  APX_B3_ROUND(v, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
  // Round 1: [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
  APX_B3_ROUND(v, m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]);
  // Round 2: [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
  APX_B3_ROUND(v, m[3], m[4], m[10], m[12], m[13], m[2], m[7], m[14], m[6], m[5], m[9], m[0], m[11], m[15], m[8], m[1]);
  // Round 3: [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
  APX_B3_ROUND(v, m[10], m[7], m[12], m[9], m[14], m[3], m[13], m[15], m[4], m[0], m[11], m[2], m[5], m[8], m[1], m[6]);
  // Round 4: [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
  APX_B3_ROUND(v, m[12], m[13], m[9], m[11], m[15], m[10], m[14], m[8], m[7], m[2], m[5], m[3], m[0], m[1], m[6], m[4]);
  // Round 5: [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
  APX_B3_ROUND(v, m[9], m[14], m[11], m[5], m[8], m[12], m[15], m[1], m[13], m[3], m[0], m[10], m[2], m[6], m[4], m[7]);
  // Round 6: [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
  APX_B3_ROUND(v, m[11], m[15], m[5], m[0], m[1], m[9], m[8], m[6], m[14], m[10], m[2], m[12], m[3], m[4], m[7], m[13]);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    cv[i] = v[i] ^ v[i + 8];
  }
}

__device__ __forceinline__ void keyed_hash64(
    const uint32_t key[8],
    const uint32_t msg[16],
    uint32_t out[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = key[i];
  compress_block(out, msg, 0ull, 64u, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);
}

} // namespace apx_b3
