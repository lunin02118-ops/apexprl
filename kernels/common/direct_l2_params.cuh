// SPDX-License-Identifier: ISC
// Shared host/device ABI parameter block for ApexPRL v0.2 Hyper-L2 Direct GEMM kernels.
#pragma once
#include <cstdint>

struct alignas(8) ApxDirectL2Params {
  const uint8_t *panelA; // [m_tiles, num_ks, 128 * 64] swizzled K_STAGE=64
  const uint8_t *panelB; // [n_tiles, num_ks, 128 * 64] swizzled K_STAGE=64
  int m_tiles;           // number of 128-row tiles in this A panel
  int n_tiles;           // number of 128-col tiles in B (N / 128)
  int num_ks;            // K / 64 (must be even >= 32, i.e. K >= 2048)
  int base_bm;           // starting 128-row tile index in full matrix A
  int total_n_warps;     // N / 64 (= n_tiles * 2)
  int hit_cap;
  uint32_t *hit_count;
  uint32_t *hit_groups;
  uint32_t key_a[8];     // inline in __grid_constant__ cmem (zero global pointer load!)
  uint32_t target_le[8]; // inline in __grid_constant__ cmem (zero global pointer load!)
};
static_assert(sizeof(ApxDirectL2Params) == 120, "ApxDirectL2Params ABI mismatch");
