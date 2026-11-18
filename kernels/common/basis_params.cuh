// SPDX-License-Identifier: ISC
#pragma once
#include <cstdint>

struct ApxBasisParams {
  const uint8_t *basA;
  const uint8_t *basB;
  const uint2   *pairs;
  int sampled_m; // 32640 (multiple of 160; strictly excludes row M-1 nonce!)
  int n;         // 32768
  int k;         // 2048..65536 (multiple of 128)
  const uint32_t *key_a;
  const uint32_t *target_le;
  uint32_t *hit_count;
  uint32_t *hit_groups;
  int hit_cap;
  const uint32_t *abort_epoch_ptr;
  uint32_t expected_epoch;
};
