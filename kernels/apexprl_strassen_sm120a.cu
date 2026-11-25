// SPDX-License-Identifier: ISC
// Native RTX 50-series Blackwell (sm_120a) v0.7.0 RAW-24K Basis-Strassen <2,2,2:7> cubin module.
// Constant-0 Pearl V3, 0 B spills, 98,432 B SMEM, L2-pinned TRACE, 0.00 GB/s DRAM at 810 MHz GDDR7!
#define APX_IS_SM120 1
#include "common/strassen24k_core.cuh"

extern "C" __global__ __launch_bounds__(64) void apx_gen_basis_arch(
    const uint32_t *__restrict__ noise_key,
    uint8_t *__restrict__ out_packed,
    int total_rows,
    int is_b) {
  apx_prep::gen_basis_body<false>(noise_key, out_packed, total_rows, is_b);
}

extern "C" __global__ __launch_bounds__(128) void apx_reorder_pairs_arch(
    uint32_t *__restrict__ pairs_soa,
    int k,
    int enable_greedy) {
  apx_strassen::reorder_pairs_soa_v07<true>(pairs_soa, k, enable_greedy != 0);
}

extern "C" __global__ __launch_bounds__(256, 1) void apx_strassen_mine_v0(
    const __grid_constant__ ApxBasisParams p,
    const uint32_t *__restrict__ pairsB,
    uint32_t *__restrict__ trace_buf) {
  apx_strassen::run_strassen_128x256(p, pairsB, trace_buf);
}
