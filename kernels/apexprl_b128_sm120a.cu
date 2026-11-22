// SPDX-License-Identifier: ISC
// Native RTX 50-series Blackwell (sm_120a) v0.3.0 Uniform-3 Basis-128x256 cubin module.
// 8.38 MiB L2-pinned working set (0 GB/s DRAM at 810 MHz!), 0 spills, 0 inner-loop WARPSYNC, 97.7% static SOL.
#define APX_IS_SM120 1
#include "common/prep_kernels.cuh"
#include "common/basis128_core.cuh"

extern "C" __global__ __launch_bounds__(64) void apx_gen_basis_arch(
    const uint32_t *__restrict__ noise_key,
    uint8_t *__restrict__ out_packed,
    int total_rows,
    int is_b) {
  apx_prep::gen_basis_body<false>(noise_key, out_packed, total_rows, is_b);
}

extern "C" __global__ __launch_bounds__(128) void apx_reorder_pairs_arch(
    uint2 *__restrict__ pairs,
    int k,
    int enable_greedy) {
  apx_prep::reorder_pairs_b128_v04<true>(pairs, k, enable_greedy != 0);
}

// Top-4 SM120a zero-spill, ZERO-PRMT (`PRMT=0`!) schedules:
extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v0(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<0, 2, 8, 15, 16, 10, 22, 1>(p); // bw128-e2 (192-cyc quiet pre-bar, 0 PRMT, 0 spills)
}
extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v1(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<2, 4, 8, 15, 19, 21, 12, 1>(p); // bw128-c9 (0 PRMT, 0 spills)
}
extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v2(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<1, 4, 8, 15, 19, 21, 12, 1>(p); // bw128-c8 (0 PRMT, 0 spills)
}
extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v3(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<1, 3, 8, 15, 19, 21, 12, 1>(p); // bw128-c0 (0 PRMT, 0 spills)
}
