// SPDX-License-Identifier: ISC
// Native RTX 40-series Ada Lovelace (sm_89) v0.3.0 Uniform-3 Basis-128x256 cubin module.
// 8.38 MiB L2-pinned working set, 0 spills, 0 inner-loop WARPSYNC.
#define APX_IS_SM120 0
#include "common/prep_kernels.cuh"
#include "common/basis128_core.cuh"

extern "C" __global__ __launch_bounds__(64) void apx_gen_basis_arch(
    const uint32_t *__restrict__ noise_key,
    uint8_t *__restrict__ out_packed,
    int total_rows,
    int is_b) {
  apx_prep::gen_basis_body<true>(noise_key, out_packed, total_rows, is_b);
}

extern "C" __global__ __launch_bounds__(128) void apx_reorder_pairs_arch(
    uint2 *__restrict__ pairs,
    int k,
    int enable_greedy) {
  apx_prep::reorder_pairs_b128_v04<false>(pairs, k, enable_greedy != 0);
}

extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v0(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<0, 2, 8, 15, 16, 10, 22, 1>(p); // ada128-e2 (v0.5 winner: 91.73% SOL, 192-cyc quiet pre-bar!)
}
extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v1(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<0, 2, 8, 15, 17, 11, 22, 1>(p); // ada128-e1
}
extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v2(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<1, 3, 8, 15, 19, 22, 12, 1>(p); // ada128-c6
}
extern "C" __global__ __launch_bounds__(256, 1) void apx_b128_mine_v3(const __grid_constant__ ApxBasisParams p) {
  apx_b128::run_basis_128x256<2, 4, 8, 15, 19, 21, 12, 1>(p); // ada128-c9
}
