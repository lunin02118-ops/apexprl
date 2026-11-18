// SPDX-License-Identifier: ISC
// Native RTX 50-series Blackwell (sm_120a) cubin module: prep kernels + 4 placement variants.
#define APX_IS_SM120 1
#include "common/prep_kernels.cuh"

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
  apx_prep::reorder_pairs_body<false>(pairs, k, enable_greedy != 0);
}

// Variant 0: Blackwell default (bw-u6)
#define APX_VARIANT_NS bw_v0
#define APX_KERNEL_ENTRY apx_basis_mine_v0
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY

// Variant 1: Blackwell P5=26 (bw-y5m1)
#define APX_P5 26
#define APX_VARIANT_NS bw_v1
#define APX_KERNEL_ENTRY apx_basis_mine_v1
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY

// Variant 2: Blackwell PL=21 (bw-w5q21)
#define APX_PL 21
#define APX_VARIANT_NS bw_v2
#define APX_KERNEL_ENTRY apx_basis_mine_v2
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY

// Variant 3: Blackwell P0=8, PL=24 (bw-u10)
#define APX_P0 8
#define APX_PL 24
#define APX_VARIANT_NS bw_v3
#define APX_KERNEL_ENTRY apx_basis_mine_v3
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY
