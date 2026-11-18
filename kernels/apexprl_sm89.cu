// SPDX-License-Identifier: ISC
// Native RTX 40-series Ada Lovelace (sm_89) cubin module: prep kernels + 4 placement variants.
#define APX_IS_SM120 0
#include "common/prep_kernels.cuh"

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
  apx_prep::reorder_pairs_body<true>(pairs, k, enable_greedy != 0);
}

// Variant 0: default balanced schedule (ada-u6)
#define APX_VARIANT_NS ada_v0
#define APX_KERNEL_ENTRY apx_basis_mine_v0
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY

// Variant 1: shifted P5=26 (ada-y5m1)
#define APX_P5 26
#define APX_VARIANT_NS ada_v1
#define APX_KERNEL_ENTRY apx_basis_mine_v1
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY

// Variant 2: shifted P0=8, PL=24 (ada-u10)
#define APX_P0 8
#define APX_PL 24
#define APX_VARIANT_NS ada_v2
#define APX_KERNEL_ENTRY apx_basis_mine_v2
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY

// Variant 3: early pair prefetch PL=16, P2=11 (ada-early)
#define APX_P2 11
#define APX_PL 16
#define APX_VARIANT_NS ada_v3
#define APX_KERNEL_ENTRY apx_basis_mine_v3
#include "common/basis_gemm_core.cuh"
#undef APX_VARIANT_NS
#undef APX_KERNEL_ENTRY
