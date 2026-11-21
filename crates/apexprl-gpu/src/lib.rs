//! Multi-GPU CUDA Driver API & NVML engine for ApexPRL v0.2
//! Supports BOTH:
//!   - **Hyper-L2 Direct 2-CTA/SM Engine (`direct`, default):** `128x128` CTA, 2 CTAs/SM, SWAR warp-transpose
//!     L2 materializer, 0 register spills, `elect.sync` TMA on Blackwell (`sm_120a`), `-O3` on Ada (`sm_89`).
//!   - **On-Chip Basis Engine (`basis`):** `160x256` CTA, 1 CTA/SM, L2-resident `8.5 MiB` basis rebuild.

pub mod cuda_drv;
pub mod nvml_drv;

use anyhow::{bail, ensure, Context, Result};
use apexprl_pow::{Constant64MerkleCache, KernelProfile, PreparedJob};
use cuda_drv::*;
use nvml_drv::{GpuSensors, NvmlHandle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

const CUBIN_B128_SM89: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/apexprl_b128_sm89.cubin"));
const CUBIN_B128_SM120A: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/apexprl_b128_sm120a.cubin"));
const CUBIN_DIRECT_SM89: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/apexprl_direct_sm89.cubin"));
const CUBIN_DIRECT_SM120A: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/apexprl_direct_sm120a.cubin"));
const CUBIN_BASIS_SM89: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/apexprl_sm89.cubin"));
const CUBIN_BASIS_SM120A: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/apexprl_sm120a.cubin"));

pub const SMEM_BASIS128_BYTES: c_int = 90240; // 16KiB A + 32KiB B + 2*12KiB stage + 16KiB trace + 128B bar
pub const SMEM_BASIS_BYTES: c_int = 96320;
pub const SMEM_DIRECT_2CTA_BYTES: c_int = 41088; // 2 * 16384 + 8192 + 128 (x2 CTAs/SM = 82,176 B <= 99 KiB)
pub const SMEM_DIRECT_3STG_BYTES: c_int = 90240; // 3 * 24576 + 16384 + 128

pub const VARIANT_NAMES_B128_ADA: [&str; 4] = ["ada128-c9", "ada128-c8", "ada128-c0", "ada128-c6"];
pub const VARIANT_NAMES_B128_BW: [&str; 4] = ["bw128-c9", "bw128-c8", "bw128-c0", "bw128-c6"];
pub const VARIANT_NAMES_DIRECT_ADA: [&str; 3] = ["hl2-2cta", "hl2-2cta-s2", "hl2-3stg"];
pub const VARIANT_NAMES_DIRECT_BW: [&str; 3] = ["hl2-2cta", "hl2-2cta-df", "hl2-3stg"];
pub const VARIANT_NAMES_BASIS_ADA: [&str; 4] = ["ada-u6", "ada-y5m1", "ada-u10", "ada-early"];
pub const VARIANT_NAMES_BASIS_BW: [&str; 4] = ["bw-u6", "bw-y5m1", "bw-w5q21", "bw-u10"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    Basis128,
    Basis160,
    DirectL2,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ApxBasisParams {
    bas_a: CUdeviceptr,
    bas_b: CUdeviceptr,
    pairs: CUdeviceptr,
    sampled_m: i32,
    n: i32,
    k: i32,
    key_a: CUdeviceptr,
    target_le: CUdeviceptr,
    hit_count: CUdeviceptr,
    hit_groups: CUdeviceptr,
    hit_cap: i32,
    abort_epoch_ptr: CUdeviceptr,
    expected_epoch: u32,
}
const _: () = assert!(std::mem::size_of::<ApxBasisParams>() == 96);

#[repr(C)]
#[derive(Clone, Copy)]
struct ApxDirectL2Params {
    panel_a: CUdeviceptr,
    panel_b: CUdeviceptr,
    m_tiles: i32,
    n_tiles: i32,
    num_ks: i32,
    base_bm: i32,
    total_n_warps: i32,
    hit_cap: i32,
    hit_count: CUdeviceptr,
    hit_groups: CUdeviceptr,
    key_a: [u32; 8],
    target_le: [u32; 8],
}
const _: () = assert!(std::mem::size_of::<ApxDirectL2Params>() == 120);

#[derive(Clone, Debug)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub sm_major: i32,
    pub sm_minor: i32,
    pub sm_count: i32,
    pub l2_bytes: usize,
    pub vram_total_mib: usize,
    pub pci_bus_id: u32,
}

impl GpuDeviceInfo {
    pub fn is_blackwell(&self) -> bool {
        self.sm_major == 12
    }
    pub fn is_ada_or_ampere(&self) -> bool {
        self.sm_major == 8 && self.sm_minor >= 9
    }
    pub fn is_supported(&self) -> bool {
        self.is_blackwell() || self.is_ada_or_ampere()
    }
    /// Estimated peak INT8 MACs/sec at ~2500 MHz for Kryptex auto-difficulty sizing.
    pub fn estimated_macs_per_sec(&self) -> f64 {
        (self.sm_count as f64) * 1024.0 * 2.5e9
    }
}

/// Computes optimal L2-resident `N` dimension (`4096`, `8192`, or `16384`) for Direct-L2 mode
/// based on the minimum L2 cache size across active GPUs so `d_panel_b` stays 100% resident in L2!
pub fn optimal_direct_l2_n(min_l2_bytes: usize, k: usize) -> usize {
    let target_b_bytes = (min_l2_bytes * 70) / 100; // leave >=30% of L2 for streaming A panels!
    let mut n = 16384usize;
    while n > 1024 && n * k > target_b_bytes {
        n >>= 1;
    }
    n.clamp(2048, 16384)
}

/// Computes wave-aligned `panel_m_tiles` (`128`-row tiles) that fits inside the remaining L2 cache
/// alongside `d_panel_b` and maximizes 2-CTA/SM wave efficiency (`2 * sm_count` CTAs/wave)!
pub fn optimal_panel_m_tiles(info: &GpuDeviceInfo, profile: &KernelProfile) -> usize {
    let total_m_tiles = profile.sampled_m() / 128;
    let n_tiles = profile.n / 128;
    let b_bytes = profile.n * profile.k;
    let tile_a_bytes = 128 * profile.k;

    let l2_budget = ((info.l2_bytes as f64) * 0.81) as usize;
    let avail_a_bytes = l2_budget.saturating_sub(b_bytes).max(tile_a_bytes * 4);
    let max_m = (avail_a_bytes / tile_a_bytes).clamp(4, total_m_tiles.min(32));
    let min_m = 8usize.min(max_m);

    let wave_ctas = (info.sm_count as usize * 2).max(1);
    let mut best_m = max_m;
    let mut best_score = -1.0f64;
    for m_t in min_m..=max_m {
        let ctas = m_t * n_tiles;
        let waves = ctas.div_ceil(wave_ctas);
        let wave_eff = (ctas as f64) / ((waves * wave_ctas) as f64);
        // Slightly prefer larger panels (fewer launches) when wave efficiency is tied
        let score = wave_eff + (m_t as f64) * 1e-4;
        if score > best_score {
            best_score = score;
            best_m = m_t;
        }
    }
    best_m
}

fn optimal_mat_k_parts(cur_m: usize, sm_count: usize, num_ks: usize) -> c_uint {
    let mut best_kp = 4usize.min(num_ks.max(1));
    let mut best_score = -1.0f64;
    for &kp in &[4usize, 8, 16] {
        if kp > num_ks || (num_ks % kp) != 0 {
            continue;
        }
        let ctas = cur_m * kp;
        let waves = ctas.div_ceil(sm_count.max(1));
        let wave_eff = (ctas as f64) / ((waves * sm_count.max(1)) as f64);
        let underfill = if ctas < sm_count { 0.5 } else { 1.0 };
        let reload_pen = if ctas >= 2 * sm_count { (kp as f64) * 0.005 } else { 0.0 };
        let score = wave_eff * underfill - reload_pen;
        if score > best_score {
            best_score = score;
            best_kp = kp;
        }
    }
    best_kp as c_uint
}

pub fn enumerate_gpus() -> Result<Vec<GpuDeviceInfo>> {
    let api = cuda()?;
    let mut count = 0;
    check_cu(api, unsafe { (api.cuDeviceGetCount)(&mut count) }, "cuDeviceGetCount")?;
    let mut out = Vec::with_capacity(count.max(0) as usize);
    for ord in 0..count {
        let mut dev = 0;
        check_cu(api, unsafe { (api.cuDeviceGet)(&mut dev, ord) }, "cuDeviceGet")?;
        let mut raw_name = [0i8; 128];
        check_cu(api, unsafe { (api.cuDeviceGetName)(raw_name.as_mut_ptr(), 128, dev) }, "cuDeviceGetName")?;
        let name = unsafe { CStr::from_ptr(raw_name.as_ptr()) }.to_string_lossy().into_owned();

        let attr = |a: c_int| -> Result<i32> {
            let mut v = 0;
            check_cu(api, unsafe { (api.cuDeviceGetAttribute)(&mut v, a, dev) }, "cuDeviceGetAttribute")?;
            Ok(v)
        };
        let sm_major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let sm_minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let sm_count = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let l2_bytes = attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE).unwrap_or(0).max(0) as usize;
        let pci_bus_id = attr(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID).unwrap_or(0).max(0) as u32;

        let mut ctx = std::ptr::null_mut();
        let mut vram_total_mib = 0;
        if unsafe { (api.cuDevicePrimaryCtxRetain)(&mut ctx, dev) } == 0 {
            let _ = unsafe { (api.cuCtxSetCurrent)(ctx) };
            let (mut free_b, mut total_b) = (0usize, 0usize);
            if unsafe { (api.cuMemGetInfo_v2)(&mut free_b, &mut total_b) } == 0 {
                let _ = free_b;
                vram_total_mib = total_b / (1024 * 1024);
            }
        }

        out.push(GpuDeviceInfo {
            ordinal: ord as usize,
            name,
            sm_major,
            sm_minor,
            sm_count,
            l2_bytes,
            vram_total_mib,
            pci_bus_id,
        });
    }
    Ok(out)
}

struct DevMem {
    ptr: CUdeviceptr,
    #[allow(dead_code)]
    bytes: usize,
}

impl DevMem {
    fn alloc(api: &CudaApi, bytes: usize) -> Result<Self> {
        let alloc_bytes = bytes.max(256);
        let mut ptr = 0u64;
        check_cu(api, unsafe { (api.cuMemAlloc_v2)(&mut ptr, alloc_bytes) }, "cuMemAlloc")?;
        Ok(Self { ptr, bytes: alloc_bytes })
    }
}

impl Drop for DevMem {
    fn drop(&mut self) {
        if let Ok(api) = cuda() {
            unsafe { let _ = (api.cuMemFree_v2)(self.ptr); }
        }
    }
}

/// Pinned + device-mapped zero-copy page holding `abort_epoch` and `HitRing`.
pub struct MappedControlPage {
    host_ptr: *mut u8,
    pub dev_ptr: CUdeviceptr,
    bytes: usize,
}

unsafe impl Send for MappedControlPage {}
unsafe impl Sync for MappedControlPage {}

impl MappedControlPage {
    fn alloc(api: &CudaApi, hit_cap: usize) -> Result<Self> {
        let bytes = (68 + hit_cap * 4 + 255) & !255;
        let mut host_void = std::ptr::null_mut();
        check_cu(
            api,
            unsafe { (api.cuMemHostAlloc)(&mut host_void, bytes, CU_MEMHOSTALLOC_DEVICEMAP) },
            "cuMemHostAlloc(DEVICEMAP)",
        )?;
        unsafe { std::ptr::write_bytes(host_void as *mut u8, 0, bytes) };
        let mut dev_ptr = 0u64;
        check_cu(
            api,
            unsafe { (api.cuMemHostGetDevicePointer_v2)(&mut dev_ptr, host_void, 0) },
            "cuMemHostGetDevicePointer",
        )?;
        Ok(Self {
            host_ptr: host_void as *mut u8,
            dev_ptr,
            bytes,
        })
    }

    #[inline]
    pub fn epoch_atomic(&self) -> &AtomicU32 {
        unsafe { &*(self.host_ptr as *const AtomicU32) }
    }

    #[inline]
    pub fn epoch_ptr_raw(&self) -> *const AtomicU32 {
        self.host_ptr as *const AtomicU32
    }

    #[inline]
    pub fn dev_abort_ptr(&self) -> CUdeviceptr {
        self.dev_ptr
    }

    #[inline]
    pub fn dev_hit_count_ptr(&self) -> CUdeviceptr {
        self.dev_ptr + 64
    }

    #[inline]
    pub fn dev_hit_groups_ptr(&self) -> CUdeviceptr {
        self.dev_ptr + 68
    }

    pub fn read_hits(&self, hit_cap: usize) -> (u32, Vec<u32>) {
        unsafe {
            let count = std::ptr::read_volatile((self.host_ptr.add(64)) as *const u32);
            let n = (count as usize).min(hit_cap);
            let groups_ptr = (self.host_ptr.add(68)) as *const u32;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(std::ptr::read_volatile(groups_ptr.add(i)));
            }
            (count, out)
        }
    }
}

impl Drop for MappedControlPage {
    fn drop(&mut self) {
        let _ = self.bytes;
        if let Ok(api) = cuda() {
            unsafe { let _ = (api.cuMemFreeHost)(self.host_ptr as *mut c_void); }
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TuneCacheFile {
    pub entries: HashMap<String, usize>,
}

impl TuneCacheFile {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(txt) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, txt);
        }
    }
}

pub struct RoundOutcome {
    pub round_ms: f32,
    pub kernel_ms: f32,
    pub total_hits: u32,
    pub hit_groups: Vec<u32>,
    pub aborted: bool,
}

fn bytes32_to_u32x8(b: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
    }
    out
}

enum EngineBackend {
    DirectL2 {
        _module: CUmodule,
        fn_merkle_leaves: CUfunction,
        fn_merkle_parents: CUfunction,
        fn_gen_basis: CUfunction,
        fn_gen_u16_pairs: CUfunction,
        fn_materialize_128: CUfunction,
        fn_variants: [CUfunction; 3],
        d_pairs_a: DevMem,
        d_pairs_b: DevMem,
        d_panel_a: DevMem,
        d_panel_b: DevMem,
        panel_m_tiles: usize,
    },
    Basis {
        _module: CUmodule,
        fn_merkle_leaves: CUfunction,
        fn_merkle_parents: CUfunction,
        fn_gen_basis: CUfunction,
        fn_gen_raw_pairs: CUfunction,
        fn_reorder_pairs: CUfunction,
        fn_variants: [CUfunction; 4],
        smem_bytes: c_int,
        d_pairs: DevMem,
    },
}

pub struct GpuWorkerEngine {
    pub info: GpuDeviceInfo,
    pub profile: KernelProfile,
    pub engine_kind: EngineKind,
    pub active_variant: usize,
    pub nvml: Option<NvmlHandle>,
    ctx: CUcontext,
    stream: CUstream,
    ev_round_start: CUevent,
    ev_start: CUevent,
    ev_end: CUevent,
    d_job_key: DevMem,
    d_key_a: DevMem,
    d_key_b: DevMem,
    d_target_le: DevMem,
    /// Single contiguous allocation holding `d_bas_a` followed immediately by `d_bas_b`
    /// (`8.38 MiB` at `32768x32768`), allowing 100% single-window L2 persistence pinning!
    d_bas_ab: DevMem,
    bas_b_offset: u64,
    l2_pinned_bytes: usize,
    d_tree: DevMem,
    ctrl: MappedControlPage,
    hit_cap: usize,
    backend: EngineBackend,
}

unsafe impl Send for GpuWorkerEngine {}

fn pin_l2_window(api: &CudaApi, dev: CUdevice, stream: CUstream, base_ptr: CUdeviceptr, num_bytes: usize) -> usize {
    let mut max_persist = 0i32;
    let mut max_win = 0i32;
    unsafe {
        let _ = (api.cuDeviceGetAttribute)(&mut max_persist, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, dev);
        let _ = (api.cuDeviceGetAttribute)(&mut max_win, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, dev);
    }
    if max_persist <= 0 || num_bytes == 0 {
        return 0;
    }
    let persist_want = num_bytes.min(max_persist as usize);
    let _ = unsafe { (api.cuCtxSetLimit)(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, persist_want) };
    if let Some(set_attr) = api.cuStreamSetAttribute {
        let win_bytes = if max_win > 0 { num_bytes.min(max_win as usize) } else { num_bytes };
        let hit_ratio = ((persist_want as f32) / (win_bytes.max(1) as f32)).clamp(0.1, 1.0);
        let val = CUstreamAttrValue {
            access_policy_window: CUaccessPolicyWindow {
                base_ptr: base_ptr as *mut c_void,
                num_bytes: win_bytes,
                hit_ratio,
                hit_prop: CU_ACCESS_PROPERTY_PERSISTING,
                miss_prop: CU_ACCESS_PROPERTY_NORMAL,
            },
        };
        if unsafe { set_attr(stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &val) } == 0 {
            return persist_want;
        }
    }
    persist_want
}

impl GpuWorkerEngine {
    #[inline]
    fn bas_a_ptr(&self) -> CUdeviceptr {
        self.d_bas_ab.ptr
    }

    #[inline]
    fn bas_b_ptr(&self) -> CUdeviceptr {
        self.d_bas_ab.ptr + self.bas_b_offset
    }

    pub fn new(
        info: GpuDeviceInfo,
        profile: KernelProfile,
        hit_cap: usize,
        lock_mem_idle: bool,
        core_lock_mhz: Option<u32>,
        power_limit_w: Option<u32>,
    ) -> Result<(Self, Vec<String>)> {
        ensure!(
            info.is_supported(),
            "GPU #{} ({}, sm_{}{}) is outside RTX 40 (sm_89) / RTX 50 (sm_120) target architectures",
            info.ordinal,
            info.name,
            info.sm_major,
            info.sm_minor
        );

        let engine_kind = if profile.is_direct_l2() {
            EngineKind::DirectL2
        } else if profile.is_basis_128() {
            EngineKind::Basis128
        } else {
            EngineKind::Basis160
        };

        let api = cuda()?;
        let mut dev = 0;
        check_cu(api, unsafe { (api.cuDeviceGet)(&mut dev, info.ordinal as c_int) }, "cuDeviceGet")?;
        let mut ctx = std::ptr::null_mut();
        check_cu(api, unsafe { (api.cuDevicePrimaryCtxRetain)(&mut ctx, dev) }, "cuDevicePrimaryCtxRetain")?;
        check_cu(api, unsafe { (api.cuCtxSetCurrent)(ctx) }, "cuCtxSetCurrent")?;

        let mut stream = std::ptr::null_mut();
        check_cu(api, unsafe { (api.cuStreamCreate)(&mut stream, CU_STREAM_NON_BLOCKING) }, "cuStreamCreate")?;

        let mut ev_round_start = std::ptr::null_mut();
        let mut ev_start = std::ptr::null_mut();
        let mut ev_end = std::ptr::null_mut();
        check_cu(api, unsafe { (api.cuEventCreate)(&mut ev_round_start, CU_EVENT_BLOCKING_SYNC) }, "cuEventCreate")?;
        check_cu(api, unsafe { (api.cuEventCreate)(&mut ev_start, CU_EVENT_BLOCKING_SYNC) }, "cuEventCreate")?;
        check_cu(api, unsafe { (api.cuEventCreate)(&mut ev_end, CU_EVENT_BLOCKING_SYNC) }, "cuEventCreate")?;

        let d_job_key = DevMem::alloc(api, 32)?;
        let d_key_a = DevMem::alloc(api, 32)?;
        let d_key_b = DevMem::alloc(api, 32)?;
        let d_target_le = DevMem::alloc(api, 32)?;
        let bas_a_bytes = profile.m * 128;
        let bas_b_bytes = profile.n * 128;
        let d_bas_ab = DevMem::alloc(api, bas_a_bytes + bas_b_bytes)?;
        let bas_b_offset = bas_a_bytes as u64;
        let d_tree = DevMem::alloc(api, Constant64MerkleCache::heap_byte_len(&profile))?;
        let ctrl = MappedControlPage::alloc(api, hit_cap)?;
        let l2_pinned_bytes;

        let backend = match engine_kind {
            EngineKind::DirectL2 => {
                let cubin = if info.is_blackwell() { CUBIN_DIRECT_SM120A } else { CUBIN_DIRECT_SM89 };
                let mut module = std::ptr::null_mut();
                check_cu(
                    api,
                    unsafe { (api.cuModuleLoadData)(&mut module, cubin.as_ptr() as *const c_void) },
                    "cuModuleLoadData(direct)",
                )?;
                let get_fn = |name: &str| -> Result<CUfunction> {
                    let cs = CString::new(name).unwrap();
                    let mut f = std::ptr::null_mut();
                    check_cu(api, unsafe { (api.cuModuleGetFunction)(&mut f, module, cs.as_ptr()) }, name)?;
                    Ok(f)
                };
                let fn_merkle_leaves = get_fn("apx_merkle_leaves_const64")?;
                let fn_merkle_parents = get_fn("apx_merkle_parents")?;
                let fn_gen_basis = get_fn("apx_gen_basis_linear")?;
                let fn_gen_u16_pairs = get_fn("apx_gen_u16_pairs")?;
                let fn_materialize_128 = get_fn("apx_materialize_panel_128")?;
                let fn_v0 = get_fn("apx_direct_l2_v0")?;
                let fn_v1 = get_fn("apx_direct_l2_v1")?;
                let fn_v2 = get_fn("apx_direct_l2_v2")?;

                check_cu(api, unsafe { (api.cuFuncSetAttribute)(fn_v0, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_DIRECT_2CTA_BYTES) }, "smem(v0)")?;
                check_cu(api, unsafe { (api.cuFuncSetAttribute)(fn_v1, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_DIRECT_2CTA_BYTES) }, "smem(v1)")?;
                check_cu(api, unsafe { (api.cuFuncSetAttribute)(fn_v2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_DIRECT_3STG_BYTES) }, "smem(v2)")?;
                for &f in &[fn_v0, fn_v1, fn_v2] {
                    let _ = unsafe { (api.cuFuncSetAttribute)(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100) };
                }

                let panel_m_tiles = optimal_panel_m_tiles(&info, &profile);
                let d_pairs_a = DevMem::alloc(api, profile.k * 2)?;
                let d_pairs_b = DevMem::alloc(api, profile.k * 2)?;
                let d_panel_a = DevMem::alloc(api, panel_m_tiles * 128 * profile.k)?;
                let d_panel_b = DevMem::alloc(api, profile.n * profile.k)?;
                l2_pinned_bytes = pin_l2_window(api, dev, stream, d_panel_b.ptr, profile.n * profile.k);

                EngineBackend::DirectL2 {
                    _module: module,
                    fn_merkle_leaves,
                    fn_merkle_parents,
                    fn_gen_basis,
                    fn_gen_u16_pairs,
                    fn_materialize_128,
                    fn_variants: [fn_v0, fn_v1, fn_v2],
                    d_pairs_a,
                    d_pairs_b,
                    d_panel_a,
                    d_panel_b,
                    panel_m_tiles,
                }
            }
            EngineKind::Basis128 | EngineKind::Basis160 => {
                let is_b128 = engine_kind == EngineKind::Basis128;
                l2_pinned_bytes = pin_l2_window(api, dev, stream, d_bas_ab.ptr, bas_a_bytes + bas_b_bytes);
                let cubin = match (is_b128, info.is_blackwell()) {
                    (true, true) => CUBIN_B128_SM120A,
                    (true, false) => CUBIN_B128_SM89,
                    (false, true) => CUBIN_BASIS_SM120A,
                    (false, false) => CUBIN_BASIS_SM89,
                };
                let mut module = std::ptr::null_mut();
                check_cu(
                    api,
                    unsafe { (api.cuModuleLoadData)(&mut module, cubin.as_ptr() as *const c_void) },
                    if is_b128 { "cuModuleLoadData(basis128)" } else { "cuModuleLoadData(basis160)" },
                )?;
                let get_fn = |name: &str| -> Result<CUfunction> {
                    let cs = CString::new(name).unwrap();
                    let mut f = std::ptr::null_mut();
                    check_cu(api, unsafe { (api.cuModuleGetFunction)(&mut f, module, cs.as_ptr()) }, name)?;
                    Ok(f)
                };
                let fn_merkle_leaves = get_fn("apx_merkle_leaves_const64")?;
                let fn_merkle_parents = get_fn("apx_merkle_parents")?;
                let fn_gen_basis = get_fn("apx_gen_basis_arch")?;
                let fn_gen_raw_pairs = get_fn("apx_gen_raw_pairs")?;
                let fn_reorder_pairs = get_fn("apx_reorder_pairs_arch")?;
                let prefix = if is_b128 { "apx_b128_mine" } else { "apx_basis_mine" };
                let fn_variants = [
                    get_fn(&format!("{prefix}_v0"))?,
                    get_fn(&format!("{prefix}_v1"))?,
                    get_fn(&format!("{prefix}_v2"))?,
                    get_fn(&format!("{prefix}_v3"))?,
                ];
                let smem_bytes = if is_b128 { SMEM_BASIS128_BYTES } else { SMEM_BASIS_BYTES };
                for &f in &fn_variants {
                    check_cu(api, unsafe { (api.cuFuncSetAttribute)(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes) }, "smem(basis)")?;
                    let _ = unsafe { (api.cuFuncSetAttribute)(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100) };
                }
                let d_pairs = DevMem::alloc(api, profile.k * 8)?;
                EngineBackend::Basis {
                    _module: module,
                    fn_merkle_leaves,
                    fn_merkle_parents,
                    fn_gen_basis,
                    fn_gen_raw_pairs,
                    fn_reorder_pairs,
                    fn_variants,
                    smem_bytes,
                    d_pairs,
                }
            }
        };

        let mut nvml = NvmlHandle::open(info.ordinal, info.pci_bus_id);
        let tune_notes = nvml
            .as_mut()
            .map(|h| h.apply_tuning(lock_mem_idle, core_lock_mhz, power_limit_w))
            .unwrap_or_default();

        Ok((
            Self {
                info,
                profile,
                engine_kind,
                active_variant: 0,
                nvml,
                ctx,
                stream,
                ev_round_start,
                ev_start,
                ev_end,
                d_job_key,
                d_key_a,
                d_key_b,
                d_target_le,
                d_bas_ab,
                bas_b_offset,
                l2_pinned_bytes,
                d_tree,
                ctrl,
                hit_cap,
                backend,
            },
            tune_notes,
        ))
    }

    pub fn bind_context(&self) -> Result<()> {
        let api = cuda()?;
        check_cu(api, unsafe { (api.cuCtxSetCurrent)(self.ctx) }, "cuCtxSetCurrent")
    }

    pub fn num_variants(&self) -> usize {
        match self.engine_kind {
            EngineKind::DirectL2 => 3,
            EngineKind::Basis128 | EngineKind::Basis160 => 4,
        }
    }

    pub fn variant_name(&self, idx: usize) -> &'static str {
        match self.engine_kind {
            EngineKind::DirectL2 => {
                if self.info.is_blackwell() {
                    VARIANT_NAMES_DIRECT_BW[idx % 3]
                } else {
                    VARIANT_NAMES_DIRECT_ADA[idx % 3]
                }
            }
            EngineKind::Basis128 => {
                if self.info.is_blackwell() {
                    VARIANT_NAMES_B128_BW[idx % 4]
                } else {
                    VARIANT_NAMES_B128_ADA[idx % 4]
                }
            }
            EngineKind::Basis160 => {
                if self.info.is_blackwell() {
                    VARIANT_NAMES_BASIS_BW[idx % 4]
                } else {
                    VARIANT_NAMES_BASIS_ADA[idx % 4]
                }
            }
        }
    }

    pub fn layout_summary(&self) -> String {
        let pin_mib = (self.l2_pinned_bytes as f64) / 1048576.0;
        match &self.backend {
            EngineBackend::DirectL2 { panel_m_tiles, .. } => {
                let b_mib = (self.profile.n * self.profile.k) as f64 / 1048576.0;
                let a_mib = (panel_m_tiles * 128 * self.profile.k) as f64 / 1048576.0;
                format!(
                    "Hyper-L2 Direct ({}x{}x{}, B={:.1}MiB+Apanel={:.1}MiB={:.1}MiB in {:.0}MiB L2, L2pin={:.1}MiB, panel_m={} tiles)",
                    self.profile.sampled_m(),
                    self.profile.n,
                    self.profile.k,
                    b_mib,
                    a_mib,
                    b_mib + a_mib,
                    self.info.l2_bytes as f64 / 1048576.0,
                    pin_mib,
                    panel_m_tiles
                )
            }
            EngineBackend::Basis { .. } => {
                let tag = if self.engine_kind == EngineKind::Basis128 {
                    "Uniform-3 Basis-128x256"
                } else {
                    "Legacy Basis-160x256"
                };
                let bas_mib = ((self.profile.m + self.profile.n) * 128) as f64 / 1048576.0;
                format!(
                    "{tag} ({}x{}x{}, L2 basis={:.2}MiB, L2pin={:.2}MiB in {:.0}MiB L2)",
                    self.profile.sampled_m(),
                    self.profile.n,
                    self.profile.k,
                    bas_mib,
                    pin_mib,
                    self.info.l2_bytes as f64 / 1048576.0
                )
            }
        }
    }

    pub fn abort_epoch_ptr(&self) -> *const AtomicU32 {
        self.ctrl.epoch_ptr_raw()
    }

    pub fn abort_epoch_ptr_usize(&self) -> usize {
        self.ctrl.epoch_ptr_raw() as usize
    }

    pub fn set_epoch(&self, epoch: u32) {
        self.ctrl.epoch_atomic().store(epoch, Ordering::Release);
    }

    pub fn sensors(&self) -> GpuSensors {
        self.nvml
            .as_ref()
            .map(|h| h.read_sensors(self.info.pci_bus_id))
            .unwrap_or(GpuSensors {
                pci_bus_id: self.info.pci_bus_id,
                ..Default::default()
            })
    }

    /// Prepares a new block job on GPU:
    /// 1. Builds the full Constant-64 Merkle heap on GPU (~0.4 ms) and verifies `root_b` on CPU.
    /// 2. Generates `d_bas_b` once per job.
    /// 3. In `DirectL2` mode, pre-materializes the ENTIRE L2-resident `d_panel_b` (`N x K`) ONCE PER JOB!
    pub fn prepare_job(&mut self, job: &PreparedJob, epoch: u32) -> Result<Constant64MerkleCache> {
        self.bind_context()?;
        self.set_epoch(epoch);
        let api = cuda()?;

        check_cu(api, unsafe { (api.cuMemcpyHtoDAsync_v2)(self.d_job_key.ptr, job.job_key.as_ptr() as *const c_void, 32, self.stream) }, "HtoD(job_key)")?;
        check_cu(api, unsafe { (api.cuMemcpyHtoDAsync_v2)(self.d_target_le.ptr, job.bound_le.as_ptr() as *const c_void, 32, self.stream) }, "HtoD(target_le)")?;

        let (fn_leaves, fn_parents, fn_gen_basis) = match &self.backend {
            EngineBackend::DirectL2 { fn_merkle_leaves, fn_merkle_parents, fn_gen_basis, .. } => {
                (*fn_merkle_leaves, *fn_merkle_parents, *fn_gen_basis)
            }
            EngineBackend::Basis { fn_merkle_leaves, fn_merkle_parents, fn_gen_basis, .. } => {
                (*fn_merkle_leaves, *fn_merkle_parents, *fn_gen_basis)
            }
        };

        let nleaves = (self.profile.m * self.profile.k) / apexprl_pow::CHUNK_BYTES;
        let mut p_key = self.d_job_key.ptr;
        let mut p_tree = self.d_tree.ptr;
        let mut nleaves_i32 = nleaves as i32;
        let mut args_leaves: [*mut c_void; 3] = [
            (&mut p_key as *mut CUdeviceptr).cast(),
            (&mut p_tree as *mut CUdeviceptr).cast(),
            (&mut nleaves_i32 as *mut i32).cast(),
        ];
        check_cu(
            api,
            unsafe {
                (api.cuLaunchKernel)(
                    fn_leaves,
                    ((nleaves as u32) + 127) / 128, 1, 1,
                    128, 1, 1,
                    0,
                    self.stream,
                    args_leaves.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "apx_merkle_leaves_const64",
        )?;

        let mut count = (nleaves >> 1) as i32;
        while count >= 2 {
            let mut first = count;
            let mut cnt = count;
            let mut args_par: [*mut c_void; 4] = [
                (&mut p_key as *mut CUdeviceptr).cast(),
                (&mut p_tree as *mut CUdeviceptr).cast(),
                (&mut first as *mut i32).cast(),
                (&mut cnt as *mut i32).cast(),
            ];
            check_cu(
                api,
                unsafe {
                    (api.cuLaunchKernel)(
                        fn_parents,
                        ((count as u32) + 127) / 128, 1, 1,
                        128, 1, 1,
                        0,
                        self.stream,
                        args_par.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                },
                "apx_merkle_parents",
            )?;
            count >>= 1;
        }

        let heap_len = Constant64MerkleCache::heap_byte_len(&self.profile);
        let mut heap = vec![0u8; heap_len];
        check_cu(api, unsafe { (api.cuMemcpyDtoHAsync_v2)(heap.as_mut_ptr() as *mut c_void, self.d_tree.ptr, heap_len, self.stream) }, "DtoH(merkle_heap)")?;
        check_cu(api, unsafe { (api.cuStreamSynchronize)(self.stream) }, "cuStreamSynchronize(job)")?;

        let cache = Constant64MerkleCache::from_gpu_heap(job, heap)
            .context("GPU Merkle heap validation failed")?;

        // Upload `key_b` and generate packed `d_bas_b` once per job
        check_cu(api, unsafe { (api.cuMemcpyHtoDAsync_v2)(self.d_key_b.ptr, cache.key_b.as_ptr() as *const c_void, 32, self.stream) }, "HtoD(key_b)")?;
        let mut p_kb = self.d_key_b.ptr;
        let mut p_bas_b = self.bas_b_ptr();
        let mut rows_b = self.profile.n as i32;
        let mut is_b: i32 = 1;
        let mut args_bb: [*mut c_void; 4] = [
            (&mut p_kb as *mut CUdeviceptr).cast(),
            (&mut p_bas_b as *mut CUdeviceptr).cast(),
            (&mut rows_b as *mut i32).cast(),
            (&mut is_b as *mut i32).cast(),
        ];
        check_cu(
            api,
            unsafe {
                (api.cuLaunchKernel)(
                    fn_gen_basis,
                    (self.profile.n / 16) as c_uint, 1, 1,
                    64, 1, 1,
                    0,
                    self.stream,
                    args_bb.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "gen_basis(B)",
        )?;

        // In DirectL2 mode: generate `d_pairs_b` and materialize the ENTIRE `d_panel_b` ONCE PER BLOCK!
        if let EngineBackend::DirectL2 {
            fn_gen_u16_pairs,
            fn_materialize_128,
            d_pairs_a,
            d_pairs_b,
            d_panel_b,
            ..
        } = &self.backend
        {
            let mut p_kb1 = self.d_key_b.ptr;
            let mut p_kb2 = self.d_key_b.ptr;
            let mut p_pa = d_pairs_a.ptr;
            let mut p_pb = d_pairs_b.ptr;
            let mut k_i32 = self.profile.k as i32;
            let mut args_up: [*mut c_void; 5] = [
                (&mut p_kb1 as *mut CUdeviceptr).cast(),
                (&mut p_kb2 as *mut CUdeviceptr).cast(),
                (&mut p_pa as *mut CUdeviceptr).cast(),
                (&mut p_pb as *mut CUdeviceptr).cast(),
                (&mut k_i32 as *mut i32).cast(),
            ];
            let hash_lines = ((self.profile.k / 8) as u32 + 127) / 128;
            check_cu(
                api,
                unsafe {
                    (api.cuLaunchKernel)(
                        *fn_gen_u16_pairs,
                        hash_lines, 1, 1,
                        128, 1, 1,
                        0,
                        self.stream,
                        args_up.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                },
                "apx_gen_u16_pairs(B)",
            )?;

            let mut p_bb = self.bas_b_ptr();
            let mut p_pb2 = d_pairs_b.ptr;
            let mut p_out_b = d_panel_b.ptr;
            let mut start_frag: i32 = 0;
            let mut k_mat = self.profile.k as i32;
            let mut args_mb: [*mut c_void; 5] = [
                (&mut p_bb as *mut CUdeviceptr).cast(),
                (&mut p_pb2 as *mut CUdeviceptr).cast(),
                (&mut p_out_b as *mut CUdeviceptr).cast(),
                (&mut start_frag as *mut i32).cast(),
                (&mut k_mat as *mut i32).cast(),
            ];
            let n_tiles = (self.profile.n / 128) as c_uint;
            check_cu(
                api,
                unsafe {
                    (api.cuLaunchKernel)(
                        *fn_materialize_128,
                        n_tiles, 4, 1,
                        256, 1, 1,
                        0,
                        self.stream,
                        args_mb.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                },
                "apx_materialize_panel_128(B)",
            )?;
        }

        check_cu(api, unsafe { (api.cuStreamSynchronize)(self.stream) }, "cuStreamSynchronize(panelB)")?;
        Ok(cache)
    }

    /// Executes one full mining round for `nonce` on `variant_idx`.
    pub fn mine_round(
        &mut self,
        job: &PreparedJob,
        cache: &Constant64MerkleCache,
        nonce: u64,
        epoch: u32,
        variant_idx: usize,
        greedy_reorder: bool,
    ) -> Result<RoundOutcome> {
        self.bind_context()?;
        let api = cuda()?;
        let (_sig, _root_a, key_a) = cache.derive_attempt(job, nonce);

        check_cu(api, unsafe { (api.cuEventRecord)(self.ev_round_start, self.stream) }, "cuEventRecord(round_start)")?;
        check_cu(api, unsafe { (api.cuMemcpyHtoDAsync_v2)(self.d_key_a.ptr, key_a.as_ptr() as *const c_void, 32, self.stream) }, "HtoD(key_a)")?;
        check_cu(api, unsafe { (api.cuMemsetD8Async)(self.ctrl.dev_hit_count_ptr(), 0, 4, self.stream) }, "Memset(hit_count)")?;

        match &self.backend {
            EngineBackend::DirectL2 {
                fn_gen_basis,
                fn_gen_u16_pairs,
                fn_materialize_128,
                fn_variants,
                d_pairs_a,
                d_pairs_b,
                d_panel_a,
                d_panel_b,
                panel_m_tiles,
                ..
            } => {
                let sampled_m = self.profile.sampled_m() as i32;
                let mut p_ka = self.d_key_a.ptr;
                let mut p_bas_a = self.bas_a_ptr();
                let mut rows_a = sampled_m;
                let mut is_b_zero: i32 = 0;
                let mut args_ba: [*mut c_void; 4] = [
                    (&mut p_ka as *mut CUdeviceptr).cast(),
                    (&mut p_bas_a as *mut CUdeviceptr).cast(),
                    (&mut rows_a as *mut i32).cast(),
                    (&mut is_b_zero as *mut i32).cast(),
                ];
                check_cu(
                    api,
                    unsafe {
                        (api.cuLaunchKernel)(
                            *fn_gen_basis,
                            (sampled_m as u32) / 16, 1, 1,
                            64, 1, 1,
                            0,
                            self.stream,
                            args_ba.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "apx_gen_basis_linear(A)",
                )?;

                let mut p_kb = self.d_key_b.ptr;
                let mut p_pa = d_pairs_a.ptr;
                let mut p_pb = d_pairs_b.ptr;
                let mut k_i32 = self.profile.k as i32;
                let mut args_up: [*mut c_void; 5] = [
                    (&mut p_ka as *mut CUdeviceptr).cast(),
                    (&mut p_kb as *mut CUdeviceptr).cast(),
                    (&mut p_pa as *mut CUdeviceptr).cast(),
                    (&mut p_pb as *mut CUdeviceptr).cast(),
                    (&mut k_i32 as *mut i32).cast(),
                ];
                let hash_lines = ((self.profile.k / 8) as u32 + 127) / 128;
                check_cu(
                    api,
                    unsafe {
                        (api.cuLaunchKernel)(
                            *fn_gen_u16_pairs,
                            hash_lines, 1, 1,
                            128, 1, 1,
                            0,
                            self.stream,
                            args_up.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "apx_gen_u16_pairs(A)",
                )?;

                check_cu(api, unsafe { (api.cuEventRecord)(self.ev_start, self.stream) }, "cuEventRecord(start)")?;

                let total_m_tiles = self.profile.sampled_m() / 128;
                let n_tiles = self.profile.n / 128;
                let v_idx = variant_idx % 3;
                let use_v2_3stg = v_idx == 2 && (n_tiles % 2 == 0);
                let chosen_fn = if use_v2_3stg { fn_variants[2] } else { fn_variants[v_idx.min(1)] };
                let block_threads: c_uint = if use_v2_3stg { 256 } else { 128 };
                let smem_bytes: c_uint = if use_v2_3stg { SMEM_DIRECT_3STG_BYTES as c_uint } else { SMEM_DIRECT_2CTA_BYTES as c_uint };
                let key_a_u32 = bytes32_to_u32x8(&key_a);
                let target_u32 = bytes32_to_u32x8(&job.bound_le);

                let mut base_bm = 0usize;
                let mut aborted = false;
                while base_bm < total_m_tiles {
                    if self.ctrl.epoch_atomic().load(Ordering::Acquire) != epoch {
                        aborted = true;
                        break;
                    }
                    let cur_m = (*panel_m_tiles).min(total_m_tiles - base_bm);

                    // 1. Materialize swizzled A panel into L2 cache (~0.015 ms)
                    let mut p_ba = self.bas_a_ptr();
                    let mut p_pa2 = d_pairs_a.ptr;
                    let mut p_out_a = d_panel_a.ptr;
                    let mut start_frag = (base_bm * 8) as i32;
                    let mut k_mat = self.profile.k as i32;
                    let mut args_ma: [*mut c_void; 5] = [
                        (&mut p_ba as *mut CUdeviceptr).cast(),
                        (&mut p_pa2 as *mut CUdeviceptr).cast(),
                        (&mut p_out_a as *mut CUdeviceptr).cast(),
                        (&mut start_frag as *mut i32).cast(),
                        (&mut k_mat as *mut i32).cast(),
                    ];
                    let mat_k_parts = optimal_mat_k_parts(
                        cur_m,
                        self.info.sm_count.max(1) as usize,
                        self.profile.k / 64,
                    );
                    check_cu(
                        api,
                        unsafe {
                            (api.cuLaunchKernel)(
                                *fn_materialize_128,
                                cur_m as c_uint, mat_k_parts, 1,
                                256, 1, 1,
                                0,
                                self.stream,
                                args_ma.as_mut_ptr(),
                                std::ptr::null_mut(),
                            )
                        },
                        "apx_materialize_panel_128(A)",
                    )?;

                    // 2. Launch Hyper-L2 Direct GEMM across `cur_m * n_tiles` CTAs
                    let mut dp = ApxDirectL2Params {
                        panel_a: d_panel_a.ptr,
                        panel_b: d_panel_b.ptr,
                        m_tiles: cur_m as i32,
                        n_tiles: n_tiles as i32,
                        num_ks: (self.profile.k / 64) as i32,
                        base_bm: base_bm as i32,
                        total_n_warps: (self.profile.n / 64) as i32,
                        hit_cap: self.hit_cap as i32,
                        hit_count: self.ctrl.dev_hit_count_ptr(),
                        hit_groups: self.ctrl.dev_hit_groups_ptr(),
                        key_a: key_a_u32,
                        target_le: target_u32,
                    };
                    let mut launch_args: [*mut c_void; 1] = [(&mut dp as *mut ApxDirectL2Params).cast()];
                    let grid_x = if use_v2_3stg {
                        (cur_m * (n_tiles / 2)) as c_uint
                    } else {
                        (cur_m * n_tiles) as c_uint
                    };
                    check_cu(
                        api,
                        unsafe {
                            (api.cuLaunchKernel)(
                                chosen_fn,
                                grid_x, 1, 1,
                                block_threads, 1, 1,
                                smem_bytes,
                                self.stream,
                                launch_args.as_mut_ptr(),
                                std::ptr::null_mut(),
                            )
                        },
                        self.variant_name(v_idx),
                    )?;

                    base_bm += cur_m;
                }

                check_cu(api, unsafe { (api.cuEventRecord)(self.ev_end, self.stream) }, "cuEventRecord(end)")?;
                check_cu(api, unsafe { (api.cuEventSynchronize)(self.ev_end) }, "cuEventSynchronize")?;

                let mut round_ms = 0.0f32;
                let mut kernel_ms = 0.0f32;
                check_cu(api, unsafe { (api.cuEventElapsedTime)(&mut round_ms, self.ev_round_start, self.ev_end) }, "cuEventElapsedTime(round)")?;
                check_cu(api, unsafe { (api.cuEventElapsedTime)(&mut kernel_ms, self.ev_start, self.ev_end) }, "cuEventElapsedTime(kernel)")?;

                aborted |= self.ctrl.epoch_atomic().load(Ordering::Acquire) != epoch;
                let (total_hits, mut hit_groups) = self.ctrl.read_hits(self.hit_cap);
                hit_groups.sort_unstable();

                Ok(RoundOutcome {
                    round_ms: round_ms.max(kernel_ms),
                    kernel_ms,
                    total_hits,
                    hit_groups,
                    aborted,
                })
            }
            EngineBackend::Basis {
                fn_gen_basis,
                fn_gen_raw_pairs,
                fn_reorder_pairs,
                fn_variants,
                smem_bytes,
                d_pairs,
                ..
            } => {
                let sampled_m = self.profile.sampled_m() as i32;
                let mut p_ka = self.d_key_a.ptr;
                let mut p_bas_a = self.bas_a_ptr();
                let mut rows_a = sampled_m;
                let mut is_b_zero: i32 = 0;
                let mut args_ba: [*mut c_void; 4] = [
                    (&mut p_ka as *mut CUdeviceptr).cast(),
                    (&mut p_bas_a as *mut CUdeviceptr).cast(),
                    (&mut rows_a as *mut i32).cast(),
                    (&mut is_b_zero as *mut i32).cast(),
                ];
                check_cu(
                    api,
                    unsafe {
                        (api.cuLaunchKernel)(
                            *fn_gen_basis,
                            (sampled_m as u32) / 16, 1, 1,
                            64, 1, 1,
                            0,
                            self.stream,
                            args_ba.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "apx_gen_basis_arch(A)",
                )?;

                let mut p_kb = self.d_key_b.ptr;
                let mut p_pairs = d_pairs.ptr;
                let mut k_i32 = self.profile.k as i32;
                let mut args_rp: [*mut c_void; 4] = [
                    (&mut p_ka as *mut CUdeviceptr).cast(),
                    (&mut p_kb as *mut CUdeviceptr).cast(),
                    (&mut p_pairs as *mut CUdeviceptr).cast(),
                    (&mut k_i32 as *mut i32).cast(),
                ];
                let hash_lines = ((self.profile.k / 8) as u32 + 127) / 128;
                check_cu(
                    api,
                    unsafe {
                        (api.cuLaunchKernel)(
                            *fn_gen_raw_pairs,
                            hash_lines, 1, 1,
                            128, 1, 1,
                            0,
                            self.stream,
                            args_rp.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "apx_gen_raw_pairs",
                )?;

                let mut enable_g: i32 = if greedy_reorder { 1 } else { 0 };
                let mut args_ro: [*mut c_void; 3] = [
                    (&mut p_pairs as *mut CUdeviceptr).cast(),
                    (&mut k_i32 as *mut i32).cast(),
                    (&mut enable_g as *mut i32).cast(),
                ];
                check_cu(
                    api,
                    unsafe {
                        (api.cuLaunchKernel)(
                            *fn_reorder_pairs,
                            ((self.profile.k / 128) as u32 + 3) / 4, 1, 1,
                            128, 1, 1,
                            0,
                            self.stream,
                            args_ro.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    "apx_reorder_pairs_arch",
                )?;

                check_cu(api, unsafe { (api.cuEventRecord)(self.ev_start, self.stream) }, "cuEventRecord(start)")?;

                let mut kparams = ApxBasisParams {
                    bas_a: self.bas_a_ptr(),
                    bas_b: self.bas_b_ptr(),
                    pairs: d_pairs.ptr,
                    sampled_m,
                    n: self.profile.n as i32,
                    k: self.profile.k as i32,
                    key_a: self.d_key_a.ptr,
                    target_le: self.d_target_le.ptr,
                    hit_count: self.ctrl.dev_hit_count_ptr(),
                    hit_groups: self.ctrl.dev_hit_groups_ptr(),
                    hit_cap: self.hit_cap as i32,
                    abort_epoch_ptr: self.ctrl.dev_abort_ptr(),
                    expected_epoch: epoch,
                };
                let mut launch_args: [*mut c_void; 1] = [(&mut kparams as *mut ApxBasisParams).cast()];
                check_cu(
                    api,
                    unsafe {
                        (api.cuLaunchKernel)(
                            fn_variants[variant_idx % 4],
                            self.info.sm_count as c_uint, 1, 1,
                            256, 1, 1,
                            *smem_bytes as c_uint,
                            self.stream,
                            launch_args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                    self.variant_name(variant_idx),
                )?;

                check_cu(api, unsafe { (api.cuEventRecord)(self.ev_end, self.stream) }, "cuEventRecord(end)")?;
                check_cu(api, unsafe { (api.cuEventSynchronize)(self.ev_end) }, "cuEventSynchronize")?;

                let mut round_ms = 0.0f32;
                let mut kernel_ms = 0.0f32;
                check_cu(api, unsafe { (api.cuEventElapsedTime)(&mut round_ms, self.ev_round_start, self.ev_end) }, "cuEventElapsedTime(round)")?;
                check_cu(api, unsafe { (api.cuEventElapsedTime)(&mut kernel_ms, self.ev_start, self.ev_end) }, "cuEventElapsedTime(kernel)")?;

                let aborted = self.ctrl.epoch_atomic().load(Ordering::Acquire) != epoch;
                let (total_hits, mut hit_groups) = self.ctrl.read_hits(self.hit_cap);
                hit_groups.sort_unstable();

                Ok(RoundOutcome {
                    round_ms: round_ms.max(kernel_ms),
                    kernel_ms,
                    total_hits,
                    hit_groups,
                    aborted,
                })
            }
        }
    }

    /// Runs an interleaved benchmark across all variants of the active engine, verifies identical
    /// hit counts across every variant, and activates the fastest one!
    pub fn run_autotune(&mut self, job: &PreparedJob, cache: &Constant64MerkleCache, epoch: u32, passes: usize) -> Result<(usize, Vec<(String, f64)>)> {
        let nv = self.num_variants();
        let test_nonce = 0x1337_c0de_2026_0001u64;
        let base = self.mine_round(job, cache, test_nonce, epoch, 0, true)?;
        for v in 1..nv {
            let chk = self.mine_round(job, cache, test_nonce, epoch, v, true)?;
            let groups_match = if (base.total_hits as usize) <= self.hit_cap {
                chk.hit_groups == base.hit_groups
            } else {
                true
            };
            if chk.total_hits != base.total_hits || !groups_match {
                bail!(
                    "Auto-tune parity failure on GPU #{} ({} vs {}): hits {} vs {}",
                    self.info.ordinal,
                    self.variant_name(v),
                    self.variant_name(0),
                    chk.total_hits,
                    base.total_hits
                );
            }
        }

        let mut sum_ms = vec![0.0f64; nv];
        let mut nonce = test_nonce + 1;
        let reps = passes.max(3);
        for _ in 0..reps {
            for v in 0..nv {
                let r = self.mine_round(job, cache, nonce, epoch, v, true)?;
                nonce += 1;
                if r.aborted {
                    bail!("Auto-tune interrupted");
                }
                sum_ms[v] += r.round_ms as f64;
            }
        }

        let macs = self.profile.macs_per_round();
        let mut summary = Vec::with_capacity(nv);
        let mut winner = 0usize;
        let mut winner_avg = f64::MAX;
        for v in 0..nv {
            let avg_ms = sum_ms[v] / (reps as f64);
            let th_s = (macs / (avg_ms * 1e-3)) * 1e-12;
            summary.push((self.variant_name(v).to_string(), th_s));
            if avg_ms < winner_avg {
                winner_avg = avg_ms;
                winner = v;
            }
        }
        self.active_variant = winner;
        Ok((winner, summary))
    }
}

impl Drop for GpuWorkerEngine {
    fn drop(&mut self) {
        if let Ok(api) = cuda() {
            unsafe {
                let _ = (api.cuCtxSetCurrent)(self.ctx);
                let _ = (api.cuStreamSynchronize)(self.stream);
                let _ = (api.cuEventDestroy_v2)(self.ev_round_start);
                let _ = (api.cuEventDestroy_v2)(self.ev_start);
                let _ = (api.cuEventDestroy_v2)(self.ev_end);
                let _ = (api.cuStreamDestroy_v2)(self.stream);
            }
        }
    }
}
