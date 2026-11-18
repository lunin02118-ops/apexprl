//! Dynamic runtime loader (`dlopen`) for NVIDIA CUDA Driver API (`libcuda.so.1`).
//! Zero link-time dependency on `libcuda.so` or `libcudart.so`, ensuring seamless
//! GLIBC 2.31 portability across all HiveOS Linux images.

#![allow(non_camel_case_types, non_snake_case, clippy::missing_safety_doc)]

use anyhow::{bail, Result};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::OnceLock;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUdeviceptr = u64;

pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_EVENT_BLOCKING_SYNC: c_uint = 1;
pub const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 2;

pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: c_int = 33;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE: c_int = 108;

pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
pub const CU_LIMIT_PERSISTING_L2_CACHE_SIZE: c_int = 0x06;

pub struct CudaApi {
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuGetErrorString: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxSetLimit: unsafe extern "C" fn(c_int, usize) -> CUresult,
    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    pub cuFuncSetAttribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint, c_uint, c_uint,
        c_uint, c_uint, c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemGetInfo_v2: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    pub cuMemHostAlloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub cuMemHostGetDevicePointer_v2: unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, c_uint) -> CUresult,
    pub cuMemFreeHost: unsafe extern "C" fn(*mut c_void) -> CUresult,
    pub cuMemsetD8Async: unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
    pub cuMemcpyHtoDAsync_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoHAsync_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamDestroy_v2: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
    pub cuEventDestroy_v2: unsafe extern "C" fn(CUevent) -> CUresult,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

static CUDA_API: OnceLock<Result<CudaApi, String>> = OnceLock::new();

unsafe fn load_sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let cname = CString::new(name).unwrap();
    let ptr = libc::dlsym(handle, cname.as_ptr());
    if ptr.is_null() {
        return Err(format!("Missing symbol {name} in libcuda.so.1"));
    }
    Ok(std::mem::transmute_copy(&ptr))
}

fn init_cuda_api() -> Result<CudaApi, String> {
    let candidates = [
        "libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
    ];
    let mut handle = std::ptr::null_mut();
    for path in candidates {
        let cpath = CString::new(path).unwrap();
        handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !handle.is_null() {
            break;
        }
    }
    if handle.is_null() {
        return Err("Could not dlopen libcuda.so.1 (is the NVIDIA driver installed?)".into());
    }

    unsafe {
        let api = CudaApi {
            cuInit: load_sym(handle, "cuInit")?,
            cuDriverGetVersion: load_sym(handle, "cuDriverGetVersion")?,
            cuGetErrorString: load_sym(handle, "cuGetErrorString")?,
            cuDeviceGetCount: load_sym(handle, "cuDeviceGetCount")?,
            cuDeviceGet: load_sym(handle, "cuDeviceGet")?,
            cuDeviceGetName: load_sym(handle, "cuDeviceGetName")?,
            cuDeviceGetAttribute: load_sym(handle, "cuDeviceGetAttribute")?,
            cuDevicePrimaryCtxRetain: load_sym(handle, "cuDevicePrimaryCtxRetain")?,
            cuCtxSetCurrent: load_sym(handle, "cuCtxSetCurrent")?,
            cuCtxSetLimit: load_sym(handle, "cuCtxSetLimit")?,
            cuModuleLoadData: load_sym(handle, "cuModuleLoadData")?,
            cuModuleGetFunction: load_sym(handle, "cuModuleGetFunction")?,
            cuFuncSetAttribute: load_sym(handle, "cuFuncSetAttribute")?,
            cuLaunchKernel: load_sym(handle, "cuLaunchKernel")?,
            cuMemAlloc_v2: load_sym(handle, "cuMemAlloc_v2")?,
            cuMemFree_v2: load_sym(handle, "cuMemFree_v2")?,
            cuMemGetInfo_v2: load_sym(handle, "cuMemGetInfo_v2")?,
            cuMemHostAlloc: load_sym(handle, "cuMemHostAlloc")?,
            cuMemHostGetDevicePointer_v2: load_sym(handle, "cuMemHostGetDevicePointer_v2")?,
            cuMemFreeHost: load_sym(handle, "cuMemFreeHost")?,
            cuMemsetD8Async: load_sym(handle, "cuMemsetD8Async")?,
            cuMemcpyHtoDAsync_v2: load_sym(handle, "cuMemcpyHtoDAsync_v2")?,
            cuMemcpyDtoHAsync_v2: load_sym(handle, "cuMemcpyDtoHAsync_v2")?,
            cuMemcpyDtoH_v2: load_sym(handle, "cuMemcpyDtoH_v2")?,
            cuStreamCreate: load_sym(handle, "cuStreamCreate")?,
            cuStreamSynchronize: load_sym(handle, "cuStreamSynchronize")?,
            cuStreamDestroy_v2: load_sym(handle, "cuStreamDestroy_v2")?,
            cuEventCreate: load_sym(handle, "cuEventCreate")?,
            cuEventRecord: load_sym(handle, "cuEventRecord")?,
            cuEventSynchronize: load_sym(handle, "cuEventSynchronize")?,
            cuEventElapsedTime: load_sym(handle, "cuEventElapsedTime")?,
            cuEventDestroy_v2: load_sym(handle, "cuEventDestroy_v2")?,
        };
        let rc = (api.cuInit)(0);
        if rc != 0 {
            return Err(format!("cuInit(0) failed with code {rc}"));
        }
        Ok(api)
    }
}

pub fn cuda() -> Result<&'static CudaApi> {
    match CUDA_API.get_or_init(init_cuda_api) {
        Ok(api) => Ok(api),
        Err(msg) => bail!("{msg}"),
    }
}

pub fn check_cu(api: &CudaApi, rc: CUresult, op: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let mut s: *const c_char = std::ptr::null();
    unsafe { (api.cuGetErrorString)(rc, &mut s) };
    let desc = if s.is_null() {
        "UNKNOWN_CUDA_ERROR".into()
    } else {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    };
    bail!("CUDA {op} failed: {desc} (code {rc})")
}
