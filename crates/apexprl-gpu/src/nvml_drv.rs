//! Dynamic runtime loader (`dlopen`) for NVIDIA Management Library (`libnvidia-ml.so.1`).
//! Powers HiveOS telemetry (`temp`, `mtemp` GDDR6X/GDDR7, `fan`, `power`, `bus_numbers`)
//! and hardware efficiency tuning (`--lock-mem-idle`, `--cclock-lock`, `--pl`).

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::sync::OnceLock;

type nvmlReturn_t = c_int;
type nvmlDevice_t = *mut c_void;

const NVML_TEMPERATURE_GPU: c_uint = 0;
const NVML_CLOCK_SM: c_uint = 1;
const NVML_CLOCK_MEM: c_uint = 2;
const NVML_FI_DEV_MEMORY_TEMP: c_uint = 82;

#[repr(C)]
struct nvmlPciInfo_v3 {
    bus_id_legacy: [c_char; 16],
    domain: c_uint,
    bus: c_uint,
    device: c_uint,
    pci_device_id: c_uint,
    pci_sub_system_id: c_uint,
    bus_id: [c_char; 32],
}

#[repr(C)]
#[derive(Copy, Clone)]
union nvmlValue_t {
    d_val: f64,
    ui_val: c_uint,
    ul_val: u64,
    ull_val: u64,
    sll_val: i64,
}

#[repr(C)]
struct nvmlFieldValue_t {
    field_id: c_uint,
    scope_id: c_uint,
    timestamp: i64,
    latency_usec: i64,
    value_type: c_uint,
    nvml_return: nvmlReturn_t,
    value: nvmlValue_t,
}

struct NvmlApi {
    get_by_pci: Option<unsafe extern "C" fn(*const c_char, *mut nvmlDevice_t) -> nvmlReturn_t>,
    get_by_idx: unsafe extern "C" fn(c_uint, *mut nvmlDevice_t) -> nvmlReturn_t,
    get_pci_info: Option<unsafe extern "C" fn(nvmlDevice_t, *mut nvmlPciInfo_v3) -> nvmlReturn_t>,
    get_temp: unsafe extern "C" fn(nvmlDevice_t, c_uint, *mut c_uint) -> nvmlReturn_t,
    get_fields: Option<unsafe extern "C" fn(nvmlDevice_t, c_int, *mut nvmlFieldValue_t) -> nvmlReturn_t>,
    get_fan: unsafe extern "C" fn(nvmlDevice_t, *mut c_uint) -> nvmlReturn_t,
    get_power: unsafe extern "C" fn(nvmlDevice_t, *mut c_uint) -> nvmlReturn_t,
    get_clock: unsafe extern "C" fn(nvmlDevice_t, c_uint, *mut c_uint) -> nvmlReturn_t,
    set_mem_lock: Option<unsafe extern "C" fn(nvmlDevice_t, c_uint, c_uint) -> nvmlReturn_t>,
    reset_mem_lock: Option<unsafe extern "C" fn(nvmlDevice_t) -> nvmlReturn_t>,
    set_gpu_lock: Option<unsafe extern "C" fn(nvmlDevice_t, c_uint, c_uint) -> nvmlReturn_t>,
    reset_gpu_lock: Option<unsafe extern "C" fn(nvmlDevice_t) -> nvmlReturn_t>,
    set_pl: Option<unsafe extern "C" fn(nvmlDevice_t, c_uint) -> nvmlReturn_t>,
}

unsafe impl Send for NvmlApi {}
unsafe impl Sync for NvmlApi {}

static NVML_API: OnceLock<Option<NvmlApi>> = OnceLock::new();

unsafe fn sym_opt<T>(h: *mut c_void, name: &str) -> Option<T> {
    let cs = CString::new(name).ok()?;
    let p = libc::dlsym(h, cs.as_ptr());
    if p.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&p))
    }
}

fn load_nvml() -> Option<NvmlApi> {
    let candidates = [
        "libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/local/nvidia/lib64/libnvidia-ml.so.1",
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
    ];
    let mut h = std::ptr::null_mut();
    for path in candidates {
        let cs = CString::new(path).unwrap();
        h = unsafe { libc::dlopen(cs.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !h.is_null() {
            break;
        }
    }
    if h.is_null() {
        return None;
    }

    unsafe {
        let init: unsafe extern "C" fn() -> nvmlReturn_t = sym_opt(h, "nvmlInit_v2")?;
        if init() != 0 {
            return None;
        }
        Some(NvmlApi {
            get_by_pci: sym_opt(h, "nvmlDeviceGetHandleByPciBusId_v2"),
            get_by_idx: sym_opt(h, "nvmlDeviceGetHandleByIndex_v2")?,
            get_pci_info: sym_opt(h, "nvmlDeviceGetPciInfo_v3"),
            get_temp: sym_opt(h, "nvmlDeviceGetTemperature")?,
            get_fields: sym_opt(h, "nvmlDeviceGetFieldValues"),
            get_fan: sym_opt(h, "nvmlDeviceGetFanSpeed")?,
            get_power: sym_opt(h, "nvmlDeviceGetPowerUsage")?,
            get_clock: sym_opt(h, "nvmlDeviceGetClockInfo")?,
            set_mem_lock: sym_opt(h, "nvmlDeviceSetMemoryLockedClocks"),
            reset_mem_lock: sym_opt(h, "nvmlDeviceResetMemoryLockedClocks"),
            set_gpu_lock: sym_opt(h, "nvmlDeviceSetGpuLockedClocks"),
            reset_gpu_lock: sym_opt(h, "nvmlDeviceResetGpuLockedClocks"),
            set_pl: sym_opt(h, "nvmlDeviceSetPowerManagementLimit"),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuSensors {
    pub temp_c: u32,
    pub mem_temp_c: u32,
    pub fan_pct: u32,
    pub power_w: f32,
    pub sm_clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub pci_bus_id: u32,
}

pub struct NvmlHandle {
    dev: nvmlDevice_t,
    locked_mem: bool,
    locked_core: bool,
}

unsafe impl Send for NvmlHandle {}
unsafe impl Sync for NvmlHandle {}

impl NvmlHandle {
    pub fn open(cuda_ordinal: usize, pci_bus: u32) -> Option<Self> {
        let api = NVML_API.get_or_init(load_nvml).as_ref()?;
        let mut dev: nvmlDevice_t = std::ptr::null_mut();
        unsafe {
            if let Some(by_pci) = api.get_by_pci {
                let bus_str = CString::new(format!("0000:{pci_bus:02x}:00.0")).ok()?;
                if by_pci(bus_str.as_ptr(), &mut dev) == 0 && !dev.is_null() {
                    return Some(Self { dev, locked_mem: false, locked_core: false });
                }
            }
            if (api.get_by_idx)(cuda_ordinal as c_uint, &mut dev) == 0 && !dev.is_null() {
                return Some(Self { dev, locked_mem: false, locked_core: false });
            }
        }
        None
    }

    pub fn apply_tuning(&mut self, lock_mem_idle: bool, core_lock_mhz: Option<u32>, power_limit_w: Option<u32>) -> Vec<String> {
        let mut notes = Vec::new();
        let Some(api) = NVML_API.get().and_then(|o| o.as_ref()) else {
            return notes;
        };
        unsafe {
            if let (Some(w), Some(set_pl)) = (power_limit_w, api.set_pl) {
                let rc = set_pl(self.dev, w * 1000);
                if rc == 0 {
                    notes.push(format!("PL={}W", w));
                } else {
                    notes.push(format!("PL={}W skipped(rc={})", w, rc));
                }
            }
            if lock_mem_idle {
                // Preserve existing external memory locks (e.g. PenguinBurner / HiveOS OC at 810 MHz),
                // while still locking unlocked GPUs that may briefly idle at 405 MHz P8 before kernel launch.
                let mut cur_mem_mhz = 0u32;
                let got_mem = (api.get_clock)(self.dev, NVML_CLOCK_MEM, &mut cur_mem_mhz) == 0;
                if got_mem && cur_mem_mhz >= 700 && cur_mem_mhz <= 1050 {
                    notes.push(format!("MemLock=preserved({}MHz)", cur_mem_mhz));
                } else if let Some(set_mem) = api.set_mem_lock {
                    // Try P8 810 MHz first, then P5 5001 MHz fallback
                    if set_mem(self.dev, 405, 810) == 0 {
                        self.locked_mem = true;
                        notes.push("MemLock=810MHz(Sleep)".into());
                    } else if set_mem(self.dev, 810, 5001) == 0 {
                        self.locked_mem = true;
                        notes.push("MemLock=5001MHz(Low)".into());
                    } else {
                        notes.push("MemLock=unsupported/non-root".into());
                    }
                }
            }
            if let (Some(mhz), Some(set_gpu)) = (core_lock_mhz, api.set_gpu_lock) {
                let rc = set_gpu(self.dev, mhz, mhz);
                if rc == 0 {
                    self.locked_core = true;
                    notes.push(format!("CoreLock={}MHz", mhz));
                } else {
                    notes.push(format!("CoreLock={}MHz skipped(rc={})", mhz, rc));
                }
            }
        }
        notes
    }

    pub fn read_sensors(&self, fallback_bus: u32) -> GpuSensors {
        let mut s = GpuSensors {
            pci_bus_id: fallback_bus,
            ..Default::default()
        };
        let Some(api) = NVML_API.get().and_then(|o| o.as_ref()) else {
            return s;
        };
        unsafe {
            let mut v = 0u32;
            if (api.get_temp)(self.dev, NVML_TEMPERATURE_GPU, &mut v) == 0 {
                s.temp_c = v;
            }
            if let Some(get_fields) = api.get_fields {
                let mut fv = nvmlFieldValue_t {
                    field_id: NVML_FI_DEV_MEMORY_TEMP,
                    scope_id: 0,
                    timestamp: 0,
                    latency_usec: 0,
                    value_type: 0,
                    nvml_return: -1,
                    value: nvmlValue_t { ui_val: 0 },
                };
                if get_fields(self.dev, 1, &mut fv) == 0 && fv.nvml_return == 0 {
                    s.mem_temp_c = fv.value.ui_val;
                }
            }
            if (api.get_fan)(self.dev, &mut v) == 0 {
                s.fan_pct = v;
            }
            if (api.get_power)(self.dev, &mut v) == 0 {
                s.power_w = (v as f32) / 1000.0;
            }
            if (api.get_clock)(self.dev, NVML_CLOCK_SM, &mut v) == 0 {
                s.sm_clock_mhz = v;
            }
            if (api.get_clock)(self.dev, NVML_CLOCK_MEM, &mut v) == 0 {
                s.mem_clock_mhz = v;
            }
            if let Some(get_pci) = api.get_pci_info {
                let mut pci: nvmlPciInfo_v3 = std::mem::zeroed();
                if get_pci(self.dev, &mut pci) == 0 {
                    s.pci_bus_id = pci.bus;
                }
            }
        }
        s
    }
}

impl Drop for NvmlHandle {
    fn drop(&mut self) {
        if let Some(api) = NVML_API.get().and_then(|o| o.as_ref()) {
            unsafe {
                if self.locked_mem {
                    if let Some(f) = api.reset_mem_lock {
                        let _ = f(self.dev);
                    }
                }
                if self.locked_core {
                    if let Some(f) = api.reset_gpu_lock {
                        let _ = f(self.dev);
                    }
                }
            }
        }
    }
}
