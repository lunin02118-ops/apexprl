//! ApexPRL — Ultra-High-Efficiency 0%-Fee Pearl (PRL) Miner for NVIDIA RTX 40 (`sm_89`) & RTX 50 (`sm_120a`)
//! Engineered specifically for HiveOS Linux & Kryptex Stratum v2 (Gzip).

mod api_server;

use anyhow::{ensure, Context, Result};
use api_server::{spawn_api_server, PerGpuStats, RigTelemetry, TelemetrySensors};
use apexprl_gpu::{
    enumerate_gpus, optimal_direct_l2_n, optimal_edcp_n, EngineKind, GpuDeviceInfo, GpuWorkerEngine, TuneCacheFile,
};
use apexprl_pow::{
    cpu_eval_group_digest, optimal_kryptex_diff, verify_and_encode_proof, verify_and_encode_share_fast,
    Constant64MerkleCache, KernelProfile, PoolNotifyJob, PreparedJob,
};
use apexprl_stratum::{StratumEvent, StratumSession};
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use primitive_types::U256;
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

static STOP_SIGNAL: AtomicBool = AtomicBool::new(false);

extern "C" fn on_unix_signal(_sig: libc::c_int) {
    STOP_SIGNAL.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGINT, on_unix_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_unix_signal as *const () as libc::sighandler_t);
    }
}

#[derive(Parser, Debug)]
#[command(name = "apexprl", version, about = "ApexPRL 0%-Fee Pearl (PRL) GPU Miner for RTX 40 & RTX 50 (HiveOS + Kryptex v2)")]
struct Cli {
    /// Mining algorithm (`pearlhash` / `pearl`)
    #[arg(long, visible_alias = "algorithm", default_value = "pearlhash")]
    algo: String,

    /// Pool endpoint(s); specify multiple times for automatic failover
    #[arg(long = "pool", visible_alias = "url")]
    pools: Vec<String>,

    /// Wallet or Kryptex account (`krxXXXXXXX.rig01` or `prl1...`)
    #[arg(long, visible_alias = "user", default_value = "")]
    wallet: String,

    /// Worker name (optional if already suffixed in `--wallet`)
    #[arg(long)]
    worker: Option<String>,

    /// Pool password (`auto` computes optimal static `d=<N>` for Kryptex based on rig SM count!)
    #[arg(long, visible_alias = "password", default_value = "auto")]
    pass: String,

    /// Compute engine (`auto` = Strassen-24K on RTX 50 Blackwell / Basis-128 on RTX 40 Ada [default], `strassen` = v0.7 RAW-24K Basis-Strassen <2,2,2:7> Constant-0, `basis128` = v0.5 Wavefront-Exact 2-Opt 128x256, `direct` = Hyper-L2 2-CTA/SM, `edcp` = EDCP-120, `basis160` = v0.1)
    #[arg(long, default_value = "auto")]
    engine: String,

    /// Inner matrix dimension K (defaults: `65536` for `strassen`/`basis128` on gzip-v2, `16384` for `basis160`, `2048` for `direct`/`edcp`)
    #[arg(long)]
    k: Option<usize>,

    /// Comma-separated CUDA device indices (default: all RTX 40 / RTX 50 GPUs)
    #[arg(long, visible_alias = "gpu-id")]
    devices: Option<String>,

    /// List detected GPUs and exit
    #[arg(long)]
    list_devices: bool,

    /// Run offline GPU-vs-CPU + official `zk-pow` verification test (both Direct-L2 & Basis engines) and exit
    #[arg(long, visible_alias = "selftest")]
    self_test: bool,

    /// Run offline sustained full-matrix benchmark for N seconds and exit
    #[arg(long)]
    benchmark: Option<u64>,

    /// Calibrate CUDA/PTX kernel variants on startup
    #[arg(long)]
    tune: bool,

    /// Path to persistent JSON kernel auto-tune cache
    #[arg(long, default_value = "/tmp/apexprl_tune.json")]
    tune_cache: PathBuf,

    /// Lock GDDR6X/GDDR7 memory clocks to idle/sleep P-state via NVML (saves 35-75W per GPU!)
    #[arg(long)]
    lock_mem_idle: bool,

    /// Lock GPU SM core clock in MHz via NVML
    #[arg(long)]
    cclock_lock: Option<u32>,

    /// Set GPU Power Limit in Watts via NVML
    #[arg(long)]
    pl: Option<u32>,

    /// Local HTTP telemetry API bind address (`/api/v1/hiveos` & dashboard)
    #[arg(long, default_value = "127.0.0.1:4079")]
    api_bind: String,

    /// Disable ANSI colors in console output
    #[arg(long)]
    no_color: bool,
}

struct SharedActiveJob {
    epoch: u32,
    job: PreparedJob,
    cache: Constant64MerkleCache,
}

struct RawHitCandidate {
    gpu_ordinal: usize,
    epoch: u32,
    job_id: String,
    nonce: u64,
    group: u32,
    gpu_macs_per_sec: u64,
}

struct VerifiedShare {
    gpu_ordinal: usize,
    job_id: String,
    wire_proof_b64: String,
    raw_len: usize,
    wire_len: usize,
    rig_macs_per_sec: u64,
}

fn select_gpus(filter: Option<&str>) -> Result<Vec<GpuDeviceInfo>> {
    let all = enumerate_gpus()?;
    if let Some(spec) = filter {
        let mut chosen = Vec::new();
        for tok in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let idx: usize = tok.parse().with_context(|| format!("Invalid GPU index '{tok}'"))?;
            let dev = all
                .iter()
                .find(|d| d.ordinal == idx)
                .with_context(|| format!("GPU #{idx} not found"))?
                .clone();
            ensure!(dev.is_supported(), "GPU #{} ({}) is not sm_89 / sm_120+", dev.ordinal, dev.name);
            chosen.push(dev);
        }
        ensure!(!chosen.is_empty(), "No GPUs selected");
        Ok(chosen)
    } else {
        let supported: Vec<GpuDeviceInfo> = all.into_iter().filter(|d| d.is_supported()).collect();
        ensure!(
            !supported.is_empty(),
            "No supported NVIDIA RTX 40 (sm_89) or RTX 50 (sm_120) GPUs detected"
        );
        Ok(supported)
    }
}

fn parse_engine(raw: &str, gpus: &[GpuDeviceInfo]) -> Result<EngineKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => {
            if !gpus.is_empty() && gpus.iter().all(|g| g.is_blackwell()) {
                Ok(EngineKind::Strassen)
            } else {
                Ok(EngineKind::Basis128)
            }
        }
        "strassen" | "strassen24k" | "raw24k" | "s24k" => Ok(EngineKind::Strassen),
        "basis128" | "b128" | "uniform3" | "u3" => Ok(EngineKind::Basis128),
        "basis160" | "basis" | "legacy" => Ok(EngineKind::Basis160),
        "direct" | "hyper-l2" | "hl2" | "l2" => Ok(EngineKind::DirectL2),
        "edcp" | "edcp120" | "k120" | "cycle" | "forest" => Ok(EngineKind::Edcp120),
        other => anyhow::bail!("Unsupported --engine '{other}'; expected 'auto', 'strassen', 'basis128', 'direct', 'edcp', or 'basis160'"),
    }
}

fn build_profile(gpus: &[GpuDeviceInfo], engine: EngineKind, k_opt: Option<usize>, gzip_v2: bool) -> Result<KernelProfile> {
    match engine {
        EngineKind::Strassen => {
            let raw_k = k_opt.unwrap_or(65536);
            let active_k = if !gzip_v2 && raw_k > 4096 {
                println!("[stratum] Pool did not negotiate gzip-v2; stepping K={raw_k} -> 4096 to fit plain proof limit");
                4096
            } else {
                raw_k
            };
            let sm_counts: Vec<usize> = gpus.iter().map(|g| g.sm_count.max(1) as usize).collect();
            KernelProfile::strassen_128x256_sm_aligned(&sm_counts, active_k)
        }
        EngineKind::Basis128 => {
            let raw_k = k_opt.unwrap_or(65536);
            let active_k = if !gzip_v2 && raw_k > 4096 {
                println!("[stratum] Pool did not negotiate gzip-v2; stepping K={raw_k} -> 4096 to fit plain proof limit");
                4096
            } else {
                raw_k
            };
            let sm_counts: Vec<usize> = gpus.iter().map(|g| g.sm_count.max(1) as usize).collect();
            KernelProfile::basis_128x256_sm_aligned(&sm_counts, active_k)
        }
        EngineKind::Basis160 => {
            let raw_k = k_opt.unwrap_or(16384);
            let active_k = if !gzip_v2 && raw_k > 4096 {
                println!("[stratum] Pool did not negotiate gzip-v2; stepping K={raw_k} -> 4096 to fit plain proof limit");
                4096
            } else {
                raw_k
            };
            KernelProfile::basis_160x256(active_k)
        }
        EngineKind::DirectL2 => {
            let raw_k = k_opt.unwrap_or(2048);
            let active_k = if !gzip_v2 && raw_k > 4096 {
                println!("[stratum] Pool did not negotiate gzip-v2; stepping K={raw_k} -> 4096");
                4096
            } else {
                raw_k
            };
            let min_l2 = gpus
                .iter()
                .map(|g| g.l2_bytes)
                .filter(|&b| b > 0)
                .min()
                .unwrap_or(32 * 1024 * 1024);
            let n = optimal_direct_l2_n(min_l2, active_k);
            KernelProfile::direct_l2_128x128(32768, n, active_k)
        }
        EngineKind::Edcp120 => {
            let raw_k = k_opt.unwrap_or(2048);
            let active_k = if !gzip_v2 && raw_k > 4096 {
                println!("[stratum] Pool did not negotiate gzip-v2; stepping K={raw_k} -> 4096");
                4096
            } else {
                raw_k
            };
            let min_l2 = gpus
                .iter()
                .map(|g| g.l2_bytes)
                .filter(|&b| b > 0)
                .min()
                .unwrap_or(32 * 1024 * 1024);
            let n = optimal_edcp_n(min_l2, active_k);
            KernelProfile::edcp_120(32768, n, active_k)
        }
    }
}

fn verify_selftest_profile(gpus: &[GpuDeviceInfo], cli: &Cli, label: &str, profile: KernelProfile) -> Result<()> {
    let header_hex = "01000000".to_string() + &"aa".repeat(32) + &"bb".repeat(32) + "78563412" + "ffff0f1e";
    let target_hex = "000001".to_string() + &"ff".repeat(29);
    let notify = PoolNotifyJob::from_stratum_params(&json!({
        "job_id": format!("selftest_{label}_v3"),
        "header": header_hex,
        "target": target_hex,
        "height": 140000,
        "cert_version": 3
    }))?;
    let job = PreparedJob::new(notify, profile)?;
    let cpu_cache = Constant64MerkleCache::build_cpu(&job)?;

    for info in gpus {
        let (mut eng, _) = GpuWorkerEngine::new(
            info.clone(),
            profile,
            4096,
            cli.lock_mem_idle,
            cli.cclock_lock,
            cli.pl,
        )?;
        let gpu_cache = eng.prepare_job(&job, 1)?;
        ensure!(
            gpu_cache.root_b == cpu_cache.root_b && gpu_cache.key_b == cpu_cache.key_b,
            "GPU #{} [{label}] Merkle root mismatch vs CPU reference!",
            info.ordinal
        );

        let nonce = 0x2026_1118_a9ef_0042u64;
        let (_sig, _root_a, key_a) = gpu_cache.derive_attempt(&job, nonce);

        let ref_base = eng.mine_round(&job, &gpu_cache, nonce, 1, 0, false)?;
        ensure!(
            ref_base.total_hits > 0,
            "Expected >0 hits in [{label}] selftest on GPU #{}, got 0",
            info.ordinal
        );

        let nv = eng.num_variants();
        for v in 0..nv {
            let r = eng.mine_round(&job, &gpu_cache, nonce, 1, v, true)?;
            ensure!(
                r.total_hits == ref_base.total_hits && r.hit_groups == ref_base.hit_groups,
                "GPU #{} [{label}] variant {} hit set mismatch vs baseline ({} vs {})!",
                info.ordinal,
                eng.variant_name(v),
                r.total_hits,
                ref_base.total_hits
            );
        }

        let bound_u256 = U256::from_little_endian(&job.bound_le);
        for (idx, &grp) in ref_base.hit_groups.iter().take(16).enumerate() {
            let (_tr, cpu_digest) = cpu_eval_group_digest(&profile, &key_a, &gpu_cache.key_b, grp);
            ensure!(
                U256::from_little_endian(&cpu_digest) <= bound_u256,
                "GPU #{} [{label}] reported false hit on group {}!",
                info.ordinal,
                grp
            );
            if idx < 3 {
                let proof = gpu_cache.build_plain_proof(grp, nonce)?;
                let (_b64, raw_b, gz_b) = verify_and_encode_proof(&job, &proof, true)?;
                if idx == 0 {
                    println!(
                        "[selftest] GPU #{} ({}, sm_{}{}) [{label} {}x{}x{}]: {} hits matched CPU + all {} variants + zk-pow V3 OK (gzip {}B -> {}B, {:.1}x)!",
                        info.ordinal,
                        info.name,
                        info.sm_major,
                        info.sm_minor,
                        profile.m,
                        profile.n,
                        profile.k,
                        ref_base.total_hits,
                        nv,
                        raw_b,
                        gz_b,
                        (raw_b as f64) / (gz_b.max(1) as f64)
                    );
                }
            }
        }
    }
    Ok(())
}

/// Runs rigorous Gate G0 quad-engine hardware verification across `Basis128`, `Edcp120`, `DirectL2`, and `Basis160`:
/// 1. Builds Merkle heap on GPU (including rectangular `M > N` subtree & `EDCP` `b_salt` plan) and checks root vs CPU.
/// 2. Runs every kernel variant (`ada128/bw128`, `edcp-2cta`, `hl2-2cta`, `v0..v3`) for 100% hit-set parity.
/// 3. Cross-checks GPU hit digests against exact CPU `cpu_eval_group_digest`.
/// 4. Verifies full `PlainProof` via official `zk-pow::verify_plain_proof` V3 + `gzip`!
fn run_hardware_selftest(gpus: &[GpuDeviceInfo], cli: &Cli) -> Result<()> {
    println!("[selftest] Starting 5-engine hardware + zk-pow V3 verification across {} GPU(s)...", gpus.len());
    // 1. v0.7 On-Chip RAW-24K Basis-Strassen <2,2,2:7> Constant-0 (power-of-two AND SM-aligned odd-leaf M < N; K=4096 exercises rotl_xor<13>)
    verify_selftest_profile(gpus, cli, "strassen-24k", KernelProfile::with_dims_strassen_128x256(1024, 1024, 4096)?)?;
    verify_selftest_profile(gpus, cli, "strassen-24k-sm", KernelProfile::with_dims_strassen_128x256(400, 512, 4096)?)?;
    // 2. v0.5 Persistent-A Unpredicated Basis-128x256 (power-of-two AND SM-aligned odd-leaf M < N! K=4096 exercises rotl_xor<13>)
    verify_selftest_profile(gpus, cli, "basis-128", KernelProfile::with_dims_128x256(1024, 1024, 4096)?)?;
    verify_selftest_profile(gpus, cli, "basis-128-sm", KernelProfile::with_dims_128x256(400, 512, 4096)?)?;
    // 3. v0.4 Zero-Outlier EDCP-120 Cycle-Packing (Constant-0 + b_salt, K=2048)
    verify_selftest_profile(gpus, cli, "edcp-120", KernelProfile::edcp_120(1024, 512, 2048)?)?;
    // 4. v0.4 Zero-Eviction Hyper-L2 Direct rectangular M > N
    verify_selftest_profile(gpus, cli, "direct-l2", KernelProfile::direct_l2_128x128(1024, 512, 2048)?)?;
    // 5. v0.1 Legacy Basis-160x256
    verify_selftest_profile(gpus, cli, "basis-160", KernelProfile::with_dims(1024, 1024, 2048)?)?;
    println!("[selftest] ALL CHECKS PASSED (BIT-EXACT GPU == CPU == ZK-POW V3 FOR ALL 5 ENGINES)!");
    Ok(())
}

fn run_offline_benchmark(gpus: &[GpuDeviceInfo], cli: &Cli, seconds: u64) -> Result<()> {
    let engine = parse_engine(&cli.engine, gpus)?;
    let profile = build_profile(gpus, engine, cli.k, true)?;
    println!(
        "[bench] ApexPRL v{} Benchmark [engine={}]: M={} (sampled {}), N={}, K={} ({:.2} TMAC/round) for {}s...",
        env!("CARGO_PKG_VERSION"),
        cli.engine,
        profile.m,
        profile.sampled_m(),
        profile.n,
        profile.k,
        profile.macs_per_round() * 1e-12,
        seconds
    );

    let header_hex = "01000000".to_string() + &"12".repeat(32) + &"34".repeat(32) + "78563412" + "ffff001d";
    // Exact Kryptex d=2097152 target: 2^203 - 1 (53 leading zero bits = 6 zero bytes + 0x07)
    let target_hex = "00000000000007".to_string() + &"ff".repeat(25);
    let notify = PoolNotifyJob::from_stratum_params(&json!({
        "job_id": "bench_kryptex_v3",
        "header": header_hex,
        "target": target_hex,
        "height": 140001,
        "cert_version": 3
    }))?;
    let job = PreparedJob::new(notify, profile)?;

    let telemetry = Arc::new(RwLock::new(RigTelemetry {
        version: env!("CARGO_PKG_VERSION").to_string(),
        pool_url: "offline://benchmark".to_string(),
        compression: "gzip".to_string(),
        active_k: profile.k,
        gpus: gpus
            .iter()
            .map(|g| PerGpuStats {
                ordinal: g.ordinal,
                name: g.name.clone(),
                arch: format!("sm_{}{}", g.sm_major, g.sm_minor),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }));
    spawn_api_server(cli.api_bind.clone(), telemetry.clone(), Instant::now());

    for (slot_i, info) in gpus.iter().enumerate() {
        let (mut eng, notes) = GpuWorkerEngine::new(
            info.clone(),
            profile,
            256,
            cli.lock_mem_idle,
            cli.cclock_lock,
            cli.pl,
        )?;
        println!("[bench] GPU #{} layout: {}", info.ordinal, eng.layout_summary());
        if !notes.is_empty() {
            println!("[bench] GPU #{} NVML tuning: {}", info.ordinal, notes.join(", "));
        }
        let cache = eng.prepare_job(&job, 1)?;
        if cli.tune {
            let (win, scores) = eng.run_autotune(&job, &cache, 1, 3)?;
            let summary: Vec<String> = scores.iter().map(|(n, r)| format!("{n}={r:.2}TH/s")).collect();
            println!("[tune] GPU #{} selected '{}' ({})", info.ordinal, eng.variant_name(win), summary.join(", "));
        }

        // Short warm-up via zero-bubble async ping-pong pipeline so boost clocks & L2 stabilize
        let mut bench_nonce = 500u64;
        let active_var = eng.active_variant;
        for _ in 0..4 {
            let _ = eng.mine_round_pipelined(
                &job,
                &cache,
                &mut || {
                    let n = bench_nonce;
                    bench_nonce += 1;
                    n
                },
                1,
                active_var,
            )?;
        }

        let t0 = Instant::now();
        let mut last_sens_at = Instant::now() - Duration::from_secs(10);
        let mut rounds = 0u64;
        let mut round_sec = 0.0f64;
        let mut kernel_sec = 0.0f64;
        let mut sum_sm_mhz = 0.0f64;
        let mut sum_power_w = 0.0f64;
        let mut sens_n = 0u64;
        let mut last_sens = eng.sensors();
        while t0.elapsed() < Duration::from_secs(seconds.max(3)) && !STOP_SIGNAL.load(Ordering::Relaxed) {
            let (_used_nonce, out) = eng.mine_round_pipelined(
                &job,
                &cache,
                &mut || {
                    let n = bench_nonce;
                    bench_nonce += 1;
                    n
                },
                1,
                active_var,
            )?;
            rounds += 1;
            round_sec += (out.round_ms as f64) * 1e-3;
            kernel_sec += (out.kernel_ms as f64) * 1e-3;
            let cur_ths_macs = ((rounds as f64) * profile.macs_per_round()) / round_sec.max(1e-6);
            if last_sens_at.elapsed() >= Duration::from_secs(4) || t0.elapsed() >= Duration::from_secs(seconds.max(3)) {
                last_sens_at = Instant::now();
                let cur_sens = eng.sensors();
                last_sens = cur_sens;
                if cur_sens.sm_clock_mhz > 0 {
                    sum_sm_mhz += cur_sens.sm_clock_mhz as f64;
                    sum_power_w += cur_sens.power_w as f64;
                    sens_n += 1;
                }
                if let Some(gs) = telemetry.write().unwrap().gpus.get_mut(slot_i) {
                    gs.variant = eng.variant_name(eng.active_variant).to_string();
                    gs.hashrate_macs = cur_ths_macs;
                    gs.sensors = TelemetrySensors::from(cur_sens);
                }
            }
        }
        let _ = eng.flush_pipeline();
        let avg_sm_mhz = if sens_n > 0 { sum_sm_mhz / (sens_n as f64) } else { last_sens.sm_clock_mhz as f64 };
        let avg_power_w = if sens_n > 0 { sum_power_w / (sens_n as f64) } else { last_sens.power_w as f64 };
        let total_macs = (rounds as f64) * profile.macs_per_round();
        let kernel_ths = (total_macs / kernel_sec.max(1e-6)) * 1e-12;
        let sustained_ths = (total_macs / round_sec.max(1e-6)) * 1e-12;
        let eff = if avg_power_w > 1.0 { sustained_ths / avg_power_w } else { 0.0 };
        let sol_ths = (info.sm_count as f64) * 1024.0 * avg_sm_mhz * 1e-6;
        let sol_pct = if sol_ths > 0.1 { (sustained_ths / sol_ths) * 100.0 } else { 0.0 };
        println!(
            "[bench] GPU #{} ({}) [{}]: Sustained={:.2} TH/s (Kernel={:.2} TH/s, Ovh={:.2}%, SOL={:.1}%) | {:.1} W | {:.3} TH/s/W | avgSM={:.0} MHz Mem={} MHz Temp={}°C",
            info.ordinal,
            info.name,
            eng.variant_name(eng.active_variant),
            sustained_ths,
            kernel_ths,
            (1.0 - sustained_ths / kernel_ths.max(1e-9)).max(0.0) * 100.0,
            sol_pct,
            avg_power_w,
            eff,
            avg_sm_mhz,
            last_sens.mem_clock_mhz,
            last_sens.temp_c
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    install_signal_handlers();
    let cli = Cli::parse();
    let _ = &cli.no_color;

    let algo_lc = cli.algo.to_ascii_lowercase();
    ensure!(
        matches!(algo_lc.as_str(), "pearl" | "pearlhash" | "prl"),
        "Unsupported --algo '{}'; expected pearlhash",
        cli.algo
    );

    if cli.list_devices {
        let all = enumerate_gpus()?;
        for d in all {
            println!(
                "GPU #{}: {} | CC {}.{} | {} SMs | L2 {} MiB | VRAM {} MiB | PCI bus {:02} | Supported={}",
                d.ordinal,
                d.name,
                d.sm_major,
                d.sm_minor,
                d.sm_count,
                d.l2_bytes / (1024 * 1024),
                d.vram_total_mib,
                d.pci_bus_id,
                d.is_supported()
            );
        }
        return Ok(());
    }

    let gpus = select_gpus(cli.devices.as_deref())?;

    if cli.self_test {
        return run_hardware_selftest(&gpus, &cli);
    }

    if let Some(sec) = cli.benchmark {
        return run_offline_benchmark(&gpus, &cli, sec);
    }

    ensure!(
        !cli.wallet.trim().is_empty(),
        "Missing --wallet <KRX_ID_OR_PRL_ADDR> (or run with --self-test / --benchmark 10)"
    );
    let _ = parse_engine(&cli.engine, &gpus)?;

    let pools = if cli.pools.is_empty() {
        vec![
            "stratum+ssl://prl-ru.kryptex.network:8048".to_string(),
            "stratum+ssl://prl-eu.kryptex.network:8048".to_string(),
        ]
    } else {
        cli.pools.clone()
    };

    // Compute Kryptex static difficulty if `--pass auto`
    let est_rig_macs: f64 = gpus.iter().map(|g| g.estimated_macs_per_sec()).sum();
    let effective_pass = if cli.pass.eq_ignore_ascii_case("auto") {
        format!("d={}", optimal_kryptex_diff(est_rig_macs))
    } else {
        cli.pass.clone()
    };

    println!(
        "=== ApexPRL v{} | 0.00% DevFee | {} GPU(s) | Target Pool: {} (pass={}) ===",
        env!("CARGO_PKG_VERSION"),
        gpus.len(),
        pools[0],
        effective_pass
    );

    let started = Instant::now();
    let telemetry = Arc::new(RwLock::new(RigTelemetry {
        version: env!("CARGO_PKG_VERSION").to_string(),
        gpus: gpus
            .iter()
            .map(|g| PerGpuStats {
                ordinal: g.ordinal,
                name: g.name.clone(),
                arch: format!("sm_{}{}", g.sm_major, g.sm_minor),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }));
    spawn_api_server(cli.api_bind.clone(), telemetry.clone(), started);

    let agent = concat!("apexprl/", env!("CARGO_PKG_VERSION"));

    // Connect initially to determine gzip v2 support before choosing K
    let mut pool_idx = 0usize;
    let mut session = loop {
        if STOP_SIGNAL.load(Ordering::SeqCst) {
            return Ok(());
        }
        let url = &pools[pool_idx % pools.len()];
        match StratumSession::connect_and_authorize(url, &cli.wallet, cli.worker.as_deref(), &effective_pass, agent) {
            Ok(s) => {
                println!(
                    "[stratum] Connected & authorized on {} (TLS={}, encoding={})",
                    s.endpoint.raw,
                    s.endpoint.use_tls,
                    if s.gzip_v2 { "gzip-v2" } else { "base64-plain" }
                );
                break s;
            }
            Err(e) => {
                eprintln!("[stratum] Connect failed on {url}: {e:#}; trying failover in 3s...");
                pool_idx += 1;
                thread::sleep(Duration::from_secs(3));
            }
        }
    };

    let engine = parse_engine(&cli.engine, &gpus)?;
    let profile = build_profile(&gpus, engine, cli.k, session.gzip_v2)?;
    let active_k = profile.k;
    let engine_tag = match engine {
        EngineKind::Strassen => "strassen24k",
        EngineKind::Basis128 => "b128",
        EngineKind::Basis160 => "b160",
        EngineKind::DirectL2 => "direct",
        EngineKind::Edcp120 => "edcp",
    };

    let global_epoch = Arc::new(AtomicU32::new(1));
    let global_nonce = Arc::new(AtomicU64::new(
        (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64)
            << 16,
    ));
    let shared_job: Arc<RwLock<Option<Arc<SharedActiveJob>>>> = Arc::new(RwLock::new(None));
    let use_gzip_flag = Arc::new(AtomicBool::new(session.gzip_v2));

    // Initialize all GPU engines & collect their mapped abort pointers for < 0.2 ms preemption!
    let mut tune_db = TuneCacheFile::load(&cli.tune_cache);
    let mut engines = Vec::with_capacity(gpus.len());
    let mut abort_ptrs_usize = Vec::with_capacity(gpus.len());

    let first_notify = session.current_job.clone().expect("initial job present");
    let first_prep = PreparedJob::new(first_notify, profile)?;
    let pending_job: Arc<RwLock<(u32, PreparedJob)>> = Arc::new(RwLock::new((1, first_prep.clone())));
    let mut first_cache: Option<Constant64MerkleCache> = None;

    for (slot_i, info) in gpus.iter().enumerate() {
        let (mut eng, notes) = GpuWorkerEngine::new(
            info.clone(),
            profile,
            256,
            cli.lock_mem_idle,
            cli.cclock_lock,
            cli.pl,
        )?;
        if slot_i == 0 {
            first_cache = Some(eng.prepare_job(&first_prep, 1)?);
        } else {
            eng.load_job_with_cache(&first_prep, first_cache.as_ref().unwrap(), 1)?;
        }
        let cache_ref = first_cache.as_ref().unwrap();
        let tune_key = format!(
            "v07_{}_{}_sm{}{}_n{}_k{}",
            engine_tag, info.name, info.sm_major, info.sm_minor, profile.n, active_k
        );
        if let Some(&cached_v) = tune_db.entries.get(&tune_key) {
            eng.active_variant = cached_v % eng.num_variants();
        } else if cli.tune {
            println!("[tune] Calibrating GPU #{} ({}) [{}]...", info.ordinal, info.name, engine_tag);
            if let Ok((win, scores)) = eng.run_autotune(&first_prep, cache_ref, 1, 4) {
                let s: Vec<String> = scores.iter().map(|(n, r)| format!("{n}:{r:.1}")).collect();
                println!("[tune] GPU #{} winner: {} ({})", info.ordinal, eng.variant_name(win), s.join(" "));
                tune_db.entries.insert(tune_key, win);
                tune_db.save(&cli.tune_cache);
            }
        }
        println!(
            "[gpu #{}] {} (sm_{}{}, {} SMs) ready | kernel={} | {} | NVML=[{}]",
            info.ordinal,
            info.name,
            info.sm_major,
            info.sm_minor,
            info.sm_count,
            eng.variant_name(eng.active_variant),
            eng.layout_summary(),
            notes.join(", ")
        );
        abort_ptrs_usize.push(eng.abort_epoch_ptr_usize());
        engines.push((slot_i, eng));
    }

    {
        let mut t = telemetry.write().unwrap();
        t.pool_url = session.endpoint.raw.clone();
        t.compression = if session.gzip_v2 { "gzip-v2".into() } else { "base64".into() };
        t.active_k = active_k;
    }

    *shared_job.write().unwrap() = Some(Arc::new(SharedActiveJob {
        epoch: 1,
        job: first_prep,
        cache: first_cache.unwrap(),
    }));

    let (hit_tx, hit_rx): (Sender<RawHitCandidate>, Receiver<RawHitCandidate>) = bounded(512);
    let (share_tx, share_rx): (Sender<VerifiedShare>, Receiver<VerifiedShare>) = bounded(256);

    // Background Proof Builder & Low-Power Canary + Exact Jackpot Verifier Thread
    {
        let shared_job_pf = shared_job.clone();
        let telemetry_pf = telemetry.clone();
        let gzip_pf = use_gzip_flag.clone();
        thread::spawn(move || {
            let mut last_canary_epoch = 0u32;
            while let Ok(cand) = hit_rx.recv() {
                let active_opt = shared_job_pf.read().unwrap().clone();
                let Some(active) = active_opt else { continue };
                if active.epoch != cand.epoch || active.job.notify.job_id != cand.job_id {
                    telemetry_pf.write().unwrap().stale_shares += 1;
                    continue;
                }
                let full_canary = active.epoch != last_canary_epoch;
                match verify_and_encode_share_fast(
                    &active.job,
                    &active.cache,
                    cand.nonce,
                    cand.group,
                    gzip_pf.load(Ordering::Relaxed),
                    full_canary,
                ) {
                    Ok((wire_b64, raw_len, wire_len)) => {
                        if full_canary {
                            last_canary_epoch = active.epoch;
                        }
                        let rig_macs: f64 = telemetry_pf.read().unwrap().gpus.iter().map(|g| g.hashrate_macs).sum();
                        let _ = share_tx.send(VerifiedShare {
                            gpu_ordinal: cand.gpu_ordinal,
                            job_id: cand.job_id,
                            wire_proof_b64: wire_b64,
                            raw_len,
                            wire_len,
                            rig_macs_per_sec: (rig_macs as u64).max(cand.gpu_macs_per_sec),
                        });
                    }
                    Err(e) => {
                        eprintln!("[zk-pow] WARNING: GPU #{} produced invalid share ({e:#}); suppressed before pool!", cand.gpu_ordinal);
                        let mut t = telemetry_pf.write().unwrap();
                        t.invalid_local += 1;
                        if let Some(gs) = t.gpus.iter_mut().find(|g| g.ordinal == cand.gpu_ordinal) {
                            gs.invalid_local += 1;
                        }
                    }
                }
            }
        });
    }

    // Spawn 1 dedicated zero-bubble async ping-pong worker thread per GPU
    let mut worker_handles = Vec::with_capacity(engines.len());
    for (slot_i, mut eng) in engines {
        let sj = shared_job.clone();
        let pj_ref = pending_job.clone();
        let gn = global_nonce.clone();
        let tx = hit_tx.clone();
        let tel = telemetry.clone();
        let handle = thread::spawn(move || {
            let mut loaded_epoch = 1u32;
            let mut ema_macs = 0.0f64;
            let mut cached_sens = eng.sensors();
            let mut last_sens_at = Instant::now();
            while !STOP_SIGNAL.load(Ordering::Relaxed) {
                let (pend_ep, pend_job) = { pj_ref.read().unwrap().clone() };
                if slot_i == 0 && pend_ep != loaded_epoch {
                    match eng.prepare_job(&pend_job, pend_ep) {
                        Ok(cache) => {
                            *sj.write().unwrap() = Some(Arc::new(SharedActiveJob {
                                epoch: pend_ep,
                                job: pend_job,
                                cache,
                            }));
                            loaded_epoch = pend_ep;
                        }
                        Err(e) => {
                            eprintln!("[gpu #{}] prepare_job error: {e:#}", eng.info.ordinal);
                            thread::sleep(Duration::from_millis(100));
                            continue;
                        }
                    }
                }

                let active = { sj.read().unwrap().clone() };
                let Some(act) = active else {
                    thread::sleep(Duration::from_millis(1));
                    continue;
                };
                if act.epoch != pend_ep {
                    // Secondary GPU (`slot_i > 0`) waiting ~2 ms for GPU #0 to publish the new block's Merkle cache
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }
                if act.epoch != loaded_epoch {
                    if let Err(e) = eng.load_job_with_cache(&act.job, &act.cache, act.epoch) {
                        eprintln!("[gpu #{}] load_job_with_cache error: {e:#}", eng.info.ordinal);
                        thread::sleep(Duration::from_millis(100));
                        continue;
                    }
                    loaded_epoch = act.epoch;
                }

                let active_var = eng.active_variant;
                match eng.mine_round_pipelined(
                    &act.job,
                    &act.cache,
                    &mut || gn.fetch_add(1, Ordering::Relaxed),
                    act.epoch,
                    active_var,
                ) {
                    Ok((nonce, out)) => {
                        if !out.aborted {
                            let sec = ((out.round_ms as f64) * 1e-3).max(1e-6);
                            let inst_macs = eng.profile.macs_per_round() / sec;
                            ema_macs = if ema_macs == 0.0 { inst_macs } else { ema_macs * 0.8 + inst_macs * 0.2 };
                            if last_sens_at.elapsed() >= Duration::from_secs(5) {
                                last_sens_at = Instant::now();
                                cached_sens = eng.sensors();
                            }
                            {
                                let mut t = tel.write().unwrap();
                                if let Some(gs) = t.gpus.get_mut(slot_i) {
                                    gs.variant = eng.variant_name(active_var).to_string();
                                    gs.hashrate_macs = ema_macs;
                                    gs.sensors = TelemetrySensors::from(cached_sens);
                                }
                            }
                            for grp in out.hit_groups {
                                let _ = tx.send(RawHitCandidate {
                                    gpu_ordinal: eng.info.ordinal,
                                    epoch: act.epoch,
                                    job_id: act.job.notify.job_id.clone(),
                                    nonce,
                                    group: grp,
                                    gpu_macs_per_sec: ema_macs as u64,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        let _ = eng.flush_pipeline();
                        eprintln!("[gpu #{}] mine_round error: {e:#}", eng.info.ordinal);
                        thread::sleep(Duration::from_millis(200));
                    }
                }
            }
            let _ = eng.flush_pipeline();
        });
        worker_handles.push(handle);
    }
    drop(hit_tx);

    // Helper to trigger instant preemption across all GPUs on new block!
    let broadcast_epoch = |ep: u32| {
        global_epoch.store(ep, Ordering::Release);
        for &ptr_u in &abort_ptrs_usize {
            let atomic_ref = unsafe { &*(ptr_u as *const AtomicU32) };
            atomic_ref.store(ep, Ordering::Release);
        }
    };

    let mut last_log = Instant::now();

    while !STOP_SIGNAL.load(Ordering::SeqCst) {
        // 1. Drain verified shares and submit immediately
        while let Ok(sh) = share_rx.try_recv() {
            if session.current_job.as_ref().map(|j| j.job_id.as_str()) != Some(sh.job_id.as_str()) {
                telemetry.write().unwrap().stale_shares += 1;
                continue;
            }
            if let Err(e) = session.submit_share(&sh.job_id, &sh.wire_proof_b64, sh.rig_macs_per_sec) {
                eprintln!("[stratum] Submit failed: {e:#}");
            } else {
                println!(
                    "[share] GPU #{} -> submitted job={} ({} -> {} chars, {:.1}x)",
                    sh.gpu_ordinal,
                    sh.job_id,
                    sh.raw_len,
                    sh.wire_len,
                    (sh.raw_len as f64) / (sh.wire_len.max(1) as f64)
                );
            }
        }

        // 2. Poll Stratum socket
        match session.poll_events(Duration::from_millis(40)) {
            Ok(events) => {
                for ev in events {
                    match ev {
                        StratumEvent::NewJob(notify) => {
                            match PreparedJob::new(notify.clone(), profile) {
                                Ok(pj) => {
                                    let new_ep = global_epoch.load(Ordering::Relaxed).wrapping_add(1).max(1);
                                    println!(
                                        "[job] New work: id={} height={} cert_v={}",
                                        notify.job_id, notify.height, notify.cert_version
                                    );
                                    *pending_job.write().unwrap() = (new_ep, pj);
                                    broadcast_epoch(new_ep);
                                }
                                Err(e) => eprintln!("[job] Invalid pool job ignored: {e:#}"),
                            }
                        }
                        StratumEvent::ShareAck { accepted, latency_ms, reason } => {
                            let mut t = telemetry.write().unwrap();
                            t.last_ping_ms = latency_ms;
                            if accepted {
                                t.accepted_shares += 1;
                                println!(
                                    "[pool] SHARE ACCEPTED ({}/{}) in {:.1} ms!",
                                    t.accepted_shares,
                                    t.accepted_shares + t.rejected_shares,
                                    latency_ms
                                );
                            } else {
                                t.rejected_shares += 1;
                                eprintln!(
                                    "[pool] SHARE REJECTED ({:.1} ms): {}",
                                    latency_ms,
                                    reason.unwrap_or_default()
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("[stratum] Disconnected ({e:#}); failing over...");
                pool_idx += 1;
                loop {
                    if STOP_SIGNAL.load(Ordering::SeqCst) {
                        break;
                    }
                    let url = &pools[pool_idx % pools.len()];
                    match StratumSession::connect_and_authorize(url, &cli.wallet, cli.worker.as_deref(), &effective_pass, agent) {
                        Ok(new_sess) => {
                            use_gzip_flag.store(new_sess.gzip_v2, Ordering::Relaxed);
                            if let Some(notify) = new_sess.current_job.clone() {
                                if let Ok(pj) = PreparedJob::new(notify, profile) {
                                    let new_ep = global_epoch.load(Ordering::Relaxed).wrapping_add(1).max(1);
                                    *pending_job.write().unwrap() = (new_ep, pj);
                                    broadcast_epoch(new_ep);
                                }
                            }
                            {
                                let mut t = telemetry.write().unwrap();
                                t.pool_url = new_sess.endpoint.raw.clone();
                                t.compression = if new_sess.gzip_v2 { "gzip-v2".into() } else { "base64".into() };
                            }
                            println!("[stratum] Reconnected to {}", new_sess.endpoint.raw);
                            session = new_sess;
                            break;
                        }
                        Err(re_err) => {
                            eprintln!("[stratum] Reconnect to {url} failed: {re_err:#}");
                            pool_idx += 1;
                            thread::sleep(Duration::from_secs(3));
                        }
                    }
                }
            }
        }

        // 3. Periodic console telemetry line every 15s
        if last_log.elapsed() >= Duration::from_secs(15) {
            last_log = Instant::now();
            let snap = telemetry.read().unwrap().clone();
            let total_ths: f64 = snap.gpus.iter().map(|g| g.hashrate_macs * 1e-12).sum();
            let total_w: f32 = snap.gpus.iter().map(|g| g.sensors.power_w).sum();
            let eff = if total_w > 1.0 { total_ths / (total_w as f64) } else { 0.0 };
            let per_card: Vec<String> = snap
                .gpus
                .iter()
                .map(|g| {
                    format!(
                        "GPU{}:{:.2}TH/s({:.0}W,{}°C)",
                        g.ordinal,
                        g.hashrate_macs * 1e-12,
                        g.sensors.power_w,
                        g.sensors.temp_c
                    )
                })
                .collect();
            println!(
                "[stats] Total: {:.2} TH/s @ {:.0} W ({:.3} TH/s/W) | A:{} R:{} S:{} | {}",
                total_ths,
                total_w,
                eff,
                snap.accepted_shares,
                snap.rejected_shares,
                snap.stale_shares,
                per_card.join(" ")
            );
        }
    }

    println!("[apexprl] Shutting down cleanly and restoring GPU clocks...");
    let final_ep = global_epoch.load(Ordering::Relaxed).wrapping_add(999);
    broadcast_epoch(final_ep);
    for h in worker_handles {
        let _ = h.join();
    }
    Ok(())
}
