//! ApexPRL — Ultra-High-Efficiency 0%-Fee Pearl (PRL) Miner for NVIDIA RTX 40 (`sm_89`) & RTX 50 (`sm_120a`)
//! Engineered specifically for HiveOS Linux & Kryptex Stratum v2 (Gzip).

mod api_server;

use anyhow::{ensure, Context, Result};
use api_server::{spawn_api_server, PerGpuStats, RigTelemetry, TelemetrySensors};
use apexprl_gpu::{enumerate_gpus, GpuDeviceInfo, GpuWorkerEngine, TuneCacheFile};
use apexprl_pow::{
    cpu_eval_group_digest, optimal_kryptex_diff, verify_and_encode_proof, Constant64MerkleCache,
    KernelProfile, PoolNotifyJob, PreparedJob,
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

    /// Inner matrix dimension K (`4096`, `8192`, `16384`, `32768`, `65536`)
    #[arg(long, default_value_t = 16384)]
    k: usize,

    /// Comma-separated CUDA device indices (default: all RTX 40 / RTX 50 GPUs)
    #[arg(long, visible_alias = "gpu-id")]
    devices: Option<String>,

    /// List detected GPUs and exit
    #[arg(long)]
    list_devices: bool,

    /// Run offline GPU-vs-CPU + official `zk-pow` verification test and exit
    #[arg(long, visible_alias = "selftest")]
    self_test: bool,

    /// Run offline sustained full-matrix benchmark for N seconds and exit
    #[arg(long)]
    benchmark: Option<u64>,

    /// Calibrate PTX instruction placement variants (`v0..v3`) on startup
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

/// Runs rigorous Gate G0 hardware verification on every selected GPU:
/// 1. Builds Merkle heap on GPU and checks root against CPU `Blake3Hasher`.
/// 2. Runs all 4 placement variants (`v0..v3`) with an easy target and checks 100% identical hit sets.
/// 3. Cross-checks GPU hit digests against exact CPU `cpu_eval_group_digest`.
/// 4. Assembles a full `PlainProof` and verifies it via official `zk-pow::verify_plain_proof` + `gzip`!
fn run_hardware_selftest(gpus: &[GpuDeviceInfo], cli: &Cli) -> Result<()> {
    println!("[selftest] Starting hardware + zk-pow verification across {} GPU(s)...", gpus.len());
    // Fast test shape: M=1024, N=1024, K=2048 -> 2048 leaves, 6 sampled M-blocks of 160 = 960 rows
    let profile = KernelProfile::with_dims(1024, 1024, 2048)?;
    let header_hex = "01000000".to_string() + &"aa".repeat(32) + &"bb".repeat(32) + "78563412" + "ffff0f1e";
    // Target <= U256::MAX / (160 * 2048): ~4% hit rate (~245 hits across 6144 groups)
    let target_hex = "000001".to_string() + &"ff".repeat(29);
    let notify = PoolNotifyJob::from_stratum_params(&json!({
        "job_id": "selftest_v3",
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
            "GPU #{} Merkle root mismatch vs CPU reference!",
            info.ordinal
        );

        let nonce = 0x2026_1118_a9ef_0042u64;
        let (_sig, _root_a, key_a) = gpu_cache.derive_attempt(&job, nonce);

        // Verify greedy reorder == non-greedy reorder == all 4 variants v0..v3!
        let ref_no_greedy = eng.mine_round(&job, &gpu_cache, nonce, 1, 0, false)?;
        ensure!(ref_no_greedy.total_hits > 0, "Expected >0 hits in selftest, got 0");

        for v in 0..4 {
            let r = eng.mine_round(&job, &gpu_cache, nonce, 1, v, true)?;
            ensure!(
                r.total_hits == ref_no_greedy.total_hits && r.hit_groups == ref_no_greedy.hit_groups,
                "GPU #{} variant {} hit set mismatch vs non-greedy baseline ({} vs {})!",
                info.ordinal,
                eng.variant_name(v),
                r.total_hits,
                ref_no_greedy.total_hits
            );
        }

        // Verify every hit against CPU reference & verify first 3 shares with official zk-pow!
        let bound_u256 = U256::from_little_endian(&job.bound_le);
        for (idx, &grp) in ref_no_greedy.hit_groups.iter().take(16).enumerate() {
            let (_tr, cpu_digest) = cpu_eval_group_digest(&profile, &key_a, &gpu_cache.key_b, grp);
            ensure!(
                U256::from_little_endian(&cpu_digest) <= bound_u256,
                "GPU #{} reported false hit on group {}!",
                info.ordinal,
                grp
            );
            if idx < 3 {
                let proof = gpu_cache.build_plain_proof(grp, nonce)?;
                let (_b64, raw_b, gz_b) = verify_and_encode_proof(&job, &proof, true)?;
                if idx == 0 {
                    println!(
                        "[selftest] GPU #{} ({}, sm_{}{}): {} hits matched CPU + all 4 variants + zk-pow V3 OK (gzip {}B -> {}B, {:.1}x)!",
                        info.ordinal,
                        info.name,
                        info.sm_major,
                        info.sm_minor,
                        ref_no_greedy.total_hits,
                        raw_b,
                        gz_b,
                        (raw_b as f64) / (gz_b.max(1) as f64)
                    );
                }
            }
        }
    }
    println!("[selftest] ALL CHECKS PASSED (BIT-EXACT GPU == CPU == ZK-POW V3)!");
    Ok(())
}

fn run_offline_benchmark(gpus: &[GpuDeviceInfo], cli: &Cli, seconds: u64) -> Result<()> {
    let profile = KernelProfile::basis_160x256(cli.k)?;
    println!(
        "[bench] Full-matrix L2-resident Basis Benchmark: M={} (sampled {}), N={}, K={} ({:.2} TMAC/round) for {}s...",
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
        if !notes.is_empty() {
            println!("[bench] GPU #{} NVML tuning: {}", info.ordinal, notes.join(", "));
        }
        let cache = eng.prepare_job(&job, 1)?;
        if cli.tune {
            let (win, scores) = eng.run_autotune(&job, &cache, 1, 2)?;
            let summary: Vec<String> = scores.iter().map(|(n, r)| format!("{n}={r:.2}TH/s")).collect();
            println!("[tune] GPU #{} selected '{}' ({})", info.ordinal, eng.variant_name(win), summary.join(", "));
        }

        let t0 = Instant::now();
        let mut rounds = 0u64;
        let mut round_sec = 0.0f64;
        let mut kernel_sec = 0.0f64;
        while t0.elapsed() < Duration::from_secs(seconds.max(3)) && !STOP_SIGNAL.load(Ordering::Relaxed) {
            let out = eng.mine_round(&job, &cache, 1000 + rounds, 1, eng.active_variant, true)?;
            rounds += 1;
            round_sec += (out.round_ms as f64) * 1e-3;
            kernel_sec += (out.kernel_ms as f64) * 1e-3;
            let cur_ths_macs = ((rounds as f64) * profile.macs_per_round()) / round_sec.max(1e-6);
            let cur_sens = eng.sensors();
            if let Some(gs) = telemetry.write().unwrap().gpus.get_mut(slot_i) {
                gs.variant = eng.variant_name(eng.active_variant).to_string();
                gs.hashrate_macs = cur_ths_macs;
                gs.sensors = TelemetrySensors::from(cur_sens);
            }
        }
        let sens = eng.sensors();
        let total_macs = (rounds as f64) * profile.macs_per_round();
        let kernel_ths = (total_macs / kernel_sec.max(1e-6)) * 1e-12;
        let sustained_ths = (total_macs / round_sec.max(1e-6)) * 1e-12;
        let eff = if sens.power_w > 1.0 { sustained_ths / (sens.power_w as f64) } else { 0.0 };
        println!(
            "[bench] GPU #{} ({}) [{}]: Sustained={:.2} TH/s (Kernel={:.2} TH/s, Overhead={:.2}%) | {:.1} W | {:.3} TH/s/W | SM={} MHz Mem={} MHz Temp={}°C",
            info.ordinal,
            info.name,
            eng.variant_name(eng.active_variant),
            sustained_ths,
            kernel_ths,
            (1.0 - sustained_ths / kernel_ths.max(1e-9)).max(0.0) * 100.0,
            sens.power_w,
            eff,
            sens.sm_clock_mhz,
            sens.mem_clock_mhz,
            sens.temp_c
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

    let active_k = if !session.gzip_v2 && cli.k > 4096 {
        println!("[stratum] Pool did not negotiate gzip-v2; stepping K={} -> 4096 to fit plain proof limit", cli.k);
        4096
    } else {
        cli.k
    };
    let profile = KernelProfile::basis_160x256(active_k)?;

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
        let cache = eng.prepare_job(&first_prep, 1)?;
        let tune_key = format!("{}_sm{}{}_k{}", info.name, info.sm_major, info.sm_minor, active_k);
        if let Some(&cached_v) = tune_db.entries.get(&tune_key) {
            eng.active_variant = cached_v % 4;
        } else if cli.tune {
            println!("[tune] Calibrating GPU #{} ({})...", info.ordinal, info.name);
            if let Ok((win, scores)) = eng.run_autotune(&first_prep, &cache, 1, 2) {
                let s: Vec<String> = scores.iter().map(|(n, r)| format!("{n}:{r:.1}")).collect();
                println!("[tune] GPU #{} winner: {} ({})", info.ordinal, eng.variant_name(win), s.join(" "));
                tune_db.entries.insert(tune_key, win);
                tune_db.save(&cli.tune_cache);
            }
        }
        println!(
            "[gpu #{}] {} (sm_{}{}, {} SMs) ready | kernel={} | NVML=[{}]",
            info.ordinal,
            info.name,
            info.sm_major,
            info.sm_minor,
            info.sm_count,
            eng.variant_name(eng.active_variant),
            notes.join(", ")
        );
        abort_ptrs_usize.push(eng.abort_epoch_ptr_usize());
        if slot_i == 0 {
            first_cache = Some(cache);
        }
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

    // Background Proof Builder & Official `zk-pow` Verifier Thread
    {
        let shared_job_pf = shared_job.clone();
        let telemetry_pf = telemetry.clone();
        let gzip_pf = use_gzip_flag.clone();
        thread::spawn(move || {
            while let Ok(cand) = hit_rx.recv() {
                let active_opt = shared_job_pf.read().unwrap().clone();
                let Some(active) = active_opt else { continue };
                if active.epoch != cand.epoch || active.job.notify.job_id != cand.job_id {
                    telemetry_pf.write().unwrap().stale_shares += 1;
                    continue;
                }
                match active.cache.build_plain_proof(cand.group, cand.nonce)
                    .and_then(|prf| verify_and_encode_proof(&active.job, &prf, gzip_pf.load(Ordering::Relaxed)).map(|v| (prf, v)))
                {
                    Ok((_prf, (wire_b64, raw_len, wire_len))) => {
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

    // Spawn 1 dedicated worker thread per GPU
    let mut worker_handles = Vec::with_capacity(engines.len());
    for (slot_i, mut eng) in engines {
        let sj = shared_job.clone();
        let gn = global_nonce.clone();
        let tx = hit_tx.clone();
        let tel = telemetry.clone();
        let handle = thread::spawn(move || {
            let mut loaded_epoch = 1u32;
            let mut ema_macs = 0.0f64;
            while !STOP_SIGNAL.load(Ordering::Relaxed) {
                let active = { sj.read().unwrap().clone() };
                let Some(act) = active else {
                    thread::sleep(Duration::from_millis(5));
                    continue;
                };
                if act.epoch != loaded_epoch {
                    if let Err(e) = eng.prepare_job(&act.job, act.epoch) {
                        eprintln!("[gpu #{}] prepare_job error: {e:#}", eng.info.ordinal);
                        thread::sleep(Duration::from_millis(100));
                        continue;
                    }
                    loaded_epoch = act.epoch;
                }

                let nonce = gn.fetch_add(1, Ordering::Relaxed);
                match eng.mine_round(&act.job, &act.cache, nonce, act.epoch, eng.active_variant, true) {
                    Ok(out) => {
                        if !out.aborted {
                            let sec = ((out.round_ms as f64) * 1e-3).max(1e-6);
                            let inst_macs = eng.profile.macs_per_round() / sec;
                            ema_macs = if ema_macs == 0.0 { inst_macs } else { ema_macs * 0.8 + inst_macs * 0.2 };
                            let sens = eng.sensors();
                            {
                                let mut t = tel.write().unwrap();
                                if let Some(gs) = t.gpus.get_mut(slot_i) {
                                    gs.variant = eng.variant_name(eng.active_variant).to_string();
                                    gs.hashrate_macs = ema_macs;
                                    gs.sensors = TelemetrySensors::from(sens);
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
                        eprintln!("[gpu #{}] mine_round error: {e:#}", eng.info.ordinal);
                        thread::sleep(Duration::from_millis(200));
                    }
                }
            }
        });
        worker_handles.push(handle);
    }
    drop(hit_tx);

    // Helper to trigger instant < 0.2 ms preemption across all GPUs on new block!
    let broadcast_epoch = |ep: u32| {
        global_epoch.store(ep, Ordering::Release);
        for &ptr_u in &abort_ptrs_usize {
            let atomic_ref = unsafe { &*(ptr_u as *const AtomicU32) };
            atomic_ref.store(ep, Ordering::Release);
        }
    };

    // Build a tiny helper CPU/GPU cache builder for new jobs?
    // Even simpler & faster: on new job, build Constant64MerkleCache::build_cpu in parallel via Rayon
    // OR note that each GPU worker's `eng.prepare_job` ALREADY builds `d_bas_b` in 1.2 ms — wait:
    // `Constant64MerkleCache::build_cpu` at M=32768, K=16384 hashes 524k leaves (~40 ms on 8-core CPU),
    // or we can build it right there so `shared_job` has the cache immediately while GPUs abort in 0.2 ms!
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
                            let new_ep = global_epoch.load(Ordering::Relaxed).wrapping_add(1).max(1);
                            // Preempt all GPUs in < 0.25 ms immediately!
                            broadcast_epoch(new_ep);
                            match PreparedJob::new(notify.clone(), profile)
                                .and_then(|pj| Constant64MerkleCache::build_cpu(&pj).map(|c| (pj, c)))
                            {
                                Ok((pj, cache)) => {
                                    println!(
                                        "[job] New work: id={} height={} cert_v={}",
                                        notify.job_id, notify.height, notify.cert_version
                                    );
                                    *shared_job.write().unwrap() = Some(Arc::new(SharedActiveJob {
                                        epoch: new_ep,
                                        job: pj,
                                        cache,
                                    }));
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
                                let new_ep = global_epoch.load(Ordering::Relaxed).wrapping_add(1).max(1);
                                broadcast_epoch(new_ep);
                                if let Ok((pj, cache)) = PreparedJob::new(notify, profile)
                                    .and_then(|pj| Constant64MerkleCache::build_cpu(&pj).map(|c| (pj, c)))
                                {
                                    *shared_job.write().unwrap() = Some(Arc::new(SharedActiveJob {
                                        epoch: new_ep,
                                        job: pj,
                                        cache,
                                    }));
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
