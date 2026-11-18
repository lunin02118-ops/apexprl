//! Lightweight zero-dependency HTTP telemetry server for HiveOS (`/api/v1/hiveos`),
//! generic JSON summary (`/api/summary`), and embedded Web UI (`/`).

use apexprl_gpu::nvml_drv::GpuSensors;
use serde::Serialize;
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, Serialize)]
pub struct PerGpuStats {
    pub ordinal: usize,
    pub name: String,
    pub arch: String,
    pub variant: String,
    pub hashrate_macs: f64,
    pub sensors: TelemetrySensors,
    pub accepted: u64,
    pub rejected: u64,
    pub invalid_local: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct TelemetrySensors {
    pub temp_c: u32,
    pub mem_temp_c: u32,
    pub fan_pct: u32,
    pub power_w: f32,
    pub sm_clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub pci_bus_id: u32,
}

impl From<GpuSensors> for TelemetrySensors {
    fn from(s: GpuSensors) -> Self {
        Self {
            temp_c: s.temp_c,
            mem_temp_c: s.mem_temp_c,
            fan_pct: s.fan_pct,
            power_w: s.power_w,
            sm_clock_mhz: s.sm_clock_mhz,
            mem_clock_mhz: s.mem_clock_mhz,
            pci_bus_id: s.pci_bus_id,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigTelemetry {
    pub version: String,
    pub pool_url: String,
    pub compression: String,
    pub active_k: usize,
    pub accepted_shares: u64,
    pub rejected_shares: u64,
    pub stale_shares: u64,
    pub invalid_local: u64,
    pub last_ping_ms: f64,
    pub gpus: Vec<PerGpuStats>,
}

pub fn spawn_api_server(bind_addr: String, state: Arc<RwLock<RigTelemetry>>, started: Instant) {
    thread::spawn(move || {
        let Ok(listener) = TcpListener::bind(&bind_addr) else {
            eprintln!("[api] Warning: could not bind telemetry server on {bind_addr}");
            return;
        };
        for stream in listener.incoming().flatten() {
            let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(1000)));
            let snap = state.read().unwrap().clone();
            let uptime = started.elapsed().as_secs();
            handle_http(stream, &snap, uptime);
        }
    });
}

fn handle_http(mut stream: std::net::TcpStream, snap: &RigTelemetry, uptime: u64) {
    let mut req = [0u8; 1024];
    let n = stream.read(&mut req).unwrap_or(0);
    if n == 0 {
        return;
    }
    let first_line = String::from_utf8_lossy(&req[..n]);
    let path = first_line
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/");

    let (content_type, body) = if path.starts_with("/api/v1/hiveos") || path.starts_with("/hiveos") {
        // HiveOS expects `khs` = H/s / 1000.0
        let total_khs: f64 = snap.gpus.iter().map(|g| g.hashrate_macs / 1000.0).sum();
        let hs: Vec<f64> = snap.gpus.iter().map(|g| g.hashrate_macs / 1000.0).collect();
        let temp: Vec<u32> = snap.gpus.iter().map(|g| g.sensors.temp_c).collect();
        let mtemp: Vec<u32> = snap.gpus.iter().map(|g| g.sensors.mem_temp_c).collect();
        let fan: Vec<u32> = snap.gpus.iter().map(|g| g.sensors.fan_pct).collect();
        let power: Vec<u32> = snap.gpus.iter().map(|g| g.sensors.power_w.round() as u32).collect();
        let bus_numbers: Vec<u32> = snap.gpus.iter().map(|g| g.sensors.pci_bus_id).collect();
        let per_gpu_inv: Vec<u64> = snap.gpus.iter().map(|g| g.invalid_local).collect();

        let payload = json!({
            "khs": total_khs,
            "stats": {
                "hs": hs,
                "hs_units": "khs",
                "temp": temp,
                "mtemp": mtemp,
                "fan": fan,
                "power": power,
                "bus_numbers": bus_numbers,
                "uptime": uptime,
                "ver": snap.version,
                "ar": [snap.accepted_shares, snap.rejected_shares + snap.stale_shares, snap.invalid_local, per_gpu_inv],
                "algo": "pearlhash"
            }
        });
        ("application/json", serde_json::to_string(&payload).unwrap_or_default())
    } else if path.starts_with("/api/") || path.starts_with("/summary") {
        let total_ths: f64 = snap.gpus.iter().map(|g| g.hashrate_macs * 1e-12).sum();
        let payload = json!({
            "miner": "ApexPRL",
            "version": snap.version,
            "uptime_sec": uptime,
            "dev_fee_percent": 0.0,
            "total_ths": total_ths,
            "pool": snap.pool_url,
            "proof_encoding": snap.compression,
            "k_dim": snap.active_k,
            "shares": {
                "accepted": snap.accepted_shares,
                "rejected": snap.rejected_shares,
                "stale": snap.stale_shares,
                "invalid_local": snap.invalid_local,
                "ping_ms": snap.last_ping_ms
            },
            "gpus": snap.gpus
        });
        ("application/json", serde_json::to_string_pretty(&payload).unwrap_or_default())
    } else {
        let total_ths: f64 = snap.gpus.iter().map(|g| g.hashrate_macs * 1e-12).sum();
        let total_w: f32 = snap.gpus.iter().map(|g| g.sensors.power_w).sum();
        let eff = if total_w > 1.0 { total_ths / (total_w as f64) } else { 0.0 };
        let rows_html: String = snap
            .gpus
            .iter()
            .map(|g| {
                format!(
                    "<tr><td>GPU #{} (bus {:02})</td><td>{}</td><td>{}</td><td><b>{:.2} TH/s</b></td><td>{:.0} W</td><td>{} / {} °C</td><td>{} / {} MHz</td></tr>",
                    g.ordinal,
                    g.sensors.pci_bus_id,
                    g.name,
                    g.variant,
                    g.hashrate_macs * 1e-12,
                    g.sensors.power_w,
                    g.sensors.temp_c,
                    g.sensors.mem_temp_c,
                    g.sensors.sm_clock_mhz,
                    g.sensors.mem_clock_mhz
                )
            })
            .collect();
        let html = format!(
            "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='5'><title>ApexPRL Dashboard</title>\
            <style>body{{background:#0d1117;color:#e6edf3;font-family:system-ui,monospace;padding:24px}}table{{border-collapse:collapse;width:100%;margin-top:16px}}th,td{{border:1px solid #30363d;padding:10px;text-align:left}}th{{background:#161b22}}.ok{{color:#3fb950}}</style></head>\
            <body><h2>ApexPRL v{} — 0% DevFee RTX 40/50 Pearl Miner</h2>\
            <p>Pool: <b>{}</b> ({}, K={}) | Uptime: {}s | Total: <b class='ok'>{:.2} TH/s</b> @ {:.0} W ({:.3} TH/s/W) | Shares: <b class='ok'>{}</b> acc / {} rej ({:.1} ms)</p>\
            <table><tr><th>Device</th><th>Model</th><th>Kernel</th><th>Hashrate</th><th>Power</th><th>Core/VRAM Temp</th><th>SM/Mem Clock</th></tr>{}</table></body></html>",
            snap.version, snap.pool_url, snap.compression, snap.active_k, uptime, total_ths, total_w, eff,
            snap.accepted_shares, snap.rejected_shares, snap.last_ping_ms, rows_html
        );
        ("text/html; charset=utf-8", html)
    };

    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
}
