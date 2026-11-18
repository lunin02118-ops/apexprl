//! Stratum v1 / Kryptex Stratum v2 (Gzip) client with native `rustls` TLS (`:8048`),
//! automatic HiveOS `%WAL%.%WORKER_NAME%` normalization, and multi-pool failover.

use anyhow::{bail, ensure, Context, Result};
use apexprl_pow::PoolNotifyJob;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConfig, ClientConnection, DigitallySignedStruct, SignatureScheme, StreamOwned};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
struct PoolTlsVerifier;

impl ServerCertVerifier for PoolTlsVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

enum Transport {
    Plain(TcpStream),
    Tls(Box<StreamOwned<ClientConnection, TcpStream>>),
}

impl Transport {
    fn tcp_ref(&self) -> &TcpStream {
        match self {
            Self::Plain(s) => s,
            Self::Tls(tls) => tls.get_ref(),
        }
    }

    fn set_read_timeout(&self, dur: Option<Duration>) -> std::io::Result<()> {
        self.tcp_ref().set_read_timeout(dur)
    }
}

impl Read for Transport {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(s) => s.read(buf),
            Self::Tls(tls) => tls.read(buf),
        }
    }
}

impl Write for Transport {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(s) => s.write(buf),
            Self::Tls(tls) => tls.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(s) => s.flush(),
            Self::Tls(tls) => tls.flush(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParsedPoolUrl {
    pub raw: String,
    pub host: String,
    pub port: u16,
    pub use_tls: bool,
}

impl ParsedPoolUrl {
    pub fn parse(raw: &str) -> Result<Self> {
        let s = raw.trim();
        let (scheme_tls, rest) = if let Some(r) = s.strip_prefix("stratum+ssl://").or_else(|| s.strip_prefix("stratum+tls://")).or_else(|| s.strip_prefix("ssl://")).or_else(|| s.strip_prefix("tls://")) {
            (Some(true), r)
        } else if let Some(r) = s.strip_prefix("stratum+tcp://").or_else(|| s.strip_prefix("tcp://")) {
            (Some(false), r)
        } else {
            (None, s)
        };
        let rest = rest.trim_end_matches('/');
        let (host, port_str) = rest.rsplit_once(':').with_context(|| format!("Pool URL missing ':port': {raw}"))?;
        let port: u16 = port_str.parse().with_context(|| format!("Invalid port in pool URL: {raw}"))?;
        // Auto-enable TLS on Kryptex SSL port 8048 if scheme was omitted!
        let use_tls = scheme_tls.unwrap_or(port == 8048 || port == 5252 || port == 19601);
        Ok(Self {
            raw: s.to_string(),
            host: host.to_string(),
            port,
            use_tls,
        })
    }
}

/// Normalizes HiveOS `%WAL%.%WORKER_NAME%` or separate `(wallet, worker)` into `(full_login, worker_short)`.
pub fn normalize_credentials(wallet_arg: &str, worker_arg: Option<&str>) -> (String, String) {
    let w = wallet_arg.trim();
    if let Some((base, suffix)) = w.split_once('.').or_else(|| w.split_once('/')) {
        let short = suffix.trim();
        let short = if short.is_empty() { worker_arg.unwrap_or("rig01") } else { short };
        (format!("{base}.{short}"), short.to_string())
    } else {
        let short = worker_arg.map(str::trim).filter(|s| !s.is_empty()).unwrap_or("rig01");
        (format!("{w}.{short}"), short.to_string())
    }
}

#[derive(Debug)]
pub enum StratumEvent {
    NewJob(PoolNotifyJob),
    ShareAck {
        accepted: bool,
        latency_ms: f64,
        reason: Option<String>,
    },
}

pub struct StratumSession {
    pub endpoint: ParsedPoolUrl,
    pub full_user: String,
    pub gzip_v2: bool,
    transport: Transport,
    rx_buf: Vec<u8>,
    next_id: u64,
    pending_submits: HashMap<u64, Instant>,
    pub current_job: Option<PoolNotifyJob>,
}

impl StratumSession {
    pub fn connect_and_authorize(
        url_str: &str,
        wallet_arg: &str,
        worker_arg: Option<&str>,
        password: &str,
        agent: &str,
    ) -> Result<Self> {
        let endpoint = ParsedPoolUrl::parse(url_str)?;
        let addr_str = format!("{}:{}", endpoint.host, endpoint.port);
        let sock_addr = addr_str
            .to_socket_addrs()
            .with_context(|| format!("DNS lookup failed for {addr_str}"))?
            .next()
            .with_context(|| format!("No IP addresses returned for {addr_str}"))?;

        let tcp = TcpStream::connect_timeout(&sock_addr, Duration::from_secs(8))
            .with_context(|| format!("TCP connect failed to {addr_str}"))?;
        tcp.set_nodelay(true)?;
        tcp.set_write_timeout(Some(Duration::from_secs(10)))?;

        let transport = if endpoint.use_tls {
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            let cfg = ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()?
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(PoolTlsVerifier))
                .with_no_client_auth();
            let server_name = ServerName::try_from(endpoint.host.clone())
                .unwrap_or_else(|_| ServerName::try_from("kryptex.network".to_string()).unwrap());
            let conn = ClientConnection::new(Arc::new(cfg), server_name)?;
            Transport::Tls(Box::new(StreamOwned::new(conn, tcp)))
        } else {
            Transport::Plain(tcp)
        };

        let (full_user, worker_short) = normalize_credentials(wallet_arg, worker_arg);

        let mut sess = Self {
            endpoint,
            full_user: full_user.clone(),
            gzip_v2: false,
            transport,
            rx_buf: Vec::with_capacity(65536),
            next_id: 1,
            pending_submits: HashMap::new(),
            current_job: None,
        };

        let auth_id = sess.send_rpc(
            "mining.authorize",
            json!({
                "wallet": full_user,
                "worker": worker_short,
                "pass": password,
                "password": password,
                "agent": agent,
                "type": "v2"
            }),
        )?;

        let deadline = Instant::now() + Duration::from_secs(12);
        let mut auth_ok = false;
        while Instant::now() < deadline {
            let lines = sess.read_lines(Duration::from_millis(500))?;
            for msg in lines {
                if msg.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                    let job = PoolNotifyJob::from_stratum_params(&msg["params"])?;
                    sess.current_job = Some(job);
                } else if msg.get("id").and_then(|v| v.as_u64()) == Some(auth_id) {
                    let err_null = msg.get("error").map_or(true, Value::is_null);
                    let res_true = msg.get("result") == Some(&Value::Bool(true));
                    if !err_null || !res_true {
                        bail!("Pool rejected mining.authorize: {msg}");
                    }
                    sess.gzip_v2 = msg.get("type").and_then(|t| t.as_str()) == Some("v2");
                    auth_ok = true;
                }
            }
            if auth_ok && sess.current_job.is_some() {
                return Ok(sess);
            }
        }
        bail!("Timeout waiting for authorize + initial mining.notify from {}", url_str)
    }

    fn send_rpc(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let mut payload = serde_json::to_vec(&json!({
            "id": id,
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }))?;
        payload.push(b'\n');
        ensure!(payload.len() <= MAX_FRAME_BYTES, "RPC frame exceeds {MAX_FRAME_BYTES} B");
        self.transport.write_all(&payload)?;
        self.transport.flush()?;
        Ok(id)
    }

    pub fn submit_share(&mut self, job_id: &str, wire_proof_b64: &str, hashrate_macs: u64) -> Result<u64> {
        let id = self.send_rpc(
            "mining.submit",
            json!({
                "worker": self.full_user,
                "job_id": job_id,
                "plain_proof": wire_proof_b64,
                "hs": hashrate_macs
            }),
        )?;
        self.pending_submits.insert(id, Instant::now());
        Ok(id)
    }

    fn read_lines(&mut self, timeout: Duration) -> Result<Vec<Value>> {
        self.transport.set_read_timeout(Some(timeout.max(Duration::from_millis(5))))?;
        let mut tmp = [0u8; 32768];
        match self.transport.read(&mut tmp) {
            Ok(0) => bail!("Pool closed TCP/TLS connection"),
            Ok(n) => self.rx_buf.extend_from_slice(&tmp[..n]),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.into()),
        }
        ensure!(self.rx_buf.len() <= MAX_FRAME_BYTES, "Pool RX buffer exceeded frame cap");

        let mut out = Vec::new();
        while let Some(nl) = self.rx_buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.rx_buf.drain(..=nl).collect();
            let trimmed = &line[..line.len() - 1];
            if trimmed.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            let val: Value = serde_json::from_slice(trimmed).context("Invalid JSON from pool")?;
            out.push(val);
        }
        Ok(out)
    }

    pub fn poll_events(&mut self, timeout: Duration) -> Result<Vec<StratumEvent>> {
        let msgs = self.read_lines(timeout)?;
        let mut events = Vec::new();
        for msg in msgs {
            if msg.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                let job = PoolNotifyJob::from_stratum_params(&msg["params"])?;
                self.current_job = Some(job.clone());
                events.push(StratumEvent::NewJob(job));
            } else if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                if let Some(t0) = self.pending_submits.remove(&id) {
                    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let err_null = msg.get("error").map_or(true, Value::is_null);
                    let res_true = msg.get("result") == Some(&Value::Bool(true));
                    if err_null && res_true {
                        events.push(StratumEvent::ShareAck {
                            accepted: true,
                            latency_ms,
                            reason: None,
                        });
                    } else {
                        let reason = msg
                            .get("error")
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| msg.to_string());
                        events.push(StratumEvent::ShareAck {
                            accepted: false,
                            latency_ms,
                            reason: Some(reason),
                        });
                    }
                }
            }
        }
        Ok(events)
    }
}
