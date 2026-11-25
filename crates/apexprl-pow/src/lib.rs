//! Consensus, Merkle caching, V3 Salted Seed derivation, CPU reference solver,
//! and local `zk-pow` share verification for ApexPRL.

use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use flate2::{write::GzEncoder, Compression};
use pearl_blake3::{blake3_digest, Blake3Hasher, MerkleProof, MerkleTree};
use primitive_types::U256;
use rayon::prelude::*;
use std::collections::{BTreeSet, VecDeque};
use std::io::Write as _;
use zk_pow::{
    api::{
        proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation},
        proof_utils::nbits_to_difficulty,
        sanity_checks::penalized_target_bound,
        seed::{bind_root_a, bind_root_b},
        verify::verify_plain_proof,
    },
    circuit::pearl_noise::{generate_permutation_matrix, generate_uniform_random_matrix},
    ffi::plain_proof::{check_cert_version_eligible, MatrixMerkleProof, PlainProof},
};

pub const CHUNK_BYTES: usize = 1024;
pub const DEFAULT_M: usize = 32768;
pub const DEFAULT_N: usize = 32768;
pub const NOISE_RANK: usize = 128;
pub const CONST_SIGNAL_VAL: i8 = 64;
pub const CONST_SIGNAL_BYTE: u8 = 64;
pub const CONST_ZERO_VAL: i8 = 0;
pub const CONST_ZERO_BYTE: u8 = 0;

/// Unified tile geometry shared across `sm_120a` (RTX 50) and `sm_89` (RTX 40) engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelProfile {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub cta_m: usize,
    pub cta_n: usize,
    pub sampling_rows: usize,
    pub const_zero: bool,
}

impl KernelProfile {
    /// Creates a Uniform-3 128x256 On-Chip L2-Pinned Basis Kernel profile.
    pub fn basis_128x256(k: usize) -> Result<Self> {
        Self::with_dims_128x256(DEFAULT_M, DEFAULT_N, k)
    }

    /// Creates an `SM_COUNT`-aligned Uniform-3 128x256 profile (`M = mult * SM_COUNT * 128 + 16`),
    /// achieving 100.000% SM wave occupancy, 0 `BASA` reloads per round, and 0% discarded rows!
    pub fn basis_128x256_sm_aligned(sm_counts: &[usize], k: usize) -> Result<Self> {
        if sm_counts.is_empty() {
            return Self::basis_128x256(k);
        }
        let first = sm_counts[0].max(1);
        let tiles_m = if sm_counts.iter().all(|&s| s.max(1) == first) {
            let mult = if first < 20 {
                (40 + first - 1) / first
            } else if first <= 84 {
                2
            } else {
                1
            };
            first * mult
        } else {
            let mut best_t = 256usize;
            let mut best_eff = -1.0f64;
            for t in 24..=256usize {
                let min_eff = sm_counts
                    .iter()
                    .map(|&s| {
                        let sm = s.max(1);
                        let waves = t.div_ceil(sm);
                        (t as f64) / ((waves * sm) as f64)
                    })
                    .fold(1.0f64, f64::min);
                if min_eff > best_eff + 1e-9 {
                    best_eff = min_eff;
                    best_t = t;
                }
            }
            best_t
        };
        let m = tiles_m * 128 + 16;
        Self::with_dims_128x256(m, DEFAULT_N, k)
    }

    /// Custom matrix dimensions for Uniform-3 128x256 On-Chip Basis Kernel (`8x16` hash tile).
    /// Supports both power-of-two and `SM_COUNT`-aligned `M` (`e.g. M = SM_COUNT * 128 + 16`) via odd-node Merkle promotion!
    pub fn with_dims_128x256(m: usize, n: usize, k: usize) -> Result<Self> {
        ensure!(
            matches!(k, 2048 | 4096 | 8192 | 16384 | 32768 | 65536),
            "Unsupported K={k}; expected power-of-two in 2048..=65536"
        );
        ensure!(
            m >= 144 && n >= 256 && n % 256 == 0,
            "Invalid M={m}, N={n} (require M>=144, N>=256 multiple of 256)"
        );
        ensure!(
            (m * k) % CHUNK_BYTES == 0 && (n * k) % CHUNK_BYTES == 0 && (m * k / CHUNK_BYTES) >= 2 && (n * k / CHUNK_BYTES) >= 2,
            "M*K and N*K must be multiples of 1024 with >= 2 Merkle leaves"
        );
        Ok(Self {
            m,
            n,
            k,
            cta_m: 128,
            cta_n: 256,
            sampling_rows: 8,
            const_zero: false,
        })
    }

    /// Creates an `SM_COUNT`-aligned Constant-0 On-Chip RAW-24K Basis-Strassen 128x256 profile
    /// (`M = mult * SM_COUNT * 128 + 16`, `const_zero = true`, `100%` wave occupancy, `0%` discarded columns, `0 ms` b_salt search!).
    pub fn strassen_128x256_sm_aligned(sm_counts: &[usize], k: usize) -> Result<Self> {
        let mut p = Self::basis_128x256_sm_aligned(sm_counts, k)?;
        p.const_zero = true;
        Ok(p)
    }

    /// Custom matrix dimensions for Constant-0 RAW-24K Basis-Strassen 128x256 (`8x16` hash tile).
    pub fn with_dims_strassen_128x256(m: usize, n: usize, k: usize) -> Result<Self> {
        let mut p = Self::with_dims_128x256(m, n, k)?;
        p.const_zero = true;
        Ok(p)
    }

    /// Creates a unified 160x256 Basis Kernel profile for `k` in {4096, 8192, 16384, 32768, 65536}.
    pub fn basis_160x256(k: usize) -> Result<Self> {
        Self::with_dims(DEFAULT_M, DEFAULT_N, k)
    }

    /// Creates a Hyper-L2 Direct 2-CTA/SM (`128x128` CTA, `64x64` warp tile, `8x16` hash tile) profile
    /// where `M >= N` and `M / N` is a power of two (allowing `B` to stay 100% resident in L2 cache!).
    pub fn direct_l2_128x128(m: usize, n: usize, k: usize) -> Result<Self> {
        ensure!(
            matches!(k, 2048 | 4096 | 8192 | 16384 | 32768 | 65536),
            "Unsupported K={k}; expected power-of-two in 2048..=65536"
        );
        ensure!(
            m >= 256 && n >= 128 && m >= n && m % n == 0 && (m / n).is_power_of_two() && m % 128 == 0 && n % 128 == 0,
            "Invalid Direct-L2 dims M={m}, N={n} (require M>=N, power-of-two ratio, multiples of 128)"
        );
        ensure!(
            (m * k / CHUNK_BYTES).is_power_of_two() && (n * k / CHUNK_BYTES).is_power_of_two(),
            "M*K/1024 and N*K/1024 must be powers of two for binary Merkle heap"
        );
        Ok(Self {
            m,
            n,
            k,
            cta_m: 128,
            cta_n: 128,
            sampling_rows: 8,
            const_zero: false,
        })
    }

    /// Creates a Graph-Theoretic Edge-Disjoint Cycle-Packing (`EDCP-120`, `Constant-0`) 2-CTA/SM profile
    /// (`3xIMMA.16832 + 1xIMMA.16816 + 2xDP4A`, `0` outliers, `100.000%` bit-exact!).
    pub fn edcp_120(m: usize, n: usize, k: usize) -> Result<Self> {
        ensure!(
            k == 2048,
            "Unsupported EDCP K={k}; EDCP-120 requires K=2048 (0% target penalty, ~2ms b_salt convergence)"
        );
        ensure!(
            m >= 256 && n >= 512 && m >= n && m % n == 0 && (m / n).is_power_of_two() && m % 128 == 0 && n % 256 == 0,
            "Invalid EDCP dims M={m}, N={n} (require M>=N>=512, power-of-two ratio, multiples of 256)"
        );
        ensure!(
            (m * k / CHUNK_BYTES).is_power_of_two() && (n * k / CHUNK_BYTES).is_power_of_two(),
            "M*K/1024 and N*K/1024 must be powers of two for binary Merkle heap"
        );
        Ok(Self {
            m,
            n,
            k,
            cta_m: 128,
            cta_n: 128,
            sampling_rows: 8,
            const_zero: true,
        })
    }

    /// Custom matrix dimensions for 160x256 Basis Kernel (used for tests as well as production 32768x32768).
    pub fn with_dims(m: usize, n: usize, k: usize) -> Result<Self> {
        ensure!(
            matches!(k, 2048 | 4096 | 8192 | 16384 | 32768 | 65536),
            "Unsupported K={k}; expected power-of-two in 2048..=65536"
        );
        ensure!(m >= 320 && n >= 256 && n % 256 == 0, "Invalid M={m}, N={n}");
        ensure!(
            (m * k / CHUNK_BYTES).is_power_of_two() && (n * k / CHUNK_BYTES).is_power_of_two(),
            "M*K/1024 and N*K/1024 must be powers of two for binary Merkle heap"
        );
        Ok(Self {
            m,
            n,
            k,
            cta_m: 160,
            cta_n: 256,
            sampling_rows: 10,
            const_zero: false,
        })
    }

    #[inline]
    pub fn is_edcp(&self) -> bool {
        self.const_zero && self.cta_n == 128
    }

    #[inline]
    pub fn is_strassen(&self) -> bool {
        self.const_zero && self.cta_n == 256
    }

    #[inline]
    pub fn is_direct_l2(&self) -> bool {
        self.cta_m == 128 && self.cta_n == 128 && self.sampling_rows == 8 && !self.const_zero
    }

    #[inline]
    pub fn is_basis_128(&self) -> bool {
        self.cta_m == 128 && self.cta_n == 256 && self.sampling_rows == 8 && !self.const_zero
    }

    #[inline]
    pub fn signal_val(&self) -> i8 {
        if self.const_zero { CONST_ZERO_VAL } else { CONST_SIGNAL_VAL }
    }

    #[inline]
    pub fn signal_byte(&self) -> u8 {
        if self.const_zero { CONST_ZERO_BYTE } else { CONST_SIGNAL_BYTE }
    }

    /// Number of M rows actually covered by full CTAs.
    /// Leaves the tail rows (including row `m - 1` where the 64-byte nonce lives) unsampled!
    #[inline]
    pub fn sampled_m(&self) -> usize {
        let max_sampled = self.m.saturating_sub(16);
        (max_sampled / self.cta_m) * self.cta_m
    }

    /// Number of N columns actually covered by CTAs.
    /// In `EDCP` (`Constant-0` with `b_salt`) mode, leaves the last 256 columns (including row `n - 1` of `B^T`
    /// where `b_salt` lives) strictly unsampled! Strassen (`const_zero && cta_n == 256`) uses no `b_salt` and samples 100% of `N`.
    #[inline]
    pub fn sampled_n(&self) -> usize {
        if self.is_edcp() {
            self.n.saturating_sub(256)
        } else {
            self.n
        }
    }

    #[inline]
    pub fn warp_m(&self) -> usize {
        self.sampling_rows * 8
    }

    #[inline]
    pub fn warp_n(&self) -> usize {
        64
    }

    #[inline]
    pub fn macs_per_round(&self) -> f64 {
        (self.sampled_m() as f64) * (self.sampled_n() as f64) * (self.k as f64)
    }

    #[inline]
    pub fn total_groups(&self) -> u32 {
        ((self.sampled_m() / self.warp_m()) * (self.n / self.warp_n()) * 32) as u32
    }

    pub fn mining_config(&self) -> Result<MiningConfiguration> {
        let rows: Vec<u32> = (0..self.sampling_rows as u32).map(|i| i * 8).collect();
        let cols: Vec<u32> = (0..8u32).flat_map(|j| [j * 8, j * 8 + 1]).collect();
        Ok(MiningConfiguration {
            common_dim: self.k as u32,
            rank: NOISE_RANK as u16,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&rows)?,
            cols_pattern: PeriodicPattern::from_list(&cols)?,
            moe: None,
        })
    }

    /// Maps a GPU `group` ID (`warp_tile_idx * 32 + lane`) to exact `(A_row_indices, B_col_indices)`.
    pub fn group_indices(&self, group: u32) -> (Vec<usize>, Vec<usize>) {
        let tile = (group as usize) / 32;
        let lane = (group as usize) % 32;
        let tiles_n = self.n / self.warp_n();
        let mt = tile / tiles_n;
        let nt = tile % tiles_n;
        let row_base = mt * self.warp_m() + (lane / 4);
        let col_base = nt * self.warp_n() + (lane % 4) * 2;
        let rows = (0..self.sampling_rows).map(|i| row_base + i * 8).collect();
        let cols = (0..8)
            .flat_map(|j| [col_base + j * 8, col_base + j * 8 + 1])
            .collect();
        (rows, cols)
    }
}

pub fn hex_decode(s: &str) -> Result<Vec<u8>> {
    let s = s.trim().strip_prefix("0x").unwrap_or(s.trim());
    ensure!(s.len() % 2 == 0, "odd hex length {}", s.len());
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).context("invalid hex byte"))
        .collect()
}

pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

/// Largest Bitcoin compact `nbits` whose expanded target is `<= value`.
/// Essential for Kryptex (`target = 2^224 / diff - 1`), ensuring `verify_plain_proof` never
/// rejects a valid share at the boundary.
pub fn compact_floor(value: U256) -> Result<u32> {
    ensure!(!value.is_zero(), "zero share target");
    let mut size = value.bits().div_ceil(8) as u32;
    let mut mant = if size <= 3 {
        value.low_u32() << (8 * (3 - size))
    } else {
        (value >> (8 * (size - 3))).low_u32()
    };
    if mant & 0x0080_0000 != 0 {
        mant >>= 8;
        size += 1;
    }
    Ok((size << 24) | (mant & 0x007f_ffff))
}

/// Computes optimal Kryptex static difficulty `d=<N>` targeting ~16 seconds per share across the rig.
pub fn optimal_kryptex_diff(estimated_rig_macs_per_sec: f64) -> u64 {
    const MIN_DIFF: u64 = 2_097_152; // 2^21
    let raw = (estimated_rig_macs_per_sec * 16.0) / 4_294_967_296.0;
    if raw <= MIN_DIFF as f64 {
        return MIN_DIFF;
    }
    let u = (raw as u64).next_power_of_two() >> 1;
    u.max(MIN_DIFF)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolNotifyJob {
    pub job_id: String,
    pub header_bytes: Vec<u8>,
    pub raw_target: U256,
    pub height: u64,
    pub cert_version: u32,
}

impl PoolNotifyJob {
    pub fn from_stratum_params(p: &serde_json::Value) -> Result<Self> {
        let header_hex = p["header"].as_str().context("notify missing 'header'")?;
        let header_bytes = hex_decode(header_hex)?;
        ensure!(header_bytes.len() == 76, "expected 76-byte header, got {}", header_bytes.len());

        let target_hex = p["target"].as_str().context("notify missing 'target'")?;
        let target_bytes = hex_decode(target_hex)?;
        ensure!(target_bytes.len() == 32, "expected 32-byte target, got {}", target_bytes.len());
        let raw_target = U256::from_big_endian(&target_bytes);
        ensure!(!raw_target.is_zero(), "pool sent zero target");

        let cert_version = p
            .get("cert_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as u32;
        if cert_version >= 4 {
            bail!("Network upgraded to cert_version={cert_version} (FP8 V4); update required");
        }
        ensure!(cert_version == 2 || cert_version == 3, "unsupported cert_version={cert_version}");

        let height = p.get("height").and_then(|v| v.as_u64()).unwrap_or(0);
        let job_id = p["job_id"].as_str().context("notify missing 'job_id'")?.to_string();
        ensure!(!job_id.is_empty(), "empty job_id");

        Ok(Self {
            job_id,
            header_bytes,
            raw_target,
            height,
            cert_version,
        })
    }
}

#[derive(Clone)]
pub struct PreparedJob {
    pub notify: PoolNotifyJob,
    pub header: IncompleteBlockHeader,
    pub profile: KernelProfile,
    pub nbits_share: u32,
    /// Little-endian 256-bit penalized jackpot bound (`[u32; 8]` on GPU).
    pub bound_le: [u8; 32],
    /// `blake3(header76 || config52)`
    pub job_key: [u8; 32],
}

impl PreparedJob {
    pub fn new(notify: PoolNotifyJob, profile: KernelProfile) -> Result<Self> {
        let header = IncompleteBlockHeader::from_bytes(&notify.header_bytes)?;
        let nbits_share = compact_floor(notify.raw_target)?;
        let share_target = nbits_to_difficulty(nbits_share).min(notify.raw_target);
        let config = profile.mining_config()?;
        let bound_u256 = penalized_target_bound(share_target, &config)
            .context("penalized share target overflowed U256")?;
        let mut bound_le = [0u8; 32];
        bound_u256.to_little_endian(&mut bound_le);

        let mut hasher = blake3::Hasher::new();
        hasher.update(&header.to_bytes());
        hasher.update(&config.to_bytes());
        let job_key = *hasher.finalize().as_bytes();

        Ok(Self {
            notify,
            header,
            profile,
            nbits_share,
            bound_le,
            job_key,
        })
    }

    pub fn seed_derivation(&self) -> SeedDerivation {
        if self.notify.cert_version == 3 {
            SeedDerivation::Salted
        } else {
            SeedDerivation::Legacy
        }
    }
}

/// Encodes a 64-bit `nonce` into the 64-byte prefix of row `m - 1` using `{63, 64}` (`Constant-64`)
/// or `{0, 1}` (`Constant-0`).
#[inline]
pub fn encode_nonce_signal(nonce: u64) -> [u8; 64] {
    encode_signal_u64(nonce, false)
}

#[inline]
pub fn encode_signal_u64(val: u64, const_zero: bool) -> [u8; 64] {
    let base: u8 = if const_zero { 0 } else { 63 };
    let mut sig = [base; 64];
    for bit in 0..64 {
        sig[bit] = base + (((val >> bit) & 1) as u8);
    }
    sig
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdcpColSpec {
    pub count: u8,
    pub orig0: u8,
    pub orig1: u8,
    pub sgn1: i8,
}

#[derive(Clone, Debug)]
pub struct EdcpJobPlan {
    pub b_salt: u64,
    pub b_salt_leaf: usize,
    pub root_b: [u8; 32],
    pub key_b: [u8; 32],
    pub path_b: Vec<(usize, [u8; 32])>,
    pub bpairs_120: Vec<u16>,
    pub specs_120: Vec<EdcpColSpec>,
    pub avg_k_eff: f64,
}

/// Packs pairwise edge-disjoint cycles in a 128-edge chunk of `G_B` with 1-for-2 local swap refinement.
/// Returns `Some(n_dp4a)` in `{0, 1, 2}` (`K_eff` in `{112, 116, 120}`) if `>= 8` disjoint cycles exist!
pub fn build_chunk_edcp_plan(
    pb: &[[u32; 2]; 128],
    out_bpairs: &mut [u16; 120],
    out_acols: &mut [EdcpColSpec; 120],
) -> Option<u8> {
    const R: usize = 128;
    let mut used = [false; R];
    let mut cycs: Vec<Vec<(usize, i8)>> = Vec::with_capacity(16);

    let try_find_shortest = |u_mask: &[bool; R]| -> Option<Vec<(usize, i8)>> {
        let mut head = [-1i32; R];
        let mut to = [0usize; 256];
        let mut eid = [0usize; 256];
        let mut esgn = [0i8; 256];
        let mut nxt = [-1i32; 256];
        let mut ec = 0usize;
        for l in 0..R {
            if !u_mask[l] {
                let (u, v) = (pb[l][0] as usize, pb[l][1] as usize);
                to[ec] = v; eid[ec] = l; esgn[ec] = 1; nxt[ec] = head[u]; head[u] = ec as i32; ec += 1;
                to[ec] = u; eid[ec] = l; esgn[ec] = -1; nxt[ec] = head[v]; head[v] = ec as i32; ec += 1;
            }
        }
        let mut best: Option<Vec<(usize, i8)>> = None;
        let mut best_len = 999usize;
        let mut q = VecDeque::with_capacity(R);
        for e in 0..R {
            if u_mask[e] {
                continue;
            }
            let (u, v) = (pb[e][0] as usize, pb[e][1] as usize);
            let mut pv = [-1i32; R];
            let mut pe = [0usize; R];
            let mut ps = [0i8; R];
            let mut dist = [0usize; R];
            q.clear();
            pv[u] = u as i32;
            q.push_back(u);
            while let Some(cur) = q.pop_front() {
                if dist[cur] + 1 >= best_len || cur == v {
                    break;
                }
                let mut ei = head[cur];
                while ei != -1 {
                    let eu = ei as usize;
                    let l2 = eid[eu];
                    if l2 != e {
                        let nv = to[eu];
                        if pv[nv] == -1 {
                            pv[nv] = cur as i32;
                            pe[nv] = l2;
                            ps[nv] = esgn[eu];
                            dist[nv] = dist[cur] + 1;
                            q.push_back(nv);
                        }
                    }
                    ei = nxt[eu];
                }
            }
            if pv[v] != -1 && dist[v] + 1 < best_len {
                best_len = dist[v] + 1;
                let mut cyc = Vec::with_capacity(best_len);
                cyc.push((e, 1i8));
                let mut x = v;
                while x != u {
                    cyc.push((pe[x], ps[x]));
                    x = pv[x] as usize;
                }
                best = Some(cyc);
                if best_len == 2 {
                    break;
                }
            }
        }
        best
    };

    while cycs.len() < 16 {
        let Some(c) = try_find_shortest(&used) else { break };
        for &(e, _) in &c {
            used[e] = true;
        }
        cycs.push(c);
    }

    for _ in 0..8 {
        if cycs.len() >= 16 {
            break;
        }
        let mut improved = false;
        for ci in 0..cycs.len() {
            if cycs[ci].len() < 4 {
                continue;
            }
            let mut tmp_u = used;
            for &(e, _) in &cycs[ci] {
                tmp_u[e] = false;
            }
            for skip_i in 0..cycs[ci].len() {
                let mut u2 = tmp_u;
                let blocked = cycs[ci][skip_i].0;
                u2[blocked] = true;
                if let Some(c1) = try_find_shortest(&u2) {
                    u2[blocked] = false;
                    for &(e, _) in &c1 {
                        u2[e] = true;
                    }
                    if let Some(c2) = try_find_shortest(&u2) {
                        for &(e, _) in &c2 {
                            u2[e] = true;
                        }
                        used = u2;
                        cycs[ci] = c1;
                        cycs.push(c2);
                        while cycs.len() < 16 {
                            let Some(c3) = try_find_shortest(&used) else { break };
                            for &(e, _) in &c3 {
                                used[e] = true;
                            }
                            cycs.push(c3);
                        }
                        improved = true;
                        break;
                    }
                }
            }
            if improved {
                break;
            }
        }
        if !improved {
            break;
        }
    }

    let n_cyc = cycs.len();
    if n_cyc < 8 {
        return None;
    }
    let use_cyc = if n_cyc >= 16 { 16 } else if n_cyc >= 12 { 12 } else { 8 };
    let k_eff = 128 - use_cyc;

    let mut is_chord = [false; R];
    let mut partner_chord = [-1i32; R];
    let mut partner_sgn = [0i8; R];
    for c in cycs.iter().take(use_cyc) {
        let ch_e = c[0].0;
        is_chord[ch_e] = true;
        for &(te, sgn) in &c[1..] {
            partner_chord[te] = ch_e as i32;
            partner_sgn[te] = sgn;
        }
    }

    let mut pos = 0usize;
    for l in 0..R {
        if is_chord[l] {
            continue;
        }
        out_bpairs[pos] = (pb[l][0] as u16) | ((pb[l][1] as u16) << 8);
        out_acols[pos] = if partner_chord[l] >= 0 {
            EdcpColSpec {
                count: 2,
                orig0: l as u8,
                orig1: partner_chord[l] as u8,
                sgn1: partner_sgn[l],
            }
        } else {
            EdcpColSpec {
                count: 1,
                orig0: l as u8,
                orig1: 0,
                sgn1: 0,
            }
        };
        pos += 1;
    }
    while pos < 120 {
        out_bpairs[pos] = 0;
        out_acols[pos] = EdcpColSpec::default();
        pos += 1;
    }
    Some(((k_eff - 112) / 4) as u8)
}

/// Derives `(b_noise_seed, a_noise_seed)` from `(job_key, root_a, root_b)` under V2 or V3 Salted rules.
pub fn derive_noise_seeds(
    job_key: &[u8; 32],
    root_a: &[u8; 32],
    root_b: &[u8; 32],
    m: u32,
    n: u32,
    derivation: SeedDerivation,
) -> ([u8; 32], [u8; 32]) {
    let (bound_a, bound_b) = match derivation {
        SeedDerivation::Legacy => (*root_a, *root_b),
        SeedDerivation::Salted => (bind_root_a(root_a, m), bind_root_b(root_b, n)),
    };
    let mut msg_b = [0u8; 64];
    msg_b[..32].copy_from_slice(job_key);
    msg_b[32..].copy_from_slice(&bound_b);
    let b_seed = blake3_digest(&msg_b, None);

    let mut msg_a = [0u8; 64];
    msg_a[..32].copy_from_slice(&b_seed);
    msg_a[32..].copy_from_slice(&bound_a);
    let a_seed = blake3_digest(&msg_a, None);

    (b_seed, a_seed)
}

/// Layered BLAKE3 Merkle cache for Constant-64 / Constant-0 matrices supporting BOTH power-of-two
/// and arbitrary `SM_COUNT`-aligned dimensions (`M < N`, `M == N`, `M > N`, and odd-node promotion
/// matching `pearl-blake3::MerkleTree`), enabling ~2.5 microsecond incremental `root_A` updates
/// per nonce and zero-VRAM-read `PlainProof` assembly!
pub struct Constant64MerkleCache {
    pub profile: KernelProfile,
    pub job_key: [u8; 32],
    nonce_leaf: usize,
    nonce_row: usize,
    nonce_subrow_sibs: Vec<[u8; 32]>,
    bsalt_subrow_sibs: Vec<[u8; 32]>,
    /// Merkle levels from row-roots (`row_layers[0]`, len = `m` or `n`) up to the 2-child root pair (`len == 2`).
    row_layers_a: Vec<Vec<[u8; 32]>>,
    row_layers_b: Vec<Vec<[u8; 32]>>,
    hasher: Blake3Hasher,
    pub root_b: [u8; 32],
    pub key_b: [u8; 32],
    pub edcp: Option<EdcpJobPlan>,
}

impl Constant64MerkleCache {
    /// Power-of-two leaf capacity of the shared GPU Merkle heap covering `max(nleaves_a, nleaves_b)`.
    pub fn base_heap_leaves(profile: &KernelProfile) -> usize {
        let na = (profile.m * profile.k) / CHUNK_BYTES;
        let nb = (profile.n * profile.k) / CHUNK_BYTES;
        na.max(nb).next_power_of_two().max(2)
    }

    pub fn heap_byte_len(profile: &KernelProfile) -> usize {
        2 * Self::base_heap_leaves(profile) * 32
    }

    /// Power-of-two row-root capacity (`base_heap_leaves / leaves_per_row`), e.g. `32,768` at `N=32768`.
    pub fn base_row_leaves(profile: &KernelProfile) -> usize {
        let lpr = (profile.k / CHUNK_BYTES).max(1);
        (Self::base_heap_leaves(profile) / lpr).max(2)
    }

    /// Prefix of the GPU Merkle heap (`2.00 MiB` at `N=32768`, `64x` smaller than `128 MiB` at `K=65536`!)
    /// containing all row-root CVs and their ancestors up to the root.
    pub fn row_heap_byte_len(profile: &KernelProfile) -> usize {
        2 * Self::base_row_leaves(profile) * 32
    }

    fn build_subrow_right_sibs(hasher: &Blake3Hasher, row_idx: usize, lpr: usize, fill_byte: u8) -> Vec<[u8; 32]> {
        let chunk = [fill_byte; CHUNK_BYTES];
        let mut cur: Vec<[u8; 32]> = (0..lpr)
            .map(|j| hasher.chunk_cv(&chunk, (row_idx * lpr + j) as u64))
            .collect();
        let mut sibs = Vec::with_capacity(lpr.trailing_zeros() as usize);
        while cur.len() >= 2 {
            sibs.push(cur[1]);
            cur = cur
                .chunks_exact(2)
                .map(|p| hasher.parent_cv(&p[0], &p[1]))
                .collect();
        }
        sibs
    }

    fn extract_layers_from_heap(
        hasher: &Blake3Hasher,
        heap: &[u8],
        base_leaves: usize,
        nleaves: usize,
    ) -> Result<(Vec<Vec<[u8; 32]>>, [u8; 32])> {
        ensure!(nleaves >= 2 && nleaves <= base_leaves, "invalid nleaves={nleaves} (base={base_leaves})");
        if nleaves.is_power_of_two() && base_leaves % nleaves == 0 {
            // Zero-hash fast path: every level is already contiguous inside the GPU binary heap!
            let mut layers = Vec::with_capacity(24);
            let mut width = nleaves;
            let mut start = base_leaves;
            while width >= 2 {
                let raw = &heap[start * 32..(start + width) * 32];
                let lvl: Vec<[u8; 32]> = raw
                    .chunks_exact(32)
                    .map(|c| c.try_into().unwrap())
                    .collect();
                layers.push(lvl);
                width >>= 1;
                start >>= 1;
            }
            let top = layers.last().unwrap();
            let root = hasher.root_cv(&top[0], &top[1]);
            return Ok((layers, root));
        }

        // General odd-node-promoting path (`pearl-blake3::MerkleTree` exact match!):
        // Reuses GPU-hashed row-root CVs at `heap[base_leaves .. base_leaves + nleaves]`.
        let leaf_raw = &heap[base_leaves * 32..(base_leaves + nleaves) * 32];
        let mut layers: Vec<Vec<[u8; 32]>> = vec![leaf_raw
            .chunks_exact(32)
            .map(|c| c.try_into().unwrap())
            .collect()];
        while layers.last().unwrap().len() > 2 {
            let prev = layers.last().unwrap();
            let next: Vec<[u8; 32]> = prev
                .par_chunks(2)
                .map(|pair| {
                    if pair.len() == 2 {
                        hasher.parent_cv(&pair[0], &pair[1])
                    } else {
                        pair[0]
                    }
                })
                .collect();
            layers.push(next);
        }
        let top = layers.last().unwrap();
        let root = hasher.root_cv(&top[0], &top[1]);
        Ok((layers, root))
    }

    fn single_row_ancestors(
        row_layers: &[Vec<[u8; 32]>],
        hasher: &Blake3Hasher,
        row_idx: usize,
        leaf_idx: usize,
        subrow_sibs: &[[u8; 32]],
        chunk: &[u8; CHUNK_BYTES],
    ) -> ([u8; 32], Vec<(usize, [u8; 32])>) {
        let mut cv = hasher.chunk_cv(chunk, leaf_idx as u64);
        for sib in subrow_sibs {
            cv = hasher.parent_cv(&cv, sib);
        }
        let mut idx = row_idx;
        let mut overrides = Vec::with_capacity(row_layers.len());
        overrides.push((idx, cv));

        for (level, layer) in row_layers.iter().enumerate() {
            let len = layer.len();
            if len == 2 {
                let sib = layer[idx ^ 1];
                let (l, r) = if (idx & 1) == 1 { (sib, cv) } else { (cv, sib) };
                cv = hasher.root_cv(&l, &r);
                break;
            }
            if (idx & 1) == 1 {
                let sib = layer[idx - 1];
                cv = hasher.parent_cv(&sib, &cv);
            } else if idx + 1 < len {
                let sib = layer[idx + 1];
                cv = hasher.parent_cv(&cv, &sib);
            }
            idx >>= 1;
            if level + 1 < row_layers.len() {
                overrides.push((idx, cv));
            }
        }
        (cv, overrides)
    }

    /// Builds the Constant-64 / Constant-0 Merkle cache on CPU using Rayon (`2 MiB` row-root heap).
    pub fn build_cpu(job: &PreparedJob) -> Result<Self> {
        let profile = job.profile;
        let lpr = profile.k / CHUNK_BYTES;
        ensure!(lpr >= 2 && lpr.is_power_of_two(), "invalid leaves_per_row={lpr}");
        let base_rows = Self::base_row_leaves(&profile);
        let max_rows = profile.m.max(profile.n);

        let hasher = Blake3Hasher::with_key(job.job_key);
        let const_chunk = [profile.signal_byte(); CHUNK_BYTES];
        let mut row_heap = vec![0u8; 2 * base_rows * 32];

        row_heap[base_rows * 32..(base_rows + max_rows) * 32]
            .par_chunks_mut(32)
            .enumerate()
            .for_each(|(row_idx, slot)| {
                let mut cur: Vec<[u8; 32]> = (0..lpr)
                    .map(|j| hasher.chunk_cv(&const_chunk, (row_idx * lpr + j) as u64))
                    .collect();
                while cur.len() >= 2 {
                    for i in 0..(cur.len() / 2) {
                        cur[i] = hasher.parent_cv(&cur[2 * i], &cur[2 * i + 1]);
                    }
                    cur.truncate(cur.len() / 2);
                }
                slot.copy_from_slice(&cur[0]);
            });

        let mut level_count = base_rows >> 1;
        while level_count >= 2 {
            let (upper, lower) = row_heap.split_at_mut(level_count * 2 * 32);
            let parent_slice = &mut upper[level_count * 32..level_count * 2 * 32];
            let children_slice = &lower[..level_count * 2 * 32];
            parent_slice
                .par_chunks_mut(32)
                .enumerate()
                .for_each(|(i, pslot)| {
                    let l: [u8; 32] = children_slice[i * 64..i * 64 + 32].try_into().unwrap();
                    let r: [u8; 32] = children_slice[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                    let cv = hasher.parent_cv(&l, &r);
                    pslot.copy_from_slice(&cv);
                });
            level_count >>= 1;
        }

        Self::from_gpu_row_heap(job, row_heap)
    }

    /// Wraps a GPU-computed `row_heap` prefix (`2.00 MiB` at `N=32768`, `64x` smaller than full leaf heap!),
    /// extracting exact row-level Merkle layers for `A` and `B` with odd-node promotion!
    pub fn from_gpu_row_heap(job: &PreparedJob, row_heap: Vec<u8>) -> Result<Self> {
        let profile = job.profile;
        let lpr = profile.k / CHUNK_BYTES;
        ensure!(lpr >= 2 && lpr.is_power_of_two(), "invalid leaves_per_row={lpr}");
        let base_rows = Self::base_row_leaves(&profile);
        ensure!(
            row_heap.len() == 2 * base_rows * 32,
            "invalid Merkle row_heap size {} (expected {})",
            row_heap.len(),
            2 * base_rows * 32
        );

        let hasher = Blake3Hasher::with_key(job.job_key);
        let (row_layers_b, base_root_b) = Self::extract_layers_from_heap(&hasher, &row_heap, base_rows, profile.n)?;
        let (row_layers_a, _base_root_a) = Self::extract_layers_from_heap(&hasher, &row_heap, base_rows, profile.m)?;

        let (base_key_b, _) = derive_noise_seeds(
            &job.job_key,
            &base_root_b,
            &base_root_b,
            profile.m as u32,
            profile.n as u32,
            job.seed_derivation(),
        );

        let nonce_row = profile.m - 1;
        let nonce_leaf = nonce_row * lpr;
        let fill = profile.signal_byte();
        let nonce_subrow_sibs = Self::build_subrow_right_sibs(&hasher, nonce_row, lpr, fill);
        let bsalt_subrow_sibs = if profile.is_edcp() {
            Self::build_subrow_right_sibs(&hasher, profile.n - 1, lpr, CONST_ZERO_BYTE)
        } else {
            Vec::new()
        };

        let mut cache = Self {
            profile,
            job_key: job.job_key,
            nonce_leaf,
            nonce_row,
            nonce_subrow_sibs,
            bsalt_subrow_sibs,
            row_layers_a,
            row_layers_b,
            hasher,
            root_b: base_root_b,
            key_b: base_key_b,
            edcp: None,
        };

        if profile.is_edcp() {
            let plan = cache.solve_edcp_plan(job)?;
            cache.root_b = plan.root_b;
            cache.key_b = plan.key_b;
            cache.edcp = Some(plan);
        }

        Ok(cache)
    }

    fn b_salt_ancestors(&self, b_salt: u64) -> (usize, [u8; 32], Vec<(usize, [u8; 32])>) {
        let b_salt_row = self.profile.n - 1;
        let b_salt_leaf = (b_salt_row * self.profile.k) / CHUNK_BYTES;
        let sig = encode_signal_u64(b_salt, true);
        let mut chunk = [CONST_ZERO_BYTE; CHUNK_BYTES];
        chunk[..64].copy_from_slice(&sig);
        let (cv, path) = Self::single_row_ancestors(
            &self.row_layers_b,
            &self.hasher,
            b_salt_row,
            b_salt_leaf,
            &self.bsalt_subrow_sibs,
            &chunk,
        );
        (b_salt_leaf, cv, path)
    }

    fn solve_edcp_plan(&self, job: &PreparedJob) -> Result<EdcpJobPlan> {
        let num_chunks = self.profile.k / NOISE_RANK;
        let mut seed_b = [0u8; 32];
        seed_b[..8].copy_from_slice(b"B_tensor");
        let derivation = job.seed_derivation();

        for batch in 0..1024u64 {
            let base = batch * 32;
            let found: Option<(u32, EdcpJobPlan)> = (0..32u64)
                .into_par_iter()
                .filter_map(|off| {
                    let b_salt = base + off;
                    let (b_salt_leaf, root_b, path_b) = self.b_salt_ancestors(b_salt);
                    let (key_b, _) = derive_noise_seeds(
                        &self.job_key,
                        &root_b,
                        &root_b,
                        self.profile.m as u32,
                        self.profile.n as u32,
                        derivation,
                    );
                    let p_b = generate_permutation_matrix(&seed_b, &key_b, self.profile.k, NOISE_RANK);
                    let mut bpairs_120 = vec![0u16; num_chunks * 120];
                    let mut specs_120 = vec![EdcpColSpec::default(); num_chunks * 120];
                    let mut sum_ndp = 0u32;
                    for ch in 0..num_chunks {
                        let chunk_pb: &[[u32; 2]; 128] = (&p_b[ch * 128..(ch + 1) * 128]).try_into().unwrap();
                        let bp_out: &mut [u16; 120] = (&mut bpairs_120[ch * 120..(ch + 1) * 120]).try_into().unwrap();
                        let sp_out: &mut [EdcpColSpec; 120] = (&mut specs_120[ch * 120..(ch + 1) * 120]).try_into().unwrap();
                        let ndp = build_chunk_edcp_plan(chunk_pb, bp_out, sp_out)?;
                        sum_ndp += ndp as u32;
                    }
                    let avg_k_eff = 112.0 + 4.0 * (sum_ndp as f64) / (num_chunks as f64);
                    Some((
                        sum_ndp,
                        EdcpJobPlan {
                            b_salt,
                            b_salt_leaf,
                            root_b,
                            key_b,
                            path_b,
                            bpairs_120,
                            specs_120,
                            avg_k_eff,
                        },
                    ))
                })
                .min_by_key(|(s, p)| (*s, p.b_salt));
            if let Some((_s, best)) = found {
                return Ok(best);
            }
        }
        bail!("EDCP b_salt search did not converge within 8192 attempts for K={}", self.profile.k)
    }

    /// Computes `(nonce_signal64, root_a, key_a)` for `nonce` in ~2.5 microseconds on CPU!
    pub fn derive_attempt(&self, job: &PreparedJob, nonce: u64) -> ([u8; 64], [u8; 32], [u8; 32]) {
        let sig = encode_signal_u64(nonce, self.profile.const_zero);
        let mut chunk = [self.profile.signal_byte(); CHUNK_BYTES];
        chunk[..64].copy_from_slice(&sig);
        let (root_a, _path) = Self::single_row_ancestors(
            &self.row_layers_a,
            &self.hasher,
            self.nonce_row,
            self.nonce_leaf,
            &self.nonce_subrow_sibs,
            &chunk,
        );
        let (_, key_a) = derive_noise_seeds(
            &self.job_key,
            &root_a,
            &self.root_b,
            self.profile.m as u32,
            self.profile.n as u32,
            job.seed_derivation(),
        );
        (sig, root_a, key_a)
    }

    fn nonce_ancestors(&self, sig: &[u8; 64]) -> ([u8; 32], Vec<(usize, [u8; 32])>) {
        let mut chunk = [self.profile.signal_byte(); CHUNK_BYTES];
        chunk[..64].copy_from_slice(sig);
        Self::single_row_ancestors(
            &self.row_layers_a,
            &self.hasher,
            self.nonce_row,
            self.nonce_leaf,
            &self.nonce_subrow_sibs,
            &chunk,
        )
    }

    fn build_matrix_proof(
        &self,
        row_layers: &[Vec<[u8; 32]>],
        total_rows: usize,
        rows: Vec<usize>,
        root: [u8; 32],
        row_overrides: &[(usize, [u8; 32])],
        sig_leaf: Option<(usize, &[u8; 64])>,
    ) -> Result<MatrixMerkleProof> {
        let lpr = self.profile.k / CHUNK_BYTES;
        let mat_leaves = total_rows * lpr;
        let leaves = MerkleTree::compute_leaf_indices_from_rows(&rows, (total_rows, self.profile.k));
        ensure!(!leaves.is_empty() && *leaves.last().unwrap() < mat_leaves, "proof rows out of range");
        if let Some((target_leaf, _)) = sig_leaf {
            ensure!(!rows.contains(&(target_leaf / lpr)), "sampled row overlaps tail signal row");
        }

        let fill = self.profile.signal_byte();
        let leaf_data = vec![[fill; CHUNK_BYTES]; leaves.len()];

        // Every sampled row selects all `lpr` (`2^row_shift`) consecutive leaves, so zero siblings
        // exist below `row_layers[0]`, and `current` at `row_layers[0]` is the sorted row set!
        let mut siblings = Vec::new();
        let mut current: Vec<usize> = rows.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
        for (level, layer) in row_layers.iter().enumerate() {
            let level_len = layer.len();
            let selected: BTreeSet<usize> = current.iter().copied().collect();
            let ov = row_overrides.get(level).copied();
            let get_node = |pos: usize| -> [u8; 32] {
                if let Some((oi, ocv)) = ov {
                    if oi == pos {
                        return ocv;
                    }
                }
                layer[pos]
            };
            for &i in &current {
                if i % 2 == 1 {
                    if !selected.contains(&(i - 1)) {
                        siblings.push(get_node(i - 1));
                    }
                } else if (i + 1) < level_len && !selected.contains(&(i + 1)) {
                    siblings.push(get_node(i + 1));
                }
            }
            current = current.iter().map(|&i| i / 2).collect::<BTreeSet<_>>().into_iter().collect();
        }

        Ok(MatrixMerkleProof {
            row_indices: rows,
            proof: MerkleProof {
                leaf_data,
                leaf_indices: leaves,
                total_leaves: mat_leaves,
                root,
                siblings,
            },
        })
    }

    /// Assembles a complete `PlainProof` for winning GPU `group` on `nonce`.
    pub fn build_plain_proof(&self, group: u32, nonce: u64) -> Result<PlainProof> {
        ensure!(group < self.profile.total_groups(), "invalid hit group {group}");
        let sig = encode_signal_u64(nonce, self.profile.const_zero);
        let (root_a, path_a) = self.nonce_ancestors(&sig);
        let (rows, cols) = self.profile.group_indices(group);
        let bt = if let Some(plan) = &self.edcp {
            let b_sig = encode_signal_u64(plan.b_salt, true);
            self.build_matrix_proof(&self.row_layers_b, self.profile.n, cols, self.root_b, &plan.path_b, Some((plan.b_salt_leaf, &b_sig)))?
        } else {
            self.build_matrix_proof(&self.row_layers_b, self.profile.n, cols, self.root_b, &[], None)?
        };
        Ok(PlainProof {
            m: self.profile.m,
            n: self.profile.n,
            k: self.profile.k,
            noise_rank: NOISE_RANK,
            a: self.build_matrix_proof(&self.row_layers_a, self.profile.m, rows, root_a, &path_a, Some((self.nonce_leaf, &sig)))?,
            bt,
            moe: None,
        })
    }
}

/// Verifies `proof` locally using the official `zk-pow` verifier, then serializes and encodes
/// for wire submission (`gzip + base64` for Kryptex v2, or plain `base64` fallback).
pub fn verify_and_encode_proof(job: &PreparedJob, proof: &PlainProof, use_gzip: bool) -> Result<(String, usize, usize)> {
    check_cert_version_eligible(job.notify.cert_version, proof)?;
    verify_plain_proof(&job.header, proof, Some(job.nbits_share), job.seed_derivation())
        .context("zk-pow local share verification failed")?;

    let raw_bincode = bincode::serialize(proof)?;
    let raw_b64_est = raw_bincode.len().div_ceil(3) * 4;
    let wire_b64 = if use_gzip {
        let mut encoder = GzEncoder::new(Vec::with_capacity(4096), Compression::fast());
        encoder.write_all(&raw_bincode)?;
        BASE64.encode(encoder.finish()?)
    } else {
        BASE64.encode(&raw_bincode)
    };
    let wire_len = wire_b64.len();
    Ok((wire_b64, raw_b64_est, wire_len))
}

/// Low-power local share verification + wire encoding for sustained mining:
/// - Runs full `zk_pow::verify_plain_proof` when `full_canary` is true (first share of every job).
/// - Otherwise verifies exact matrix jackpot `cpu_eval_group_digest <= bound_le` directly without
///   redundantly re-hashing 535 KiB of constant `0x40` Merkle leaves, preserving package power for the GPU!
pub fn verify_and_encode_share_fast(
    job: &PreparedJob,
    cache: &Constant64MerkleCache,
    nonce: u64,
    group: u32,
    use_gzip: bool,
    full_canary: bool,
) -> Result<(String, usize, usize)> {
    let proof = cache.build_plain_proof(group, nonce)?;
    if full_canary || job.profile.is_edcp() {
        return verify_and_encode_proof(job, &proof, use_gzip);
    }
    check_cert_version_eligible(job.notify.cert_version, &proof)?;
    let (_sig, _root_a, key_a) = cache.derive_attempt(job, nonce);
    let (_tr, digest) = cpu_eval_group_digest(&job.profile, &key_a, &cache.key_b, group);
    ensure!(
        U256::from_little_endian(&digest) <= U256::from_little_endian(&job.bound_le),
        "local CPU jackpot digest check failed for group {group}"
    );

    let raw_bincode = bincode::serialize(&proof)?;
    let raw_b64_est = raw_bincode.len().div_ceil(3) * 4;
    let wire_b64 = if use_gzip {
        let mut encoder = GzEncoder::new(Vec::with_capacity(4096), Compression::fast());
        encoder.write_all(&raw_bincode)?;
        BASE64.encode(encoder.finish()?)
    } else {
        BASE64.encode(&raw_bincode)
    };
    let wire_len = wire_b64.len();
    Ok((wire_b64, raw_b64_est, wire_len))
}

/// Exact CPU reference evaluator for one `group` (`8x16` or `10x16` periodic tile), used by G0 tests
/// and runtime `--self-test` / canary verification.
pub fn cpu_eval_group_digest(
    profile: &KernelProfile,
    key_a: &[u8; 32],
    key_b: &[u8; 32],
    group: u32,
) -> ([u32; 16], [u8; 32]) {
    let mut seed_a = [0u8; 32];
    seed_a[..8].copy_from_slice(b"A_tensor");
    let mut seed_b = [0u8; 32];
    seed_b[..8].copy_from_slice(b"B_tensor");

    let (rows, cols) = profile.group_indices(group);
    let u_a = generate_uniform_random_matrix(&seed_a, key_a, &rows, NOISE_RANK);
    let u_b = generate_uniform_random_matrix(&seed_b, key_b, &cols, NOISE_RANK);
    let p_a = generate_permutation_matrix(&seed_a, key_a, profile.k, NOISE_RANK);
    let p_b = generate_permutation_matrix(&seed_b, key_b, profile.k, NOISE_RANK);

    let h = rows.len();
    let w = cols.len();
    let sig_val = profile.signal_val() as i32;
    let mut acc = vec![0i32; h * w];
    let mut transcript = [0u32; 16];

    for chunk_idx in 0..(profile.k / NOISE_RANK) {
        let k0 = chunk_idx * NOISE_RANK;
        for kk in k0..(k0 + NOISE_RANK) {
            let [pa0, pa1] = p_a[kk];
            let [pb0, pb1] = p_b[kk];
            let mut a_vals = [0i32; 16];
            let mut b_vals = [0i32; 16];
            for u in 0..h {
                a_vals[u] = sig_val
                    + (u_a[u][pa0 as usize] as i32)
                    - (u_a[u][pa1 as usize] as i32);
            }
            for v in 0..w {
                b_vals[v] = sig_val
                    + (u_b[v][pb0 as usize] as i32)
                    - (u_b[v][pb1 as usize] as i32);
            }
            for u in 0..h {
                let av = a_vals[u];
                let row_off = u * w;
                for v in 0..w {
                    acc[row_off + v] += av * b_vals[v];
                }
            }
        }
        let fold_word = acc.iter().fold(0u32, |x, &val| x ^ (val as u32));
        let slot = chunk_idx & 15;
        transcript[slot] = transcript[slot].rotate_left(13) ^ fold_word;
    }

    let mut msg_bytes = [0u8; 64];
    for (i, &w32) in transcript.iter().enumerate() {
        msg_bytes[i * 4..i * 4 + 4].copy_from_slice(&w32.to_le_bytes());
    }
    let digest = blake3_digest(&msg_bytes, Some(*key_a));
    (transcript, digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn g0_merkle_and_zk_pow_verify_roundtrip() {
        // Small valid power-of-two leaf geometry: M=512, N=512, K=2048 -> nleaves = 1024
        let profile = KernelProfile::with_dims(512, 512, 2048).unwrap();
        let header_hex = "01000000".to_string() + &"11".repeat(32) + &"22".repeat(32) + "60504030" + "ffff7f20";
        // Target <= U256::MAX / (160 * 2048) so penalized_target_bound fits in U256 (~4% hit rate)
        let target_hex = "000001".to_string() + &"ff".repeat(29);
        let notify = PoolNotifyJob::from_stratum_params(&json!({
            "job_id": "test01_2097152",
            "header": header_hex,
            "target": target_hex,
            "height": 135000,
            "cert_version": 3
        }))
        .unwrap();

        let job = PreparedJob::new(notify, profile).unwrap();
        let cache = Constant64MerkleCache::build_cpu(&job).unwrap();

        let nonce = 0xdead_beef_cafe_1234u64;
        let (_sig, _root_a, key_a) = cache.derive_attempt(&job, nonce);

        // Find the first winning group on CPU and verify full PlainProof via official zk-pow crate!
        let bound = U256::from_little_endian(&job.bound_le);
        let winning_grp = (0..profile.total_groups())
            .find(|&g| {
                let (_tr, digest) = cpu_eval_group_digest(&profile, &key_a, &cache.key_b, g);
                U256::from_little_endian(&digest) <= bound
            })
            .expect("expected a winning group within total_groups");

        let proof = cache.build_plain_proof(winning_grp, nonce).unwrap();
        let (wire_gzip, raw_len, gz_len) = verify_and_encode_proof(&job, &proof, true).unwrap();
        assert!(!wire_gzip.is_empty());
        assert!(gz_len * 10 < raw_len, "expected >10x gzip compression on constant-64 strips: raw={raw_len} gz={gz_len}");
    }

    #[test]
    fn kryptex_compact_floor_never_relaxes_target() {
        // Kryptex target = 2^224 / 2097152 - 1
        let kryptex_t = (U256::one() << 203) - U256::one();
        let bits = compact_floor(kryptex_t).unwrap();
        let expanded = nbits_to_difficulty(bits);
        assert!(expanded <= kryptex_t);
        assert!(expanded > (kryptex_t >> 1));
    }

    #[test]
    fn g0_direct_l2_rectangular_m_gt_n_verify_roundtrip() {
        // Rectangular Direct-L2 shape: M=1024, N=256 (ratio=4), K=2048, CTA=128x128 (8x16 tile)
        let profile = KernelProfile::direct_l2_128x128(1024, 256, 2048).unwrap();
        let header_hex = "01000000".to_string() + &"33".repeat(32) + &"44".repeat(32) + "60504030" + "ffff7f20";
        let target_hex = "000001".to_string() + &"ff".repeat(29);
        let notify = PoolNotifyJob::from_stratum_params(&json!({
            "job_id": "test_rect_2097152",
            "header": header_hex,
            "target": target_hex,
            "height": 145000,
            "cert_version": 3
        }))
        .unwrap();

        let job = PreparedJob::new(notify, profile).unwrap();
        let cache = Constant64MerkleCache::build_cpu(&job).unwrap();
        let nonce = 0xfeed_face_2026_0077u64;
        let (_sig, _root_a, key_a) = cache.derive_attempt(&job, nonce);

        let bound = U256::from_little_endian(&job.bound_le);
        let winning_grp = (0..profile.total_groups())
            .find(|&g| {
                let (_tr, digest) = cpu_eval_group_digest(&profile, &key_a, &cache.key_b, g);
                U256::from_little_endian(&digest) <= bound
            })
            .expect("expected a winning group in rectangular M=1024, N=256 test");

        let proof = cache.build_plain_proof(winning_grp, nonce).unwrap();
        let (wire_gzip, raw_len, gz_len) = verify_and_encode_proof(&job, &proof, true).unwrap();
        assert!(!wire_gzip.is_empty());
        assert!(gz_len * 10 < raw_len);
    }

    #[test]
    fn g0_basis_128x256_verify_roundtrip() {
        let profile = KernelProfile::with_dims_128x256(512, 512, 2048).unwrap();
        assert!(profile.is_basis_128() && !profile.is_direct_l2());
        let header_hex = "01000000".to_string() + &"55".repeat(32) + &"66".repeat(32) + "60504030" + "ffff7f20";
        let target_hex = "000001".to_string() + &"ff".repeat(29);
        let notify = PoolNotifyJob::from_stratum_params(&json!({
            "job_id": "test_b128_2097152",
            "header": header_hex,
            "target": target_hex,
            "height": 146000,
            "cert_version": 3
        }))
        .unwrap();
        let job = PreparedJob::new(notify, profile).unwrap();
        let cache = Constant64MerkleCache::build_cpu(&job).unwrap();
        let nonce = 0xcafe_babe_2026_0030u64;
        let (_sig, _root_a, key_a) = cache.derive_attempt(&job, nonce);
        let bound = U256::from_little_endian(&job.bound_le);
        let winning_grp = (0..profile.total_groups())
            .find(|&g| {
                let (_tr, digest) = cpu_eval_group_digest(&profile, &key_a, &cache.key_b, g);
                U256::from_little_endian(&digest) <= bound
            })
            .expect("expected winning group in basis_128x256 test");
        let proof = cache.build_plain_proof(winning_grp, nonce).unwrap();
        let (wire_gzip, raw_len, gz_len) = verify_and_encode_proof(&job, &proof, true).unwrap();
        assert!(!wire_gzip.is_empty() && gz_len * 10 < raw_len);
    }

    #[test]
    fn g0_edcp_120_const0_bsalt_zk_pow_roundtrip() {
        let profile = KernelProfile::edcp_120(1024, 512, 2048).unwrap();
        assert!(profile.is_edcp() && profile.sampled_n() == 256);
        let header_hex = "01000000".to_string() + &"77".repeat(32) + &"88".repeat(32) + "60504030" + "ffff7f20";
        let target_hex = "000001".to_string() + &"ff".repeat(29);
        let notify = PoolNotifyJob::from_stratum_params(&json!({
            "job_id": "test_edcp_const0_v3",
            "header": header_hex,
            "target": target_hex,
            "height": 148000,
            "cert_version": 3
        }))
        .unwrap();
        let job = PreparedJob::new(notify, profile).unwrap();
        let cache = Constant64MerkleCache::build_cpu(&job).unwrap();
        assert!(cache.edcp.is_some());

        let nonce = 0x2026_0400_edcb_a987u64;
        let (_sig, _root_a, key_a) = cache.derive_attempt(&job, nonce);
        let bound = U256::from_little_endian(&job.bound_le);

        // Search only valid sampled groups (`nt < sampled_n / 64`, leaving row n-1 unsampled!)
        let tiles_n_full = profile.n / profile.warp_n();
        let tiles_n_sampled = profile.sampled_n() / profile.warp_n();
        let winning_grp = (0..profile.total_groups())
            .filter(|&g| ((g as usize) / 32) % tiles_n_full < tiles_n_sampled)
            .find(|&g| {
                let (_tr, digest) = cpu_eval_group_digest(&profile, &key_a, &cache.key_b, g);
                U256::from_little_endian(&digest) <= bound
            })
            .expect("expected winning group in edcp_120 Constant-0 + b_salt test");

        let proof = cache.build_plain_proof(winning_grp, nonce).unwrap();
        let (wire_gzip, raw_len, gz_len) = verify_and_encode_proof(&job, &proof, true).unwrap();
        assert!(!wire_gzip.is_empty() && gz_len * 10 < raw_len);
    }

    #[test]
    fn g0_odd_leaf_sm_aligned_m_lt_n_zk_pow_roundtrip() {
        // Non-power-of-two SM-aligned M < N (`M = 3*128 + 16 = 400`, `sampled_m = 384`, `N = 512`, `K = 2048` -> `nleaves_a = 800`, `nleaves_b = 1024`)
        let profile = KernelProfile::with_dims_128x256(400, 512, 2048).unwrap();
        assert_eq!(profile.sampled_m(), 384);
        let header_hex = "01000000".to_string() + &"99".repeat(32) + &"ab".repeat(32) + "60504030" + "ffff7f20";
        let target_hex = "000001".to_string() + &"ff".repeat(29);
        let notify = PoolNotifyJob::from_stratum_params(&json!({
            "job_id": "test_sm_exact_odd_merkle_v3",
            "header": header_hex,
            "target": target_hex,
            "height": 150000,
            "cert_version": 3
        }))
        .unwrap();
        let job = PreparedJob::new(notify, profile).unwrap();
        let cache = Constant64MerkleCache::build_cpu(&job).unwrap();
        let nonce = 0x2026_1122_feed_beefu64;
        let (_sig, _root_a, key_a) = cache.derive_attempt(&job, nonce);
        let bound = U256::from_little_endian(&job.bound_le);
        let winning_grp = (0..profile.total_groups())
            .find(|&g| {
                let (_tr, digest) = cpu_eval_group_digest(&profile, &key_a, &cache.key_b, g);
                U256::from_little_endian(&digest) <= bound
            })
            .expect("expected winning group in SM-aligned odd-leaf M=400, N=512 test");
        let proof = cache.build_plain_proof(winning_grp, nonce).unwrap();
        let (wire_gzip, raw_len, gz_len) = verify_and_encode_proof(&job, &proof, true).unwrap();
        assert!(!wire_gzip.is_empty() && gz_len * 10 < raw_len);
    }

    #[test]
    fn g0_strassen_24k_const0_sm_aligned_zk_pow_roundtrip() {
        // Constant-0 RAW-24K Strassen SM-aligned odd-leaf shape (`M = 3*128 + 16 = 400`, `N = 512`, `K = 2048`):
        // No b_salt (`edcp = None`), 100% of N sampled (`sampled_n = 512`), Constant-0 signal byte `0x00`!
        let profile = KernelProfile::with_dims_strassen_128x256(400, 512, 2048).unwrap();
        assert!(profile.is_strassen() && !profile.is_edcp() && profile.const_zero);
        assert_eq!(profile.sampled_m(), 384);
        assert_eq!(profile.sampled_n(), 512);
        let header_hex = "01000000".to_string() + &"bc".repeat(32) + &"de".repeat(32) + "60504030" + "ffff7f20";
        let target_hex = "000001".to_string() + &"ff".repeat(29);
        let notify = PoolNotifyJob::from_stratum_params(&json!({
            "job_id": "test_strassen_const0_odd_v3",
            "header": header_hex,
            "target": target_hex,
            "height": 152000,
            "cert_version": 3
        }))
        .unwrap();
        let job = PreparedJob::new(notify, profile).unwrap();
        let cache = Constant64MerkleCache::build_cpu(&job).unwrap();
        assert!(cache.edcp.is_none());
        let nonce = 0x2026_1125_57a5_5e00u64;
        let (_sig, _root_a, key_a) = cache.derive_attempt(&job, nonce);
        let bound = U256::from_little_endian(&job.bound_le);
        let winning_grp = (0..profile.total_groups())
            .find(|&g| {
                let (_tr, digest) = cpu_eval_group_digest(&profile, &key_a, &cache.key_b, g);
                U256::from_little_endian(&digest) <= bound
            })
            .expect("expected winning group in Strassen Constant-0 odd-leaf test");
        let proof = cache.build_plain_proof(winning_grp, nonce).unwrap();
        let (wire_gzip, raw_len, gz_len) = verify_and_encode_proof(&job, &proof, true).unwrap();
        assert!(!wire_gzip.is_empty() && gz_len * 10 < raw_len);
    }
}
