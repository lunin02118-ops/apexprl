//! Consensus, Merkle caching, V3 Salted Seed derivation, CPU reference solver,
//! and local `zk-pow` share verification for ApexPRL.

use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use flate2::{write::GzEncoder, Compression};
use pearl_blake3::{blake3_digest, Blake3Hasher, MerkleProof, MerkleTree};
use primitive_types::U256;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
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

/// Unified tile geometry shared identically by both `sm_120a` (RTX 50) and `sm_89` (RTX 40)
/// Basis Kernels: CTA = 160 x 256 (8 warps in 2x4, warp tile = 80 x 64 = 5 x 8 m16n8 atoms).
/// Each thread owns 10 rows x 16 cols = 160 accumulators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelProfile {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub cta_m: usize,
    pub cta_n: usize,
    pub sampling_rows: usize,
}

impl KernelProfile {
    /// Creates a unified 160x256 Basis Kernel profile for `k` in {4096, 8192, 16384, 32768, 65536}.
    pub fn basis_160x256(k: usize) -> Result<Self> {
        Self::with_dims(DEFAULT_M, DEFAULT_N, k)
    }

    /// Custom matrix dimensions (used for fast G0 unit tests as well as production 32768x32768).
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
        })
    }

    /// Number of M rows actually covered by full 160-row CTAs.
    /// Leaves the tail rows (including row `m - 1` where the 64-byte nonce lives) unsampled!
    #[inline]
    pub fn sampled_m(&self) -> usize {
        let max_sampled = self.m.saturating_sub(16); // strictly exclude row m-1
        (max_sampled / self.cta_m) * self.cta_m
    }

    #[inline]
    pub fn warp_m(&self) -> usize {
        self.sampling_rows * 8 // 80
    }

    #[inline]
    pub fn warp_n(&self) -> usize {
        64
    }

    #[inline]
    pub fn macs_per_round(&self) -> f64 {
        (self.sampled_m() as f64) * (self.n as f64) * (self.k as f64)
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

/// Encodes a 64-bit `nonce` into the 64-byte prefix of row `m - 1` using values `{63, 64}`.
#[inline]
pub fn encode_nonce_signal(nonce: u64) -> [u8; 64] {
    let mut sig = [CONST_SIGNAL_BYTE; 64];
    for bit in 0..64 {
        sig[bit] = 63 + (((nonce >> bit) & 1) as u8);
    }
    sig
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

/// Binary Merkle heap for the Constant-64 matrix (`m == n`), enabling ~3 microsecond
/// incremental `root_A` updates per nonce and zero-VRAM-read `PlainProof` assembly.
pub struct Constant64MerkleCache {
    pub profile: KernelProfile,
    pub job_key: [u8; 32],
    nleaves: usize,
    nonce_leaf: usize,
    /// Heap order: `2 * nleaves * 32` bytes (`node 1` unused, `nodes 2..3` children of root, `nleaves..2*nleaves` leaves).
    heap: Vec<u8>,
    hasher: Blake3Hasher,
    pub root_b: [u8; 32],
    pub key_b: [u8; 32],
}

impl Constant64MerkleCache {
    pub fn heap_byte_len(profile: &KernelProfile) -> usize {
        let nleaves = (profile.m * profile.k) / CHUNK_BYTES;
        2 * nleaves * 32
    }

    /// Builds the entire Constant-64 Merkle heap on CPU using Rayon (used for tests or CPU fallback).
    pub fn build_cpu(job: &PreparedJob) -> Result<Self> {
        let profile = job.profile;
        ensure!(profile.m == profile.n, "Constant64MerkleCache requires square M == N");
        let nleaves = (profile.m * profile.k) / CHUNK_BYTES;
        ensure!(nleaves.is_power_of_two() && nleaves >= 2, "nleaves must be power of two >= 2");

        let hasher = Blake3Hasher::with_key(job.job_key);
        let const_chunk = [CONST_SIGNAL_BYTE; CHUNK_BYTES];
        let mut heap = vec![0u8; 2 * nleaves * 32];

        // Compute leaves in parallel
        heap[nleaves * 32..]
            .par_chunks_mut(32)
            .enumerate()
            .for_each(|(leaf_idx, slot)| {
                let cv = hasher.chunk_cv(&const_chunk, leaf_idx as u64);
                slot.copy_from_slice(&cv);
            });

        // Reduce internal levels bottom-up
        let mut level_count = nleaves >> 1;
        while level_count >= 2 {
            let (upper, lower) = heap.split_at_mut(level_count * 2 * 32);
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

        Self::from_gpu_heap(job, heap)
    }

    /// Wraps a GPU-computed heap, verifying its root integrity on CPU in < 1 microsecond.
    pub fn from_gpu_heap(job: &PreparedJob, heap: Vec<u8>) -> Result<Self> {
        let profile = job.profile;
        let nleaves = (profile.m * profile.k) / CHUNK_BYTES;
        ensure!(heap.len() == 2 * nleaves * 32, "invalid Merkle heap size {}", heap.len());
        let hasher = Blake3Hasher::with_key(job.job_key);
        let l2: [u8; 32] = heap[2 * 32..3 * 32].try_into().unwrap();
        let r3: [u8; 32] = heap[3 * 32..4 * 32].try_into().unwrap();
        let root_b = hasher.root_cv(&l2, &r3);

        let (key_b, _) = derive_noise_seeds(
            &job.job_key,
            &root_b,
            &root_b,
            profile.m as u32,
            profile.n as u32,
            job.seed_derivation(),
        );

        Ok(Self {
            profile,
            job_key: job.job_key,
            nleaves,
            nonce_leaf: ((profile.m - 1) * profile.k) / CHUNK_BYTES,
            heap,
            hasher,
            root_b,
            key_b,
        })
    }

    #[inline]
    fn node(&self, idx: usize) -> [u8; 32] {
        self.heap[idx * 32..(idx + 1) * 32].try_into().unwrap()
    }

    /// Computes `(nonce_signal64, root_a, key_a)` for `nonce` in ~3 microseconds on CPU!
    pub fn derive_attempt(&self, job: &PreparedJob, nonce: u64) -> ([u8; 64], [u8; 32], [u8; 32]) {
        let sig = encode_nonce_signal(nonce);
        let mut chunk = [CONST_SIGNAL_BYTE; CHUNK_BYTES];
        chunk[..64].copy_from_slice(&sig);

        let mut node = self.nleaves + self.nonce_leaf;
        let mut cv = self.hasher.chunk_cv(&chunk, self.nonce_leaf as u64);
        while node > 1 {
            let sib = self.node(node ^ 1);
            let (l, r) = if (node & 1) == 1 { (sib, cv) } else { (cv, sib) };
            cv = if node < 4 {
                self.hasher.root_cv(&l, &r)
            } else {
                self.hasher.parent_cv(&l, &r)
            };
            node >>= 1;
        }
        let root_a = cv;
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

    fn nonce_ancestors(&self, sig: &[u8; 64]) -> ([u8; 32], HashMap<usize, [u8; 32]>) {
        let mut chunk = [CONST_SIGNAL_BYTE; CHUNK_BYTES];
        chunk[..64].copy_from_slice(sig);
        let mut path = HashMap::with_capacity(24);
        let mut node = self.nleaves + self.nonce_leaf;
        let mut cv = self.hasher.chunk_cv(&chunk, self.nonce_leaf as u64);
        path.insert(node, cv);
        while node > 1 {
            let sib = self.node(node ^ 1);
            let (l, r) = if (node & 1) == 1 { (sib, cv) } else { (cv, sib) };
            cv = if node < 4 {
                self.hasher.root_cv(&l, &r)
            } else {
                self.hasher.parent_cv(&l, &r)
            };
            node >>= 1;
            path.insert(node, cv);
        }
        (cv, path)
    }

    fn build_matrix_proof(
        &self,
        rows: Vec<usize>,
        root: [u8; 32],
        overrides: &HashMap<usize, [u8; 32]>,
        sig: Option<&[u8; 64]>,
    ) -> Result<MatrixMerkleProof> {
        let leaves = MerkleTree::compute_leaf_indices_from_rows(&rows, (self.profile.m, self.profile.k));
        ensure!(!leaves.is_empty() && *leaves.last().unwrap() < self.nleaves, "proof rows out of range");

        let leaf_data = leaves
            .iter()
            .map(|&leaf_idx| match sig {
                Some(s) if leaf_idx == self.nonce_leaf => {
                    let mut c = [CONST_SIGNAL_BYTE; CHUNK_BYTES];
                    c[..64].copy_from_slice(s);
                    c
                }
                _ => [CONST_SIGNAL_BYTE; CHUNK_BYTES],
            })
            .collect();

        let mut siblings = Vec::new();
        let mut current = leaves.clone();
        let (mut width, mut base) = (self.nleaves, self.nleaves);
        while width > 1 {
            let selected: BTreeSet<usize> = current.iter().copied().collect();
            for &i in &current {
                let other = i ^ 1;
                if other < width && !selected.contains(&other) {
                    let node = base + other;
                    siblings.push(overrides.get(&node).copied().unwrap_or_else(|| self.node(node)));
                }
            }
            current = current.iter().map(|i| i / 2).collect::<BTreeSet<_>>().into_iter().collect();
            width = width.div_ceil(2);
            base >>= 1;
        }

        Ok(MatrixMerkleProof {
            row_indices: rows,
            proof: MerkleProof {
                leaf_data,
                leaf_indices: leaves,
                total_leaves: self.nleaves,
                root,
                siblings,
            },
        })
    }

    /// Assembles a complete `PlainProof` for winning GPU `group` on `nonce`.
    pub fn build_plain_proof(&self, group: u32, nonce: u64) -> Result<PlainProof> {
        ensure!(group < self.profile.total_groups(), "invalid hit group {group}");
        let sig = encode_nonce_signal(nonce);
        let (root_a, path_a) = self.nonce_ancestors(&sig);
        let (rows, cols) = self.profile.group_indices(group);
        Ok(PlainProof {
            m: self.profile.m,
            n: self.profile.n,
            k: self.profile.k,
            noise_rank: NOISE_RANK,
            a: self.build_matrix_proof(rows, root_a, &path_a, Some(&sig))?,
            bt: self.build_matrix_proof(cols, self.root_b, &HashMap::new(), None)?,
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

/// Exact CPU reference evaluator for one `group` (`10 x 16` periodic tile), used by G0 tests
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
                a_vals[u] = (CONST_SIGNAL_VAL as i32)
                    + (u_a[u][pa0 as usize] as i32)
                    - (u_a[u][pa1 as usize] as i32);
            }
            for v in 0..w {
                b_vals[v] = (CONST_SIGNAL_VAL as i32)
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
}
