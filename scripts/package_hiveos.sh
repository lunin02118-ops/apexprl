#!/usr/bin/env bash
# Builds portable GLIBC 2.31 Linux binary + HiveOS Custom Miner package (`dist/apexprl-hiveos.tar.gz`).
set -euo pipefail

export PATH="/root/.cargo/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/root/apexprl-target}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/4] Auditing CUDA SASS Kernels (v0.2 Hyper-L2 Direct + Legacy Basis; SM89 Ada + SM120a Blackwell) ==="
mkdir -p /root/apexprl-cubins
nvcc -O3 -std=c++20 --expt-relaxed-constexpr -diag-suppress 549 -arch=sm_89 -Xptxas -O3,-v --cubin kernels/apexprl_direct_sm89.cu -o /root/apexprl-cubins/apexprl_direct_sm89.cubin
nvcc -O3 -std=c++20 --expt-relaxed-constexpr -diag-suppress 549 -arch=sm_120a -Xptxas -O1,-v --cubin kernels/apexprl_direct_sm120a.cu -o /root/apexprl-cubins/apexprl_direct_sm120a.cubin
nvcc -O3 -std=c++20 --expt-relaxed-constexpr -diag-suppress 549 -arch=sm_89 -Xptxas -O1,-v --cubin kernels/apexprl_sm89.cu -o /root/apexprl-cubins/apexprl_sm89.cubin
nvcc -O3 -std=c++20 --expt-relaxed-constexpr -diag-suppress 549 -arch=sm_120a -Xptxas -O1,-v --cubin kernels/apexprl_sm120a.cu -o /root/apexprl-cubins/apexprl_sm120a.cubin
python3 tools/check_sass.py /root/apexprl-cubins

echo "=== [2/4] Building Portable HiveOS Binary (x86_64-unknown-linux-gnu.2.31) ==="
cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.31 -p apexprl-cli

BIN_PATH="${CARGO_TARGET_DIR}/x86_64-unknown-linux-gnu/release/apexprl"
echo "=== [3/4] Auditing Dynamic Symbols & GLIBC Compatibility ==="
ldd "$BIN_PATH" || true
MAX_GLIBC=$(objdump -T "$BIN_PATH" | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -V | tail -n 1 || echo "NONE")
echo "Highest required GLIBC version: ${MAX_GLIBC}"
if echo "$MAX_GLIBC" | grep -Eq 'GLIBC_2\.(3[2-9]|[4-9][0-9])'; then
    echo "ERROR: Binary requires $MAX_GLIBC (> GLIBC_2.31)!"
    exit 1
fi

echo "=== [4/4] Assembling HiveOS Custom Miner Archive ==="
STAGE_DIR="/tmp/apexprl-hiveos-stage/apexprl"
rm -rf "/tmp/apexprl-hiveos-stage"
mkdir -p "$STAGE_DIR" "$PROJECT_ROOT/dist"

cp "$BIN_PATH" "$STAGE_DIR/apexprl"
cp "$PROJECT_ROOT/hiveos/h-manifest.conf" "$STAGE_DIR/"
cp "$PROJECT_ROOT/hiveos/h-config.sh" "$STAGE_DIR/"
cp "$PROJECT_ROOT/hiveos/h-run.sh" "$STAGE_DIR/"
cp "$PROJECT_ROOT/hiveos/h-stats.sh" "$STAGE_DIR/"
cp "$PROJECT_ROOT/hiveos/flight-sheet-kryptex.json" "$PROJECT_ROOT/dist/"
cp "$BIN_PATH" "$PROJECT_ROOT/dist/apexprl"

sed -i 's/\r$//' "$STAGE_DIR"/h-*.sh "$STAGE_DIR"/h-manifest.conf
chmod 0755 "$STAGE_DIR/apexprl" "$STAGE_DIR"/h-*.sh
chmod 0644 "$STAGE_DIR/h-manifest.conf"

tar -C "/tmp/apexprl-hiveos-stage" -czf "$PROJECT_ROOT/dist/apexprl-hiveos.tar.gz" apexprl
rm -rf "/tmp/apexprl-hiveos-stage"
(cd "$PROJECT_ROOT/dist" && sha256sum apexprl apexprl-hiveos.tar.gz > SHA256SUMS.txt)

echo "=== DONE! Package ready at dist/apexprl-hiveos.tar.gz ==="
cat "$PROJECT_ROOT/dist/SHA256SUMS.txt"
