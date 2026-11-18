#!/usr/bin/env bash
# Launches ApexPRL on HiveOS Linux with pinned memory limits unlocked.

cd "$(dirname "$0")" || exit 1
. ./h-manifest.conf

if [[ ! -f "$CUSTOM_CONFIG_FILENAME" ]]; then
    echo "ERROR: Missing config $CUSTOM_CONFIG_FILENAME (run h-config.sh first)"
    exit 1
fi

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

# Allow locked/pinned host memory pages for zero-copy CUDA HitRing & DMA
ulimit -l unlimited 2>/dev/null || true

# Ensure NVIDIA driver libraries (CUDA Driver API & NVML) are on library path
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

exec ./apexprl $(< "$CUSTOM_CONFIG_FILENAME") 2>&1 | tee --append "${CUSTOM_LOG_BASENAME}.log"
