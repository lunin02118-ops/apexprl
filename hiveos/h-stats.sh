#!/usr/bin/env bash
# HiveOS telemetry collector for ApexPRL.
# Sourced by HiveOS agent: must only populate $khs and $stats without calling exit.

. "$(dirname "${BASH_SOURCE[0]}")/h-manifest.conf"

raw=$(curl -s --connect-timeout 2 --max-time 3 "http://127.0.0.1:${MINER_API_PORT}/api/v1/hiveos")

if [[ -z "$raw" ]] || ! echo "$raw" | jq -e '.stats' >/dev/null 2>&1; then
    khs=0
    stats=""
else
    khs=$(echo "$raw" | jq -r '.khs // 0')
    stats=$(echo "$raw" | jq -c '.stats')
fi

if [[ -z "$khs" ]]; then
    khs=0
fi
