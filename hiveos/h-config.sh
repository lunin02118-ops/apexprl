#!/usr/bin/env bash
# Generates ApexPRL CLI configuration from HiveOS Flight Sheet variables.
# Sourced by HiveOS agent; uses BASH_SOURCE to locate h-manifest.conf safely.

. "$(dirname "${BASH_SOURCE[0]}")/h-manifest.conf"

conf="--algo ${CUSTOM_ALGO:-pearlhash}"

# Support multiple failover URLs separated by whitespace or newlines
for url in ${CUSTOM_URL:-}; do
    conf+=" --pool ${url}"
done

# Default to Kryptex TLS endpoints if no URL was specified
if [[ -z "${CUSTOM_URL:-}" ]]; then
    conf+=" --pool stratum+ssl://prl-ru.kryptex.network:8048 --pool stratum+ssl://prl-eu.kryptex.network:8048"
fi

if [[ -n "${CUSTOM_TEMPLATE:-}" ]]; then
    conf+=" --wallet ${CUSTOM_TEMPLATE}"
fi
if [[ -n "${CUSTOM_PASS:-}" ]]; then
    conf+=" --pass ${CUSTOM_PASS}"
fi

# Persist kernel auto-tune profile across reboots in /hive-config/
conf+=" --api-bind 127.0.0.1:${MINER_API_PORT} --tune-cache ${CUSTOM_TUNE_CACHE} --no-color"

if [[ -n "${CUSTOM_USER_CONFIG:-}" ]]; then
    conf+=" ${CUSTOM_USER_CONFIG}"
fi

mkdir -p "$(dirname "$CUSTOM_CONFIG_FILENAME")"
echo "$conf" > "$CUSTOM_CONFIG_FILENAME"
