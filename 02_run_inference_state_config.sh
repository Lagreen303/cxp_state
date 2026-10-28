#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/hpc/02_run_inference_state_config.sh" "$@"
