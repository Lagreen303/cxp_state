#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/hpc/01_run_train_state_config.sh" "$@"
