#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/hpc/run_train_state.sh" "$@"
