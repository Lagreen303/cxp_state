#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/hpc/00_preprocess_train.sh" "$@"
