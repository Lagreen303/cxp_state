#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/hpc/install_in_apptainer.sh" "$@"
