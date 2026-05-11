#!/bin/bash
# UV environment setup for State project
# Source this file: source setup_uv_env.sh
#
# Defaults are under this repository. Override any variable before sourcing
# if your site stores caches on scratch, for example:
#   export UV_CACHE_DIR=/scratch/$USER/uv-cache && source setup_uv_env.sh

_STATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$_STATE_ROOT/uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$_STATE_ROOT/uv/pythons}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$_STATE_ROOT/uv/tools}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$_STATE_ROOT/.venv}"

unset _STATE_ROOT

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")" 2>/dev/null || true

# sanity checks
echo "UV Cache Directory:"
uv cache dir
echo "UV Tool Directory:"
uv tool dir
echo "UV Python Install Directory:"
echo "$UV_PYTHON_INSTALL_DIR"
echo "UV Project Environment:"
echo "$UV_PROJECT_ENVIRONMENT"
