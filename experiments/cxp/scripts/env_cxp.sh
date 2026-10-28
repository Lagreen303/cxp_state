# Source from Slurm drivers: resolves repo root for CxP STATE fork.
# Override with STATE_ROOT if you run from a non-standard checkout.
if [[ -z "${STATE_ROOT:-}" ]]; then
  _cxp_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  STATE_ROOT="$(cd "${_cxp_env_dir}/../../.." && pwd)"
  unset _cxp_env_dir
fi
export STATE_ROOT
export UV_OFFLINE="${UV_OFFLINE:-1}"
