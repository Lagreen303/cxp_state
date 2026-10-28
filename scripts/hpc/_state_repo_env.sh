# Source from scripts/hpc/*.sh: repo root + setup_uv_env (not SLURM submit dir).
_hpc_lib="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_REPO_ROOT="$(cd "${_hpc_lib}/../.." && pwd)"
cd "${STATE_REPO_ROOT}"
if [[ -f "${STATE_REPO_ROOT}/setup_uv_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${STATE_REPO_ROOT}/setup_uv_env.sh"
fi
export UV_OFFLINE="${UV_OFFLINE:-1}"
unset _hpc_lib
