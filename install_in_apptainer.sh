#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./install_inside_apptainer.sh scgpt
#   ./install_inside_apptainer.sh gears scgpt glob2
# or:
#   PACKAGES="gears scgpt glob2" ./install_inside_apptainer.sh

# Collect packages from CLI args or PACKAGES env var, default to 'scgpt'
PACKAGES="omnipath"

# Make local dirs (same as your original)
mkdir -p .local/lib/python3.11/site-packages .cache/pip tmp

apptainer exec \
  --env PACKAGES="$PACKAGES" \
  -H "$PWD" \
  --bind /iridisfs/ddnb/Luke:/workspace \
  --bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
  --bind "$PWD/tmp":/tmp \
  --nv --fakeroot /iridisfs/ddnb/Luke/helical_sb \
  bash --noprofile --norc -lc '
    set -euo pipefail
    export HOME="$PWD"
    export TMPDIR="$PWD/tmp"
    export XDG_CACHE_HOME="$PWD/.cache"
    export PIP_CACHE_DIR="$PWD/.cache/pip"

    echo "Packages to install: $PACKAGES"
    # Install to user site: $HOME/.local/lib/python3.11/site-packages
    # Accept space-separated list
    python3 -m pip install --user --no-cache-dir --no-build-isolation $PACKAGES

    echo "---- verify ----"
    python3 - <<PY
import os, importlib, site, sys
print("user site:", site.getusersitepackages())

raw = os.environ.get("PACKAGES","")
# Try to import each; translate dashes to underscores for module names
pkgs = [p for p in raw.split() if p.strip()]
ok = True
for p in pkgs:
    mod = p.replace("-", "_")
    try:
        m = importlib.import_module(mod)
        path = getattr(m, "__file__", "built-in / namespace")
        print(f"{mod} @ {path}")
    except Exception as e:
        ok = False
        print(f"FAILED to import {mod}: {e}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
  '
