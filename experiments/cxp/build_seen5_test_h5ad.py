#!/usr/bin/env python3
"""Deprecated: seen5 eval uses fewshot test perts on train1 (see cxp_nested_v2_collectri_seen5.toml)."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "build_seen5_test_h5ad.py is deprecated; "
        "use tomls/cxp_nested_v2_collectri_seen5.toml (train1 + fewshot test cytokines).",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
