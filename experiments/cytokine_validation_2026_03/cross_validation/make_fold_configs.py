#!/usr/bin/env python3
"""
Create per-fold YAML configs that point to the correct TOML and set run names.

For each K = 1..9, this script:
  - Reads the base experiment config.yaml in cytokine_validation_2026/
  - Sets training.data.toml_config_path to cross_validation/tomls/cytokine_foldK.toml
  - Sets training.output.name to run_2026_02_05_foldK
  - Writes the result to cross_validation/configs/config_foldK.yaml

Use these configs with run_train_state.sh, e.g.:
  sbatch run_train_state.sh experiments/cytokine_validation_2026/cross_validation/configs/config_fold1.yaml
"""

from copy import deepcopy
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent  # .../cytokine_validation_2026
CV_DIR = BASE_DIR / "cross_validation"
TOML_DIR = CV_DIR / "tomls"
CONFIGS_DIR = CV_DIR / "configs"

BASE_CONFIG = BASE_DIR / "config.yaml"

N_FOLDS = 9


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Base YAML config not found: {BASE_CONFIG}")

    if not TOML_DIR.exists():
        raise FileNotFoundError(f"TOML directory not found: {TOML_DIR}")

    CONFIGS_DIR.mkdir(exist_ok=True)

    # Load base YAML config once
    with BASE_CONFIG.open("r") as f:
        base_cfg = yaml.safe_load(f)

    for k in range(1, N_FOLDS + 1):
        cfg_k = deepcopy(base_cfg)

        # 1) Point to the correct fold TOML
        toml_path = (
            BASE_DIR
            / "cross_validation"
            / "tomls"
            / f"cytokine_fold{k}.toml"
        )

        training_block = cfg_k.setdefault("training", {})
        data_block = training_block.setdefault("data", {})
        data_block["toml_config_path"] = str(toml_path)

        # 2) Give a fold-specific run name (leave output_dir as in base config)
        output_block = training_block.setdefault("output", {})
        base_output_dir = output_block.get(
            "output_dir",
            str(BASE_DIR),
        )
        output_block["output_dir"] = base_output_dir
        output_block["name"] = f"run_2026_02_05_fold{k}"

        # 3) Write fold-specific YAML into cross_validation/configs
        out_yaml = CONFIGS_DIR / f"config_fold{k}.yaml"
        with out_yaml.open("w") as f:
            yaml.safe_dump(cfg_k, f, sort_keys=False)

        print(f"Wrote {out_yaml} with TOML {toml_path}")

    print("Done generating fold configs.")


if __name__ == "__main__":
    main()

