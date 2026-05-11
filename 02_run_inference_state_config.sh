#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8

# This script reads configuration from a YAML file and runs STATE inference
# Usage: sbatch 02_run_inference_state_config.sh [config.yaml]

CONFIG_FILE="${1:-config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Parse YAML and build command using Python, then execute
CMD=$(python3 << 'PYTHON_SCRIPT'
import yaml
import sys
import os
from pathlib import Path
config_file = os.environ.get('CONFIG_FILE', 'config.yaml')

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading YAML: {e}", file=sys.stderr)
    sys.exit(1)

# Extract configs
train_cfg = (config.get('training') or {})
train_out = (train_cfg.get('output') or {})
inf_cfg = (config.get('inference') or {}) or {}
viz_cfg = (config.get('visualization') or {}) or {}
config_dir = Path(config_file).parent

# # # Derive run_dir from training output if possible
# output_dir = train_out.get('output_dir')
# # name = train_out.get('name')
# # run_dir = None
# # if output_dir and name:
# #     run_dir = f"{output_dir}/{name}"
# # elif output_dir:
# #     # Fallback to legacy convention where name == output_dir
# #     run_dir = f"{output_dir}/{output_dir}"

# # # Resolve model_dir if missing
#  model_dir = inf_cfg.get('model_dir')
# # if (not model_dir or str(model_dir).strip() == "") and run_dir:
# #     model_dir = run_dir
# #     inf_cfg['model_dir'] = model_dir

# # Resolve checkpoint path if missing or only a filename
# ckpt = inf_cfg.get('checkpoint')
# if not ckpt or os.sep not in str(ckpt):
#     ckpt_name = str(ckpt).strip() if ckpt else "last.ckpt"
#     if run_dir:
#         ckpt = str(Path(run_dir) / "checkpoints" / ckpt_name)
#         inf_cfg['checkpoint'] = ckpt

# # Resolve adata: if missing, try to use an existing control template; otherwise error
# adata = inf_cfg.get('adata')
# # if (not adata or str(adata).strip() == ""):
# #     candidates = []
# #     if output_dir:
# #         candidates.append(Path(output_dir) / "control_template.h5ad")
# #     if run_dir:
# #         candidates.append(Path(run_dir) / "control_template.h5ad")
# #     chosen = next((str(p) for p in candidates if p and p.exists()), None)
# #     if chosen:
# #         inf_cfg['adata'] = chosen
# #     else:
# #         print("Error: inference.adata is not set and no control template found at "
# #               f"{[str(p) for p in candidates]}. Set inference.adata or pre-generate the template.",
# #               file=sys.stderr)
#         sys.exit(2)

# # If inference output missing, create a sensible default inside run_dir
# if not inf_cfg.get('output') and run_dir:
#     # inf_cfg['output'] = f"{output_dir}/vcc_prediction.h5ad" if output_dir else f"{run_dir}/prediction.h5ad"

# # Build command arguments
args = []

if 'output' in inf_cfg:
    args.append(f'--output "{inf_cfg["output"]}"')
if 'model_dir' in inf_cfg:
    args.append(f'--model-dir "{inf_cfg["model_dir"]}"')
if 'checkpoint' in inf_cfg:
    args.append(f'--checkpoint "{inf_cfg["checkpoint"]}"')
if 'adata' in inf_cfg:
    args.append(f'--adata "{inf_cfg["adata"]}"')
if 'pert_col' in inf_cfg:
    args.append(f'--pert-col "{inf_cfg["pert_col"]}"')
if 'control_pert' in inf_cfg:
    args.append(f'--control-pert "{inf_cfg["control_pert"]}"')

# Build and print command
cmd = 'uv run state tx infer ' + ' '.join(args)
print(cmd)
PYTHON_SCRIPT
)

# Execute the command
eval "$CMD"

