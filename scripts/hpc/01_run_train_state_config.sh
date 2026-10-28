#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --partition=l40
#SBATCH --export=ALL
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12

# This script reads configuration from a YAML file and runs STATE training
# Usage: sbatch 01_run_train_state_config.sh [config.yaml]

CONFIG_FILE="${1:-config.yaml}"
export CONFIG_FILE

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Parse YAML and build command using Python, then execute
CMD=$(python3 << 'PYTHON_SCRIPT'
import yaml
import sys
import os

config_file = os.environ.get('CONFIG_FILE', 'config.yaml')

try:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading YAML: {e}", file=sys.stderr)
    sys.exit(1)

# Extract training config
train_cfg = config.get('training', {})

# Build command arguments
args = []

# Data arguments
data_cfg = train_cfg.get('data', {})
if 'toml_config_path' in data_cfg:
    args.append(f'data.kwargs.toml_config_path="{data_cfg["toml_config_path"]}"')
if 'num_workers' in data_cfg:
    args.append(f'data.kwargs.num_workers={data_cfg["num_workers"]}')
if 'batch_col' in data_cfg:
    args.append(f'data.kwargs.batch_col="{data_cfg["batch_col"]}"')
if 'pert_col' in data_cfg:
    args.append(f'data.kwargs.pert_col="{data_cfg["pert_col"]}"')
if 'cell_type_key' in data_cfg:
    args.append(f'data.kwargs.cell_type_key="{data_cfg["cell_type_key"]}"')
if 'control_pert' in data_cfg:
    args.append(f'data.kwargs.control_pert="{data_cfg["control_pert"]}"')
if data_cfg.get('perturbation_features_file'):
    args.append(f'data.kwargs.perturbation_features_file="{data_cfg["perturbation_features_file"]}"')
if 'embed_key' in data_cfg:
    args.append(f'data.kwargs.embed_key="{data_cfg["embed_key"]}"')


# Training arguments
train_params = train_cfg.get('training', {})
if 'batch_size' in train_params:
    args.append(f'training.batch_size={train_params["batch_size"]}')
if 'max_steps' in train_params:
    args.append(f'training.max_steps={train_params["max_steps"]}')
if 'ckpt_every_n_steps' in train_params:
    args.append(f'training.ckpt_every_n_steps={train_params["ckpt_every_n_steps"]}')

# Model
if 'model' in train_cfg:
    args.append(f'model={train_cfg["model"]}')

# Wandb
wandb_cfg = train_cfg.get('wandb', {})
if 'tags' in wandb_cfg:
    args.append(f'wandb.tags={wandb_cfg["tags"]}')
if 'project' in wandb_cfg:
    args.append(f'wandb.project={wandb_cfg["project"]}')
if 'entity' in wandb_cfg:
    args.append(f'wandb.entity={wandb_cfg["entity"]}')

# Output
output_cfg = train_cfg.get('output', {})
if 'output_dir' in output_cfg:
    args.append(f'output_dir="{output_cfg["output_dir"]}"')
if 'name' in output_cfg:
    args.append(f'name="{output_cfg["name"]}"')

# Extra args from environment (e.g. set by run_train_state.sh)
extra = os.environ.get('EXTRA_TRAIN_ARGS', '').strip()
if extra:
    args.extend(extra.split())

# Build and print command
cmd = 'uv run state tx train ' + ' '.join(args)
print(cmd)
PYTHON_SCRIPT
) || exit 1

# Execute the command
eval "$CMD"

