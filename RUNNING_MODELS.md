# Running STATE Models with YAML/TOML Configuration

For SLURM wrapper scripts under `scripts/hpc/` (`run_train_state.sh`, Apptainer helpers, etc.), see **[docs/HPC_SLURM_WRAPPERS.md](docs/HPC_SLURM_WRAPPERS.md)**.

This guide explains how to run STATE models using the YAML and TOML configuration format. The typical workflow consists of four main steps: preprocessing training data, preprocessing inference templates, training the model, and running in-silico perturbation predictions.

## Workflow Overview

The standard workflow for running STATE models follows these steps:

1. **Preprocess Training Data** (`00_preprocess_train.sh`) - Process raw training data
2. **Preprocess Inference Template** (e.g. `02a_preprocess_infer.sh` in this fork) - Create control template for inference
3. **Train Model** (`01_run_train_state_config.sh` or `run_train_state.sh`) - Train the STATE model
4. **Run In-Silico Perturbation** (`run_insilicoperturbation_state.sh`) - Generate perturbation predictions

## Step-by-Step Guide

### Step 1: Preprocess Training Data

The first step processes your raw training data to normalize, log-transform, and select highly variable genes.

**Script:** `00_preprocess_train.sh`

**Example:**
```bash
uv run state tx preprocess_train \
    --adata "/path/to/raw_data.h5ad" \
    --output /path/to/processed_training_data.h5ad \
    --num_hvgs 2000
```

**Parameters:**
- `--adata`: Path to input AnnData file (raw training data)
- `--output`: Path to output processed AnnData file
- `--num_hvgs`: Number of highly variable genes to select (default: 2000)

This command:
- Normalizes total counts per cell
- Applies log1p transformation
- Identifies highly variable genes
- Stores the HVG expression matrix in `.obsm['X_hvg']`

### Step 2: Preprocess Inference Template

Create a control template from your processed data that will be used for inference. This template contains control cells that serve as the baseline for perturbation predictions.

**Script:** Use any job script that calls `state tx preprocess_infer` (this repo includes `02a_preprocess_infer.sh` as a SLURM example with site-specific paths).

**Example:**
```bash
uv run state tx preprocess_infer \
  --adata /path/to/processed_training_data.h5ad \
  --output control_template.h5ad \
  --control-condition "PBS" \
  --pert-col "cytokine" \
  --seed 42
```

**Parameters:**
- `--adata`: Path to processed training data (from Step 1)
- `--output`: Path to output control template file
- `--control-condition`: Name of the control condition in your data (e.g., "PBS", "control", "DMSO")
- `--pert-col`: Column name in AnnData that contains perturbation labels
- `--seed`: Random seed for reproducibility

The control template preserves perturbation annotations but uses control cell expression, allowing the model to predict perturbation effects from control states.

### Step 3: Create Configuration Files

Before training, you need to create two configuration files:

#### A. TOML Configuration File

The TOML file defines your datasets, training splits, and evaluation strategy (zeroshot/fewshot).

**Example: `cytokine.toml`** (from `cytokine_perturbation/cytokine.toml`):
```toml
# Dataset paths - maps dataset names to their directories
[datasets]
cytokine = "/path/to/processed_training_data.h5ad"

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
cytokine = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]

# Fewshot specifications - explicit perturbation lists
[fewshot]
[fewshot."cytokine.B Intermediate/Memory"]
val = ["APRIL", "IFN-epsilon", "TL1A", "M-CSF", "TGF-beta1"]
test = ["HGF", "IL-35", "LT-alpha1-beta2", "FLT3L", "IL-23"]
```

**Example: `replogle.toml`** (from `replogle_run/replogle.toml`):
```toml
[datasets]
replogle = "/path/to/replogle_processed_data.h5ad"

[training]
replogle = "train"

[zeroshot]

[fewshot]
[fewshot."replogle.k562"]
val = ["SLC39A9"]
test = ["GAB2", "RPL3"]
```

**TOML Structure:**
- `[datasets]`: Maps dataset names to file paths
- `[training]`: Specifies which datasets participate in training
- `[zeroshot]`: Reserves entire cell types for validation/testing
- `[fewshot]`: Specifies perturbation-level splits within cell types

#### B. YAML Configuration File

The YAML file contains all training and inference parameters.

**Example: `config.yaml`** (from `cytokine_perturbation/config.yaml`):
```yaml
# Training Configuration
training:
  # Data configuration
  data:
    toml_config_path: "/path/to/cytokine.toml"
    num_workers: 8
    batch_col: "sample"
    pert_col: "cytokine"
    cell_type_key: "cell_type"
    control_pert: "PBS"
  
  # Training hyperparameters
  training:
    max_steps: 71000
    ckpt_every_n_steps: 10000
  
  # Model configuration
  model: "state_sm"
  
  # Weights & Biases configuration
  wandb:
    tags: "[experiment_name]"
    project: "your_project"
    entity: "your_entity"
  
  # Output configuration
  output:
    output_dir: "cytokine_perturbation"
    name: "experiment_name"

# Inference Configuration
inference:
  # Inference paths and settings
  output: "cytokine_perturbation/experiment_name/cytokine_prediction.h5ad"
  model_dir: "cytokine_perturbation/experiment_name"
  checkpoint: "final.ckpt"
  adata: "cytokine_perturbation/cytokine_control_template.h5ad"
  pert_col: "cytokine"
```

**Example: `config.yaml`** (from `replogle_run/config.yaml`):
```yaml
training:
  data:
    toml_config_path: "/path/to/replogle.toml"
    num_workers: 8
    batch_col: "batch"
    pert_col: "perturbation"
    cell_type_key: "cell_line"
    control_pert: "control"
  
  training:
    max_steps: 100000
    ckpt_every_n_steps: 10000
  
  model: "state_sm"
  
  wandb:
    tags: "[replogle_experiment]"
    project: "your_project"
    entity: "your_entity"
  
  output:
    output_dir: "replogle_run"
    name: "replogle_experiment"

inference:
  output: "replogle_run/replogle_prediction.h5ad"
  model_dir: "replogle_run/replogle_experiment"
  checkpoint: "last.ckpt"
  adata: "replogle_run/replogle_control_template.h5ad"
  pert_col: "perturbation"
```

**YAML Parameters:**

**Training Section:**
- `data.toml_config_path`: Path to your TOML configuration file
- `data.num_workers`: Number of data loading workers
- `data.batch_col`: Column name for batch information
- `data.pert_col`: Column name for perturbation labels
- `data.cell_type_key`: Column name for cell type information
- `data.control_pert`: Label for control/untreated condition
- `training.max_steps`: Maximum training steps
- `training.ckpt_every_n_steps`: Frequency of checkpoint saving
- `model`: Model architecture (e.g., "state_sm", "state_lg")
- `wandb.*`: Weights & Biases logging configuration
- `output.output_dir`: Directory for model outputs
- `output.name`: Experiment name (creates subdirectory)

**Inference Section:**
- `output`: Path to output prediction file
- `model_dir`: Directory containing trained model
- `checkpoint`: Checkpoint filename (e.g., "final.ckpt", "last.ckpt")
- `adata`: Path to control template (from Step 2)
- `pert_col`: Column name for perturbation labels (must match training)

### Step 4: Train the Model

Train the STATE model using your configuration files.

**Option A: Using the config script directly**
```bash
sbatch scripts/hpc/01_run_train_state_config.sh config.yaml
# or
bash scripts/hpc/01_run_train_state_config.sh config.yaml
```

**Option B: Using the master training script**
```bash
sbatch scripts/hpc/run_train_state.sh config.yaml
# or
bash scripts/hpc/run_train_state.sh config.yaml
```

The training script will:
1. Parse your YAML configuration
2. Extract training parameters
3. Build and execute the training command
4. Save checkpoints in `{output_dir}/{name}/checkpoints/`

**Checkpoints:**
- `last.ckpt`: Latest checkpoint
- `final.ckpt`: Final checkpoint (if training completes)
- Checkpoints are saved every `ckpt_every_n_steps`

### Step 5: Run In-Silico Perturbation

Generate perturbation predictions using your trained model.

**Script:** `run_insilicoperturbation_state.sh`

```bash
sbatch scripts/hpc/run_insilicoperturbation_state.sh config.yaml
# or
bash scripts/hpc/run_insilicoperturbation_state.sh config.yaml
```

This script:
1. Reads inference configuration from your YAML file
2. Loads the trained model checkpoint
3. Generates predictions for perturbations specified in the control template
4. Saves predictions to the specified output path

**Inference Configuration:**
The inference section in your YAML file should specify:
- `output`: Where to save predictions
- `model_dir`: Directory containing your trained model
- `checkpoint`: Which checkpoint to use (typically "final.ckpt" or "last.ckpt")
- `adata`: Path to control template from Step 2
- `pert_col`: Perturbation column name

## Complete Example Workflow

Here's a complete example using the cytokine perturbation dataset:

### 1. Preprocess Training Data
```bash
uv run state tx preprocess_train \
    --adata "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings_ensembl.h5ad" \
    --output /iridisfs/ddnb/Luke/data/parse-10M-90cytokines_processed_state.h5ad \
    --num_hvgs 2000
```

### 2. Preprocess Inference Template
```bash
uv run state tx preprocess_infer \
  --adata /iridisfs/ddnb/Luke/data/parse-10M-90cytokines_processed_state.h5ad \
  --output cytokine_perturbation/cytokine_control_template.h5ad \
  --control-condition "PBS" \
  --pert-col "cytokine" \
  --seed 42
```

### 3. Create TOML File (`cytokine_perturbation/cytokine.toml`)
```toml
[datasets]
cytokine = "/iridisfs/ddnb/Luke/data/parse-10M-90cytokines_processed_state.h5ad"

[training]
cytokine = "train"

[zeroshot]

[fewshot]
# Add your fewshot configurations here
```

### 4. Create YAML File (`cytokine_perturbation/config.yaml`)
```yaml
training:
  data:
    toml_config_path: "/iridisfs/ddnb/Luke/scFMs/state/cytokine_perturbation/cytokine.toml"
    num_workers: 8
    batch_col: "sample"
    pert_col: "cytokine"
    cell_type_key: "cell_type"
    control_pert: "PBS"
  training:
    max_steps: 71000
    ckpt_every_n_steps: 10000
  model: "state_sm"
  wandb:
    tags: "[experiment]"
    project: "vcc"
    entity: "arcinstitute"
  output:
    output_dir: "cytokine_perturbation"
    name: "my_experiment"

inference:
  output: "cytokine_perturbation/my_experiment/cytokine_prediction.h5ad"
  model_dir: "cytokine_perturbation/my_experiment"
  checkpoint: "final.ckpt"
  adata: "cytokine_perturbation/cytokine_control_template.h5ad"
  pert_col: "cytokine"
```

### 5. Train Model
```bash
sbatch scripts/hpc/run_train_state.sh cytokine_perturbation/config.yaml
```

### 6. Run Inference
```bash
sbatch scripts/hpc/run_insilicoperturbation_state.sh cytokine_perturbation/config.yaml
```

## Reference Examples

Two complete examples are available in this repository:

### 1. Cytokine Perturbation Example
- **Location:** `cytokine_perturbation/`
- **Config:** `cytokine_perturbation/config.yaml`
- **TOML:** `cytokine_perturbation/cytokine.toml`
- **Use case:** Predicting cellular responses to cytokine perturbations across multiple cell types

### 2. Replogle Example
- **Location:** `replogle_run/`
- **Config:** `replogle_run/config.yaml`
- **TOML:** `replogle_run/replogle.toml`
- **Use case:** Genetic perturbation prediction in K562 cells

## Tips and Best Practices

1. **Path Consistency**: Ensure paths in your TOML and YAML files are absolute or relative to where you run the scripts
2. **Control Condition**: Make sure your `control_pert` value exactly matches the control label in your data
3. **Column Names**: Verify that `pert_col`, `cell_type_key`, and `batch_col` match your AnnData column names
4. **Checkpoint Selection**: Use `final.ckpt` for completed training runs, `last.ckpt` for interrupted runs
5. **Resource Requirements**: Adjust SLURM parameters in the scripts based on your dataset size and available resources
6. **Experiment Naming**: Use descriptive names in `output.name` to track different experiments

## Troubleshooting

### Common Issues

1. **Config file not found**: Ensure you're running scripts from the correct directory or use absolute paths
2. **Control template missing**: Make sure Step 2 completed successfully before running inference
3. **Checkpoint not found**: Verify the `model_dir` and `checkpoint` paths in your YAML file
4. **Column name mismatches**: Check that all column names in your config match your AnnData object
5. **TOML parsing errors**: Validate your TOML syntax, especially in fewshot sections

### Getting Help

- Check the example configurations in `cytokine_perturbation/` and `replogle_run/`
- Review the main README.md for general STATE documentation
- Verify your data format matches the expected AnnData structure

