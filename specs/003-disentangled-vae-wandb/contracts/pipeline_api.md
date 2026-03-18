# Pipeline API: Stage 003 (Experiment CLI)

## CLI Interface

The `main.py` script serves as the primary entry point for all model operations.

### Training Command
```bash
python main.py train [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--arch` | String | `baseline` | Architecture type: `baseline` or `beta_vae`. |
| `--run-id` | String | (Generated) | Unique identifier for the experiment. |
| `--beta` | Float | 1.0 | KL weight. Use > 1.0 for disentanglement. |
| `--anneal-end` | Int | 0 | Epoch where β reaches target value. |
| `--latent-dim` | Int | 16 | Size of latent space. |
| `--wandb` | Flag | False | Enable Weights & Biases logging. |
| `--gkl` | Flag | False | Enable experimental Generalized KL loss. |

### Inference & Reporting Command
```bash
python main.py infer --run-id [ID] [OPTIONS]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--run-id` | String | Yes | Loads model, config, and metrics from `experiments/[run_id]/`. |
| `--data` | Path | Yes | Path to `.parquet` data for analysis. |
| `--out` | Path | No | Output path for the generated report. |

## File Contracts

### Model Directory Structure
Each `run-id` must resolve to:
```text
experiments/[run_id]/
├── config.json    # Architecture & Hyperparameters
├── weights.pt     # PyTorch model weights
├── metrics.json   # Final evaluation scores
└── report.md      # Summary of analysis (if run)
```

### Config Schema (JSON)
```json
{
  "run_id": "vae-beta-001",
  "arch": "beta_vae",
  "latent_dim": 32,
  "beta": 4.0,
  "anneal_epochs": 20,
  "input_dim": 612
}
```
