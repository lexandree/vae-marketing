"""Main entry point for the VAE Marketing Impact Analysis pipeline."""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.beta_vae import beta_vae_loss
from src.models.factory import ModelFactory
from src.services.baseline import get_household_profile, vae_loss
from src.services.impact_analysis import (
    analyze_persistence,
    calculate_deviation,
    categorize_shift,
)
from src.services.reporting_baseline import generate_aggregate_report
from src.utils.metrics import setup_logger
from src.utils.seed import set_seed
from src.utils.wandb_logger import finish_logging, init_wandb, log_metrics, save_artifact

set_seed(42)
logger = setup_logger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: Dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace
) -> Tuple[float, float]:
    """Executes the training loop for the given model and data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss = float('inf')
    avg_kl = 0.0

    model.train()
    for epoch in range(args.epochs):
        total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
        current_beta = args.beta
        if args.arch == "beta_vae":
            current_beta = model.get_beta(epoch, args.anneal_end, args.beta)

        for batch_x, batch_t in dataloader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_t)

            if args.arch == "beta_vae":
                loss, mse, kl = beta_vae_loss(
                    recon_x, batch_x, mu, logvar, current_beta, use_gkl=args.gkl
                )
                total_mse += mse.item()
                total_kl += kl.item()
            else:
                loss = vae_loss(recon_x, batch_x, mu, logvar)
                total_mse += loss.item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            torch.save(model.state_dict(), run_dir / "best_model.pth")

        # LOGGING: respect verbosity
        should_log_wandb = args.wandb and (
            (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == args.epochs or args.verbosity > 1
        )
        
        if should_log_wandb:
            log_metrics({
                "epoch": epoch, "loss": avg_loss, "mse_loss": avg_mse,
                "kl_loss": avg_kl if args.arch == "beta_vae" else 0.0,
                "beta": current_beta if args.arch == "beta_vae" else 1.0
            }, step=epoch)

        should_log_console = (
            (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == args.epochs or args.verbosity > 1
        ) and args.verbosity > 0

        if should_log_console:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                f"MSE: {avg_mse:.4f} | KL: {avg_kl:.4f}"
            )

    return best_loss, avg_kl


def train_command(args: argparse.Namespace) -> None:
    """Train the VAE model (Baseline or Beta)."""
    if args.verbosity > 0:
        logger.info(f"Loading training data from {args.data}")
    
    train_df = pd.read_parquet(args.data)

    with open(args.vocab, 'r') as f:
        vocab = json.load(f)

    num_categories = len(vocab) * 2
    category_cols = [c for c in train_df.columns if c.endswith('_SPEND') or c.endswith('_QTY')]
    temporal_cols = [c for c in train_df.columns if c.startswith('TEMPORAL_')]

    run_id = args.run_id if args.run_id else f"{args.arch}-{uuid.uuid4().hex[:8]}"
    run_dir = Path("experiments") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id, "arch": args.arch, "latent_dim": args.latent_dim,
        "beta": args.beta, "anneal_epochs": args.anneal_end, "use_gkl": args.gkl,
        "num_categories": num_categories, "num_temporal_features": len(temporal_cols),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "vocabulary_path": str(args.vocab), "train_data_path": str(args.data)
    }

    if args.wandb:
        init_wandb("vae_marketing", run_id, config, verbosity=args.verbosity)

    model = ModelFactory.create_model(config).to(device)
    ModelFactory.save_config(config, run_dir)

    x_tensor = torch.tensor(train_df[category_cols].values, dtype=torch.float32).to(device)
    t_tensor = torch.tensor(train_df[temporal_cols].values, dtype=torch.float32).to(device)
    
    dataloader = DataLoader(
        TensorDataset(x_tensor, t_tensor), 
        batch_size=config["batch_size"], 
        shuffle=True
    )

    best_loss, last_kl = run_training_loop(model, dataloader, config, run_dir, args)

    ModelFactory.save_metrics({
        "mse_loss": best_loss, "kl_divergence": last_kl,
        "mig_score": 0.0, "sap_score": 0.0
    }, run_dir)

    if args.wandb:
        # Final artifact upload (once per run) if requested
        if args.upload_model and (run_dir / "best_model.pth").exists():
            save_artifact(run_dir / "best_model.pth", "best_model", "model")
        finish_logging(verbosity=args.verbosity)
        
    if args.verbosity > 0:
        logger.info(f"Training complete. Artifacts saved in {run_dir}")


def infer_command(args: argparse.Namespace) -> None:
    """Run impact analysis inference using trained model."""
    run_dir = Path("experiments") / args.run_id
    model = ModelFactory.load_model(run_dir).to(device)
    model.eval()

    target_df = pd.read_parquet(args.data)
    
    baseline_path = args.baseline
    if not baseline_path:
        with open(run_dir / "config.json", "r") as f:
            config = json.load(f)
            baseline_path = config.get("train_data_path")
            
    if not baseline_path or not Path(baseline_path).exists():
        raise FileNotFoundError("Baseline data not found. Use --baseline.")
        
    logger.info(f"Loading baseline from {baseline_path}")
    base_df = pd.read_parquet(baseline_path)

    baseline_profiles = {}
    valid_households = sorted(list(set(base_df["HOUSEHOLD_KEY"].unique()).intersection(
        set(target_df["HOUSEHOLD_KEY"].unique())
    )))
    
    if args.limit and len(valid_households) > args.limit:
        import random
        random.seed(42)
        valid_households = random.sample(valid_households, args.limit)
        
    logger.info(f"Analyzing impact for {len(valid_households)} households...")

    shift_results = []
    stimulus_end_day = (
        int(base_df["WINDOW_START_DAY"].max()) if "WINDOW_START_DAY" in base_df else 0
    )

    for h_id in valid_households:
        h_base = base_df[base_df["HOUSEHOLD_KEY"] == h_id]
        base_prof = get_household_profile(model, h_base)
        baseline_profiles[h_id] = base_prof

        h_post = target_df[target_df["HOUSEHOLD_KEY"] == h_id]
        dev = calculate_deviation(base_prof, h_post, model)
        # Collect full vector for reporting
        dev_vec = calculate_deviation(base_prof, h_post, model, return_vector=True)
        
        cat = categorize_shift(h_base, h_post, baseline_profile=base_prof, post_profile=(base_prof + dev_vec))
        pers = analyze_persistence(h_post, stimulus_end_day, model, base_prof, threshold=2.0)

        shift_results.append({
            "household_id": h_id, "stimulus_id": "VAL_PERIOD",
            "quantitative_magnitude": dev, 
            "latent_vector": dev_vec.tolist(),
            "qualitative_nature": cat,
            "persistence_duration_days": pers,
        })

    profiles_data = [
        {"household_id": k, "baseline_profile": v}
        for k, v in baseline_profiles.items()
    ]
    report = generate_aggregate_report(
        pd.DataFrame(profiles_data),
        pd.DataFrame(shift_results),
        target_df
    )

    report_path = run_dir / "inference_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, default=str)

    print("\n" + "=" * 50 + "\nPIPELINE INFERENCE SUMMARY\n" + "=" * 50)
    print(f"Total Households Analyzed: {report['total_households_analyzed']}")
    print(f"Average Latent Deviation: {report['average_magnitude']:.4f}")
    print(f"Average Persistence: {report['average_persistence_days']:.1f} days")
    print(f"Report saved to: {report_path}")
    print("=" * 50 + "\n")


def compare_command(args: argparse.Namespace) -> None:
    """Compare metrics across multiple Run-IDs."""
    results = []
    for run_id in args.run_ids:
        run_dir = Path("experiments") / run_id
        if (run_dir / "metrics.json").exists() and (run_dir / "config.json").exists():
            with open(run_dir / "metrics.json", "r") as f:
                m = json.load(f)
            with open(run_dir / "config.json", "r") as f:
                c = json.load(f)
            m.update({"run_id": run_id, "arch": c.get("arch", "baseline")})
            results.append(m)

    if not results:
        return logger.error("No valid runs found to compare.")

    df = pd.DataFrame(results)
    cols = ["run_id", "arch", "mse_loss", "kl_divergence", "mig_score", "sap_score"]
    print("\n" + "=" * 50 + "\nMODEL COMPARISON SUMMARY\n" + "=" * 50)
    print(df[[c for c in cols if c in df.columns]].to_markdown(index=False))
    print("=" * 50 + "\n")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="VAE Marketing Impact Analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train baseline or Beta VAE")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--vocab", type=Path, required=True)
    train_parser.add_argument(
        "--arch", type=str, default="baseline", choices=["baseline", "beta_vae"]
    )
    train_parser.add_argument("--run-id", type=str, default=None)
    train_parser.add_argument("--beta", type=float, default=1.0)
    train_parser.add_argument("--anneal-end", type=int, default=0)
    train_parser.add_argument("--latent-dim", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--wandb", action="store_true")
    train_parser.add_argument("--upload-model", action="store_true")
    train_parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2])
    train_parser.add_argument("--gkl", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Run inference on data")
    infer_parser.add_argument("--run-id", type=str, required=True)
    infer_parser.add_argument("--data", type=Path, required=True)
    infer_parser.add_argument("--baseline", type=Path, default=None)
    infer_parser.add_argument("--limit", type=int, default=None)

    compare_parser = subparsers.add_parser("compare", help="Compare across Run-IDs")
    compare_parser.add_argument("run_ids", nargs="+")

    args = parser.parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "compare":
        compare_command(args)


if __name__ == "__main__":
    main()
