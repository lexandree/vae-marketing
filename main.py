"""Main entry point for the VAE Marketing Impact Analysis pipeline."""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.campaign_dataset import (
    build_campaign_analysis_dataset,
    write_validation_artifacts,
)
from src.data.dataset import load_validation_source
from src.models.beta_vae import beta_vae_loss
from src.models.factory import ModelFactory
from src.services.baseline import get_household_profile, vae_loss
from src.services.campaign_latent_bridge import build_campaign_latent_bridge
from src.services.campaign_sensitivity import run_campaign_sensitivity
from src.services.campaign_validation import validate_campaign_effects
from src.services.impact_analysis import (
    analyze_persistence,
    calculate_deviation,
    categorize_shift,
)
from src.services.latent_validation import validate_latent_factors
from src.services.reporting_baseline import generate_aggregate_report
from src.services.validation_reporting import generate_validation_report
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


def build_validation_data_command(args: argparse.Namespace) -> None:
    """Build the campaign validation dataset."""
    transactions = load_validation_source(args.transactions, "transactions")
    products = load_validation_source(args.products, "products")
    campaign_table = load_validation_source(args.campaign_table, "campaign_table")
    campaign_desc = load_validation_source(args.campaign_desc, "campaign_desc")
    coupon = load_validation_source(args.coupon, "coupon")
    coupon_redempt = load_validation_source(args.coupon_redempt, "coupon_redempt")
    demographics = (
        load_validation_source(args.demographics, "demographics")
        if args.demographics is not None
        else None
    )
    causal_data = (
        load_validation_source(args.causal_data, "causal_data")
        if args.causal_data is not None
        else None
    )

    analysis_df, comparison_df, attributes_df, summary = build_campaign_analysis_dataset(
        transactions=transactions,
        products=products,
        campaign_table=campaign_table,
        campaign_desc=campaign_desc,
        coupon=coupon,
        coupon_redempt=coupon_redempt,
        demographics=demographics,
        causal_data=causal_data,
        campaign_ids=args.campaign_ids,
        pre_weeks=args.pre_weeks,
        post_weeks=args.post_weeks,
    )
    write_validation_artifacts(
        args.output_dir,
        analysis_df,
        comparison_df,
        attributes_df,
        summary,
    )

    print("\n" + "=" * 50 + "\nVALIDATION DATASET SUMMARY\n" + "=" * 50)
    print(f"Campaigns: {summary['selected_campaigns']}")
    print(f"Analysis records: {summary['analysis_records']}")
    print(f"Treated records: {summary['treated_records']}")
    print(f"Comparison records: {summary['comparison_records']}")
    print(f"Validation attribute rows: {summary['validation_attribute_rows']}")
    print(f"Artifacts saved to: {args.output_dir}")
    print("=" * 50 + "\n")


def validate_campaigns_command(args: argparse.Namespace) -> None:
    """Run quasi-causal campaign validation."""
    validate_campaign_effects(args=args)


def analyze_campaign_sensitivity_command(args: argparse.Namespace) -> None:
    """Run reproducible sensitivity analysis for campaign validation."""
    run_campaign_sensitivity(args=args)


def validate_latents_command(args: argparse.Namespace) -> None:
    """Run latent-factor validation."""
    validate_latent_factors(args=args)


def generate_validation_report_command(args: argparse.Namespace) -> None:
    """Generate the final validation report."""
    generate_validation_report(args=args)


def build_campaign_latent_bridge_command(args: argparse.Namespace) -> None:
    """Connect campaign shifts to validated latent mappings."""
    build_campaign_latent_bridge(args=args)


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

    build_validation_parser = subparsers.add_parser(
        "build-validation-data",
        help="Build campaign-linked validation datasets",
    )
    build_validation_parser.add_argument("--transactions", type=Path, required=True)
    build_validation_parser.add_argument("--products", type=Path, required=True)
    build_validation_parser.add_argument("--campaign-table", type=Path, required=True)
    build_validation_parser.add_argument("--campaign-desc", type=Path, required=True)
    build_validation_parser.add_argument("--coupon", type=Path, required=True)
    build_validation_parser.add_argument("--coupon-redempt", type=Path, required=True)
    build_validation_parser.add_argument("--demographics", type=Path, default=None)
    build_validation_parser.add_argument("--causal-data", type=Path, default=None)
    build_validation_parser.add_argument("--campaign-ids", type=int, nargs="+", required=True)
    build_validation_parser.add_argument("--output-dir", type=Path, required=True)
    build_validation_parser.add_argument("--pre-weeks", type=int, default=8)
    build_validation_parser.add_argument("--post-weeks", type=int, default=8)
    build_validation_parser.add_argument("--seed", type=int, default=42)

    validate_campaigns_parser = subparsers.add_parser(
        "validate-campaigns",
        help="Validate campaign effects with quasi-causal diagnostics",
    )
    validate_campaigns_parser.add_argument("--analysis-data", type=Path, required=True)
    validate_campaigns_parser.add_argument("--campaign-ids", type=int, nargs="+", required=True)
    validate_campaigns_parser.add_argument("--method", type=str, required=True)
    validate_campaigns_parser.add_argument("--output-dir", type=Path, required=True)
    validate_campaigns_parser.add_argument("--outcomes", nargs="*", default=None)
    validate_campaigns_parser.add_argument("--min-treated", type=int, default=30)
    validate_campaigns_parser.add_argument("--min-comparison", type=int, default=30)
    validate_campaigns_parser.add_argument(
        "--matching-method",
        type=str,
        choices=["none", "propensity"],
        default="none",
    )
    validate_campaigns_parser.add_argument("--propensity-caliper", type=float, default=0.02)
    validate_campaigns_parser.add_argument("--seed", type=int, default=42)

    sensitivity_parser = subparsers.add_parser(
        "analyze-campaign-sensitivity",
        help="Run a sensitivity grid for campaign validation",
    )
    sensitivity_parser.add_argument("--transactions", type=Path, required=True)
    sensitivity_parser.add_argument("--products", type=Path, required=True)
    sensitivity_parser.add_argument("--campaign-table", type=Path, required=True)
    sensitivity_parser.add_argument("--campaign-desc", type=Path, required=True)
    sensitivity_parser.add_argument("--coupon", type=Path, required=True)
    sensitivity_parser.add_argument("--coupon-redempt", type=Path, required=True)
    sensitivity_parser.add_argument("--demographics", type=Path, default=None)
    sensitivity_parser.add_argument("--campaign-ids", type=int, nargs="+", required=True)
    sensitivity_parser.add_argument("--weeks-grid", type=int, nargs="+", required=True)
    sensitivity_parser.add_argument("--output-dir", type=Path, required=True)
    sensitivity_parser.add_argument("--outcomes", nargs="*", default=None)
    sensitivity_parser.add_argument("--matching-methods", nargs="*", default=["propensity"])
    sensitivity_parser.add_argument("--propensity-calipers", type=float, nargs="*", default=[0.02])
    sensitivity_parser.add_argument("--min-treated", type=int, default=30)
    sensitivity_parser.add_argument("--min-comparison", type=int, default=30)

    validate_latents_parser = subparsers.add_parser(
        "validate-latents",
        help="Validate latent-factor semantics",
    )
    validate_latents_parser.add_argument("--analysis-data", type=Path, required=True)
    validate_latents_parser.add_argument("--attributes", type=Path, required=True)
    validate_latents_parser.add_argument("--run-ids", nargs="+", required=True)
    validate_latents_parser.add_argument("--output-dir", type=Path, required=True)
    validate_latents_parser.add_argument("--model-types", nargs="*", default=None)
    validate_latents_parser.add_argument("--holdout-split", type=str, default="validation")
    validate_latents_parser.add_argument("--seeds", type=int, nargs="*", default=None)
    validate_latents_parser.add_argument("--top-k-attributes", type=int, default=5)
    validate_latents_parser.add_argument(
        "--mig-method",
        type=str,
        choices=["sklearn", "binned"],
        default="sklearn",
    )
    validate_latents_parser.add_argument("--mig-bins", type=int, default=16)
    validate_latents_parser.add_argument(
        "--mig-binning",
        type=str,
        choices=["quantile", "uniform"],
        default="quantile",
    )
    validate_latents_parser.add_argument(
        "--sap-method",
        type=str,
        choices=["sklearn", "vectorized"],
        default="sklearn",
    )

    latent_bridge_parser = subparsers.add_parser(
        "build-campaign-latent-bridge",
        help="Connect campaign attribute shifts to validated latent mappings",
    )
    latent_bridge_parser.add_argument("--campaign-results", type=Path, required=True)
    latent_bridge_parser.add_argument("--factor-mappings", type=Path, required=True)
    latent_bridge_parser.add_argument("--attributes", type=Path, required=True)
    latent_bridge_parser.add_argument("--output-dir", type=Path, required=True)
    latent_bridge_parser.add_argument("--top-k-attributes", type=int, default=5)

    validation_report_parser = subparsers.add_parser(
        "generate-validation-report",
        help="Generate the validation research report",
    )
    validation_report_parser.add_argument("--campaign-results", type=Path, required=True)
    validation_report_parser.add_argument("--campaign-diagnostics", type=Path, required=True)
    validation_report_parser.add_argument("--latent-results", type=Path, required=True)
    validation_report_parser.add_argument("--latent-metrics", type=Path, required=True)
    validation_report_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "compare":
        compare_command(args)
    elif args.command == "build-validation-data":
        build_validation_data_command(args)
    elif args.command == "validate-campaigns":
        validate_campaigns_command(args)
    elif args.command == "analyze-campaign-sensitivity":
        analyze_campaign_sensitivity_command(args)
    elif args.command == "validate-latents":
        validate_latents_command(args)
    elif args.command == "build-campaign-latent-bridge":
        build_campaign_latent_bridge_command(args)
    elif args.command == "generate-validation-report":
        generate_validation_report_command(args)


if __name__ == "__main__":
    main()
