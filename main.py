"""Main entry point for the VAE Marketing Impact Analysis pipeline."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.vae import build_vae_model
from src.services.baseline import get_household_profile, train_baseline_vae
from src.services.impact_analysis import (
    analyze_persistence,
    calculate_deviation,
    categorize_shift,
)
from src.services.reporting import generate_aggregate_report
from src.utils.metrics import setup_logger
from src.utils.seed import set_seed

set_seed(42)
logger = setup_logger(__name__)


def train_command(args: argparse.Namespace) -> None:
    """Train the VAE baseline model."""
    logger.info(f"Loading training data from {args.data}")
    train_df = pd.read_parquet(args.data)
    
    logger.info(f"Loading vocabulary from {args.vocab}")
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
        
    # The true shape is defined by the vocabulary (which acts as a contract)
    # The vocabulary extracted by prepare.py already includes 'UNKNOWN'
    num_categories = len(vocab) * 2
    num_temporal = 6
    
    # We enforce that the dataset strictly matches the vocabulary contract
    category_cols = [c for c in train_df.columns if c.endswith('_SPEND') or c.endswith('_QTY')]
    if len(category_cols) != num_categories:
        logger.error(f"Schema mismatch: vocabulary expects {num_categories} category features, "
                     f"but data contains {len(category_cols)}. Ensure data was prepared with the same vocabulary.")
        raise ValueError("Data schema does not match vocabulary contract.")
        
    logger.info(f"Initializing VAE Model (categories={num_categories}, temporal={num_temporal})...")
    model = build_vae_model(
        latent_dim=16, 
        num_categories=num_categories, 
        num_temporal_features=num_temporal
    )
    
    logger.info("Training VAE on baseline data...")
    trained_model = train_baseline_vae(
        data=train_df,
        model=model,
        epochs=args.epochs,
        batch_size=64,
        learning_rate=1e-3
    )
    
    logger.info(f"Saving trained model to {args.model_out}")
    # Ensure directory exists
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), args.model_out)


def infer_command(args: argparse.Namespace) -> None:
    """Run impact analysis inference using trained model."""
    logger.info(f"Loading post-stimulus (inference) data from {args.data}")
    val_df = pd.read_parquet(args.data)
    
    logger.info(f"Loading vocabulary from {args.vocab}")
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
        
    num_categories = len(vocab) * 2
    num_temporal = 6
    
    # We enforce that the dataset strictly matches the vocabulary contract
    category_cols = [c for c in val_df.columns if c.endswith('_SPEND') or c.endswith('_QTY')]
    if len(category_cols) != num_categories:
        logger.error(f"Schema mismatch: vocabulary expects {num_categories} category features, "
                     f"but inference data contains {len(category_cols)}.")
        raise ValueError("Inference data schema does not match vocabulary contract.")
        
    logger.info("Initializing VAE Model...")
    model = build_vae_model(
        latent_dim=16, 
        num_categories=num_categories, 
        num_temporal_features=num_temporal
    )
    
    logger.info(f"Loading model weights from {args.model_in}")
    model.load_state_dict(torch.load(args.model_in))
    model.eval()

    logger.info("Loading baseline reference profiles from training data...")
    # To calculate deviation, we need the baseline profile for each household.
    # In a real scenario, this would be pre-calculated and stored.
    # For now, we load train.parquet to compute the baseline profile on the fly.
    train_df = pd.read_parquet(args.train_data)
    baseline_profiles = {}
    
    # We only analyze households that exist in both train and inference data
    valid_households = set(train_df["HOUSEHOLD_KEY"].unique()).intersection(set(val_df["HOUSEHOLD_KEY"].unique()))
    
    logger.info(f"Analyzing impact for {len(valid_households)} households...")
    
    shift_results = []
    
    for h_id in valid_households:
        h_base = train_df[train_df["HOUSEHOLD_KEY"] == h_id]
        base_prof = get_household_profile(model, h_base)
        baseline_profiles[h_id] = base_prof
        
        h_post = val_df[val_df["HOUSEHOLD_KEY"] == h_id]
        
        dev = calculate_deviation(base_prof, h_post, model)
        cat = categorize_shift(h_base, h_post)
        
        # Determine stimulus end day (simplified assumption: max day in train)
        stimulus_end_day = int(train_df["WINDOW_START_DAY"].max())
        pers = analyze_persistence(h_post, stimulus_end_day, model, base_prof, threshold=2.0)

        shift_results.append(
            {
                "household_id": h_id,
                "stimulus_id": "VAL_PERIOD",
                "quantitative_magnitude": dev,
                "qualitative_nature": cat,
                "persistence_duration_days": pers,
            }
        )

    shifts_df = pd.DataFrame(shift_results)
    
    # Generate reporting format
    logger.info("Generating aggregate impact report...")
    
    # Mocking profiles_df for reporting compatibility
    profiles_df = pd.DataFrame([
        {"household_id": k, "baseline_profile": v} 
        for k, v in baseline_profiles.items()
    ])
    
    report = generate_aggregate_report(profiles_df, shifts_df, val_df)

    # Output results
    print("\n" + "=" * 50)
    print("PIPELINE INFERENCE SUMMARY")
    print("=" * 50)
    print(f"Total Households Analyzed: {report['total_households_analyzed']}")
    print(f"Average Latent Deviation: {report['average_magnitude']:.4f}")
    print(f"Average Persistence: {report['average_persistence_days']:.1f} days")
    print("\nSegment Distribution:")
    for cluster, count in report["segments"].items():
        print(f"  Cluster {cluster}: {count} households")
    print("\nTop Sensitive Categories (Placeholder until Phase 4):")
    for cat in report["top_sensitive_categories"]:
        print(f"  {cat['product_category']}: Score {cat['sensitivity_score']:.2f}")
    print("=" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE Marketing Impact Analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train baseline VAE")
    train_parser.add_argument("--data", type=Path, required=True, help="Path to train.parquet")
    train_parser.add_argument("--vocab", type=Path, required=True, help="Path to vocabulary.json")
    train_parser.add_argument("--model-out", type=Path, required=True, help="Path to save model.pt")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")

    # Infer subparser
    infer_parser = subparsers.add_parser("infer", help="Run inference on validation/test data")
    infer_parser.add_argument("--data", type=Path, required=True, help="Path to val.parquet or test.parquet")
    infer_parser.add_argument("--train-data", type=Path, required=True, help="Path to train.parquet for baseline references")
    infer_parser.add_argument("--vocab", type=Path, required=True, help="Path to vocabulary.json")
    infer_parser.add_argument("--model-in", type=Path, required=True, help="Path to load model.pt")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)


if __name__ == "__main__":
    main()
