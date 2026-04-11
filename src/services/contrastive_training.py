"""Training service for the Contrastive VAE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.splits import filter_households, household_ids_for_role, load_household_splits
from src.models.contrastive_vae import ContrastiveVAE, contrastive_vae_loss
from src.models.factory import ModelFactory
from src.services.latent_validation import validate_latent_runs
from src.utils.metrics import setup_logger
from src.utils.wandb_logger import finish_logging, init_wandb, log_metrics, save_artifact


logger = setup_logger(__name__)


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split prepared frame columns into commodity and temporal features."""
    category_cols = [c for c in df.columns if c.endswith("_SPEND") or c.endswith("_QTY")]
    temporal_cols = [c for c in df.columns if c.startswith("TEMPORAL_")]
    if not category_cols:
        raise ValueError("Contrastive training data requires commodity spend/qty columns.")
    return category_cols, temporal_cols


def _build_loader(
    df: pd.DataFrame,
    category_cols: list[str],
    temporal_cols: list[str],
    batch_size: int,
) -> DataLoader:
    """Build a dataloader for one contrastive side."""
    x_tensor = torch.tensor(df[category_cols].values, dtype=torch.float32)
    t_tensor = torch.tensor(df[temporal_cols].values, dtype=torch.float32)
    return DataLoader(TensorDataset(x_tensor, t_tensor), batch_size=batch_size, shuffle=True, drop_last=True)


def train_contrastive_vae(*, args: Any) -> None:
    """Train a Contrastive VAE on target/background weekly windows."""
    target_df = pd.read_parquet(args.target_data)
    background_df = pd.read_parquet(args.background_data)
    category_cols, temporal_cols = _feature_columns(target_df)
    run_id = args.run_id if getattr(args, "run_id", None) else f"contrastive-vae-{uuid.uuid4().hex[:8]}"
    run_dir = Path("experiments") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "arch": "contrastive_vae",
        "latent_dim": args.shared_dim + args.salient_dim,
        "shared_dim": args.shared_dim,
        "salient_dim": args.salient_dim,
        "num_categories": len(category_cols),
        "num_temporal_features": len(temporal_cols),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "beta_shared": args.beta_shared,
        "beta_salient": args.beta_salient,
        "salient_background_weight": args.salient_background_weight,
        "target_data_path": str(args.target_data),
        "background_data_path": str(args.background_data),
    }
    if getattr(args, "latent_split", None):
        config["latent_split"] = args.latent_split
    if getattr(args, "eval_analysis_data", None):
        config["eval_analysis_data"] = str(args.eval_analysis_data)
    if getattr(args, "eval_attributes", None):
        config["eval_attributes"] = str(args.eval_attributes)
    ModelFactory.save_config(config, run_dir)

    if getattr(args, "wandb", False):
        init_wandb("vae_marketing", run_id, config, verbosity=args.verbosity)

    model = ContrastiveVAE(
        shared_dim=args.shared_dim,
        salient_dim=args.salient_dim,
        num_categories=len(category_cols),
        num_temporal_features=len(temporal_cols),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    target_loader = _build_loader(target_df, category_cols, temporal_cols, args.batch_size)
    background_loader = _build_loader(background_df, category_cols, temporal_cols, args.batch_size)

    best_loss = float("inf")
    final_metrics: dict[str, float] = {}
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        metric_sums = {
            "recon_target": 0.0,
            "recon_background": 0.0,
            "kl_target_shared": 0.0,
            "kl_target_salient": 0.0,
            "kl_background_shared": 0.0,
            "salient_background_penalty": 0.0,
        }
        steps = 0
        for (target_x, target_t), (background_x, background_t) in zip(target_loader, background_loader):
            optimizer.zero_grad()
            target_outputs = model.forward_target(target_x, target_t)
            background_outputs = model.forward_background(background_x, background_t)
            loss, metrics = contrastive_vae_loss(
                target_recon=target_outputs[0],
                target_x=target_x,
                target_shared_mu=target_outputs[1],
                target_shared_logvar=target_outputs[2],
                target_salient_mu=target_outputs[3],
                target_salient_logvar=target_outputs[4],
                background_recon=background_outputs[0],
                background_x=background_x,
                background_shared_mu=background_outputs[1],
                background_shared_logvar=background_outputs[2],
                background_salient_mu=background_outputs[3],
                background_salient_logvar=background_outputs[4],
                beta_shared=args.beta_shared,
                beta_salient=args.beta_salient,
                salient_background_weight=args.salient_background_weight,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            for key, value in metrics.items():
                metric_sums[key] += value
            steps += 1

        if steps == 0:
            raise ValueError("Contrastive training produced zero optimization steps; increase data size or lower batch size.")

        avg_loss = epoch_loss / steps
        averaged = {key: value / steps for key, value in metric_sums.items()}
        final_metrics = {"loss": avg_loss, **averaged}
        if getattr(args, "wandb", False):
            log_metrics(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    **averaged,
                },
                step=epoch,
            )
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), run_dir / "best_model.pth")
        if args.verbosity > 0 and ((epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == args.epochs):
            logger.info(
                "Epoch %s/%s | Loss: %.4f | Target Recon: %.4f | Background Recon: %.4f | Salient Bg: %.4f",
                epoch + 1,
                args.epochs,
                avg_loss,
                averaged["recon_target"],
                averaged["recon_background"],
                averaged["salient_background_penalty"],
            )

    ModelFactory.save_metrics(final_metrics, run_dir)
    with open(run_dir / "contrastive_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    if getattr(args, "eval_analysis_data", None) and getattr(args, "eval_attributes", None):
        eval_analysis_df = pd.read_parquet(args.eval_analysis_data)
        eval_attributes_df = pd.read_parquet(args.eval_attributes)
        if getattr(args, "eval_household_splits", None) is not None:
            splits = load_household_splits(args.eval_household_splits)
            allowed = household_ids_for_role(splits, getattr(args, "eval_split_role", "eval"))
            eval_analysis_df = filter_households(eval_analysis_df, allowed)
            eval_attributes_df = filter_households(eval_attributes_df, allowed)

        metrics_rows, mapping_rows, stability_rows = validate_latent_runs(
            analysis_df=eval_analysis_df,
            attributes_df=eval_attributes_df,
            run_dirs=[run_dir],
            mig_method=getattr(args, "eval_mig_method", "binned"),
            mig_bins=getattr(args, "eval_mig_bins", 32),
            mig_binning=getattr(args, "eval_mig_binning", "quantile"),
            sap_method=getattr(args, "eval_sap_method", "vectorized"),
            latent_mode=getattr(args, "eval_latent_mode", "salient"),
        )
        eval_metrics = dict(metrics_rows[0]) if metrics_rows else {}
        best_total_spend = 0.0
        best_category_diversity = 0.0
        validated_count = 0
        for row in mapping_rows:
            if row.get("mapping_decision") == "validated":
                validated_count += 1
            if row.get("candidate_attribute") == "total_spend":
                best_total_spend = max(best_total_spend, float(row.get("association_strength", 0.0)))
            if row.get("candidate_attribute") == "category_diversity":
                best_category_diversity = max(
                    best_category_diversity,
                    float(row.get("association_strength", 0.0)),
                )

        eval_summary = {
            **eval_metrics,
            "validated_count": validated_count,
            "best_total_spend_association": best_total_spend,
            "best_category_diversity_association": best_category_diversity,
            "salient_eval_score": (
                0.6 * float(eval_metrics.get("mig_score", 0.0))
                + 0.4 * float(eval_metrics.get("sap_score", 0.0))
            ),
        }
        with open(run_dir / "contrastive_eval_metrics.json", "w") as f:
            json.dump(eval_summary, f, indent=2)
        if getattr(args, "wandb", False):
            log_metrics(eval_summary)
            if (run_dir / "best_model.pth").exists():
                save_artifact(run_dir / "best_model.pth", f"{run_id}-best_model", "model")
        if args.verbosity > 0:
            logger.info(
                "Held-out %s evaluation | MIG: %.4f | SAP: %.4f | score: %.4f",
                getattr(args, "eval_latent_mode", "salient"),
                float(eval_summary.get("mig_score", 0.0)),
                float(eval_summary.get("sap_score", 0.0)),
                float(eval_summary.get("salient_eval_score", 0.0)),
            )

    if getattr(args, "wandb", False):
        finish_logging(verbosity=args.verbosity)
    if args.verbosity > 0:
        logger.info("Contrastive training complete. Artifacts saved in %s", run_dir)
