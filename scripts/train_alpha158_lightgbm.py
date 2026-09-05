#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM Alpha158 Training Pipeline for US Equities (Russell 1000)
=================================================================
Automates data initialization, feature computation, model training,
metric evaluation (IC, Rank IC, ICIR), artifact serialization, and
cross-sectional score generation using Microsoft Qlib.

Reference Architecture:
    examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainAlpha158LightGBM")

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qlib
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.model.trainer import task_train


def ensure_universe_file(qlib_dir: Path, market_name: str = "russell1000") -> Path:
    """Ensure the specified instrument universe file exists in Qlib provider."""
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst_file = inst_dir / f"{market_name}.txt"

    if inst_file.exists() and inst_file.stat().st_size > 0:
        return inst_file

    # Check repository data/instruments/russell1000.txt
    repo_inst = REPO_ROOT / "data" / "instruments" / f"{market_name}.txt"
    if repo_inst.exists() and repo_inst.stat().st_size > 0:
        logger.info(f"Copying universe file from {repo_inst} to {inst_file}...")
        inst_file.write_text(repo_inst.read_text(encoding="utf-8"), encoding="utf-8")
        return inst_file

    # Check fallback all.txt
    all_file = inst_dir / "all.txt"
    if all_file.exists() and all_file.stat().st_size > 0:
        logger.warning(f"Universe {inst_file} not found. Linking all.txt as {market_name}.txt fallback...")
        inst_file.write_text(all_file.read_text(encoding="utf-8"), encoding="utf-8")
        return inst_file

    # Create dynamic seed instrument file if missing
    logger.info(f"Generating default {market_name}.txt seed universe...")
    from scripts.get_russell1000_symbols import get_curated_russell1000_universe, write_qlib_instrument_file
    symbols = get_curated_russell1000_universe(seed_only=True)
    write_qlib_instrument_file(symbols, inst_file)
    return inst_file


def calculate_ic_metrics(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate daily Information Coefficient (IC) and Rank IC metrics
    between predicted score and actual forward return.
    """
    try:
        # Align prediction and label on (datetime, instrument)
        merged = pd.concat([pred_df.rename(columns={pred_df.columns[0]: "pred"}),
                            label_df.rename(columns={label_df.columns[0]: "label"})], axis=1).dropna()
        if merged.empty:
            return {"mean_ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0}

        daily_ics = []
        daily_rank_ics = []

        for date, group in merged.groupby(level=0):
            if len(group) >= 3:
                ic = group["pred"].corr(group["label"], method="pearson")
                rank_ic = group["pred"].corr(group["label"], method="spearman")
                if not np.isnan(ic):
                    daily_ics.append(ic)
                if not np.isnan(rank_ic):
                    daily_rank_ics.append(rank_ic)

        ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
        ic_std = float(np.std(daily_ics)) if daily_ics else 1.0
        rank_ic_mean = float(np.mean(daily_rank_ics)) if daily_rank_ics else 0.0
        rank_ic_std = float(np.std(daily_rank_ics)) if daily_rank_ics else 1.0

        icir = float(ic_mean / (ic_std + 1e-12))
        rank_icir = float(rank_ic_mean / (rank_ic_std + 1e-12))

        return {
            "mean_ic": round(ic_mean, 5),
            "rank_ic": round(rank_ic_mean, 5),
            "icir": round(icir, 4),
            "rank_icir": round(rank_icir, 4),
            "daily_observations": len(daily_ics),
        }
    except Exception as e:
        logger.warning(f"Failed calculating IC metrics: {e}")
        return {"mean_ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0}


def train_alpha158_model(
    config_path: Path,
    qlib_dir: Path,
    market: str = "russell1000",
    experiment_name: str = "lightgbm_alpha158_us_russell1000",
    model_output_dir: Optional[Path] = None,
    scores_output_dir: Optional[Path] = None,
    num_boost_round: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute full LightGBM Alpha158 model training and artifact serialization.
    """
    start_time = time.time()
    # Configure MLflow execution flags locally
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    logger.info(f"Loading configuration from: {config_path.resolve()}")
    yaml = YAML(typ="safe", pure=True)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    # Override qlib_init if needed
    qlib_dir_str = str(qlib_dir.expanduser().resolve())
    config["qlib_init"]["provider_uri"] = qlib_dir_str
    config["qlib_init"]["region"] = "us"
    config["market"] = market

    # Ensure universe file exists
    ensure_universe_file(qlib_dir, market)

    # Initialize Qlib
    logger.info(f"Initializing Qlib (provider_uri={qlib_dir_str}, region=us)...")
    exp_uri = "file:" + str(REPO_ROOT / "mlruns")
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = exp_uri
    qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    # Fast run override if requested
    if num_boost_round is not None:
        config["task"]["model"]["kwargs"]["num_boost_round"] = num_boost_round

    # Execute training via Qlib's task_train
    logger.info(f"Starting Qlib model training (experiment={experiment_name})...")
    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)

    # Retrieve trained model and test predictions
    trained_model = recorder.load_object("params.pkl")
    pred_df = recorder.load_object("pred.pkl")
    try:
        label_df = recorder.load_object("label.pkl")
    except Exception:
        label_df = None

    # Calculate Information Coefficient metrics
    ic_metrics = {}
    if pred_df is not None and label_df is not None:
        ic_metrics = calculate_ic_metrics(pred_df, label_df)
        logger.info(f"Validation IC Metrics: {ic_metrics}")

    # Resolve output directories
    if model_output_dir is None:
        model_output_dir = REPO_ROOT / "models" / "lightgbm"
    if scores_output_dir is None:
        scores_output_dir = REPO_ROOT / "output" / "scores"

    model_output_dir.mkdir(parents=True, exist_ok=True)
    scores_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save production model binary (.pkl)
    prod_model_pkl = model_output_dir / "alpha158_russell1000_latest.pkl"
    import pickle
    with open(prod_model_pkl, "wb") as f:
        pickle.dump(trained_model, f)
    logger.info(f"Saved production model pickle to: {prod_model_pkl.resolve()}")

    # 2. Save native LightGBM booster text (.txt) if available
    prod_model_txt = model_output_dir / "alpha158_russell1000_latest.txt"
    feature_names = []
    feature_importances = []
    if hasattr(trained_model, "model") and trained_model.model is not None:
        try:
            trained_model.model.save_model(str(prod_model_txt))
            logger.info(f"Saved native LightGBM booster text to: {prod_model_txt.resolve()}")
            raw_feature_names = trained_model.model.feature_name()
            importances = trained_model.model.feature_importance(importance_type="gain")
            num_trees = trained_model.model.num_trees()
            logger.info(f"Model Quality Check: num_trees={num_trees}")

            # Map to canonical Alpha158 factor names if generic Column_i returned
            from qlib.contrib.data.loader import Alpha158DL
            _, canonical_names = Alpha158DL.get_feature_config()

            feature_names = []
            for i, fn in enumerate(raw_feature_names):
                if fn.startswith("Column_"):
                    try:
                        col_idx = int(fn.split("_")[1])
                        if col_idx < len(canonical_names):
                            fn = canonical_names[col_idx]
                    except (ValueError, IndexError):
                        pass
                feature_names.append(fn)

            feature_importances = [
                {"feature": fn, "gain": float(gain)}
                for fn, gain in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            ]
        except Exception as e:
            logger.warning(f"Could not dump native booster text: {e}")

    # 3. Save model metadata (.json)
    date_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta_data = {
        "model_name": "LightGBM_Alpha158_Russell1000",
        "trained_at_utc": datetime.datetime.utcnow().isoformat(),
        "training_duration_seconds": round(time.time() - start_time, 2),
        "experiment_name": experiment_name,
        "recorder_id": recorder.id,
        "market": market,
        "provider_uri": qlib_dir_str,
        "segments": config["task"]["dataset"]["kwargs"]["segments"],
        "hyperparameters": config["task"]["model"]["kwargs"],
        "metrics": ic_metrics,
        "features_count": len(feature_names),
        "top_10_features": feature_importances[:10] if feature_importances else [],
    }
    prod_model_meta = model_output_dir / "alpha158_russell1000_latest_meta.json"
    with open(prod_model_meta, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4, default=str)
    logger.info(f"Saved model metadata to: {prod_model_meta.resolve()}")

    # 4. Save versioned checkpoint
    checkpoint_dir = model_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_pkl = checkpoint_dir / f"alpha158_russell1000_{date_tag}.pkl"
    with open(checkpoint_pkl, "wb") as f:
        pickle.dump(trained_model, f)

    # 5. Export out-of-sample cross-sectional score table
    scores_exported = False
    if pred_df is not None:
        try:
            scores_df = pred_df.copy()
            if isinstance(scores_df, pd.Series):
                scores_df = scores_df.to_frame("score")
            elif "score" not in scores_df.columns:
                scores_df.columns = ["score"]

            scores_df = scores_df.reset_index()
            # Standardize column names
            if "datetime" in scores_df.columns:
                scores_df.rename(columns={"datetime": "date"}, inplace=True)
            if "instrument" in scores_df.columns:
                scores_df.rename(columns={"instrument": "symbol"}, inplace=True)

            # Compute cross-sectional ranks and percentiles per date
            scores_df["rank"] = scores_df.groupby("date")["score"].rank(ascending=False, method="dense").astype(int)
            scores_df["percentile"] = (
                scores_df.groupby("date")["score"].rank(pct=True, ascending=True) * 100.0
            ).round(2)

            # Save Parquet & CSV
            parquet_path = scores_output_dir / "alpha158_russell1000_latest.parquet"
            csv_path = scores_output_dir / "alpha158_russell1000_latest.csv"
            scores_df.to_parquet(parquet_path, index=False)
            scores_df.to_csv(csv_path, index=False)
            logger.info(f"Exported {len(scores_df)} scores to {parquet_path.resolve()} and {csv_path.resolve()}")
            scores_exported = True
        except Exception as e:
            logger.warning(f"Failed exporting scores: {e}")

    # Summary Display
    print("\n" + "=" * 80)
    print(f"{'LIGHTGBM ALPHA158 US TRAINING SUMMARY':^80}")
    print("=" * 80)
    print(f"Status:             SUCCESS")
    print(f"Universe:           {market} (US Equities)")
    print(f"Experiment ID:      {recorder.experiment_id} | Run: {recorder.id}")
    print(f"Mean IC:            {ic_metrics.get('mean_ic', 'N/A')} (Rank IC: {ic_metrics.get('rank_ic', 'N/A')})")
    print(f"ICIR:               {ic_metrics.get('icir', 'N/A')} (Rank ICIR: {ic_metrics.get('rank_icir', 'N/A')})")
    print(f"Production Model:   {prod_model_pkl}")
    print(f"Model Metadata:     {prod_model_meta}")
    if scores_exported:
        print(f"Latest Scores:      {scores_output_dir / 'alpha158_russell1000_latest.parquet'}")
    print("=" * 80 + "\n")

    return {
        "status": "success",
        "model_path": str(prod_model_pkl),
        "metadata_path": str(prod_model_meta),
        "ic_metrics": ic_metrics,
        "recorder_id": recorder.id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train LightGBM on Alpha158 for US Russell 1000 equities."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_russell1000.yaml",
        help="Path to workflow YAML configuration.",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/us_data",
        help="Qlib data directory path.",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="russell1000",
        help="Market universe instrument file prefix.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="lightgbm_alpha158_us_russell1000",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--num_boost_round",
        type=int,
        default=None,
        help="Override number of boosting rounds for quick testing.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config_p = Path(args.config)
    if not config_p.is_absolute():
        config_p = REPO_ROOT / config_p
    config_p = config_p.resolve()
    qlib_p = Path(args.qlib_dir).expanduser().resolve()

    train_alpha158_model(
        config_path=config_p,
        qlib_dir=qlib_p,
        market=args.market,
        experiment_name=args.experiment_name,
        num_boost_round=args.num_boost_round,
    )


if __name__ == "__main__":
    main()
