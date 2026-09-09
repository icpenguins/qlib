#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib GBDT Multi-Model Ensemble & Alpha Blending Pipeline
========================================================
Trains LightGBM, XGBoost, and CatBoost on a unified Alpha158 dataset,
extracts out-of-sample prediction signals, standardizes them via
cross-sectional Z-score normalization, and generates a blended alpha signal.

This script is a thin CLI/orchestrator over ``scripts/ensemble_lib.py`` (shared
dataset/model/backtest plumbing) and ``scripts/diagnose_signal.py`` (the pre-training
diagnostic gate; see ``.team-code/plans/pretraining-diagnostic-gate.md``). See that plan
for the full rationale behind ``--diagnose``, ``--allow_undertrained``, and
``--label_variant``.

Usage:
------
1. Install dependencies:
   pip install xgboost catboost

2. Run ensemble training on S&P 500 (real point-in-time membership; see audit_universe_bias()
   and scripts/data_collector/us_index/collector.py -- 'russell1000' has no free, non-fabricated
   point-in-time source and will be rejected by default, see --allow_survivorship_bias):
   python scripts/train_ensemble.py

3. Quick test run on 5 symbols:
   python scripts/train_ensemble.py --instruments AAPL,MSFT,NVDA,SMH,SPY --num_boost_round 30 --save_predictions

4. Run the pre-training diagnostic gate (gate items #1-#7, #10 -- see the plan) alongside a run:
   python scripts/train_ensemble.py --diagnose

5. Run the paired label/execution variant (gate item #10) with its own full diagnostic suite:
   python scripts/train_ensemble.py --diagnose --label_variant open_shift --output_dir models/gbdt_ensemble_openshift
"""

import os
import sys
import json
import random
import datetime
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Ensure this script's own directory is importable regardless of invocation cwd, so
# `import ensemble_lib` / `import diagnose_signal` resolve as sibling modules.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Ensure MLflow allows file-store on local filesystem
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import qlib
from qlib.data.dataset.handler import DataHandlerLP

from ensemble_lib import (
    DEFAULT_LEARNING_RATE,
    assert_sufficient_training_budget,
    audit_universe_bias,
    blend_predictions,
    build_and_train_models,
    build_dataset,
    calc_ic_metrics,
    find_available_benchmark,
    parse_instruments,
    run_portfolio_backtest,
    _json_safe,
)
import diagnose_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EnsembleAlpha")


# Gate item #10: paired label/execution variant. `label_config=None` means "use Alpha158's own
# get_label_config() default unchanged" -- NOT a new Alpha158 subclass, just a `label=` kwarg
# override applied inside ensemble_lib.build_dataset() at its Alpha158(...) call site. See the
# plan's Section 4/7 for why `Ref($open,-1)/$close - 1` was considered and rejected (look-ahead:
# with TopkDropoutStrategy's hardcoded shift=1, the earliest transactable price is open(s+1), so a
# label anchored at close(s) is not realizable) and why TopkDropoutStrategy's shift=1
# (qlib/contrib/strategy/signal_strategy.py:141-142, :351-352) is a hardcoded literal inside
# generate_trade_decision, not a constructor kwarg, and is therefore NOT exposed as a variant knob.
LABEL_VARIANTS = {
    "default": {
        "label_config": None,
        "deal_price": "close",
    },
    "open_shift": {
        "label_config": (["Ref($open,-2)/Ref($open,-1) - 1"], ["LABEL0"]),
        "deal_price": "open",
    },
}

TABLE_WIDTH = 118  # must stay >= the longest formatted data row (verified below); recheck if columns change


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LightGBM, XGBoost, and CatBoost ensemble in Qlib.")
    parser.add_argument("--data_dir", type=str, default="D:\\trading\\qlib\\qlib_data", help="Qlib data directory.")
    parser.add_argument(
        "--market",
        type=str,
        default="sp500",
        help=(
            "Market universe (e.g. 'sp500' or ticker list). Default changed from 'russell1000' to "
            "'sp500': the bundled russell1000.txt has no real historical add/drop/delisting dates "
            "(see audit_universe_bias) and there is no free point-in-time Russell 1000 source; "
            "'sp500' is built by scripts/data_collector/us_index/collector.py from Wikipedia's "
            "genuinely point-in-time S&P 500 membership history."
        ),
    )
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark symbol.")
    parser.add_argument(
        "--benchmark_candidates",
        type=str,
        default="SPY,^GSPC,VOO,IVV,QQQ",
        help=(
            "Comma-separated fallback symbols to search (in order) if --benchmark has no usable "
            "$close data in --data_dir. Checked with qlib.data.D.features against the actual "
            "provider before training starts, so a bad/missing benchmark (e.g. SPY absent from a "
            "constituents-only universe download, since the ETF isn't itself an index constituent) "
            "fails fast with a clear message instead of deep inside the backtest step after a full "
            "training run has already completed. The first candidate with real data wins; pass a "
            "single known-good symbol here (or in --benchmark) to skip searching."
        ),
    )
    parser.add_argument("--train_start", type=str, default="2020-01-01", help="Train start date.")
    parser.add_argument("--train_end", type=str, default="2023-12-31", help="Train end date.")
    parser.add_argument("--valid_start", type=str, default="2024-01-01", help="Validation start date.")
    parser.add_argument("--valid_end", type=str, default="2024-12-31", help="Validation end date.")
    parser.add_argument("--test_start", type=str, default="2025-01-01", help="Test start date.")
    parser.add_argument("--test_end", type=str, default="2026-09-04", help="Test end date.")
    parser.add_argument("--models", type=str, default="lgb,xgb,cat", help="Comma-separated models: 'lgb,xgb,cat'.")
    parser.add_argument("--num_boost_round", type=int, default=300, help="Boosting rounds per model.")
    parser.add_argument("--num_threads", type=int, default=16, help="Thread count.")
    parser.add_argument("--output_dir", type=str, default="models/gbdt_ensemble", help="Output directory.")
    parser.add_argument("--save_predictions", action="store_true", default=False, help="Save individual and blended signals to CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed applied to all three learners (and numpy/random) for reproducible runs.")
    parser.add_argument(
        "--allow_survivorship_bias",
        action="store_true",
        default=False,
        help=(
            "Proceed even if the resolved universe file looks like it was back-projected from "
            "today's index membership (no real historical add/drop/delisting dates). Off by "
            "default: such a universe materially inflates backtest metrics."
        ),
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        default=False,
        help=(
            "Run the pre-training diagnostic gate (see .team-code/plans/pretraining-diagnostic-gate.md) "
            "after training: gate items #1 (budget guard, always checked), #2 (noise-floor framing), "
            "#3 (linear baseline factors), #4 (feature/label lag audit), #5 (train/valid/test IC + "
            "dispersion), #6 (IC decay curve), #7 (turnover/cost stress). Writes diagnostics_report.json "
            "next to ensemble_metadata.json and prints a GATE SUMMARY console block. Advisory only -- "
            "never sys.exit()s on a failing item."
        ),
    )
    parser.add_argument(
        "--allow_undertrained",
        action="store_true",
        default=False,
        help=(
            "Override gate item #1's training-budget guard (num_boost_round * learning_rate must "
            "otherwise be >= a minimum threshold). Off by default; needed for fast smoke tests with "
            "a small --num_boost_round."
        ),
    )
    parser.add_argument(
        "--label_variant",
        type=str,
        default="default",
        choices=sorted(LABEL_VARIANTS.keys()),
        help=(
            "Gate item #10: 'default' uses Alpha158's own close-based label "
            "(Ref($close,-2)/Ref($close,-1)-1) and deal_price='close' (unchanged pipeline "
            "behavior). 'open_shift' uses Ref($open,-2)/Ref($open,-1)-1 and deal_price='open' -- "
            "the only label/execution pairing whose return interval matches what "
            "TopkDropoutStrategy's hardcoded shift=1 execution can actually realize."
        ),
    )
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    # Seed every RNG this script touches so a reported result can actually be reproduced.
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data_dir).expanduser().resolve()
    output_path = (REPO_ROOT / args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    variant = LABEL_VARIANTS[args.label_variant]

    # Gate item #1: fail loudly (or warn, if explicitly overridden) before spending any time
    # training on a budget too small to support any conclusion about signal presence/absence.
    assert_sufficient_training_budget(
        num_boost_round=args.num_boost_round,
        learning_rate=DEFAULT_LEARNING_RATE,
        allow_override=args.allow_undertrained,
    )

    # Initialize Qlib
    logger.info(f"Initializing Qlib (Provider: {data_path}, Region: us)...")
    qlib.init(provider_uri=str(data_path), region="us")

    # Resolve and verify the benchmark BEFORE spending any time training: a bad benchmark
    # otherwise fails deep inside qlib.backtest.report.PortfolioMetrics.init_bench, after a full
    # training run has already completed (see --benchmark_candidates' help text for why this is a
    # real, recurring case -- not hypothetical -- with a constituents-only universe download).
    resolved_benchmark = find_available_benchmark(
        [args.benchmark], start_time=args.train_start, end_time=args.test_end
    )
    if resolved_benchmark is None:
        candidates = [c.strip() for c in args.benchmark_candidates.split(",") if c.strip()]
        logger.warning(
            f"Benchmark '{args.benchmark}' has no usable $close data in {data_path} -- "
            f"searching --benchmark_candidates ({candidates})..."
        )
        resolved_benchmark = find_available_benchmark(candidates, start_time=args.train_start, end_time=args.test_end)
        if resolved_benchmark is None:
            raise RuntimeError(
                f"No usable benchmark found. Tried '{args.benchmark}' and candidates {candidates}, none of "
                f"which have $close data in {data_path} over {args.train_start}..{args.test_end}. Pass "
                f"--benchmark <symbol already present in this data directory> (e.g. a ticker from your "
                f"--market universe itself), or download the desired benchmark symbol into this data_dir first."
            )
        logger.info(f"Resolved benchmark: '{resolved_benchmark}' (--benchmark '{args.benchmark}' was unavailable).")
    args.benchmark = resolved_benchmark

    # Resolve instruments
    instruments = parse_instruments(args.market, data_path)
    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    # Sanity-check the resolved universe for survivorship/look-ahead bias before spending any
    # time training. Raises by default if severe bias is detected; pass --allow_survivorship_bias
    # to override once you understand the risk (see audit_universe_bias docstring).
    universe_audit = audit_universe_bias(instruments, data_path, allow_survivorship_bias=args.allow_survivorship_bias)

    # 1. Prepare Alpha158 Dataset ONCE
    logger.info(
        f"Preparing shared Alpha158 Dataset for market '{args.market}' ({args.train_start} to {args.test_end}), "
        f"label_variant='{args.label_variant}'..."
    )
    dataset = build_dataset(args, instruments, label_config=variant["label_config"])

    # Extract test labels for IC evaluation
    test_df = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_label = test_df["label"].iloc[:, 0]

    # 2. Train Models
    trained_models, predictions, library_versions = build_and_train_models(
        dataset=dataset,
        models_to_run=models_to_run,
        num_boost_round=args.num_boost_round,
        num_threads=args.num_threads,
        seed=args.seed,
    )
    # GBDT library versions alone are not sufficient provenance: qlib's own Alpha158 feature
    # definitions and `risk_analysis`'s scaler/mode semantics, plus numpy/pandas's floating-point
    # and groupby behavior, all affect the reported numbers just as much.
    library_versions["qlib"] = getattr(qlib, "__version__", "unknown")
    library_versions["numpy"] = np.__version__
    library_versions["pandas"] = pd.__version__
    library_versions["python"] = sys.version.split()[0]

    if not predictions:
        logger.error("No models were trained successfully.")
        sys.exit(1)

    # 3. Blend Predictions
    logger.info("--> Applying Cross-Sectional Z-Score Blending across model predictions...")
    blended_pred = blend_predictions(predictions)
    predictions["Ensemble_Blended"] = blended_pred

    # 4. Evaluate Metrics
    results_table = []
    for model_name, pred_series in predictions.items():
        ic_stats = calc_ic_metrics(pred_series, test_label)
        bt_stats = run_portfolio_backtest(
            pred_series, benchmark=args.benchmark, codes=instruments, deal_price=variant["deal_price"]
        )
        results_table.append({
            "Model": model_name,
            "IC": ic_stats["IC"],
            "Rank IC": ic_stats["Rank IC"],
            "Rank ICIR": ic_stats["Rank ICIR"],
            "Excess Return (gross)": bt_stats["annualized_return_gross"],
            "Info Ratio (gross)": bt_stats["information_ratio_gross"],
            "Excess Return (net)": bt_stats["annualized_return_net"],
            "Info Ratio (net)": bt_stats["information_ratio_net"],
            "Max Relative Drawdown (net)": bt_stats["max_relative_drawdown_net"],
            "Max Absolute Drawdown (net)": bt_stats["max_absolute_drawdown_net"],
            "Excess Return (absolute, net, CAGR)": bt_stats["annualized_return_absolute_net"],
            "Backtest Status": bt_stats["status"],
            "Backtest Error": bt_stats["error"],
        })

    # Print Summary Table
    print("\n" + "=" * TABLE_WIDTH)
    print("           QLIB MULTI-MODEL GBDT ENSEMBLE PERFORMANCE EVALUATION           ")
    print("=" * TABLE_WIDTH)
    print(f"Market Universe: {args.market} | Benchmark: {args.benchmark} | Seed: {args.seed}")
    print(f"Train: {args.train_start} -> {args.train_end} | Test: {args.test_start} -> {args.test_end}")
    print(f"Label variant: {args.label_variant} (deal_price='{variant['deal_price']}')")
    print("NOTE: 'gross'/'net' excess return & IR are vs. benchmark, excluding/including trading cost.")
    print("NOTE: 'RelDD' is drawdown of the benchmark-relative curve (arithmetic); 'AbsDD' is the true")
    print("      compounded peak-to-trough % drawdown of the account's own net-of-cost capital curve")
    print("      (see 'Excess Return (absolute, net, CAGR)' in ensemble_metadata.json for its return).")
    print("NOTE: annualized using N=252 (US trading days) -- qlib's own default for freq='day' is")
    print("      238; figures here will NOT tie out directly against qlib's built-in PortAnaRecord.")
    print("-" * TABLE_WIDTH)
    print(
        f"{'Model':<18} | {'IC':>7} | {'RankIC':>7} | {'RankICIR':>8} | "
        f"{'AnnRet(g)':>9} | {'IR(g)':>7} | {'AnnRet(n)':>9} | {'IR(n)':>7} | {'RelDD(n)':>8} | {'AbsDD(n)':>8}"
    )
    print("-" * TABLE_WIDTH)
    for r in results_table:
        is_ens = r["Model"] == "Ensemble_Blended"
        prefix = ">> " if is_ens else "   "
        if r["Backtest Status"] != "ok":
            print(f"{prefix + r['Model']:<18} | {r['IC']:>7.4f} | {r['Rank IC']:>7.4f} | {r['Rank ICIR']:>8.4f} | "
                  f"{'BACKTEST FAILED: ' + str(r['Backtest Error']):<50}")
            continue
        print(
            f"{prefix + r['Model']:<18} | {r['IC']:>7.4f} | {r['Rank IC']:>7.4f} | {r['Rank ICIR']:>8.4f} | "
            f"{r['Excess Return (gross)']:>9.4f} | {r['Info Ratio (gross)']:>7.4f} | "
            f"{r['Excess Return (net)']:>9.4f} | {r['Info Ratio (net)']:>7.4f} | "
            f"{r['Max Relative Drawdown (net)']:>8.4f} | {r['Max Absolute Drawdown (net)']:>8.4f}"
        )
    print("=" * TABLE_WIDTH + "\n")

    # 5. Export Artifacts
    meta_file = output_path / "ensemble_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        metadata = _json_safe({
            "market": args.market,
            "benchmark": args.benchmark,
            "seed": args.seed,
            "label_variant": args.label_variant,
            "universe_audit": universe_audit,
            "segments": {"train": [args.train_start, args.train_end], "test": [args.test_start, args.test_end]},
            "results": results_table,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            # Everything needed to actually reproduce a reported run: the seed alone is not
            # sufficient -- num_threads/num_boost_round and library versions materially affect
            # LightGBM/XGBoost/CatBoost output even with a fixed seed.
            "args": vars(args),
            "library_versions": library_versions,
        })
        json.dump(metadata, f, indent=4, allow_nan=False)
    logger.info(f"Ensemble metadata saved to: {meta_file}")

    if args.save_predictions:
        for m_name, s in predictions.items():
            fname = output_path / f"{m_name.lower()}_predictions.csv"
            s.to_csv(fname)
            logger.info(f"Exported predictions to: {fname}")

    # 6. Pre-training diagnostic gate (gate items #1 [already checked above], #2-#7, #10)
    if args.diagnose:
        logger.info(
            f"--> [--diagnose] Running pre-training diagnostic gate "
            f"(price_field='{variant['deal_price']}', label_variant='{args.label_variant}')..."
        )
        report = diagnose_signal.run_diagnostics(
            dataset=dataset,
            trained_models=trained_models,
            predictions=predictions,
            args=args,
            price_field=variant["deal_price"],
        )
        diag_file = output_path / "diagnostics_report.json"
        with open(diag_file, "w", encoding="utf-8") as f:
            json.dump(_json_safe(report), f, indent=4, allow_nan=False)
        logger.info(f"Diagnostics report saved to: {diag_file}")

        print_gate_summary(report)


def print_gate_summary(report: Dict[str, Any]) -> None:
    """Console "GATE SUMMARY" block, printed after the main results table when --diagnose is
    set. Matches the existing summary table's TABLE_WIDTH/NOTE-line style (Section 10 of the
    diagnostic gate plan). One row per active item that was actually run this pass: #1-#7, #10.
    Item #8 (walk-forward/purged CV) is a separate, not-yet-implemented milestone
    (scripts/walk_forward_cv.py) and is therefore never listed here, per the plan's own
    convention that its row appears "only when walk_forward_cv.py was also run". Item #9
    (sector/month attribution) is deferred and out of scope entirely.
    """
    ITEM_LABELS = {
        "1": "Training budget guard",
        "2": "Noise-floor framing",
        "3": "Linear baseline factors",
        "4": "Lag/label audit",
        "5": "Train/valid/test IC + dispersion",
        "6": "IC decay curve",
        "7": "Turnover / cost stress",
        "10": "Paired label/execution variant",
    }
    gate_verdict = report.get("gate_verdict", {})
    print("=" * TABLE_WIDTH)
    print("                                  GATE SUMMARY                                  ")
    print("=" * TABLE_WIDTH)
    for item_id in ("1", "2", "3", "4", "5", "6", "7", "10"):
        entry = gate_verdict.get(item_id)
        if entry is None:
            continue
        label = ITEM_LABELS.get(item_id, item_id)
        status = str(entry.get("status", "NOT_RUN")).upper()
        rationale = entry.get("rationale", "")
        print(f"[{item_id:>2}] {label:<34} {status:<12} ({rationale})")
    print("-" * TABLE_WIDTH)
    print("NOTE: Advisory only -- no gate item triggers a non-zero exit code. This report")
    print("      distinguishes 'too little signal in this window/universe to detect' from")
    print("      'the model/pipeline is broken' from 'the label horizon is miscalibrated'.")
    print("NOTE: Item #8 (walk-forward/purged CV) is a separate, deferred milestone and is not")
    print("      included above -- see .team-code/plans/pretraining-diagnostic-gate.md Section 9.")
    print("=" * TABLE_WIDTH + "\n")


if __name__ == "__main__":
    main()
