"""Shared CLI arguments and runtime setup for training and inference entrypoints."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from run_paths import default_feature_run_id

from model_pipeline.multilabel_core import (
    DEFAULT_FEATURE_IMPORTANCE_TOP_K,
    DEFAULT_MODEL_TYPE,
    DEFAULT_THRESHOLD_SWEEP,
    FEATURE_SET_CHOICES,
    GRID_SEARCH_METRIC_CHOICES,
    HOLDOUT_SPLITS,
    LR_PARAMS,
    MODEL_TYPE_CHOICES,
    PREDICT_STAGES,
    RF_PARAMS,
    _feature_components,
    load_schema_paths,
    logger,
    resolve_exp_id,
)


def build_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Experiment id under model_bank/experiments/{exp_id}/ (preferred over --run-id).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Deprecated alias for --exp-id (model experiment id).",
    )
    parser.add_argument(
        "--feature-run-id",
        default=None,
        help="Feature run under model_bank/features/ and gold/runs/ for TF-IDF/DCW (e.g. run001).",
    )
    parser.add_argument(
        "--gold-run-id",
        default=None,
        help="Gold run for assembled X_train / X_val_test_oot (e.g. run004). Default: --feature-run-id.",
    )
    parser.add_argument(
        "--x-run-id",
        default=None,
        help="Deprecated alias for --gold-run-id.",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default="tfidf_dcw_embeddings",
        help=(
            "Feature columns to use (default: tfidf_dcw_embeddings = tfidf + dcw + embeddings; "
            "all = log_tfidf + dcw + embeddings)"
        ),
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train on train split only; save models and X_train. Skip holdout scoring and predictions.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Load saved per-label models; run holdout pipeline (see --predict-stage).",
    )
    parser.add_argument(
        "--predict-stage",
        choices=PREDICT_STAGES,
        default=None,
        help=(
            "Required with --predict-only. Recommended flow: features, then predict, then eval. "
            "features=save X_val_test_oot; predict=score Delta; metrics=prediction_*.pkl+json; "
            "threshold_sweep=prob sweep to R2; feature_importance=backfill FI json if missing; "
            "eval=metrics+sweep+FI; all=deprecated one-shot features+predict+eval"
        ),
    )
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="Prediction batch date YYYY-MM-DD or YYYYMMDD (default: today UTC).",
    )
    parser.add_argument(
        "--prediction-suffix",
        default=None,
        help="Deprecated: use --exp-id. Optional fallback folder name under prediction_date= partition.",
    )
    parser.add_argument(
        "--model-date",
        default=None,
        help="Model manifest date YYYYMMDD for --predict-only (default: latest {model_type}_*.pkl).",
    )
    parser.add_argument(
        "--holdout-splits",
        nargs="+",
        choices=HOLDOUT_SPLITS,
        default=None,
        help="Score only these holdout splits (default: val, test, and oot). Example: --holdout-splits val test",
    )
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPE_CHOICES,
        default=DEFAULT_MODEL_TYPE,
        help="Binary relevance classifier per label (default: random_forest)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=f"Logistic Regression maxIter (default: {LR_PARAMS['maxIter']})",
    )
    parser.add_argument(
        "--reg-param",
        type=float,
        default=None,
        help=f"Logistic Regression L2 regParam (default: {LR_PARAMS['regParam']})",
    )
    parser.add_argument(
        "--elastic-net-param",
        type=float,
        default=None,
        help=f"Logistic Regression elasticNetParam (default: {LR_PARAMS['elasticNetParam']})",
    )
    parser.add_argument(
        "--num-trees",
        type=int,
        default=None,
        help=f"Random Forest numTrees (default: {RF_PARAMS['numTrees']})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help=f"Random Forest maxDepth (default: {RF_PARAMS['maxDepth']})",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        help=f"Random Forest maxBins (default: {RF_PARAMS['maxBins']})",
    )
    parser.add_argument(
        "--multilabel-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary relevance predictions (default: 0.5)",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Cap the number of labels trained (top by train frequency)",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help=(
            "Tune hyperparameters on the val split: LR regParam × elasticNetParam; "
            "RF numTrees × maxDepth × maxBins"
        ),
    )
    parser.add_argument(
        "--grid-search-metric",
        choices=GRID_SEARCH_METRIC_CHOICES,
        default="micro_f1",
        help="Metric to maximize during grid search on val (default: micro_f1)",
    )
    parser.add_argument(
        "--feature-importance-top-k",
        type=int,
        default=DEFAULT_FEATURE_IMPORTANCE_TOP_K,
        help=(
            "Save top-K feature importance (LR: |coefficient|; RF: Gini importance). "
            f"Default: {DEFAULT_FEATURE_IMPORTANCE_TOP_K}; set 0 to skip"
        ),
    )
    parser.add_argument(
        "--force-feature-importance",
        action="store_true",
        help="With --predict-stage feature_importance/eval: overwrite existing feature_importance_*.json",
    )
    parser.add_argument(
        "--threshold-sweep",
        default=None,
        help=(
            "Comma-separated probability thresholds for --predict-stage threshold_sweep/eval "
            f"(default: {','.join(str(t) for t in DEFAULT_THRESHOLD_SWEEP)})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training/evaluation locally without writing models, predictions, metrics, or metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows after joins for smoke testing (may reduce join counts)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def validate_training_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.predict_only:
        parser.error("model_training.py does not support --predict-only; use model_inference.py")
    if args.predict_stage is not None:
        parser.error("--predict-stage requires model_inference.py (--predict-only)")
    if args.feature_importance_top_k < 0:
        parser.error("--feature-importance-top-k must be >= 0")


def validate_inference_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.train_only:
        parser.error("model_inference.py does not support --train-only; use model_training.py")
    if args.grid_search:
        parser.error("model_inference.py does not support --grid-search")
    if not args.predict_only:
        parser.error("model_inference.py requires --predict-only")
    if args.predict_stage is None:
        parser.error(
            "--predict-only requires --predict-stage. "
            "Recommended: run features, then predict, then eval as separate jobs."
        )
    if args.feature_importance_top_k < 0:
        parser.error("--feature-importance-top-k must be >= 0")


def configure_logging(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_run_context(
    args: argparse.Namespace,
) -> tuple[str, str, str, dict[str, str], dict[str, bool]]:
    exp_id = resolve_exp_id(args)
    feature_run_id = args.feature_run_id or default_feature_run_id()
    gold_run_id = args.gold_run_id or args.x_run_id or feature_run_id
    paths = load_schema_paths(feature_run_id, gold_run_id=gold_run_id)
    components = _feature_components(args.feature_set)
    return exp_id, feature_run_id, gold_run_id, paths, components


def log_run_banner(
    args: argparse.Namespace,
    exp_id: str,
    feature_run_id: str,
    gold_run_id: str,
    *,
    mode: str,
) -> None:
    logger.info(
        "Experiment: %s | Feature run: %s | Gold run (X): %s",
        exp_id,
        feature_run_id,
        gold_run_id,
    )
    logger.info("Model: %s | Feature set: %s", args.model_type, args.feature_set)
    if mode == "train":
        if args.grid_search:
            logger.info("Grid search enabled (metric=%s)", args.grid_search_metric)
        if args.train_only:
            logger.info("Train-only mode: holdout load, scoring, and predictions will be skipped")
    else:
        logger.info("Predict-only mode: stage=%s", args.predict_stage)
    if args.dry_run:
        logger.info("DRY RUN enabled: skipping all writes to R2/S3A/model_bank.")
