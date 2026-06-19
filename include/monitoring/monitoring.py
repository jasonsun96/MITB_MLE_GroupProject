'''
Batch-inference monitoring for the Legal Topic Tagger.

================================================================================
ASSUMPTIONS & THINGS THAT PROBABLY NEED CHANGING
================================================================================
This script stitches together tables/artifacts produced by other parts of the
pipeline. Where those aren't built yet, I made assumptions — listed here with the
function/constant to edit. Please sanity-check these against your work:

UPSTREAM DATA CONTRACTS (the script reads these; some don't exist yet):
  • inference_features (gold) MUST carry a `dcw_features` map<string,double>
    keyed by DCW *lemma* (e.g. "fighting"), next to document_id + the model
    `features` vector. CSI's production side reads it. This table is currently
    EMPTY on R2 and the assembly job isn't written — whoever builds it must add
    that column.                                   → load_csi_production_values()
  • published_predictions (gold) MUST be written by batch inference with columns
    batch_id, document_id, predicted_labels (array<string>), deployment_group.
    Performance / PSI / CSI all read it. Currently empty.
  • Ground truth: the lawyer-reviewed 10% must be appended to label_store with
    category='production'. Ingesting those reviewed labels is NOT done here.
                                                   → REVIEWED_CATEGORY, load_ground_truth()

MODEL RESOLUTION (hardcoded, but need to be wired to correct path once done):
  • T0_EXP_ID = "exp004_LR_tfidf_dcw_gs" is the assumed PRODUCTION model. Replace
    with a lookup of the 'production' alias (include/inference/model_registry.py
    get_alias) once monitoring should track whatever is actually promoted.
  • SHADOW_MODEL_PATH = None disables the optional shadow model. To explore one,
    hardcode the R2 path to its predictions table there; its performance then plots
    as a second line on the performance dashboard.

PATHS / SCHEMA (schema.yaml):
  • Requires a top-level `monitoring:` entry (output path) — I added it; make sure
    it's committed.                                            → load_paths()
  • The feature-importance JSON path is NOT in schema.yaml; it's built in code as
    model_bank/experiments/{exp_id}/model/feature_importance_*.json. Adjust if the
    model_training layout differs.                       → load_global_features()
  • Baselines come from the v2 gold layout (built in code, not schema tables):
    CSI training features = gold/runs/{feature_run_id}/dcw_train; PSI training
    predictions = gold/model_predictions/prediction_date=*/{exp_id}_train (latest).
                  → load_csi_baseline_values(), resolve_train_predictions_path()

================================================================================
 WHAT THIS CODE DOES
================================================================================
Runs once per batch (right after batch inference) and writes, to
monitoring/{batch_id}/ on R2, a metrics.json plus the dashboard PNGs:

  performance.png               Macro F1 (P0) + Hamming Loss (P1) vs each model's T=0
                                baseline, GYR-banded, as a time series.
  stability.png                 PSI (label-distribution drift) + CSI (top-50 feature
                                drift) time series, GYR-banded.
  psi_distribution.png          Reference: per-label expected-vs-actual prevalence.
  csi_distribution.png          Reference: per-feature baseline-vs-production value
                                histograms for all 50 global features.
  csi_distribution_top3.png     Same, for the 3 most globally-important features.
  csi_feature_trends.png        Per-feature CSI over time, all 50 (grid).
  csi_feature_trends_top3.png   Per-feature CSI over time, top 3 (overlaid).

The production champion is tracked as a trend line; an optional shadow model adds a
second line on the performance plot. Lines break at a model swap (promotion) so a
change of model never looks like drift; each segment restarts at its own T=0.

Metric families:
  • Performance (needs ground truth → reviewed 10% only): Macro F1, Hamming Loss.
  • PSI (predictions only → full batch): predicted-label distribution vs OOT.
  • CSI (features only → full batch): top-50 global feature values vs OOT.

History for the trend lines is rebuilt each run by reading prior metrics.json
files back from monitoring/ (load_metric_history); there is no central index.
'''
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from include.inference.model_registry import hadoop_path_exists, read_json, write_json

logger = logging.getLogger(__name__)

# T=0 reference model. This is the model currently in PRODUCTION; its holdout
# metrics are the baseline that production drift is measured against. Should be
# kept in sync with whatever model_registry.py promotes to the 'production' alias.
T0_EXP_ID = "exp004_LR_tfidf_dcw_gs"

# Reviewed-subset performance is measured on the out-of-time holdout split, the
# closest training-time analogue to future production data (see design notes).
T0_METRICS_SPLIT = "holdout_oot"

# Documents reviewed by lawyers (the 10% with ground truth) are written back to
# label_store tagged with this category. [Assumption: ground-truth ingestion
# appends reviewed production docs to label_store with category='production'.]
REVIEWED_CATEGORY = "production"

# Optional SHADOW model: a second model scored alongside production for comparison
# (exploratory). When set, its performance is plotted as a separate line on the
# performance dashboard; when None (default) no shadow is tracked or plotted.
#
# >>> HARDCODE THE SHADOW MODEL PATH HERE (R2 path to the shadow model's
#     predictions table for the batch; None disables the shadow entirely). <<<
SHADOW_MODEL_PATH: str | None = None

# v2 gold-layout directories (under gold/) that monitoring reads for its TRAINING
# baselines. These are NOT in this branch's schema.yaml, so they are hardcoded here.
# Reconcile with schema.yaml if/when the model_training (v2) branch is merged in.
GOLD_RUNS_DIR = "runs"                          # gold/runs/{feature_run_id}/dcw_train (CSI baseline)
GOLD_MODEL_PREDICTIONS_DIR = "model_predictions"  # gold/model_predictions/prediction_date=*/{exp_id}_train (PSI baseline)


def load_schema() -> dict:
    with (PROJECT_ROOT / "schema.yaml").open() as schema_file:
        return yaml.safe_load(schema_file)


def load_paths(schema: dict) -> dict:
    gold = schema["gold"]
    gold_base = gold["path"].rstrip("/")
    tables = gold["tables"]
    model_bank_base = schema["model_bank"]["path"].rstrip("/")
    return {
        "gold_base": gold_base,
        # Ground-truth labels for the reviewed 10% (label_store, filtered to
        # category='production'). 'labels' is an alias to this path.
        "label_store": f"{gold_base}/{tables['label_store']['path']}",
        # Served predictions; the reviewed 10% are a subset of each batch.
        "published_predictions": f"{gold_base}/{tables['published_predictions']['path']}",
        "model_bank_base": model_bank_base,
        # v2 layout bases. These dirs are NOT in this branch's schema.yaml, so they
        # are hardcoded here (see GOLD_RUNS_DIR / GOLD_MODEL_PREDICTIONS_DIR).
        # CSI training baseline = {runs_base}/{feature_run_id}/dcw_train;
        # PSI training baseline = {model_predictions_base}/prediction_date=*/{exp_id}_train.
        "runs_base": f"{gold_base}/{GOLD_RUNS_DIR}",
        "model_predictions_base": f"{gold_base}/{GOLD_MODEL_PREDICTIONS_DIR}",
        # CSI production: assembled inference inputs (carries a dcw_features map for monitoring).
        "inference_features": f"{gold_base}/{tables['inference_features']['path']}",
        # Per-batch monitoring artifacts: monitoring/{batch_id}/metrics.json + dashboard.png
        "monitoring_base": schema["monitoring"]["path"].rstrip("/"),
    }


def load_ground_truth(spark, paths: dict):
    """Reviewed 10% ground-truth labels: document_id + semicolon-delimited labels."""
    from pyspark.sql import functions as F

    label_store = spark.read.format("delta").load(paths["label_store"])
    return (
        label_store.filter(F.col("category") == REVIEWED_CATEGORY)
        .select("document_id", "labels")
    )


def load_batch_predictions(spark, paths: dict, batch_id: str):
    """Predictions for one batch: document_id + predicted_labels array."""
    from pyspark.sql import functions as F

    predictions = spark.read.format("delta").load(paths["published_predictions"])
    return (
        predictions.filter(F.col("batch_id") == batch_id)
        .select("document_id", "predicted_labels")
    )


# ── PSI ingestion (prediction stability) ───────────────────────────────────────
#
# PSI compares the model's PREDICTED label distribution at baseline vs production.
# Both sides are predicted_labels (not ground truth): this isolates distribution
# shift and avoids contaminating it with the model's inherent prediction bias.
# Baseline = predicted labels on the production model's TRAINING set (its in-sample
# reference scoring); production = predicted labels on the live batch.

# Split that defines the PSI baseline distribution: the model's predictions on its
# own training set (the dedicated {exp_id}_train predictions table, category='train').
PSI_BASELINE_SPLIT = "train"


def _predicted_label_counts(df) -> dict:
    """Per-label predicted prevalence for a df with a predicted_labels array column."""
    from pyspark.sql import functions as F

    total = df.count()
    counts = {
        row["label"]: row["count"]
        for row in (
            df
            # predicted_labels is an array<string> already normalised to model labels.
            .withColumn("label", F.explode_outer(F.col("predicted_labels")))
            .filter(F.col("label").isNotNull())
            .groupBy("label").count()
            .collect()
        )
    }
    return {"total": total, "counts": counts}


def load_baseline_prediction_counts(spark, reference_path: str) -> dict:
    """
    PSI baseline (expected): the production model's predicted-label prevalence on its
    TRAINING set. reference_path is the model's train-predictions table
    (gold/model_predictions/prediction_date=*/{exp_id}_train), pinned to production.

    Returns {"total": <#train docs>, "counts": {label: <#docs predicted label>}}.
    """
    from pyspark.sql import functions as F

    if not reference_path:
        raise ValueError("No train-predictions path for the production model (PSI baseline)")

    reference = spark.read.format("delta").load(reference_path)
    train = reference.filter(F.col("category") == PSI_BASELINE_SPLIT)
    result = _predicted_label_counts(train)
    logger.info(
        "PSI baseline: %d %s docs from %s, %d distinct predicted labels",
        result["total"], PSI_BASELINE_SPLIT, reference_path, len(result["counts"]),
    )
    return result


def resolve_train_predictions_path(spark, paths: dict, exp_id: str) -> str | None:
    """
    Find the production model's train-predictions table by scanning
    gold/model_predictions/prediction_date=*/{exp_id}_train and taking the latest
    date. Returns None if none exists. (This table isn't referenced by the model's
    metrics JSON, hence the scan; convention is the '{exp_id}_train' folder name.)
    """
    model_predictions_base = paths["model_predictions_base"]
    candidates = []
    for date_dir in _list_hadoop_children(spark, model_predictions_base):
        candidate = f"{date_dir.rstrip('/')}/{exp_id}_train"
        if hadoop_path_exists(spark, candidate):
            candidates.append(candidate)
    return max(candidates) if candidates else None


def load_production_prediction_counts(spark, paths: dict, batch_id: str) -> dict:
    """
    PSI actual (production): predicted-label prevalence for this batch.

    Returns {"total": <#docs in batch>, "counts": {label: <#docs predicted label>}}.
    Uses the full batch (no ground truth needed — these are model outputs).
    """
    from pyspark.sql import functions as F

    predictions = spark.read.format("delta").load(paths["published_predictions"])
    batch = predictions.filter(F.col("batch_id") == batch_id)
    result = _predicted_label_counts(batch)
    logger.info(
        "PSI production: %d batch docs, %d distinct predicted labels",
        result["total"], len(result["counts"]),
    )
    return result


# PSI GYR thresholds (lower is better): GREEN < 0.10, YELLOW 0.10–0.25, RED > 0.25.
PSI_GREEN, PSI_YELLOW = 0.10, 0.25
# Floor for zero rates so ln(A/E) stays finite when a label is absent on one side.
PSI_EPSILON = 1e-4


def _classify_lower_is_better(value: float, green: float, yellow: float) -> str:
    if value <= green:
        return "GREEN"
    if value <= yellow:
        return "YELLOW"
    return "RED"


def _one_bin_psi(actual_rate: float, expected_rate: float) -> float:
    """One-bin PSI contribution for a single label's prevalence: (A-E)*ln(A/E)."""
    a = max(actual_rate, PSI_EPSILON)
    e = max(expected_rate, PSI_EPSILON)
    return (a - e) * math.log(a / e)


def compute_psi(baseline_counts: dict, production_counts: dict, label_list: list[str]) -> dict:
    """
    Per-label one-bin PSI summed into an overall score, comparing the production
    model's predicted-label prevalence in production against its OOT baseline.

    Aligns both distributions to the model's full label universe (label_list) so a
    label present on one side but not the other still contributes (epsilon-floored).
    Overall PSI = sum of per-label contributions; GYR is applied to the overall.
    Per-label GYR is included as a heuristic for *which* label is drifting.
    """
    expected_total = baseline_counts.get("total") or 0
    actual_total = production_counts.get("total") or 0
    if expected_total == 0 or actual_total == 0:
        raise ValueError("PSI requires non-empty baseline and production distributions")

    per_label: dict[str, dict] = {}
    overall = 0.0
    for label in label_list:
        expected_rate = baseline_counts["counts"].get(label, 0) / expected_total
        actual_rate = production_counts["counts"].get(label, 0) / actual_total
        psi = _one_bin_psi(actual_rate, expected_rate)
        overall += psi
        per_label[label] = {
            "psi": round(psi, 6),
            "expected_rate": round(expected_rate, 6),
            "actual_rate": round(actual_rate, 6),
            "gyr": _classify_lower_is_better(psi, PSI_GREEN, PSI_YELLOW),
        }

    return {
        "overall_psi": round(overall, 6),
        "overall_gyr": _classify_lower_is_better(overall, PSI_GREEN, PSI_YELLOW),
        "per_label": per_label,
    }


# ── CSI ingestion (covariate / feature stability) ──────────────────────────────
#
# CSI watches the production model's top-50 global features for distribution shift.
# The feature LIST comes from the model's stored feature-importance JSON (global_top);
# the baseline VALUE distribution is the model's TRAINING features (dcw_train);
# production is the same features on the live batch. All on full data — no labels.


def load_global_features(spark, paths: dict, exp_id: str = T0_EXP_ID) -> list[dict]:
    """
    The production model's global top-50 features (the CSI watch-list), read from its
    stored feature-importance JSON (model_bank/experiments/{exp_id}/model/
    feature_importance_*.json). Returns [{"name": "dcw:fighting", "lemma": "fighting"}]
    — lemma is the dcw_features map key (the "dcw:" prefix is stripped).
    """
    model_dir = f"{paths['model_bank_base']}/experiments/{exp_id}/model"
    fi_path = _latest_feature_importance(spark, model_dir)
    if fi_path is None:
        raise FileNotFoundError(f"No feature_importance_*.json found under {model_dir}")

    feature_importance = read_json(spark, fi_path)
    features = []
    for row in feature_importance["global_top"]:
        name = row["feature"]
        lemma = name.split(":", 1)[1] if ":" in name else name
        features.append({"name": name, "lemma": lemma})
    logger.info("CSI: %d global features from %s", len(features), fi_path)
    return features


def _latest_feature_importance(spark, model_dir: str) -> str | None:
    """Return the lexicographically latest feature_importance_*.json path, or None."""
    candidates = [
        child for child in _list_hadoop_children(spark, model_dir)
        if child.rsplit("/", 1)[-1].startswith("feature_importance_") and child.endswith(".json")
    ]
    return max(candidates) if candidates else None


def _extract_map_values(df, lemmas: list[str]) -> dict[str, list[float]]:
    """
    Collect per-document values of each lemma from the df's dcw_features map.
    The map is sparse, so a missing key is a genuine 0 (feature did not fire) and is
    coalesced to 0.0 rather than dropped — the zeros are part of the distribution.
    """
    from pyspark.sql import functions as F

    columns = [F.coalesce(F.col("dcw_features")[lemma], F.lit(0.0)).alias(lemma) for lemma in lemmas]
    rows = df.select(*columns).collect()
    values: dict[str, list[float]] = {lemma: [] for lemma in lemmas}
    for row in rows:
        for lemma in lemmas:
            values[lemma].append(float(row[lemma]))
    return values


def load_csi_baseline_values(spark, paths: dict, features: list[dict], feature_run_id: str) -> dict[str, list[float]]:
    """
    CSI baseline (expected): the production model's TRAINING feature values for the
    top-50 features, from gold/runs/{feature_run_id}/dcw_train (v2 layout). The whole
    table is the train split, so no category filter is needed; dcw_features is a map
    keyed by lemma (same shape as the production side).

    Returns {lemma: [values across train docs]}.
    """
    dcw_train_path = f"{paths['runs_base']}/{feature_run_id}/dcw_train"
    train = spark.read.format("delta").load(dcw_train_path)
    values = _extract_map_values(train, [f["lemma"] for f in features])
    logger.info("CSI baseline: %d train docs from %s over %d features",
                len(next(iter(values.values()), [])), dcw_train_path, len(features))
    return values


def load_csi_production_values(spark, paths: dict, batch_id: str, features: list[dict]) -> dict[str, list[float]]:
    """
    CSI actual (production): the same top-50 feature values on the live batch.
    Uses the full batch (10% reviewed subset is irrelevant — features need no labels):
    the batch's documents come from published_predictions, joined to inference_features.

    Returns {lemma: [values across batch docs]}.
    [Assumption: inference_features carries a dcw_features map<string,double> keyed by
    lemma, mirroring the holdout table, so the same extraction works on both sides.]
    """
    from pyspark.sql import functions as F

    batch_ids = (
        spark.read.format("delta").load(paths["published_predictions"])
        .filter(F.col("batch_id") == batch_id)
        .select("document_id").distinct()
    )
    inference = spark.read.format("delta").load(paths["inference_features"])
    batch_features = inference.join(batch_ids, on="document_id", how="inner")
    values = _extract_map_values(batch_features, [f["lemma"] for f in features])
    logger.info("CSI production: %d batch docs over %d features", len(next(iter(values.values()), [])), len(features))
    return values


# CSI uses the same GYR thresholds as PSI (lower is better): GREEN < 0.10, etc.
CSI_GREEN, CSI_YELLOW = 0.10, 0.25
CSI_BINS = 10
CSI_EPSILON = 1e-4


def _feature_csi(baseline_values: list[float], production_values: list[float], n_bins: int = CSI_BINS) -> float:
    """
    CSI for one continuous feature: quantile-bin the baseline, score both sides into
    those bins, then Σ (actual_frac - expected_frac) * ln(actual/expected).

    DCW features are sparse (mostly 0). If the baseline is constant (e.g. never fires
    in OOT), quantile edges collapse, so we fall back to zero-vs-nonzero bins to still
    catch a firing-rate shift. Returns 0.0 when there is nothing to compare.
    """
    base = np.asarray(baseline_values, dtype=float)
    prod = np.asarray(production_values, dtype=float)
    if base.size == 0 or prod.size == 0:
        return 0.0

    edges = np.unique(np.quantile(base, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:  # degenerate baseline (typically all-zero) → presence/absence bins
        edges = np.array([-np.inf, 1e-12, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf  # capture production values outside the baseline range

    base_hist, _ = np.histogram(base, bins=edges)
    prod_hist, _ = np.histogram(prod, bins=edges)
    expected = np.clip(base_hist / base_hist.sum(), CSI_EPSILON, None)
    actual = np.clip(prod_hist / prod_hist.sum(), CSI_EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_csi(baseline_values: dict, production_values: dict, features: list[dict]) -> dict:
    """
    Per-feature CSI over the production model's top-50 global features, plus an
    overall score. Overall CSI = the worst (max) single-feature CSI, so the GYR
    thresholds (0.10 / 0.25) stay interpretable per feature and the worst-drifting
    feature drives the alert. per_feature keeps each feature's CSI + GYR (and rank)
    for the distribution plots and drill-down.
    """
    per_feature: dict[str, dict] = {}
    overall = 0.0
    for rank, feature in enumerate(features, start=1):
        lemma = feature["lemma"]
        csi = _feature_csi(baseline_values.get(lemma, []), production_values.get(lemma, []))
        overall = max(overall, csi)
        per_feature[lemma] = {
            "name": feature["name"],
            "rank": rank,  # global-importance rank (features arrive in global_top order)
            "csi": round(csi, 6),
            "gyr": _classify_lower_is_better(csi, CSI_GREEN, CSI_YELLOW),
        }

    return {
        "overall_csi": round(overall, 6),
        "overall_gyr": _classify_lower_is_better(overall, CSI_GREEN, CSI_YELLOW),
        "per_feature": per_feature,
    }


def compute_performance(ground_truth_df, predictions_df, label_list: list[str]) -> dict | None:
    """
    Compute live Macro F1 (P0) and Hamming Loss (P1) for one batch on the reviewed
    10%, by joining ground truth with predictions on document_id.

    Macro F1 is averaged over `label_list` (the production model's label universe)
    so it is directly comparable to the T=0 baseline. Returns None when no reviewed
    documents overlap this batch (nothing to score).
    """
    joined = ground_truth_df.join(predictions_df, on="document_id", how="inner")
    rows = joined.select("labels", "predicted_labels").collect()
    if not rows:
        logger.warning("No reviewed ground-truth documents overlap this batch; skipping performance")
        return None

    label_index = {label: i for i, label in enumerate(label_list)}
    n_docs, n_labels = len(rows), len(label_list)
    y_true = np.zeros((n_docs, n_labels), dtype=int)
    y_pred = np.zeros((n_docs, n_labels), dtype=int)

    for i, row in enumerate(rows):
        # Ground-truth labels: semicolon-delimited string (label_store format).
        for label in {part.strip().lower() for part in (row["labels"] or "").split(";") if part.strip()}:
            if label in label_index:
                y_true[i, label_index[label]] = 1
        # Predicted labels: array<string>, already normalised to the model labels.
        for label in (row["predicted_labels"] or []):
            if label in label_index:
                y_pred[i, label_index[label]] = 1

    hamming_loss = float(np.mean(y_true != y_pred))

    per_label_f1: dict[str, float] = {}
    for j, label in enumerate(label_list):
        tp = int(np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 1)))
        fp = int(np.sum((y_true[:, j] == 0) & (y_pred[:, j] == 1)))
        fn = int(np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label_f1[label] = round(f1, 6)

    macro_f1 = float(np.mean(list(per_label_f1.values())))
    return {
        "reviewed_count": n_docs,
        "macro_f1": round(macro_f1, 6),
        "hamming_loss": round(hamming_loss, 6),
        "per_label_f1": per_label_f1,
    }


def load_t0_baseline(spark, paths: dict) -> dict:
    """
    Load the T=0 baseline performance metrics (macro_f1, hamming_loss) from the
    production model's holdout evaluation. Uses the most recent prediction_*.json
    under the experiment's metrics directory.
    """
    metrics_dir = f"{paths['model_bank_base']}/experiments/{T0_EXP_ID}/metrics"
    metrics_path = _latest_prediction_metrics(spark, metrics_dir)
    if metrics_path is None:
        raise FileNotFoundError(
            f"No prediction_*.json baseline metrics found under {metrics_dir}"
        )

    logger.info("Loading T=0 baseline metrics from %s", metrics_path)
    metrics = read_json(spark, metrics_path)
    split_metrics = metrics["metrics"][T0_METRICS_SPLIT]
    # The model's canonical label set defines the universe over which Macro F1 is
    # averaged. Live metrics must use this same set to be comparable to the baseline.
    labels = sorted(metrics["probability_columns"].keys())
    return {
        "exp_id": T0_EXP_ID,
        "split": T0_METRICS_SPLIT,
        "macro_f1": split_metrics["macro_f1"],
        "hamming_loss": split_metrics["hamming_loss"],
        "labels": labels,
        # Evaluation date (from the prediction_YYYYMMDD filename) becomes the
        # x-position of the T=0 point that anchors the start of the model's line.
        "date": _parse_metrics_date(metrics_path),
        # Feature run id (e.g. "run001") locates the model's gold/runs/{id}/dcw_train
        # table used as the CSI training baseline.
        "feature_run_id": metrics.get("feature_run_id"),
    }


def _parse_metrics_date(metrics_path: str) -> str | None:
    """Extract YYYY-MM-DD from a prediction_YYYYMMDD.json metrics path."""
    match = re.search(r"prediction_(\d{8})\.json$", metrics_path)
    if not match:
        return None
    digits = match.group(1)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def load_shadow_performance(spark, paths: dict, batch_id: str, label_list: list[str]) -> dict | None:
    """
    Optional shadow-model performance for this batch (exploratory). Returns None when
    no shadow model is configured (SHADOW_MODEL_PATH is None), so the shadow line is
    simply absent. When configured, SHADOW_MODEL_PATH is the shadow model's
    predictions table (expected columns: document_id, predicted_labels, and batch_id
    if multi-batch); its predictions on the reviewed 10% are scored against ground
    truth, exactly like production.
    """
    if SHADOW_MODEL_PATH is None:
        return None

    from pyspark.sql import functions as F

    shadow_predictions = spark.read.format("delta").load(SHADOW_MODEL_PATH)
    if "batch_id" in shadow_predictions.columns:
        shadow_predictions = shadow_predictions.filter(F.col("batch_id") == batch_id)
    shadow_predictions = shadow_predictions.select("document_id", "predicted_labels")
    return compute_performance(load_ground_truth(spark, paths), shadow_predictions, label_list)


# Models tracked as their own trend line. Production is the champion; shadow is an
# optional comparison model (see SHADOW_MODEL_PATH) — its series stays empty unless
# a shadow model is configured, and only carries performance metrics.
TRACKED_MODELS = ("production", "shadow")


# Per-model metrics carried through the readback into the trend plots. Performance
# metrics (macro_f1, hamming_loss) need ground truth; psi does not, so a point may
# carry some metrics and not others. Each plot filters for the metric it draws.
TRACKED_METRICS = ("macro_f1", "hamming_loss", "psi", "csi")


def load_metric_history(spark, paths: dict) -> dict[str, list[dict]]:
    """
    Read back prior runs' metric points from monitoring/{batch_id}/metrics.json so
    each daily run can plot time-series lines (performance and PSI) rather than dots.

    Returns one series per tracked model: {"production": [...], "shadow": [...]}.
    Each point carries run_id (so a plot can break the line at a model swap) plus
    every tracked metric (macro_f1, hamming_loss, psi); any metric absent that day
    is None and is filtered out by the plot that draws it. Points are sorted
    oldest-first. A model's block is skipped only on days the model was absent.
    Returns empty series when no history exists.
    """
    monitoring_base = paths["monitoring_base"]
    history: dict[str, list[dict]] = {model: [] for model in TRACKED_MODELS}

    for batch_dir in _list_hadoop_children(spark, monitoring_base):
        metrics_path = f"{batch_dir.rstrip('/')}/metrics.json"
        if not hadoop_path_exists(spark, metrics_path):
            continue
        try:
            report = read_json(spark, metrics_path)
        except Exception as exc:  # tolerate a single corrupt/partial file
            logger.warning("Skipping unreadable monitoring file %s: %s", metrics_path, exc)
            continue

        for model in TRACKED_MODELS:
            block = report.get(model)
            if not block or not block.get("run_id"):  # model absent that day
                continue
            point = {
                "batch_id": report.get("batch_id"),
                "monitored_at": report.get("monitored_at"),
                "run_id": block.get("run_id"),
            }
            point.update({metric: block.get(metric) for metric in TRACKED_METRICS})
            history[model].append(point)

    # batch_id is a ts_nodash stamp; monitored_at is ISO. Either sorts chronologically.
    for model in TRACKED_MODELS:
        history[model].sort(key=lambda point: point.get("monitored_at") or point.get("batch_id") or "")
        logger.info("Loaded %d prior %s point(s) from %s", len(history[model]), model, monitoring_base)

    return history


def load_csi_feature_history(spark, paths: dict, model: str = "production") -> dict[str, list[dict]]:
    """
    Read back the PER-FEATURE CSI over time from monitoring/{batch_id}/metrics.json
    (block[model]["csi_per_feature"]), so each of the top-50 features can be plotted
    as its own CSI trend. Returns {lemma: [{monitored_at, batch_id, run_id, csi,
    gyr}, ...]} sorted oldest-first. Empty when no history (or no CSI) exists yet.
    """
    monitoring_base = paths["monitoring_base"]
    history: dict[str, list[dict]] = {}

    for batch_dir in _list_hadoop_children(spark, monitoring_base):
        metrics_path = f"{batch_dir.rstrip('/')}/metrics.json"
        if not hadoop_path_exists(spark, metrics_path):
            continue
        try:
            report = read_json(spark, metrics_path)
        except Exception as exc:  # tolerate a single corrupt/partial file
            logger.warning("Skipping unreadable monitoring file %s: %s", metrics_path, exc)
            continue

        block = report.get(model)
        if not block:
            continue
        run_id = block.get("run_id")
        for lemma, detail in (block.get("csi_per_feature") or {}).items():
            if detail.get("csi") is None:
                continue
            history.setdefault(lemma, []).append({
                "monitored_at": report.get("monitored_at"),
                "batch_id": report.get("batch_id"),
                "run_id": run_id,
                "csi": detail["csi"],
                "gyr": detail.get("gyr"),
            })

    for lemma in history:
        history[lemma].sort(key=lambda point: point.get("monitored_at") or point.get("batch_id") or "")
    logger.info("Loaded per-feature CSI history for %d features", len(history))
    return history


def _list_hadoop_children(spark, path: str) -> list[str]:
    """List immediate child paths of a Hadoop directory; empty if it does not exist."""
    jvm = spark.sparkContext._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    if not fs.exists(hadoop_path):
        return []
    return [str(status.getPath()) for status in fs.listStatus(hadoop_path)]


def _latest_prediction_metrics(spark, metrics_dir: str) -> str | None:
    """Return the lexicographically latest prediction_*.json path, or None."""
    jvm = spark.sparkContext._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(metrics_dir)
    fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    if not fs.exists(hadoop_path):
        return None
    candidates = [
        str(status.getPath())
        for status in fs.listStatus(hadoop_path)
        if str(status.getPath()).rsplit("/", 1)[-1].startswith("prediction_")
        and str(status.getPath()).endswith(".json")
    ]
    return max(candidates) if candidates else None


# ── Performance trend plot ──────────────────────────────────────────────────

# GYR colours and the absolute pass/fail bands per metric (from the project's
# Green-Yellow-Red criteria). Bands apply to every model line identically; they
# are not relative to any T=0 baseline.
GYR_COLORS = {"GREEN": "#4CAF50", "YELLOW": "#FFC107", "RED": "#F44336"}

# (low, high, gyr) shaded zones. Macro F1: higher is better; Hamming: lower is better.
MACRO_F1_BANDS = [(0.00, 0.60, "RED"), (0.60, 0.65, "YELLOW"), (0.65, 1.01, "GREEN")]
HAMMING_BANDS = [(0.00, 0.15, "GREEN"), (0.15, 0.20, "YELLOW"), (0.20, 1.01, "RED")]

# One subplot per metric: (point key, title, bands, y-axis range).
PERF_METRICS = [
    ("macro_f1", "Macro F1 (P0)", MACRO_F1_BANDS, (0.40, 1.00)),
    ("hamming_loss", "Hamming Loss (P1)", HAMMING_BANDS, (0.00, 0.30)),
]

# Per-model line styling. Production is the champion; shadow is the optional
# comparison model (drawn only when a shadow model is configured).
MODEL_STYLES = {
    "production": {"color": "#1565C0", "marker": "o", "linestyle": "-", "label": "Production"},
    "shadow": {"color": "#6A1B9A", "marker": "D", "linestyle": "--", "label": "Shadow"},
}


def _point_time(point: dict) -> datetime | None:
    """Parse a history point's timestamp for the x-axis (ISO, else ts_nodash batch_id)."""
    iso = point.get("monitored_at")
    if iso:
        try:
            # Drop tzinfo so T=0 (naive date) and live points (tz-aware) plot together.
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    batch_id = point.get("batch_id") or ""
    try:
        return datetime.strptime(batch_id, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def _segments_by_run_id(series: list[dict]) -> list[list[dict]]:
    """Split a time-sorted series into contiguous runs of the same run_id, so the
    plotted line breaks at a model swap instead of bridging two different models."""
    segments: list[list[dict]] = []
    for point in series:
        if segments and segments[-1][-1].get("run_id") == point.get("run_id"):
            segments[-1].append(point)
        else:
            segments.append([point])
    return segments


def _prepend_t0(series: list[dict], baseline: dict | None, key: str):
    """
    Return (series_with_t0, t0_time) where the model's T=0 baseline is inserted as
    the first vertex, sharing the first segment's run_id so the line stays
    continuous. The T=0 x-position is the baseline's eval date, or one day before
    the first live point if no date is available. Returns the series unchanged
    (and t0_time=None) when there is no baseline value for this metric.
    """
    if not baseline or baseline.get(key) is None:
        return list(series), None

    first_run_id = series[0].get("run_id") if series else baseline.get("exp_id")
    if baseline.get("date"):
        t0_time = datetime.fromisoformat(baseline["date"])
    elif series:
        first_time = _point_time(series[0])
        t0_time = (first_time - timedelta(days=1)) if first_time else None
    else:
        t0_time = None

    t0_point = {
        "monitored_at": t0_time.isoformat() if t0_time else baseline.get("date"),
        "batch_id": None,
        "run_id": first_run_id,
        "macro_f1": baseline.get("macro_f1"),
        "hamming_loss": baseline.get("hamming_loss"),
    }
    return [t0_point, *series], t0_time


def build_performance_plot(
    series_by_model: dict[str, list[dict]],
    baselines: dict[str, dict | None],
    batch_id: str,
    generated_at: str,
) -> bytes:
    """
    Render the performance time-series dashboard as PNG bytes.

    series_by_model: {"production": [...], "shadow": [...]} time-sorted points, each
        {monitored_at, batch_id, run_id, macro_f1, hamming_loss}. The shadow series
        is empty (and no shadow line drawn) unless a shadow model is configured.
    baselines: T=0 reference keyed by run_id, {run_id: {macro_f1, hamming_loss,
        date}}. Each model version has its own baseline; when production is
        promoted, the new model's segment restarts at its own T=0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"Performance Monitoring — batch {batch_id}  ({generated_at[:19]} UTC)",
        fontsize=12, fontweight="bold",
    )

    for ax, (key, title, bands, ylim) in zip(axes, PERF_METRICS):
        # GYR zones
        for low, high, gyr in bands:
            ax.axhspan(low, high, color=GYR_COLORS[gyr], alpha=0.12, zorder=0)

        # One line per model, broken at model swaps. Each segment is one model
        # version and restarts at its own T=0 baseline (a new champion after a
        # promotion has a fresh training baseline, not the old model's).
        for model, raw_series in series_by_model.items():
            style = MODEL_STYLES[model]
            # Drop days missing this metric (e.g. no ground truth -> no macro_f1).
            series = [point for point in raw_series if point.get(key) is not None]
            for seg in _segments_by_run_id(series):
                run_id = seg[0].get("run_id")
                seg_with_t0, t0_time = _prepend_t0(seg, baselines.get(run_id), key)

                xs = [_point_time(p) for p in seg_with_t0]
                ys = [p.get(key) for p in seg_with_t0]
                ax.plot(
                    xs, ys,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=5, linewidth=1.6,
                    label=style["label"], zorder=3,
                )

                # Emphasise this segment's T=0 anchor with a star
                if t0_time is not None:
                    ax.scatter(
                        [t0_time], [baselines[run_id][key]],
                        marker="*", s=140, color=style["color"],
                        edgecolor="white", linewidth=0.6, zorder=4,
                    )

        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.2)

        # De-duplicate legend entries (segments repeat the same label)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            ax.legend(unique.values(), unique.keys(), fontsize=7.5, loc="best")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


# ── Stability trend plot (PSI + CSI) ────────────────────────────────────────────

# GYR zones for stability metrics (lower is better); same thresholds for PSI and CSI.
STABILITY_BANDS = [(0.00, PSI_GREEN, "GREEN"), (PSI_GREEN, PSI_YELLOW, "YELLOW"), (PSI_YELLOW, 100.0, "RED")]

# One subplot per stability metric: (point key, subplot title).
STABILITY_METRICS = [
    ("psi", "PSI — prediction stability (label distribution vs OOT)"),
    ("csi", "CSI — feature stability (worst of top-50 features vs OOT)"),
]


def build_stability_plot(
    series_by_model: dict[str, list[dict]],
    batch_id: str,
    generated_at: str,
) -> bytes:
    """
    Render the stability time-series (PSI on top, CSI below) as PNG bytes — same style
    as the performance plot but its own figure and no T=0 anchor (stability is a
    production-vs-baseline drift score that only exists once batches start; ≈0 = no
    shift). One line per model, broken at model swaps.

    series_by_model: {"production": [...], "shadow": [...]} time-sorted points, each
        carrying run_id, psi (overall summed PSI) and csi (overall worst-feature CSI).
        Shadow only carries performance, so it has no PSI/CSI line here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"Stability Monitoring — batch {batch_id}  ({generated_at[:19]} UTC)",
        fontsize=12, fontweight="bold",
    )

    for ax, (key, title) in zip(axes, STABILITY_METRICS):
        y_max = PSI_YELLOW * 1.2  # keep GYR bands visible even when the metric is tiny
        for low, high, gyr in STABILITY_BANDS:
            ax.axhspan(low, high, color=GYR_COLORS[gyr], alpha=0.12, zorder=0)

        for model, raw_series in series_by_model.items():
            style = MODEL_STYLES[model]
            series = [point for point in raw_series if point.get(key) is not None]
            for seg in _segments_by_run_id(series):
                xs = [_point_time(p) for p in seg]
                ys = [p[key] for p in seg]
                if ys:
                    y_max = max(y_max, max(ys))
                ax.plot(
                    xs, ys,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=5, linewidth=1.6,
                    label=style["label"], zorder=3,
                )

        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel(key.upper())
        ax.set_ylim(0, y_max * 1.1)
        ax.grid(True, axis="y", alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            ax.legend(unique.values(), unique.keys(), fontsize=7.5, loc="best")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_psi_distribution_plot(
    psi_result: dict,
    batch_id: str,
    generated_at: str,
    model_name: str = "production",
) -> bytes:
    """
    Reference-only companion to the PSI trend: a grouped bar chart of each label's
    prevalence, expected (OOT baseline) vs actual (this batch), so the shift behind
    the PSI score is visible. Labels are ordered by PSI contribution (largest
    movers on top); the actual bar is tinted by that label's GYR.

    psi_result: a compute_psi(...) output with a per_label block.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    items = sorted(psi_result["per_label"].items(), key=lambda kv: kv[1]["psi"], reverse=True)
    labels = [label for label, _ in items]
    expected = [v["expected_rate"] for _, v in items]
    actual = [v["actual_rate"] for _, v in items]
    actual_colors = [GYR_COLORS[v["gyr"]] for _, v in items]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(11, max(4.0, n * 0.42)))
    fig.suptitle(
        f"PSI Distribution (reference) — {model_name} — batch {batch_id}  ({generated_at[:19]} UTC)",
        fontsize=12, fontweight="bold",
    )

    y = np.arange(n)
    bar_h = 0.4
    ax.barh(y - bar_h / 2, expected, height=bar_h, color="#9E9E9E",
            label="Expected (OOT baseline)", zorder=3)
    ax.barh(y + bar_h / 2, actual, height=bar_h, color=actual_colors,
            edgecolor="white", linewidth=0.4, label="Actual (this batch)", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()  # largest PSI contributor at the top
    ax.set_xlabel("Label prevalence (fraction of documents)", fontsize=9)
    ax.set_title("Predicted-label prevalence: expected vs actual (actual tinted by GYR)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, axis="x", alpha=0.2)

    legend_handles = [
        mpatches.Patch(color="#9E9E9E", label="Expected (OOT baseline)"),
        mpatches.Patch(facecolor="white", edgecolor="#777", label="Actual (tinted GREEN/YELLOW/RED by PSI)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_csi_distribution_plot(
    baseline_values: dict,
    production_values: dict,
    csi_result: dict,
    features: list[dict],
    batch_id: str,
    generated_at: str,
    top_n: int | None = None,
) -> bytes:
    """
    Reference companion to the CSI trend: overlaid baseline (OOT) vs production value
    histograms per feature, so the distribution shift behind each CSI is visible.

    top_n=None  -> all features in a compact grid (the full top-50 reference).
    top_n=3     -> just the most globally important features, large (presentation).

    Each panel is titled with the feature name + its CSI, and the production
    histogram is tinted by that feature's GYR. features arrive in global-importance
    order, so features[:top_n] are the top-N.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = features[:top_n] if top_n else features
    n = len(selected)
    ncols = min(n, 3) if top_n else 5
    nrows = math.ceil(n / ncols)
    scope = f"top {n}" if top_n else f"all {n}"

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 2.6), squeeze=False)
    fig.suptitle(
        f"CSI Distribution (reference) — {scope} features — batch {batch_id}  ({generated_at[:19]} UTC)",
        fontsize=12, fontweight="bold",
    )

    for idx, feature in enumerate(selected):
        ax = axes[idx // ncols][idx % ncols]
        lemma = feature["lemma"]
        detail = csi_result["per_feature"].get(lemma, {})
        base = np.asarray(baseline_values.get(lemma, []), dtype=float)
        prod = np.asarray(production_values.get(lemma, []), dtype=float)

        combined = np.concatenate([base, prod]) if base.size + prod.size else np.array([0.0, 1.0])
        lo, hi = float(combined.min()), float(combined.max())
        bins = np.linspace(lo, hi, 21) if hi > lo else np.linspace(lo - 0.5, lo + 0.5, 3)
        actual_color = GYR_COLORS.get(detail.get("gyr", "GREEN"), MODEL_STYLES["production"]["color"])

        base_n = prod_n = None
        if base.size:
            base_n, _, _ = ax.hist(base, bins=bins, density=True, color="#9E9E9E", alpha=0.55, label="Baseline (OOT)")
        if prod.size:
            prod_n, _, _ = ax.hist(prod, bins=bins, density=True, color=actual_color, alpha=0.55, label="Production")

        # The 0-bin (feature absent) dwarfs everything; cap the y-axis to the tallest
        # non-zero bin so the actual-value differences are visible. The 0 bar clips off.
        nonzero_peak = 0.0
        for heights in (base_n, prod_n):
            if heights is not None and len(heights) > 1:
                nonzero_peak = max(nonzero_peak, float(np.max(heights[1:])))
        if nonzero_peak > 0:
            ax.set_ylim(0, nonzero_peak * 1.15)

        ax.set_title(f"{feature['name']}  (CSI={detail.get('csi', 0):.3f})", fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        ax.set_yticks([])

    # blank any unused grid cells
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # one shared legend + shared axis labels (per-panel labels would be too dense)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.supxlabel("Feature value (DCW weight per document; 0 = feature absent)", fontsize=9)
    fig.supylabel("Density (y capped to non-zero bins; 0-bar clipped)", fontsize=9)

    fig.tight_layout(rect=[0.02, 0.03, 1, 0.95])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


# ── Per-feature CSI trend plots ─────────────────────────────────────────────────

# Distinct line colours for the top-N feature comparison.
TOP_FEATURE_COLORS = ["#1565C0", "#6A1B9A", "#C62828", "#2E7D32", "#EF6C00"]


def build_csi_feature_grid_plot(
    feature_history: dict[str, list[dict]],
    features: list[dict],
    batch_id: str,
    generated_at: str,
) -> bytes:
    """
    A grid of mini CSI trends — one panel per top-50 feature — so every feature's
    drift over time is visible at a glance (the per-feature counterpart to the single
    aggregate CSI line on the stability plot). Panels are in global-importance order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    n = len(features)
    ncols = 5
    nrows = math.ceil(n / ncols)
    color = MODEL_STYLES["production"]["color"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.2), squeeze=False, sharex=True)
    fig.suptitle(
        f"CSI per feature (trend) — all {n} features — batch {batch_id}  ({generated_at[:19]} UTC)",
        fontsize=12, fontweight="bold",
    )

    for idx, feature in enumerate(features):
        ax = axes[idx // ncols][idx % ncols]
        for low, high, gyr in STABILITY_BANDS:
            ax.axhspan(low, high, color=GYR_COLORS[gyr], alpha=0.12, zorder=0)

        series = feature_history.get(feature["lemma"], [])
        y_max = PSI_YELLOW * 1.2
        for seg in _segments_by_run_id(series):
            xs = [_point_time(p) for p in seg]
            ys = [p["csi"] for p in seg]
            if ys:
                y_max = max(y_max, max(ys))
            ax.plot(xs, ys, color=color, marker="o", markersize=2.5, linewidth=1.0, zorder=3)

        ax.set_ylim(0, y_max * 1.1)
        ax.set_title(feature["name"], fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    axes[-1][0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=45)
    fig.supylabel("CSI (GYR bands 0.10 / 0.25)", fontsize=9)
    fig.tight_layout(rect=[0.01, 0, 1, 0.96])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_csi_top_features_plot(
    feature_history: dict[str, list[dict]],
    features: list[dict],
    batch_id: str,
    generated_at: str,
    top_n: int = 3,
) -> bytes:
    """
    The top-N most globally-important features' CSI trends overlaid on one chart, so
    they can be compared directly. One coloured line per feature, broken at model
    swaps, over the shared GYR bands. features arrive in global-importance order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    selected = features[:top_n]
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    fig.suptitle(
        f"CSI — top {len(selected)} features by global importance — batch {batch_id}  ({generated_at[:19]} UTC)",
        fontsize=12, fontweight="bold",
    )

    y_max = PSI_YELLOW * 1.2
    for low, high, gyr in STABILITY_BANDS:
        ax.axhspan(low, high, color=GYR_COLORS[gyr], alpha=0.12, zorder=0)

    for index, feature in enumerate(selected):
        color = TOP_FEATURE_COLORS[index % len(TOP_FEATURE_COLORS)]
        series = feature_history.get(feature["lemma"], [])
        labelled = False
        for seg in _segments_by_run_id(series):
            xs = [_point_time(p) for p in seg]
            ys = [p["csi"] for p in seg]
            if ys:
                y_max = max(y_max, max(ys))
            ax.plot(
                xs, ys, color=color, marker="o", markersize=5, linewidth=1.6,
                label=feature["name"] if not labelled else None, zorder=3,
            )
            labelled = True

    ax.set_title("Per-feature CSI vs OOT baseline", fontsize=10, fontweight="bold", loc="left")
    ax.set_ylabel("CSI")
    ax.set_ylim(0, y_max * 1.1)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="best")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def write_bytes(spark, path: str, data: bytes) -> None:
    """Write raw bytes (e.g. a PNG) to a Hadoop/R2 path, overwriting if present."""
    jvm = spark.sparkContext._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    stream = fs.create(hadoop_path, True)
    try:
        stream.write(bytearray(data))
    finally:
        stream.close()
    logger.info("Wrote %d bytes to %s", len(data), path)


# ── Orchestration ───────────────────────────────────────────────────────────

# Performance GYR thresholds. Macro F1: higher is better; Hamming Loss: lower.
MACRO_F1_GREEN, MACRO_F1_YELLOW = 0.65, 0.60
HAMMING_GREEN, HAMMING_YELLOW = 0.15, 0.20


def _classify_higher_is_better(value: float, green: float, yellow: float) -> str:
    if value >= green:
        return "GREEN"
    if value >= yellow:
        return "YELLOW"
    return "RED"


def _safe_plot(spark, path: str, builder) -> None:
    """Build a PNG and write it; log and continue if it fails (monitoring must not
    crash a whole run because one chart couldn't render)."""
    try:
        write_bytes(spark, path, builder())
    except Exception as exc:
        logger.warning("Could not render %s: %s", path, exc)


def _monitor_production(spark, paths: dict, batch_id: str, t0: dict) -> tuple[dict, dict | None, dict | None, dict | None]:
    """
    Compute the production model's metrics for this batch. Returns
    (block, psi_result, csi_result, csi_values) where block is the per-model record
    for metrics.json and the *_result objects feed the distribution plots.
    Each metric family is best-effort: if an upstream table is missing (e.g. the
    inference_features dcw_features map, or empty published_predictions) that family
    is skipped with a warning rather than failing the whole run.
    """
    labels = t0["labels"]
    block: dict = {"run_id": T0_EXP_ID}
    psi_result = csi_result = csi_values = None

    # Performance (needs ground truth → reviewed 10% only)
    try:
        performance = compute_performance(
            load_ground_truth(spark, paths),
            load_batch_predictions(spark, paths, batch_id),
            labels,
        )
        if performance:
            block["macro_f1"] = performance["macro_f1"]
            block["hamming_loss"] = performance["hamming_loss"]
            block["macro_f1_gyr"] = _classify_higher_is_better(performance["macro_f1"], MACRO_F1_GREEN, MACRO_F1_YELLOW)
            block["hamming_loss_gyr"] = _classify_lower_is_better(performance["hamming_loss"], HAMMING_GREEN, HAMMING_YELLOW)
            block["reviewed_count"] = performance["reviewed_count"]
            block["per_label_f1"] = performance["per_label_f1"]
    except Exception as exc:
        logger.warning("Performance metrics unavailable: %s", exc)

    # PSI (predictions only → full batch); baseline = predictions on the train set.
    try:
        psi_result = compute_psi(
            load_baseline_prediction_counts(spark, resolve_train_predictions_path(spark, paths, T0_EXP_ID)),
            load_production_prediction_counts(spark, paths, batch_id),
            labels,
        )
        block["psi"] = psi_result["overall_psi"]
        block["psi_gyr"] = psi_result["overall_gyr"]
        block["psi_per_label"] = psi_result["per_label"]
    except Exception as exc:
        logger.warning("PSI unavailable: %s", exc)

    # CSI (features only → full batch); baseline = training DCW feature values.
    try:
        features = load_global_features(spark, paths, T0_EXP_ID)
        baseline_values = load_csi_baseline_values(spark, paths, features, t0["feature_run_id"])
        production_values = load_csi_production_values(spark, paths, batch_id, features)
        csi_result = compute_csi(baseline_values, production_values, features)
        csi_values = {"features": features, "baseline": baseline_values, "production": production_values}
        block["csi"] = csi_result["overall_csi"]
        block["csi_gyr"] = csi_result["overall_gyr"]
        block["csi_per_feature"] = csi_result["per_feature"]
    except Exception as exc:
        logger.warning("CSI unavailable: %s", exc)

    return block, psi_result, csi_result, csi_values


def run_monitoring(spark, batch_id: str) -> dict:
    """
    Daily entrypoint (runs right after batch inference). Computes performance / PSI /
    CSI for the production model on this batch, writes metrics.json, then rebuilds the
    trend history (now including this batch) and renders the dashboard PNGs — all
    under monitoring/{batch_id}/ on R2.

    If a shadow model is configured (SHADOW_MODEL_PATH), its performance is also
    computed and appears as a second line on the performance plot; otherwise only
    production is tracked.
    """
    schema = load_schema()
    paths = load_paths(schema)
    monitored_at = datetime.now(timezone.utc).isoformat()
    base_dir = f"{paths['monitoring_base']}/{batch_id}"
    logger.info("Monitoring batch %s -> %s", batch_id, base_dir)

    t0 = load_t0_baseline(spark, paths)
    production, psi_result, csi_result, csi_values = _monitor_production(spark, paths, batch_id, t0)

    # Optional shadow model: performance only (a second line on the performance plot).
    shadow = None
    try:
        shadow_performance = load_shadow_performance(spark, paths, batch_id, t0["labels"])
        if shadow_performance:
            shadow = {
                "run_id": "shadow",
                "macro_f1": shadow_performance["macro_f1"],
                "hamming_loss": shadow_performance["hamming_loss"],
                "macro_f1_gyr": _classify_higher_is_better(shadow_performance["macro_f1"], MACRO_F1_GREEN, MACRO_F1_YELLOW),
                "hamming_loss_gyr": _classify_lower_is_better(shadow_performance["hamming_loss"], HAMMING_GREEN, HAMMING_YELLOW),
                "reviewed_count": shadow_performance["reviewed_count"],
                "per_label_f1": shadow_performance["per_label_f1"],
            }
    except Exception as exc:
        logger.warning("Shadow performance unavailable: %s", exc)

    report = {
        "batch_id": batch_id,
        "monitored_at": monitored_at,
        "production": production,
        "shadow": shadow,
    }

    # Persist first, so the history readback for the trend plots includes this batch.
    write_json(spark, f"{base_dir}/metrics.json", report, overwrite=True)

    # Trend plots: production champion (+ shadow line if configured) over time.
    history = load_metric_history(spark, paths)
    # T=0 baselines keyed by run_id (each model version anchors its own segment).
    # NOTE: only the current production model's baseline is loaded here; historical
    # promoted segments won't get a T=0 star until per-run_id baseline loading is added.
    baselines = {T0_EXP_ID: {"macro_f1": t0["macro_f1"], "hamming_loss": t0["hamming_loss"], "date": t0["date"]}}

    _safe_plot(spark, f"{base_dir}/performance.png",
               lambda: build_performance_plot(history, baselines, batch_id, monitored_at))
    _safe_plot(spark, f"{base_dir}/stability.png",
               lambda: build_stability_plot(history, batch_id, monitored_at))
    if psi_result:
        _safe_plot(spark, f"{base_dir}/psi_distribution.png",
                   lambda: build_psi_distribution_plot(psi_result, batch_id, monitored_at))
    if csi_result and csi_values:
        _safe_plot(spark, f"{base_dir}/csi_distribution.png",
                   lambda: build_csi_distribution_plot(
                       csi_values["baseline"], csi_values["production"], csi_result,
                       csi_values["features"], batch_id, monitored_at))
        _safe_plot(spark, f"{base_dir}/csi_distribution_top3.png",
                   lambda: build_csi_distribution_plot(
                       csi_values["baseline"], csi_values["production"], csi_result,
                       csi_values["features"], batch_id, monitored_at, top_n=3))

        # Per-feature CSI trends over time (this batch's per-feature CSI is now in
        # history): one panel per feature, plus the top-3 overlaid for comparison.
        feature_history = load_csi_feature_history(spark, paths)
        if feature_history:
            features = csi_values["features"]
            _safe_plot(spark, f"{base_dir}/csi_feature_trends.png",
                       lambda: build_csi_feature_grid_plot(feature_history, features, batch_id, monitored_at))
            _safe_plot(spark, f"{base_dir}/csi_feature_trends_top3.png",
                       lambda: build_csi_top_features_plot(feature_history, features, batch_id, monitored_at, top_n=3))

    logger.info("Monitoring complete for %s", batch_id)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference monitoring")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from utils.spark_session import create_spark_session

    spark = create_spark_session("batch-inference-monitoring")
    try:
        print(json.dumps(run_monitoring(spark, args.batch_id), indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
