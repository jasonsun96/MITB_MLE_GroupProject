from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from include.inference.model_registry import hadoop_path_exists, read_json

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

# Challenger model trained by the scheduled AutoML job (3-monthly). It is
# evaluated on the same reviewed 10% as production; promotion to champion
# requires Macro F1 to exceed production by >=15% for 1 month.
# No AutoML model exists yet (training is scheduled), so this is None for now.
# When the AutoML job lands, set this to its experiment id (or resolve it from a
# model_registry alias, e.g. get_alias(..., "automl")).
AUTOML_EXP_ID: str | None = None

# How the AutoML challenger's served predictions are distinguished within
# published_predictions once it runs. Placeholder until the AutoML job defines it.
AUTOML_DEPLOYMENT_GROUP = "automl"


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
# Baseline = predicted labels on the production model's OOT holdout (its reference
# scoring); production = predicted labels on the live batch.

# Holdout split that defines the PSI baseline distribution. OOT is the closest
# analogue to future/production data and matches the performance T=0 (holdout_oot).
PSI_BASELINE_SPLIT = "oot"


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
    PSI baseline (expected): the production model's predicted-label prevalence on
    its OOT holdout. reference_path is the model's prediction_delta_path (from its
    metrics JSON), so the baseline is pinned to whatever model is in production.

    Returns {"total": <#oot docs>, "counts": {label: <#docs predicted label>}}.
    """
    from pyspark.sql import functions as F

    if not reference_path:
        raise ValueError("No reference prediction path for the production model (PSI baseline)")

    reference = spark.read.format("delta").load(reference_path)
    oot = reference.filter(F.col("category") == PSI_BASELINE_SPLIT)
    result = _predicted_label_counts(oot)
    logger.info(
        "PSI baseline: %d %s docs from %s, %d distinct predicted labels",
        result["total"], PSI_BASELINE_SPLIT, reference_path, len(result["counts"]),
    )
    return result


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
        # Reference prediction table (holdout predictions) for this model; used as
        # the PSI baseline so the expected distribution is the model's own outputs.
        "prediction_delta_path": metrics.get("prediction_delta_path"),
    }


def _parse_metrics_date(metrics_path: str) -> str | None:
    """Extract YYYY-MM-DD from a prediction_YYYYMMDD.json metrics path."""
    match = re.search(r"prediction_(\d{8})\.json$", metrics_path)
    if not match:
        return None
    digits = match.group(1)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def load_automl_baseline(spark, paths: dict) -> dict | None:
    """
    Load the AutoML challenger's T=0 baseline metrics, mirroring load_t0_baseline.

    PLACEHOLDER: the AutoML job is scheduled and no model exists yet, so this
    returns None until AUTOML_EXP_ID is set. Once wired up, this reads the
    challenger's holdout metrics so its T=0 can be compared against production.
    """
    if AUTOML_EXP_ID is None:
        logger.info("No AutoML model configured (AUTOML_EXP_ID is None); skipping AutoML baseline")
        return None

    metrics_dir = f"{paths['model_bank_base']}/experiments/{AUTOML_EXP_ID}/metrics"
    metrics_path = _latest_prediction_metrics(spark, metrics_dir)
    if metrics_path is None:
        logger.warning("AutoML model %s has no baseline metrics yet at %s", AUTOML_EXP_ID, metrics_dir)
        return None

    logger.info("Loading AutoML baseline metrics from %s", metrics_path)
    metrics = read_json(spark, metrics_path)
    split_metrics = metrics["metrics"][T0_METRICS_SPLIT]
    return {
        "exp_id": AUTOML_EXP_ID,
        "split": T0_METRICS_SPLIT,
        "macro_f1": split_metrics["macro_f1"],
        "hamming_loss": split_metrics["hamming_loss"],
    }


def load_automl_predictions(spark, paths: dict, batch_id: str):
    """
    Load the AutoML challenger's predictions on the reviewed 10% for one batch,
    mirroring load_batch_predictions, so its live Macro F1 can be computed.

    PLACEHOLDER: returns None until the scheduled AutoML job runs and serves
    predictions (expected within published_predictions, distinguished by
    deployment_group=AUTOML_DEPLOYMENT_GROUP, or a dedicated table TBD).
    """
    if AUTOML_EXP_ID is None:
        logger.info("No AutoML model configured (AUTOML_EXP_ID is None); skipping AutoML predictions")
        return None

    from pyspark.sql import functions as F

    predictions = spark.read.format("delta").load(paths["published_predictions"])
    automl_preds = predictions.filter(
        (F.col("batch_id") == batch_id)
        & (F.col("deployment_group") == AUTOML_DEPLOYMENT_GROUP)
    ).select("document_id", "predicted_labels")

    if automl_preds.limit(1).count() == 0:
        logger.warning("No AutoML predictions found for batch %s", batch_id)
        return None
    return automl_preds


# Models whose daily performance is tracked as its own trend line. Production is
# the champion; automl is the latest challenger (the 'automl' alias is repointed
# to a fresh model every 3-month cycle, so this line tracks "latest AutoML").
TRACKED_MODELS = ("production", "automl")


# Per-model metrics carried through the readback into the trend plots. Performance
# metrics (macro_f1, hamming_loss) need ground truth; psi does not, so a point may
# carry some metrics and not others. Each plot filters for the metric it draws.
TRACKED_METRICS = ("macro_f1", "hamming_loss", "psi")


def load_metric_history(spark, paths: dict) -> dict[str, list[dict]]:
    """
    Read back prior runs' metric points from monitoring/{batch_id}/metrics.json so
    each daily run can plot time-series lines (performance and PSI) rather than dots.

    Returns one series per tracked model: {"production": [...], "automl": [...]}.
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

# Per-model line styling. Production is the champion; automl is the latest challenger.
MODEL_STYLES = {
    "production": {"color": "#1565C0", "marker": "o", "linestyle": "-", "label": "Production"},
    "automl": {"color": "#6A1B9A", "marker": "D", "linestyle": "--", "label": "AutoML (latest)"},
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

    series_by_model: {"production": [...], "automl": [...]} time-sorted points,
        each {monitored_at, batch_id, run_id, macro_f1, hamming_loss}. The AutoML
        series is expected pre-filtered to the latest model only.
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


# ── PSI trend plot ─────────────────────────────────────────────────────────────

# GYR zones for the overall PSI (lower is better). Mirrors the performance bands.
PSI_BANDS = [(0.00, PSI_GREEN, "GREEN"), (PSI_GREEN, PSI_YELLOW, "YELLOW"), (PSI_YELLOW, 100.0, "RED")]


def build_psi_plot(
    series_by_model: dict[str, list[dict]],
    batch_id: str,
    generated_at: str,
) -> bytes:
    """
    Render the PSI (prediction stability) time-series as PNG bytes — same style as
    the performance plot but its own figure. One line per model, broken at model
    swaps. No T=0 anchor: PSI is a production-vs-baseline drift score that only
    exists once batches start (PSI≈0 means no shift).

    series_by_model: {"production": [...], "automl": [...]} time-sorted points,
        each carrying run_id and psi (the overall summed PSI for that batch).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    fig.suptitle(
        f"PSI Monitoring — prediction stability — batch {batch_id}  ({generated_at[:19]} UTC)",
        fontsize=12, fontweight="bold",
    )

    y_max = PSI_YELLOW * 1.2  # ensure GYR bands are visible even when PSI is tiny
    for low, high, gyr in PSI_BANDS:
        ax.axhspan(low, high, color=GYR_COLORS[gyr], alpha=0.12, zorder=0)

    for model, raw_series in series_by_model.items():
        style = MODEL_STYLES[model]
        series = [point for point in raw_series if point.get("psi") is not None]
        for seg in _segments_by_run_id(series):
            xs = [_point_time(p) for p in seg]
            ys = [p["psi"] for p in seg]
            if ys:
                y_max = max(y_max, max(ys))
            ax.plot(
                xs, ys,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], markersize=5, linewidth=1.6,
                label=style["label"], zorder=3,
            )

    ax.set_title(
        "Overall PSI (one-bin per label, summed) — production vs OOT baseline",
        fontsize=10, fontweight="bold", loc="left",
    )
    ax.set_ylabel("PSI")
    ax.set_ylim(0, y_max * 1.1)
    ax.grid(True, axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=7.5, loc="best")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

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


def run_monitoring(spark, batch_id: str) -> dict:
    raise NotImplementedError


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
