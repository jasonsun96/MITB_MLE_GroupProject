"""
Spark ML multi-label training for legal document topic classification.

Consumes precomputed Gold feature tables (TF-IDF, log-TF-IDF, DCW, embeddings)
and gold/labels. Does not refit any feature extractor.

Upstream feature jobs (run separately):
  - ngram_processing.py / tfidf_processing.py  → tfidf_features_*
  - DCW job                           → dcw_features_*
  - embeddings job                    → gold/embeddings

Outputs per run_id:
  - gold/model_prediction/{run_id}
  - gold/runs/{run_id}/X_train              (optional, --save-assembled)
  - gold/runs/{run_id}/X_val_test_oot       (optional, --save-assembled)
  - model_bank/runs/{run_id}/models/{label} (binary relevance, one model per label)
  - model_bank/runs/{run_id}/label_mapping.json
  - model_bank/runs/{run_id}/metadata.json
  - model_bank/runs/{run_id}/metrics.json

Assumptions (static layout for now):
  - All input/output paths are defined in schema.yaml and follow a fixed R2 layout.
  - Train/val/test/OOT membership is frozen upstream in feature tables.
  - gold/labels is a static snapshot; category is split metadata only.
  - Embeddings are document-level (full corpus); joined by document_id.
  - Re-running upstream jobs overwrites Gold tables; this script reads the current snapshot.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector, Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType, StringType

_TRAINING_DIR = Path(__file__).resolve().parent
_GOLD_DIR = _TRAINING_DIR.parent / "gold"
_PROJECT_ROOT = _TRAINING_DIR.parents[1]
for _path in (_PROJECT_ROOT, _GOLD_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from gold_io import PROJECT_ROOT, bootstrap_paths, save_bytes_to_path, write_delta
from utils.spark_session import create_spark_session

bootstrap_paths()

logger = logging.getLogger(__name__)

FEATURE_SET_CHOICES = (
    "tfidf",
    "log_tfidf",
    "dcw",
    "embeddings",
    "tfidf_dcw",
    "log_tfidf_dcw",
    "tfidf_embeddings",
    "log_tfidf_embeddings",
    "dcw_embeddings",
    "tfidf_dcw_embeddings",
    "log_tfidf_dcw_embeddings",
    "all",
)
MODEL_TYPE = "random_forest"
SOURCE_LABEL_COL = "label"
EMBEDDING_COL = "embedding"  # column name written by legal_embeddings.py / wiki_embeddings.py
HOLDOUT_SPLITS = ("val", "test", "oot")
MULTILABEL_STRATEGY = "binary_relevance"

SPLIT_COL = "category"
DOCUMENT_ID_COL = "document_id"
FEATURES_COL = "features"
TARGET_LABELS_COL = "target_labels"
PREDICTED_LABELS_COL = "predicted_labels"
EMBEDDING_VECTOR_COL = "embedding_vector"
DCW_VECTOR_COL = "dcw_vector"

RF_PARAMS = {"numTrees": 50, "maxDepth": 10}

LABEL_NORMALIZATION = "lowercase_trim_dedupe"


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_label_name(label: str) -> str:
    return re.sub(r"[^\w.-]", "_", label)[:128]


def _feature_components(feature_set: str) -> dict[str, bool]:
    """Resolve which feature blocks are required. ``all`` = log_tfidf + dcw + embeddings."""
    if feature_set == "all":
        return {"log_tfidf": True, "dcw": True, "embeddings": True, "tfidf": False}

    uses_log_tfidf = feature_set in (
        "log_tfidf",
        "log_tfidf_dcw",
        "log_tfidf_embeddings",
        "log_tfidf_dcw_embeddings",
    )
    uses_tfidf = feature_set in ("tfidf", "tfidf_dcw", "tfidf_embeddings", "tfidf_dcw_embeddings")
    uses_dcw = feature_set in (
        "dcw",
        "tfidf_dcw",
        "log_tfidf_dcw",
        "dcw_embeddings",
        "tfidf_dcw_embeddings",
        "log_tfidf_dcw_embeddings",
    )
    uses_embeddings = feature_set in (
        "embeddings",
        "tfidf_embeddings",
        "log_tfidf_embeddings",
        "dcw_embeddings",
        "tfidf_dcw_embeddings",
        "log_tfidf_dcw_embeddings",
    )
    return {
        "tfidf": uses_tfidf,
        "log_tfidf": uses_log_tfidf,
        "dcw": uses_dcw,
        "embeddings": uses_embeddings,
    }


def load_schema_paths() -> dict[str, str]:
    # Paths assume the fixed Gold / model_bank folder tree in schema.yaml (see module docstring).
    with open(PROJECT_ROOT / "schema.yaml") as f:
        schema = yaml.safe_load(f)

    gold_path = schema["gold"]["path"]
    gold_tables = schema["gold"]["tables"]
    model_bank_path = schema["model_bank"]["path"]
    runs_path = schema["model_bank"].get("runs", {}).get("path", "runs")
    gold_runs_path = gold_tables.get("runs", {}).get("path", "runs")

    def gold_table(name: str) -> str:
        return f"{gold_path}/{gold_tables[name]['path']}"

    return {
        "gold_path": gold_path,
        "gold_runs_base": f"{gold_path}/{gold_runs_path}",
        "tfidf_features_train": gold_table("tfidf_features_train"),
        "tfidf_features_val_test_oot": gold_table("tfidf_features_val_test_oot"),
        "dcw_features_train": gold_table("dcw_features_train"),
        "dcw_features_val_test_oot": gold_table("dcw_features_val_test_oot"),
        "embeddings": gold_table("embeddings"),
        "labels": gold_table("labels"),
        "model_prediction_base": gold_table("model_prediction"),
        "model_bank_runs": f"{model_bank_path}/{runs_path}",
    }


def _read_delta(spark, path: str, name: str) -> DataFrame:
    logger.info("Reading %s from %s", name, path)
    return spark.read.format("delta").load(path)


def _ensure_embedding_vector(df: DataFrame) -> DataFrame:
    dtype = dict(df.dtypes).get(EMBEDDING_COL, "")
    if dtype.startswith("vector") or "Vector" in dtype:
        return df.withColumn(EMBEDDING_VECTOR_COL, F.col(EMBEDDING_COL))

    @udf(VectorUDT())
    def array_to_vector(values: list[float] | None) -> Vector:
        if not values:
            return Vectors.dense([])
        return Vectors.dense([float(v) for v in values])

    logger.info("Converting %s from array to Spark ML vector", EMBEDDING_COL)
    return df.withColumn(EMBEDDING_VECTOR_COL, array_to_vector(F.col(EMBEDDING_COL)))


@udf(ArrayType(StringType()))
def _normalize_labels_udf(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []

    if isinstance(raw_value, (list, tuple)):
        parts = [str(x) for x in raw_value]
    else:
        parts = re.split(r"[,|;]", str(raw_value))

    seen: set[str] = set()
    normalized: list[str] = []
    for part in parts:
        label = part.strip().lower()
        if label and label not in seen:
            seen.add(label)
            normalized.append(label)
    return normalized


def normalize_target_labels(df: DataFrame, source_col: str) -> DataFrame:
    """Build target_labels: array<string>, lowercase, trimmed, deduplicated."""
    return (
        df.withColumn(TARGET_LABELS_COL, _normalize_labels_udf(F.col(source_col)))
        .filter(F.size(F.col(TARGET_LABELS_COL)) > 0)
    )


def _labels_scaffold(labels: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Document-id skeletons for train vs holdout when TF-IDF columns are not required."""
    train_ids = (
        labels.filter(F.col(SPLIT_COL) == "train")
        .select(DOCUMENT_ID_COL)
        .dropDuplicates([DOCUMENT_ID_COL])
    )
    holdout_ids = (
        labels.filter(F.col(SPLIT_COL) != "train")
        .select(DOCUMENT_ID_COL)
        .dropDuplicates([DOCUMENT_ID_COL])
    )
    return train_ids, holdout_ids


def load_features(
    spark,
    paths: dict[str, str],
    feature_set: str,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame | None]:
    components = _feature_components(feature_set)
    train_df: DataFrame | None = None
    holdout_df: DataFrame | None = None
    embeddings_df: DataFrame | None = None

    labels_raw = _read_delta(spark, paths["labels"], "labels")
    missing = {DOCUMENT_ID_COL, SPLIT_COL, SOURCE_LABEL_COL} - set(labels_raw.columns)
    if missing:
        raise ValueError(f"gold/labels missing required column(s): {sorted(missing)}")

    labels = (
        labels_raw.select(
            F.col(DOCUMENT_ID_COL),
            F.col(SPLIT_COL),
            F.col(SOURCE_LABEL_COL).alias("_raw_target"),
        )
        .dropDuplicates([DOCUMENT_ID_COL])
        .transform(lambda df: normalize_target_labels(df, "_raw_target"))
        .drop("_raw_target")
    )

    if components["tfidf"] or components["log_tfidf"]:
        train_df = _read_delta(spark, paths["tfidf_features_train"], "train TF-IDF")
        holdout_df = _read_delta(spark, paths["tfidf_features_val_test_oot"], "holdout TF-IDF")
        if DOCUMENT_ID_COL not in train_df.columns:
            raise ValueError("tfidf_features_train missing document_id")

    if components["dcw"]:
        dcw_train = _read_delta(spark, paths["dcw_features_train"], "train DCW")
        dcw_holdout = _read_delta(spark, paths["dcw_features_val_test_oot"], "holdout DCW")
        dcw_cols = sorted(c for c in dcw_train.columns if c.startswith("dcw_"))
        if not dcw_cols:
            raise ValueError("No dcw_* columns found in dcw_features_train")
        dcw_select = [DOCUMENT_ID_COL, *dcw_cols]

        if components["tfidf"] or components["log_tfidf"]:
            assert train_df is not None and holdout_df is not None
            train_df = train_df.join(dcw_train.select(*dcw_select), on=DOCUMENT_ID_COL, how="inner")
            holdout_df = holdout_df.join(
                dcw_holdout.select(*dcw_select), on=DOCUMENT_ID_COL, how="inner"
            )
        else:
            train_df = dcw_train.select(*dcw_select)
            holdout_df = dcw_holdout.select(*dcw_select)

    if train_df is None and components["embeddings"]:
        train_df, holdout_df = _labels_scaffold(labels)

    if components["embeddings"]:
        embeddings_raw = _read_delta(spark, paths["embeddings"], "embeddings")
        if EMBEDDING_COL not in embeddings_raw.columns:
            raise ValueError(
                f"gold/embeddings must contain column {EMBEDDING_COL!r}. "
                f"Found: {sorted(embeddings_raw.columns)}"
            )
        embeddings_df = (
            embeddings_raw.select(DOCUMENT_ID_COL, EMBEDDING_COL)
            .dropDuplicates([DOCUMENT_ID_COL])
            .transform(_ensure_embedding_vector)
            .select(DOCUMENT_ID_COL, EMBEDDING_VECTOR_COL)
        )

    if train_df is None or holdout_df is None:
        raise ValueError(f"Could not load features for feature_set={feature_set!r}")

    logger.info(
        "Loaded feature tables for feature_set=%s (joins and limits applied later)",
        feature_set,
    )
    return train_df, holdout_df, labels, embeddings_df


def prepare_training_data(
    train_features: DataFrame,
    holdout_features: DataFrame,
    labels: DataFrame,
    embeddings_df: DataFrame | None,
) -> tuple[DataFrame, DataFrame]:
    label_subset = labels.select(DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL)

    train_df = train_features.join(label_subset, on=DOCUMENT_ID_COL, how="inner")
    holdout_df = holdout_features.join(label_subset, on=DOCUMENT_ID_COL, how="inner")

    if embeddings_df is not None:
        train_df = train_df.join(embeddings_df, on=DOCUMENT_ID_COL, how="inner")
        holdout_df = holdout_df.join(embeddings_df, on=DOCUMENT_ID_COL, how="inner")

    n_train = train_df.count()
    n_holdout = holdout_df.count()
    if n_train == 0:
        raise ValueError("No training rows after joining features with labels")

    logger.info(
        "Prepared data: %s train documents, %s holdout documents",
        f"{n_train:,}",
        f"{n_holdout:,}",
    )
    return train_df, holdout_df


def build_feature_column(df: DataFrame, feature_set: str) -> DataFrame:
    components = _feature_components(feature_set)
    vector_cols: list[str] = []

    if components["tfidf"]:
        if "tfidf" not in df.columns:
            raise ValueError("tfidf column missing from feature table")
        vector_cols.append("tfidf")

    if components["log_tfidf"]:
        if "log_tfidf" not in df.columns:
            raise ValueError("log_tfidf column missing from feature table")
        vector_cols.append("log_tfidf")

    if components["dcw"]:
        dcw_cols = sorted(c for c in df.columns if c.startswith("dcw_"))
        if not dcw_cols:
            raise ValueError(f"No dcw_* columns found for feature_set={feature_set!r}")
        df = VectorAssembler(inputCols=dcw_cols, outputCol=DCW_VECTOR_COL).transform(df)
        vector_cols.append(DCW_VECTOR_COL)

    if components["embeddings"]:
        if EMBEDDING_VECTOR_COL not in df.columns:
            raise ValueError("embedding_vector missing; join embeddings before build_feature_column")
        vector_cols.append(EMBEDDING_VECTOR_COL)

    if not vector_cols:
        raise ValueError(f"No feature vectors resolved for feature_set={feature_set!r}")

    if len(vector_cols) == 1:
        return df.withColumn(FEATURES_COL, F.col(vector_cols[0]))

    return VectorAssembler(inputCols=vector_cols, outputCol=FEATURES_COL).transform(df)


def _collect_training_labels(train_df: DataFrame, max_labels: int | None) -> list[str]:
    label_counts = (
        train_df.select(F.explode(F.col(TARGET_LABELS_COL)).alias("lbl"))
        .groupBy("lbl")
        .count()
        .orderBy(F.desc("count"), "lbl")
    )
    rows = label_counts.collect()
    labels = [row.lbl for row in rows]
    if max_labels is not None:
        labels = labels[:max_labels]
    if not labels:
        raise ValueError("No training labels found after normalization")
    logger.info("Training binary relevance for %s unique labels", f"{len(labels):,}")
    return labels


def _build_binary_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        featuresCol=FEATURES_COL,
        labelCol="binary_label",
        **RF_PARAMS,
    )


@udf(DoubleType())
def _prob_positive(probability: Vector | None) -> float:
    if probability is None:
        return 0.0
    values = probability.toArray()
    return float(values[1]) if len(values) > 1 else float(values[0])


def train_multilabel_model(
    train_df: DataFrame,
    max_labels: int | None,
) -> tuple[dict[str, Any], list[str]]:
    logger.warning(
        "RandomForestClassifier may be slow or memory-heavy on high-dimensional sparse text features."
    )

    label_list = _collect_training_labels(train_df, max_labels)
    models: dict[str, Any] = {}

    for label in label_list:
        safe = _safe_label_name(label)
        binary_train = train_df.withColumn(
            "binary_label",
            F.when(F.array_contains(F.col(TARGET_LABELS_COL), label), 1.0).otherwise(0.0),
        )
        classifier = _build_binary_classifier()
        models[label] = classifier.fit(binary_train)
        logger.debug("Trained binary model for label=%s", label)

    logger.info(
        "Binary relevance training complete: %s per-label models",
        f"{len(models):,}",
    )
    return models, label_list


def predict_multilabel(
    df: DataFrame,
    models: dict[str, Any],
    label_list: list[str],
    threshold: float,
) -> DataFrame:
    if df.limit(1).count() == 0:
        return df.withColumn(PREDICTED_LABELS_COL, F.array().cast(ArrayType(StringType())))

    prob_cols: list[str] = []
    scored = df

    for label in label_list:
        safe = _safe_label_name(label)
        prob_col = f"_prob_{safe}"
        transformed = (
            models[label]
            .transform(scored)
            .select(DOCUMENT_ID_COL, F.col("probability").alias("_prob_vector"))
            .withColumn(prob_col, _prob_positive(F.col("_prob_vector")))
            .drop("_prob_vector")
        )
        scored = scored.join(transformed, on=DOCUMENT_ID_COL, how="left")
        prob_cols.append(prob_col)

    threshold_exprs = [
        F.when(F.col(col) >= F.lit(threshold), F.lit(label))
        for label, col in zip(label_list, prob_cols)
    ]
    scored = scored.withColumn(
        PREDICTED_LABELS_COL,
        F.array_compact(F.array(*threshold_exprs)),
    )
    return scored.drop(*prob_cols)


def _empty_metrics() -> dict[str, float]:
    nan = float("nan")
    return {
        "documents": 0,
        "exact_match_ratio": nan,
        "micro_precision": nan,
        "micro_recall": nan,
        "micro_f1": nan,
        "macro_precision": nan,
        "macro_recall": nan,
        "macro_f1": nan,
        "hamming_loss": nan,
    }


def _compute_multilabel_metrics(df: DataFrame, label_universe: list[str]) -> dict[str, float]:
    if df.limit(1).count() == 0:
        return _empty_metrics()

    scored = (
        df.withColumn("_tp", F.size(F.array_intersect(F.col(TARGET_LABELS_COL), F.col(PREDICTED_LABELS_COL))))
        .withColumn("_pred_n", F.size(F.col(PREDICTED_LABELS_COL)))
        .withColumn("_true_n", F.size(F.col(TARGET_LABELS_COL)))
        .withColumn("_fp", F.col("_pred_n") - F.col("_tp"))
        .withColumn("_fn", F.col("_true_n") - F.col("_tp"))
        .withColumn(
            "_exact",
            F.when(
                F.size(F.array_except(F.col(TARGET_LABELS_COL), F.col(PREDICTED_LABELS_COL))) == 0,
                F.when(
                    F.size(F.array_except(F.col(PREDICTED_LABELS_COL), F.col(TARGET_LABELS_COL))) == 0,
                    1,
                ).otherwise(0),
            ).otherwise(0),
        )
        .withColumn(
            "_hamming",
            (F.col("_fp") + F.col("_fn")) / F.lit(max(len(label_universe), 1)),
        )
    )

    micro = scored.agg(
        F.sum("_tp").alias("tp"),
        F.sum("_fp").alias("fp"),
        F.sum("_fn").alias("fn"),
        F.sum("_exact").alias("exact"),
        F.count("*").alias("n"),
        F.avg("_hamming").alias("hamming"),
    ).collect()[0]

    tp, fp, fn = float(micro.tp or 0), float(micro.fp or 0), float(micro.fn or 0)
    n_docs = int(micro.n or 0)
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    per_label_ps: list[float] = []
    per_label_rs: list[float] = []
    per_label_f1s: list[float] = []

    for label in label_universe:
        lbl_df = scored.withColumn(
            "_true", F.array_contains(F.col(TARGET_LABELS_COL), F.lit(label)).cast("double")
        ).withColumn(
            "_pred", F.array_contains(F.col(PREDICTED_LABELS_COL), F.lit(label)).cast("double")
        )
        stats = lbl_df.agg(
            F.sum(F.when((F.col("_true") == 1) & (F.col("_pred") == 1), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("_true") == 0) & (F.col("_pred") == 1), 1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col("_true") == 1) & (F.col("_pred") == 0), 1).otherwise(0)).alias("fn"),
        ).collect()[0]
        ltp, lfp, lfn = float(stats.tp or 0), float(stats.fp or 0), float(stats.fn or 0)
        lp = ltp / (ltp + lfp) if (ltp + lfp) > 0 else 0.0
        lr = ltp / (ltp + lfn) if (ltp + lfn) > 0 else 0.0
        lf1 = (2 * lp * lr / (lp + lr)) if (lp + lr) > 0 else 0.0
        per_label_ps.append(lp)
        per_label_rs.append(lr)
        per_label_f1s.append(lf1)

    macro_p = sum(per_label_ps) / len(per_label_ps) if per_label_ps else 0.0
    macro_r = sum(per_label_rs) / len(per_label_rs) if per_label_rs else 0.0
    macro_f1 = sum(per_label_f1s) / len(per_label_f1s) if per_label_f1s else 0.0

    return {
        "documents": n_docs,
        "exact_match_ratio": float(micro.exact or 0) / n_docs if n_docs else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "hamming_loss": float(micro.hamming or 0),
    }


def evaluate_multilabel(
    holdout_df: DataFrame,
    label_universe: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {
        "holdout_overall": _compute_multilabel_metrics(holdout_df, label_universe),
    }
    for split in HOLDOUT_SPLITS:
        split_df = holdout_df.filter(F.col(SPLIT_COL) == split)
        key = f"holdout_{split}"
        metrics[key] = _compute_multilabel_metrics(split_df, label_universe)
        logger.info(
            "Metrics [%s]: micro_f1=%.4f exact_match=%.4f docs=%s",
            split,
            metrics[key]["micro_f1"],
            metrics[key]["exact_match_ratio"],
            metrics[key]["documents"],
        )

    overall = metrics["holdout_overall"]
    logger.info(
        "Metrics [holdout_overall]: micro_f1=%.4f exact_match=%.4f docs=%s",
        overall["micro_f1"],
        overall["exact_match_ratio"],
        overall["documents"],
    )
    return metrics


def _split_row_counts(holdout_df: DataFrame) -> dict[str, int]:
    counts = {"holdout": holdout_df.count()}
    for split in HOLDOUT_SPLITS:
        counts[split] = holdout_df.filter(F.col(SPLIT_COL) == split).count()
    return counts


def _log_metric_block(title: str, block: dict[str, float]) -> None:
    logger.info(
        "%s: docs=%s exact_match=%.4f micro_f1=%.4f macro_f1=%.4f hamming_loss=%.4f",
        title,
        block.get("documents", 0),
        block.get("exact_match_ratio", float("nan")),
        block.get("micro_f1", float("nan")),
        block.get("macro_f1", float("nan")),
        block.get("hamming_loss", float("nan")),
    )
    logger.info(
        "%s detail: micro_p=%.4f micro_r=%.4f macro_p=%.4f macro_r=%.4f",
        title,
        block.get("micro_precision", float("nan")),
        block.get("micro_recall", float("nan")),
        block.get("macro_precision", float("nan")),
        block.get("macro_recall", float("nan")),
    )


def print_dry_run_summary(
    run_id: str,
    args: argparse.Namespace,
    components: dict[str, bool],
    train_df: DataFrame,
    holdout_df: DataFrame,
    label_list: list[str],
    metrics: dict[str, dict[str, float]],
    predictions: DataFrame,
) -> None:
    logger.info("DRY RUN enabled: skipping all writes to R2/S3A/model_bank.")

    n_train = train_df.count()
    n_holdout = holdout_df.count()
    labels_preview = label_list[:20]
    labels_suffix = f" ... (+{len(label_list) - 20} more)" if len(label_list) > 20 else ""

    logger.info("=== Dry-run summary ===")
    logger.info("run_id: %s", run_id)
    logger.info("model: %s", MODEL_TYPE)
    logger.info("feature_set: %s", args.feature_set)
    logger.info(
        "features: tfidf=%s log_tfidf=%s dcw=%s embeddings=%s",
        components["tfidf"],
        components["log_tfidf"],
        components["dcw"],
        components["embeddings"],
    )
    logger.info("train rows: %s | holdout rows: %s", f"{n_train:,}", f"{n_holdout:,}")
    logger.info("labels trained: %s", f"{len(label_list):,}")
    logger.info("label list (up to 20): %s%s", labels_preview, labels_suffix)
    logger.info("multilabel_threshold: %s", args.multilabel_threshold)
    if args.max_labels is not None:
        logger.info("max_labels: %s", args.max_labels)

    for key, title in (
        ("holdout_overall", "holdout_overall"),
        ("holdout_val", "val"),
        ("holdout_test", "test"),
        ("holdout_oot", "oot"),
    ):
        _log_metric_block(title, metrics.get(key, _empty_metrics()))

    if predictions.limit(1).count() == 0:
        logger.warning("Dry-run: holdout predictions are empty; skipping sample output")
        return

    logger.info("Sample predictions (up to 10 rows):")
    predictions.select(
        DOCUMENT_ID_COL,
        SPLIT_COL,
        TARGET_LABELS_COL,
        PREDICTED_LABELS_COL,
    ).show(10, truncate=False)


def save_outputs(
    spark,
    run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
    components: dict[str, bool],
    train_df: DataFrame,
    holdout_df: DataFrame,
    models: dict[str, Any],
    label_list: list[str],
    metrics: dict[str, dict[str, float]],
    predictions: DataFrame,
    hyperparameters: dict[str, Any],
) -> dict[str, str]:
    run_root = f"{paths['model_bank_runs']}/{run_id}"
    models_dir = f"{run_root}/models"
    metadata_path = f"{run_root}/metadata.json"
    metrics_path = f"{run_root}/metrics.json"
    label_mapping_path = f"{run_root}/label_mapping.json"
    prediction_path = f"{paths['model_prediction_base']}/{run_id}"

    model_paths: dict[str, str] = {}
    for label, model in models.items():
        safe = _safe_label_name(label)
        model_path = f"{models_dir}/{safe}"
        model.write().overwrite().save(model_path)
        model_paths[label] = model_path

    label_mapping = {
        "labels": label_list,
        "safe_names": {_safe_label_name(lbl): lbl for lbl in label_list},
        "model_paths": model_paths,
        "num_labels": len(label_list),
        "label_normalization": LABEL_NORMALIZATION,
    }
    save_bytes_to_path(label_mapping_path, json.dumps(label_mapping, indent=2).encode("utf-8"), spark)

    assembled_paths: dict[str, str] = {}
    if args.save_assembled:
        assembled_cols = [DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL, FEATURES_COL]
        x_train_path = f"{paths['gold_runs_base']}/{run_id}/X_train"
        x_holdout_path = f"{paths['gold_runs_base']}/{run_id}/X_val_test_oot"
        write_delta(train_df.select(*assembled_cols), x_train_path)
        write_delta(holdout_df.select(*assembled_cols), x_holdout_path)
        assembled_paths = {"X_train": x_train_path, "X_val_test_oot": x_holdout_path}
        logger.info("Saved assembled X_train to %s", x_train_path)
        logger.info("Saved assembled X_val_test_oot to %s", x_holdout_path)

    holdout_counts = _split_row_counts(holdout_df)
    n_train = train_df.count()

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": MODEL_TYPE,
        "feature_set": args.feature_set,
        "uses_tfidf": components["tfidf"],
        "uses_log_tfidf": components["log_tfidf"],
        "uses_dcw": components["dcw"],
        "uses_embeddings": components["embeddings"],
        "multilabel_strategy": MULTILABEL_STRATEGY,
        "multilabel_threshold": args.multilabel_threshold,
        "max_labels": args.max_labels,
        "label_normalization": LABEL_NORMALIZATION,
        "input_feature_paths": {
            "tfidf_features_train": paths["tfidf_features_train"],
            "tfidf_features_val_test_oot": paths["tfidf_features_val_test_oot"],
            "dcw_features_train": paths["dcw_features_train"],
            "dcw_features_val_test_oot": paths["dcw_features_val_test_oot"],
            "embeddings": paths["embeddings"],
        },
        "labels_path": paths["labels"],
        "prediction_output_path": prediction_path,
        "models_directory": models_dir,
        "label_mapping_path": label_mapping_path,
        "assembled_dataset_paths": assembled_paths or None,
        "hyperparameters": hyperparameters,
        "row_counts": {
            "train_documents": n_train,
            "holdout_documents": holdout_counts["holdout"],
            "val_documents": holdout_counts["val"],
            "test_documents": holdout_counts["test"],
            "oot_documents": holdout_counts["oot"],
        },
        "num_unique_labels": len(label_list),
        "split_column": SPLIT_COL,
        "notes": (
            "TF-IDF, DCW, POS, and embeddings were precomputed in upstream Gold jobs "
            "and were not refit in this training script. "
            "Multi-label training uses binary relevance (one Random Forest per label). "
            "Predicted labels are selected with --multilabel-threshold."
        ),
    }
    save_bytes_to_path(metadata_path, json.dumps(metadata, indent=2).encode("utf-8"), spark)
    save_bytes_to_path(metrics_path, json.dumps(metrics, indent=2).encode("utf-8"), spark)

    prediction_ts = datetime.now(timezone.utc).isoformat()
    if predictions.limit(1).count() > 0:
        pred_out = predictions.select(
            F.lit(run_id).alias("run_id"),
            F.col(DOCUMENT_ID_COL),
            F.col(SPLIT_COL),
            F.col(TARGET_LABELS_COL),
            F.col(PREDICTED_LABELS_COL),
            F.lit(prediction_ts).alias("prediction_ts"),
        )
        write_delta(pred_out, prediction_path)
    else:
        logger.warning("No holdout predictions to write for run_id=%s", run_id)

    logger.info("Saved per-label models under %s", models_dir)
    logger.info("Saved predictions to %s", prediction_path)
    logger.info("Saved metadata to %s", metadata_path)
    logger.info("Saved metrics to %s", metrics_path)

    return {
        "models_directory": models_dir,
        "prediction_path": prediction_path,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a multi-label Spark ML classifier on precomputed Gold features"
    )
    parser.add_argument("--run-id", default=None, help="Run identifier (default: UTC timestamp)")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default="log_tfidf",
        help="Feature columns to use (default: log_tfidf; all = log_tfidf + dcw + embeddings)",
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
        "--save-assembled",
        action="store_true",
        help="Save document-level X_train and X_val_test_oot under gold/runs/{run_id}/",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_id = args.run_id or default_run_id()
    paths = load_schema_paths()
    components = _feature_components(args.feature_set)

    logger.info("Run ID: %s", run_id)
    logger.info("Model: %s | Feature set: %s", MODEL_TYPE, args.feature_set)
    if args.dry_run:
        logger.info("DRY RUN enabled: skipping all writes to R2/S3A/model_bank.")
        if args.save_assembled:
            logger.warning("--save-assembled is ignored when --dry-run is set")

    spark = create_spark_session("gold-model-training")

    train_features, holdout_features, labels, embeddings_df = load_features(
        spark, paths, args.feature_set
    )
    train_df, holdout_df = prepare_training_data(
        train_features, holdout_features, labels, embeddings_df
    )

    if args.limit:
        train_df = train_df.limit(args.limit)
        holdout_df = holdout_df.limit(args.limit)
        logger.info("Smoke test: limited to %s rows per split after joins", f"{args.limit:,}")

    train_df = build_feature_column(train_df, args.feature_set)
    holdout_df = build_feature_column(holdout_df, args.feature_set)

    models, label_list = train_multilabel_model(train_df, args.max_labels)
    predictions = predict_multilabel(holdout_df, models, label_list, args.multilabel_threshold)
    metrics = evaluate_multilabel(predictions, label_list)

    if args.dry_run:
        print_dry_run_summary(
            run_id,
            args,
            components,
            train_df,
            holdout_df,
            label_list,
            metrics,
            predictions,
        )
    else:
        hyperparameters = {
            **RF_PARAMS,
            "multilabel_threshold": args.multilabel_threshold,
            "max_labels": args.max_labels,
        }
        save_outputs(
            spark,
            run_id,
            paths,
            args,
            components,
            train_df,
            holdout_df,
            models,
            label_list,
            metrics,
            predictions,
            hyperparameters,
        )

    logger.info("Model training complete for run_id=%s", run_id)


if __name__ == "__main__":
    main()
