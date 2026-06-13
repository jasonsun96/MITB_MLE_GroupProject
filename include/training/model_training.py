"""
Spark ML multi-label training for legal document topic classification.

Consumes precomputed Gold feature tables (TF-IDF, log-TF-IDF, DCW, embeddings)
and gold/labels. Does not refit any feature extractor.

Upstream feature jobs (run separately):
  - ngram_processing.py / tfidf_processing.py  → tfidf_features_*
  - DCW job                           → dcw_features_*
  - embeddings job                    → gold/embeddings

Outputs per training run_id (feature tables read from --feature-run-id):
  - gold/runs/{feature_run_id}/X_train  (+ X_val_test_oot when not --train-only)
    Use --x-run-id to store assembled X under a different gold run without overwriting.
  - gold/model_predictions/...            (skipped with --train-only; run inference later)
  - model_bank/runs/{run_id}/model/{model}_{date}.pkl
  - model_bank/runs/{run_id}/model/per_label/{label}/  (Spark ML binary models)

Use --feature-set tfidf_dcw_embeddings (default) for tfidf + dcw + embeddings.
Use --train-only to fit and save models on train only; defer val/test/oot scoring.
Use --predict-only to load saved models and evaluate val/test/oot without retraining.
Split holdout work with --predict-stage:
  features  → assemble and save gold/runs/{id}/X_val_test_oot only
  predict   → score from saved X and checkpoint prediction Delta
  metrics   → compute metrics from checkpointed predictions
  all       → run features, predict, and metrics in one job (default)

Assumptions (static layout for now):
  - All input/output paths are defined in schema.yaml and follow a fixed R2 layout.
  - Train/val/test/OOT membership is frozen upstream in feature tables.
  - gold/labels is a static snapshot; category is split metadata only.
  - Embeddings are document-level (full corpus); joined by document_id.
  - Re-running upstream jobs overwrites Gold tables; this script reads the current snapshot.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.ml.classification import (
    LogisticRegression,
    LogisticRegressionModel,
    RandomForestClassifier,
    RandomForestClassificationModel,
)
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

from gold_io import bootstrap_paths, load_bytes_from_path, load_pickle, save_json, save_pickle, write_delta
from run_paths import (
    gold_run_table_path,
    load_schema,
    model_bank_model_manifest_path,
    model_bank_per_label_models_dir,
    model_bank_run_root,
    normalize_prediction_suffix,
    prediction_batch_name,
    prediction_delta_path,
    prediction_manifest_path,
    prediction_manifest_path_for_batch,
    resolve_feature_run_paths,
)
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
MODEL_TYPE_CHOICES = ("random_forest", "logistic_regression")
DEFAULT_MODEL_TYPE = "random_forest"
MODEL_TYPE = DEFAULT_MODEL_TYPE  # backward-compatible alias
SOURCE_LABEL_COL = "label"
SOURCE_LABEL_FALLBACK_COL = "labels"  # gold/label_store column name on R2
EMBEDDING_COL = "embedding"  # column name written by legal_embeddings.py / wiki_embeddings.py
HOLDOUT_SPLITS = ("val", "test", "oot")
PREDICT_STAGES = ("features", "predict", "metrics", "all")
MULTILABEL_STRATEGY = "binary_relevance"

SPLIT_COL = "category"
DOCUMENT_ID_COL = "document_id"
FEATURES_COL = "features"
TARGET_LABELS_COL = "target_labels"
PREDICTED_LABELS_COL = "predicted_labels"
PROB_COL_PREFIX = "prob_"
EMBEDDING_VECTOR_COL = "embedding_vector"
DCW_FEATURES_COL = "dcw_features"  # map<string,double> from domain_concept_weight.py
DCW_VECTOR_COL = "dcw_vector"

RF_PARAMS = {"numTrees": 50, "maxDepth": 10}
LR_PARAMS = {"maxIter": 100, "regParam": 0.0, "elasticNetParam": 0.0}

LABEL_NORMALIZATION = "lowercase_trim_dedupe"


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_label_name(label: str) -> str:
    return re.sub(r"[^\w.-]", "_", label)[:128]


def _prob_column_name(label: str) -> str:
    return f"{PROB_COL_PREFIX}{_safe_label_name(label)}"


def prob_columns_in_df(df: DataFrame) -> list[str]:
    return sorted(col for col in df.columns if col.startswith(PROB_COL_PREFIX))


def label_prob_column_map(label_list: list[str]) -> dict[str, str]:
    return {label: _prob_column_name(label) for label in label_list}


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


def load_schema_paths(feature_run_id: str, x_run_id: str | None = None) -> dict[str, str]:
    """Resolve gold paths: read TF-IDF/DCW from feature_run_id; X matrices from x_run_id."""
    paths = resolve_feature_run_paths(feature_run_id)
    assembled_id = x_run_id or feature_run_id
    if assembled_id != feature_run_id:
        paths = {
            **paths,
            "assembled_run_id": assembled_id,
            "X_train": gold_run_table_path(assembled_id, "X_train"),
            "X_val_test_oot": gold_run_table_path(assembled_id, "X_val_test_oot"),
            "X_unlabelled": gold_run_table_path(assembled_id, "X_unlabelled"),
        }
    return {
        **paths,
        "tfidf_features_train": paths["tfidf_train"],
        "tfidf_features_val_test_oot": paths["tfidf_val_test_oot"],
        "dcw_features_train": paths["dcw_train"],
        "dcw_features_val_test_oot": paths["dcw_val_test_oot"],
    }


def _read_delta(spark, path: str, name: str) -> DataFrame:
    logger.info("Reading %s from %s", name, path)
    return spark.read.format("delta").load(path)


def load_dcw_vocab(spark, paths: dict[str, str]) -> list[str]:
    """Frozen lemma order from dcw_score (same vocabulary as dcw_features maps)."""
    score_path = paths["dcw_score_path"]
    logger.info("Loading DCW vocabulary from %s", score_path)
    vocab = [row.lemma for row in spark.read.format("delta").load(score_path).select("lemma").orderBy("lemma").collect()]
    if not vocab:
        raise ValueError(f"No lemmas found in DCW score table at {score_path}")
    logger.info("DCW vocabulary size: %s", f"{len(vocab):,}")
    return vocab


def _map_to_dcw_vector_udf(lemma_to_idx: dict[str, int], vocab_size: int):
    @udf(VectorUDT())
    def _convert(dcw_map: dict[str, float] | None) -> Vector:
        if not dcw_map:
            return Vectors.sparse(vocab_size, [], [])
        pairs: list[tuple[int, float]] = []
        for lemma, val in dcw_map.items():
            idx = lemma_to_idx.get(lemma)
            if idx is not None:
                pairs.append((idx, float(val)))
        if not pairs:
            return Vectors.sparse(vocab_size, [], [])
        pairs.sort(key=lambda x: x[0])
        return Vectors.sparse(vocab_size, [p[0] for p in pairs], [p[1] for p in pairs])

    return _convert


def add_dcw_vector_column(df: DataFrame, lemma_vocab: list[str]) -> DataFrame:
    """Convert sparse dcw_features map to a fixed-size Spark ML vector."""
    if DCW_FEATURES_COL not in df.columns:
        raise ValueError(f"{DCW_FEATURES_COL!r} column missing from DCW feature table")
    lemma_to_idx = {lemma: idx for idx, lemma in enumerate(lemma_vocab)}
    udf_fn = _map_to_dcw_vector_udf(lemma_to_idx, len(lemma_vocab))
    return df.withColumn(DCW_VECTOR_COL, udf_fn(F.col(DCW_FEATURES_COL)))


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
    include_holdout: bool = True,
) -> tuple[DataFrame, DataFrame | None, DataFrame, DataFrame | None]:
    components = _feature_components(feature_set)
    train_df: DataFrame | None = None
    holdout_df: DataFrame | None = None
    embeddings_df: DataFrame | None = None

    labels_raw = _read_delta(spark, paths["labels"], "labels")
    missing = {DOCUMENT_ID_COL, SPLIT_COL} - set(labels_raw.columns)
    if missing:
        raise ValueError(f"gold/labels missing required column(s): {sorted(missing)}")

    if SOURCE_LABEL_COL in labels_raw.columns:
        source_label_col = SOURCE_LABEL_COL
    elif SOURCE_LABEL_FALLBACK_COL in labels_raw.columns:
        source_label_col = SOURCE_LABEL_FALLBACK_COL
    else:
        raise ValueError(
            f"gold/labels missing a label text column; expected {SOURCE_LABEL_COL!r} "
            f"or {SOURCE_LABEL_FALLBACK_COL!r}"
        )

    labels = (
        labels_raw.select(
            F.col(DOCUMENT_ID_COL),
            F.col(SPLIT_COL),
            F.col(source_label_col).alias("_raw_target"),
        )
        .dropDuplicates([DOCUMENT_ID_COL])
        .transform(lambda df: normalize_target_labels(df, "_raw_target"))
        .drop("_raw_target")
    )

    if components["tfidf"] or components["log_tfidf"]:
        train_df = _read_delta(spark, paths["tfidf_features_train"], "train TF-IDF")
        if DOCUMENT_ID_COL not in train_df.columns:
            raise ValueError("tfidf_features_train missing document_id")
        if include_holdout:
            holdout_df = _read_delta(spark, paths["tfidf_features_val_test_oot"], "holdout TF-IDF")

    if components["dcw"]:
        dcw_train = _read_delta(spark, paths["dcw_features_train"], "train DCW")
        if DCW_FEATURES_COL not in dcw_train.columns:
            raise ValueError(
                f"dcw_train missing {DCW_FEATURES_COL!r}; expected map column from domain_concept_weight.py"
            )
        dcw_select = [DOCUMENT_ID_COL, DCW_FEATURES_COL]

        if components["tfidf"] or components["log_tfidf"]:
            assert train_df is not None
            train_df = train_df.join(dcw_train.select(*dcw_select), on=DOCUMENT_ID_COL, how="inner")
            if include_holdout:
                assert holdout_df is not None
                dcw_holdout = _read_delta(spark, paths["dcw_features_val_test_oot"], "holdout DCW")
                holdout_df = holdout_df.join(
                    dcw_holdout.select(*dcw_select), on=DOCUMENT_ID_COL, how="inner"
                )
        else:
            train_df = dcw_train.select(*dcw_select)
            if include_holdout:
                dcw_holdout = _read_delta(spark, paths["dcw_features_val_test_oot"], "holdout DCW")
                holdout_df = dcw_holdout.select(*dcw_select)

    if train_df is None and components["embeddings"]:
        train_ids, holdout_ids = _labels_scaffold(labels)
        train_df = train_ids
        if include_holdout:
            holdout_df = holdout_ids

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

    if train_df is None:
        raise ValueError(f"Could not load features for feature_set={feature_set!r}")
    if include_holdout and holdout_df is None:
        raise ValueError(f"Could not load holdout features for feature_set={feature_set!r}")

    logger.info(
        "Loaded feature tables for feature_set=%s (holdout=%s; joins and limits applied later)",
        feature_set,
        include_holdout,
    )
    return train_df, holdout_df, labels, embeddings_df


def prepare_training_data(
    train_features: DataFrame,
    holdout_features: DataFrame | None,
    labels: DataFrame,
    embeddings_df: DataFrame | None,
) -> tuple[DataFrame, DataFrame | None]:
    label_subset = labels.select(DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL)

    train_df = (
        train_features.join(label_subset, on=DOCUMENT_ID_COL, how="inner")
        .filter(F.col(SPLIT_COL) == "train")
    )
    holdout_df: DataFrame | None = None
    if holdout_features is not None:
        holdout_df = (
            holdout_features.join(label_subset, on=DOCUMENT_ID_COL, how="inner")
            .filter(F.col(SPLIT_COL) != "train")
        )

    if embeddings_df is not None:
        train_df = train_df.join(embeddings_df, on=DOCUMENT_ID_COL, how="inner")
        if holdout_df is not None:
            holdout_df = holdout_df.join(embeddings_df, on=DOCUMENT_ID_COL, how="inner")

    n_train = train_df.count()
    if n_train == 0:
        raise ValueError("No training rows after joining features with labels")

    if holdout_df is not None:
        n_holdout = holdout_df.count()
        logger.info(
            "Prepared data: %s train documents, %s holdout documents",
            f"{n_train:,}",
            f"{n_holdout:,}",
        )
    else:
        logger.info("Prepared data: %s train documents (holdout skipped)", f"{n_train:,}")
    return train_df, holdout_df


def prepare_holdout_data(
    holdout_features: DataFrame,
    labels: DataFrame,
    embeddings_df: DataFrame | None,
) -> DataFrame:
    """Join holdout feature tables with labels (val / test / oot only)."""
    label_subset = labels.select(DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL)
    holdout_df = (
        holdout_features.join(label_subset, on=DOCUMENT_ID_COL, how="inner")
        .filter(F.col(SPLIT_COL) != "train")
    )
    if embeddings_df is not None:
        holdout_df = holdout_df.join(embeddings_df, on=DOCUMENT_ID_COL, how="inner")

    n_holdout = holdout_df.count()
    if n_holdout == 0:
        raise ValueError("No holdout rows after joining features with labels")
    logger.info("Prepared holdout data: %s documents", f"{n_holdout:,}")
    return holdout_df


def _list_hadoop_child_names(spark, dir_path: str) -> list[str]:
    jvm = spark.sparkContext._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(dir_path)
    fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    if not fs.exists(hadoop_path):
        return []
    return [st.getPath().getName() for st in fs.listStatus(hadoop_path)]


def resolve_prediction_delta_path(
    spark,
    prediction_date: str | None = None,
    prediction_suffix: str | None = None,
) -> str:
    """Return prediction Delta path; default is latest prediction_* folder under model_predictions."""
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    mp = schema["gold"]["model_predictions"]
    base = f"{gold_path}/{mp['base']}"
    prefix = mp["file_prefix"]
    norm_suffix = normalize_prediction_suffix(prediction_suffix)
    if prediction_date or norm_suffix:
        batch = prediction_batch_name(prediction_date, prediction_suffix)
        path = f"{base}/{batch}"
        logger.info("Using prediction Delta path: %s", path)
        return path
    candidates = sorted(
        name
        for name in _list_hadoop_child_names(spark, base)
        if name.startswith(prefix) and not name.endswith((".pkl", ".json"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prediction Delta found under {base}. Run --predict-stage predict first."
        )
    path = f"{base}/{candidates[-1]}"
    logger.info("Resolved latest prediction Delta: %s", path)
    return path


def _prediction_batch_name_from_delta_path(pred_delta_path: str) -> str | None:
    prefix = load_schema()["gold"]["model_predictions"]["file_prefix"]
    marker = f"/{prefix}"
    if marker not in pred_delta_path:
        return None
    batch = pred_delta_path.rsplit(marker, 1)[-1]
    return batch.split(".", 1)[0] if batch else None


def _prediction_batch_date_from_delta_path(pred_delta_path: str) -> str | None:
    batch = _prediction_batch_name_from_delta_path(pred_delta_path)
    if not batch:
        return None
    prefix = load_schema()["gold"]["model_predictions"]["file_prefix"]
    body = batch[len(prefix) :] if batch.startswith(prefix) else batch
    date_part = body.split("_", 1)[0]
    return date_part if date_part.isdigit() and len(date_part) == 8 else None


def resolve_model_manifest_path(
    spark,
    run_id: str,
    model_date: str | None = None,
    model_type: str | None = None,
) -> str:
    model_name = model_type or load_schema()["model_bank"]["runs"].get(
        "default_model_name", DEFAULT_MODEL_TYPE
    )
    if model_date:
        return model_bank_model_manifest_path(run_id, model_name, model_date)
    model_dir = f"{model_bank_run_root(run_id)}/{load_schema()['model_bank']['runs']['model_dir']}"
    candidates = sorted(
        name
        for name in _list_hadoop_child_names(spark, model_dir)
        if name.startswith(f"{model_name}_") and name.endswith(".pkl")
    )
    if not candidates:
        raise FileNotFoundError(f"No model manifest found under {model_dir}")
    manifest_path = f"{model_dir}/{candidates[-1]}"
    logger.info("Resolved latest model manifest: %s", manifest_path)
    return manifest_path


def _reconstruct_manifest_from_per_label_models(
    spark,
    run_id: str,
    paths: dict[str, str],
    feature_set: str,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict[str, Any]:
    """Fallback when manifest .pkl on R2 is missing/corrupt; map per_label dirs to train labels."""
    per_label_dir = model_bank_per_label_models_dir(run_id)
    dir_names = set(_list_hadoop_child_names(spark, per_label_dir))
    if not dir_names:
        raise FileNotFoundError(f"No per-label models found under {per_label_dir}")

    labels_raw = _read_delta(spark, paths["labels"], "labels")
    if SOURCE_LABEL_COL in labels_raw.columns:
        source_col = SOURCE_LABEL_COL
    elif SOURCE_LABEL_FALLBACK_COL in labels_raw.columns:
        source_col = SOURCE_LABEL_FALLBACK_COL
    else:
        raise ValueError("labels table missing label column for manifest reconstruction")

    train_labels = (
        labels_raw.filter(F.col(SPLIT_COL) == "train")
        .select(DOCUMENT_ID_COL, F.col(source_col).alias("_raw_target"))
        .dropDuplicates([DOCUMENT_ID_COL])
        .transform(lambda df: normalize_target_labels(df, "_raw_target"))
    )
    candidate_labels = [
        row.lbl
        for row in (
            train_labels.select(F.explode(F.col(TARGET_LABELS_COL)).alias("lbl"))
            .groupBy("lbl")
            .count()
            .orderBy(F.desc("count"), "lbl")
            .collect()
        )
    ]

    per_label_paths: dict[str, str] = {}
    unmatched_dirs = set(dir_names)
    for label in candidate_labels:
        safe = _safe_label_name(label)
        if safe in dir_names:
            per_label_paths[label] = f"{per_label_dir}/{safe}"
            unmatched_dirs.discard(safe)

    if unmatched_dirs:
        logger.warning("Unmatched per_label model dirs: %s", sorted(unmatched_dirs))
    if not per_label_paths:
        raise ValueError(
            f"Could not map any train labels to model dirs under {per_label_dir}"
        )

    logger.info(
        "Reconstructed manifest from %s per-label models (%s train labels matched)",
        f"{len(per_label_paths):,}",
        f"{len(candidate_labels):,}",
    )
    return {
        "feature_set": feature_set,
        "model_type": model_type,
        "per_label_model_paths": per_label_paths,
        "multilabel_threshold": 0.5,
        "hyperparameters": _default_hyperparameters(model_type),
        "reconstructed_from_per_label": True,
    }


def load_training_manifest(
    spark,
    run_id: str,
    model_date: str | None = None,
    paths: dict[str, str] | None = None,
    feature_set: str = "tfidf_dcw_embeddings",
    model_type: str = DEFAULT_MODEL_TYPE,
) -> tuple[dict[str, Any], str]:
    model_name = model_type or load_schema()["model_bank"]["runs"].get(
        "default_model_name", DEFAULT_MODEL_TYPE
    )
    try:
        manifest_path = resolve_model_manifest_path(spark, run_id, model_date, model_type=model_type)
    except FileNotFoundError:
        per_label_dir = model_bank_per_label_models_dir(run_id)
        if paths is None or not _list_hadoop_child_names(spark, per_label_dir):
            raise
        manifest_path = model_bank_model_manifest_path(run_id, model_name, model_date)
        logger.warning("No manifest pickle under model dir; reconstructing from %s", per_label_dir)
        return (
            _reconstruct_manifest_from_per_label_models(
                spark, run_id, paths, feature_set, model_type=model_type
            ),
            manifest_path,
        )

    logger.info("Loading training manifest from %s", manifest_path)
    try:
        manifest = load_pickle(manifest_path, spark)
        return manifest, manifest_path
    except Exception as exc:
        logger.warning("Failed to load manifest pickle (%s); using per_label fallback", exc)
        if paths is None:
            raise
        return (
            _reconstruct_manifest_from_per_label_models(
                spark, run_id, paths, feature_set, model_type=model_type
            ),
            manifest_path,
        )


def _read_spark_ml_class_name(spark, model_path: str) -> str | None:
    """Read Spark ML class from model metadata/part-00000 (if present)."""
    import json

    metadata_path = f"{model_path.rstrip('/')}/metadata/part-00000"
    try:
        raw = load_bytes_from_path(metadata_path, spark).decode("utf-8")
        return json.loads(raw).get("class")
    except Exception:
        return None


def detect_model_type_from_path(spark, model_path: str) -> str | None:
    class_name = _read_spark_ml_class_name(spark, model_path)
    if not class_name:
        return None
    if class_name.endswith("LogisticRegressionModel"):
        return "logistic_regression"
    if class_name.endswith("RandomForestClassificationModel"):
        return "random_forest"
    return None


def resolve_model_type_for_run(
    spark,
    run_id: str,
    per_label_paths: dict[str, str],
    cli_model_type: str,
) -> str:
    """Prefer CLI model_type; auto-detect from saved Spark model metadata when mismatched."""
    if not per_label_paths:
        return cli_model_type
    sample_path = next(iter(per_label_paths.values()))
    detected = detect_model_type_from_path(spark, sample_path)
    if detected and detected != cli_model_type:
        logger.warning(
            "Saved models under %s are %r but CLI had %r; using detected type",
            run_id,
            detected,
            cli_model_type,
        )
        return detected
    return cli_model_type


def load_per_label_classification_model(path: str, model_type: str) -> Any:
    if model_type == "logistic_regression":
        return LogisticRegressionModel.load(path)
    if model_type == "random_forest":
        return RandomForestClassificationModel.load(path)
    raise ValueError(f"Unsupported model_type={model_type!r}")


def load_trained_models(per_label_paths: dict[str, str], model_type: str = DEFAULT_MODEL_TYPE) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label, path in per_label_paths.items():
        logger.info("Loading model for label=%r", label)
        models[label] = load_per_label_classification_model(path, model_type)
    logger.info("Loaded %s per-label models from model_bank", f"{len(models):,}")
    return models


def build_feature_column(
    df: DataFrame,
    feature_set: str,
    dcw_vocab: list[str] | None = None,
) -> DataFrame:
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
        if DCW_VECTOR_COL not in df.columns:
            if DCW_FEATURES_COL not in df.columns:
                raise ValueError(
                    f"{DCW_FEATURES_COL!r} or {DCW_VECTOR_COL!r} required for feature_set={feature_set!r}"
                )
            if not dcw_vocab:
                raise ValueError("dcw_vocab is required to convert dcw_features map to vector")
            df = add_dcw_vector_column(df, dcw_vocab)
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


def _default_hyperparameters(model_type: str) -> dict[str, Any]:
    if model_type == "logistic_regression":
        return dict(LR_PARAMS)
    return dict(RF_PARAMS)


def resolve_model_params(args: argparse.Namespace) -> dict[str, Any]:
    """Merge module defaults with optional CLI overrides for the selected model type."""
    model_type = args.model_type
    if model_type == "logistic_regression":
        params: dict[str, Any] = dict(LR_PARAMS)
        if args.max_iter is not None:
            params["maxIter"] = args.max_iter
        if args.reg_param is not None:
            params["regParam"] = args.reg_param
        if args.elastic_net_param is not None:
            params["elasticNetParam"] = args.elastic_net_param
        return params

    params = dict(RF_PARAMS)
    if args.num_trees is not None:
        params["numTrees"] = args.num_trees
    if args.max_depth is not None:
        params["maxDepth"] = args.max_depth
    return params


def resolve_rf_params(args: argparse.Namespace) -> dict[str, int]:
    """Backward-compatible alias for Random Forest param resolution."""
    params = resolve_model_params(args)
    return {k: int(v) for k, v in params.items()}


def _build_binary_classifier(model_type: str, model_params: dict[str, Any]) -> Any:
    if model_type == "logistic_regression":
        return LogisticRegression(
            featuresCol=FEATURES_COL,
            labelCol="binary_label",
            family="binomial",
            maxIter=int(model_params["maxIter"]),
            regParam=float(model_params["regParam"]),
            elasticNetParam=float(model_params["elasticNetParam"]),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            featuresCol=FEATURES_COL,
            labelCol="binary_label",
            numTrees=int(model_params["numTrees"]),
            maxDepth=int(model_params["maxDepth"]),
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")


@udf(DoubleType())
def _prob_positive(probability: Vector | None) -> float:
    if probability is None:
        return 0.0
    values = probability.toArray()
    return float(values[1]) if len(values) > 1 else float(values[0])


def train_multilabel_model(
    train_df: DataFrame,
    max_labels: int | None,
    model_type: str,
    model_params: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if model_type == "random_forest":
        logger.warning(
            "RandomForestClassifier may be slow or memory-heavy on high-dimensional sparse text features."
        )
    logger.info("%s hyperparameters: %s", model_type, model_params)

    label_list = _collect_training_labels(train_df, max_labels)
    models: dict[str, Any] = {}

    for label in label_list:
        binary_train = train_df.withColumn(
            "binary_label",
            F.when(F.array_contains(F.col(TARGET_LABELS_COL), label), 1.0).otherwise(0.0),
        )
        classifier = _build_binary_classifier(model_type, model_params)
        models[label] = classifier.fit(binary_train)
        logger.debug("Trained binary model for label=%s", label)

    logger.info(
        "Binary relevance training complete: %s per-label models",
        f"{len(models):,}",
    )
    return models, label_list


def predict_multilabel(
    df: DataFrame,
    label_list: list[str],
    threshold: float,
    *,
    models: dict[str, Any] | None = None,
    per_label_paths: dict[str, str] | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> DataFrame:
    """Score holdout rows. Keeps metadata + prob columns only (drops fat feature vectors)."""
    if models is None and per_label_paths is None:
        raise ValueError("predict_multilabel requires models or per_label_paths")

    meta = df.select(DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL)
    if meta.limit(1).count() == 0:
        return meta.withColumn(PREDICTED_LABELS_COL, F.array().cast(ArrayType(StringType())))

    features_df = df.select(DOCUMENT_ID_COL, FEATURES_COL)
    result = meta
    prob_cols: list[str] = []
    n_labels = len(label_list)

    for idx, label in enumerate(label_list, start=1):
        prob_col = _prob_column_name(label)
        if models is not None:
            model = models[label]
        else:
            assert per_label_paths is not None
            logger.info("Scoring label %s/%s: %r", idx, n_labels, label)
            model = load_per_label_classification_model(per_label_paths[label], model_type)

        prob_df = (
            model.transform(features_df)
            .select(DOCUMENT_ID_COL, F.col("probability").alias("_prob_vector"))
            .withColumn(prob_col, _prob_positive(F.col("_prob_vector")))
            .drop("_prob_vector")
        )
        result = result.join(prob_df, on=DOCUMENT_ID_COL, how="left")
        prob_cols.append(prob_col)

    threshold_exprs = [
        F.when(F.col(col) >= F.lit(threshold), F.lit(label))
        for label, col in zip(label_list, prob_cols)
    ]
    return result.withColumn(
        PREDICTED_LABELS_COL,
        F.array_compact(F.array(*threshold_exprs)),
    )


def apply_multilabel_threshold(
    predictions: DataFrame,
    label_list: list[str],
    threshold: float,
    *,
    prob_column_map: dict[str, str] | None = None,
) -> DataFrame:
    """Rebuild predicted_labels from saved prob_* columns (no model re-scoring)."""
    prob_column_map = prob_column_map or label_prob_column_map(label_list)
    threshold_exprs = []
    for label in label_list:
        prob_col = prob_column_map.get(label)
        if prob_col and prob_col in predictions.columns:
            threshold_exprs.append(F.when(F.col(prob_col) >= F.lit(threshold), F.lit(label)))
    if not threshold_exprs:
        raise ValueError(
            "No prob_* columns on predictions. Re-run --predict-stage predict to checkpoint probabilities."
        )
    return predictions.withColumn(
        PREDICTED_LABELS_COL,
        F.array_compact(F.array(*threshold_exprs)),
    )


def format_prediction_delta_df(
    predictions: DataFrame,
    run_id: str,
    feature_run_id: str,
    prediction_ts: str,
    *,
    multilabel_threshold: float | None = None,
) -> DataFrame:
    """Slim prediction row for Delta: metadata, predicted labels, and per-label probabilities."""
    prob_cols = prob_columns_in_df(predictions)
    select_exprs = [
        F.lit(run_id).alias("run_id"),
        F.lit(feature_run_id).alias("feature_run_id"),
        F.col(DOCUMENT_ID_COL),
        F.col(SPLIT_COL),
        F.col(TARGET_LABELS_COL),
        F.col(PREDICTED_LABELS_COL),
        *[F.col(col) for col in prob_cols],
    ]
    if multilabel_threshold is not None:
        select_exprs.append(F.lit(multilabel_threshold).alias("multilabel_threshold"))
    select_exprs.append(F.lit(prediction_ts).alias("prediction_ts"))
    return predictions.select(*select_exprs)


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


def _split_row_counts(holdout_df: DataFrame | None) -> dict[str, int]:
    if holdout_df is None:
        return {"holdout": 0, "val": 0, "test": 0, "oot": 0}
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
    train_df: DataFrame | None,
    holdout_df: DataFrame | None,
    label_list: list[str],
    metrics: dict[str, dict[str, float]] | None,
    predictions: DataFrame | None,
    model_params: dict[str, Any],
) -> None:
    logger.info("DRY RUN enabled: skipping all writes to R2/S3A/model_bank.")

    n_train = train_df.count() if train_df is not None else 0
    n_holdout = holdout_df.count() if holdout_df is not None else 0
    labels_preview = label_list[:20]
    labels_suffix = f" ... (+{len(label_list) - 20} more)" if len(label_list) > 20 else ""

    logger.info("=== Dry-run summary ===")
    logger.info("run_id: %s", run_id)
    logger.info("model: %s", args.model_type)
    logger.info("feature_set: %s", args.feature_set)
    logger.info("train_only: %s", args.train_only)
    logger.info("model_params: %s", model_params)
    logger.info(
        "features: tfidf=%s log_tfidf=%s dcw=%s embeddings=%s",
        components["tfidf"],
        components["log_tfidf"],
        components["dcw"],
        components["embeddings"],
    )
    if args.predict_only or train_df is None:
        logger.info("holdout rows: %s", f"{n_holdout:,}")
    else:
        logger.info("train rows: %s | holdout rows: %s", f"{n_train:,}", f"{n_holdout:,}")
    logger.info("labels trained: %s", f"{len(label_list):,}")
    logger.info("label list (up to 20): %s%s", labels_preview, labels_suffix)
    logger.info("multilabel_threshold: %s", args.multilabel_threshold)
    if args.max_labels is not None:
        logger.info("max_labels: %s", args.max_labels)

    if args.train_only:
        logger.info("Train-only dry run: holdout metrics and predictions were skipped")
    elif args.predict_only:
        logger.info("Predict-only dry run: holdout metrics below (no writes)")
    elif metrics is not None:
        for key, title in (
            ("holdout_overall", "holdout_overall"),
            ("holdout_val", "val"),
            ("holdout_test", "test"),
            ("holdout_oot", "oot"),
        ):
            _log_metric_block(title, metrics.get(key, _empty_metrics()))

    if predictions is None or predictions.limit(1).count() == 0:
        if not args.train_only:
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
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
    components: dict[str, bool],
    train_df: DataFrame,
    holdout_df: DataFrame | None,
    models: dict[str, Any],
    label_list: list[str],
    metrics: dict[str, dict[str, float]] | None,
    predictions: DataFrame | None,
    hyperparameters: dict[str, Any],
) -> dict[str, str]:
    train_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    model_name = args.model_type
    per_label_dir = model_bank_per_label_models_dir(run_id)
    model_manifest_path = model_bank_model_manifest_path(run_id, model_name, train_date)
    pred_pkl_path = prediction_manifest_path(train_date, args.prediction_suffix)
    pred_delta_path = prediction_delta_path(train_date, args.prediction_suffix)

    model_paths: dict[str, str] = {}
    for label, model in models.items():
        safe = _safe_label_name(label)
        model_path = f"{per_label_dir}/{safe}"
        model.write().overwrite().save(model_path)
        model_paths[label] = model_path

    assembled_cols = [DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL, FEATURES_COL]
    write_delta(train_df.select(*assembled_cols), paths["X_train"])
    logger.info("Saved X_train to %s", paths["X_train"])
    if holdout_df is not None and not args.train_only:
        write_delta(holdout_df.select(*assembled_cols), paths["X_val_test_oot"])
        logger.info("Saved X_val_test_oot to %s", paths["X_val_test_oot"])
    else:
        logger.info("Skipped X_val_test_oot (train-only mode)")

    holdout_counts = _split_row_counts(holdout_df)
    n_train = train_df.count()
    prediction_ts = datetime.now(timezone.utc).isoformat()
    metrics = metrics or {}

    metadata = {
        "run_id": run_id,
        "feature_run_id": feature_run_id,
        "timestamp": prediction_ts,
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "uses_tfidf": components["tfidf"],
        "uses_log_tfidf": components["log_tfidf"],
        "uses_dcw": components["dcw"],
        "uses_embeddings": components["embeddings"],
        "train_only": args.train_only,
        "multilabel_strategy": MULTILABEL_STRATEGY,
        "multilabel_threshold": args.multilabel_threshold,
        "max_labels": args.max_labels,
        "label_normalization": LABEL_NORMALIZATION,
        "input_feature_paths": {
            "tfidf_train": paths["tfidf_train"],
            "tfidf_val_test_oot": paths["tfidf_val_test_oot"],
            "dcw_train": paths["dcw_train"],
            "dcw_val_test_oot": paths["dcw_val_test_oot"],
            "embeddings": paths["embeddings"],
            "tfidf_pkl": paths["tfidf_pkl"],
            "dcw_pkl": paths["dcw_pkl"],
        },
        "labels_path": paths["labels"],
        "assembled_dataset_paths": {
            "X_train": paths["X_train"],
            "X_val_test_oot": paths["X_val_test_oot"],
            "X_unlabelled": paths["X_unlabelled"],
        },
        "per_label_model_paths": model_paths,
        "probability_columns": label_prob_column_map(label_list),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
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
            "TF-IDF, DCW, and embeddings were precomputed in upstream Gold jobs "
            "and were not refit in this training script. "
            "Multi-label training uses binary relevance (one binary classifier per label). "
            "Spark ML models live under model/per_label/; this .pkl holds metadata and metrics."
            + (" Holdout scoring deferred (--train-only)." if args.train_only else "")
        ),
    }
    save_pickle(model_manifest_path, metadata, spark)
    model_manifest_json = model_manifest_path.replace(".pkl", ".json")
    save_json(model_manifest_json, metadata, spark)
    logger.info("Saved model manifest JSON to %s", model_manifest_json)

    if (
        not args.train_only
        and predictions is not None
        and predictions.limit(1).count() > 0
    ):
        pred_out = format_prediction_delta_df(
            predictions,
            run_id,
            feature_run_id,
            prediction_ts,
            multilabel_threshold=args.multilabel_threshold,
        )
        write_delta(pred_out, pred_delta_path)
        save_pickle(
            pred_pkl_path,
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "prediction_ts": prediction_ts,
                "prediction_delta_path": pred_delta_path,
                "row_count": holdout_counts["holdout"],
                "metrics": metrics,
            },
            spark,
        )
    elif args.train_only:
        logger.info("Train-only mode: skipped holdout predictions for run_id=%s", run_id)
    else:
        logger.warning("No holdout predictions to write for run_id=%s", run_id)

    logger.info("Saved per-label Spark models under %s", per_label_dir)
    logger.info("Saved model manifest to %s", model_manifest_path)
    if not args.train_only:
        logger.info("Saved prediction manifest to %s", pred_pkl_path)

    out = {
        "per_label_models_dir": per_label_dir,
        "model_manifest_path": model_manifest_path,
    }
    if not args.train_only:
        out["prediction_manifest_path"] = pred_pkl_path
        out["prediction_delta_path"] = pred_delta_path
    return out


def save_holdout_features(holdout_df: DataFrame, paths: dict[str, str]) -> None:
    """Persist assembled holdout X before scoring (survives predict/metrics OOM)."""
    assembled_cols = [DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL, FEATURES_COL]
    write_delta(holdout_df.select(*assembled_cols), paths["X_val_test_oot"])
    logger.info("Saved assembled holdout features to %s", paths["X_val_test_oot"])


def checkpoint_predictions(
    spark,
    predictions: DataFrame,
    run_id: str,
    feature_run_id: str,
    pred_delta_path: str | None = None,
    *,
    multilabel_threshold: float | None = None,
) -> tuple[DataFrame, str, str]:
    """Write slim predictions to Delta and reload to break heavy scoring lineage."""
    prediction_ts = datetime.now(timezone.utc).isoformat()
    delta_path = pred_delta_path or prediction_delta_path()
    pred_out = format_prediction_delta_df(
        predictions,
        run_id,
        feature_run_id,
        prediction_ts,
        multilabel_threshold=multilabel_threshold,
    )
    write_delta(pred_out, delta_path)
    materialized = spark.read.format("delta").load(delta_path)
    n_rows = materialized.count()
    logger.info("Checkpointed %s prediction rows to %s", f"{n_rows:,}", delta_path)
    spark.catalog.clearCache()
    return materialized, prediction_ts, delta_path


def save_evaluation_outputs(
    spark,
    run_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    model_manifest_path: str,
    label_list: list[str],
    metrics: dict[str, dict[str, float]],
    predictions: DataFrame,
    multilabel_threshold: float,
    prediction_ts: str,
    pred_delta_path: str,
    *,
    skip_x_write: bool = False,
    skip_pred_delta_write: bool = False,
    holdout_df: DataFrame | None = None,
) -> dict[str, str]:
    """Write holdout X (optional), predictions (optional), and metrics manifests."""
    if not skip_x_write:
        if holdout_df is None:
            raise ValueError("holdout_df is required when skip_x_write=False")
        save_holdout_features(holdout_df, paths)

    holdout_counts = _split_row_counts(predictions)
    batch_name = _prediction_batch_name_from_delta_path(pred_delta_path)
    pred_pkl_path = (
        prediction_manifest_path_for_batch(batch_name)
        if batch_name
        else prediction_manifest_path(_prediction_batch_date_from_delta_path(pred_delta_path))
    )
    pred_json_path = pred_pkl_path.replace(".pkl", ".json")

    if not skip_pred_delta_write:
        pred_out = format_prediction_delta_df(
            predictions,
            run_id,
            feature_run_id,
            prediction_ts,
            multilabel_threshold=multilabel_threshold,
        )
        write_delta(pred_out, pred_delta_path)
    eval_manifest = {
        "run_id": run_id,
        "feature_run_id": feature_run_id,
        "model_manifest_path": model_manifest_path,
        "prediction_ts": prediction_ts,
        "prediction_delta_path": pred_delta_path,
        "multilabel_threshold": multilabel_threshold,
        "probability_columns": label_prob_column_map(label_list),
        "row_count": holdout_counts["holdout"],
        "val_documents": holdout_counts["val"],
        "test_documents": holdout_counts["test"],
        "oot_documents": holdout_counts["oot"],
        "metrics": metrics,
        "num_unique_labels": len(label_list),
    }
    save_pickle(pred_pkl_path, eval_manifest, spark)
    save_json(pred_json_path, eval_manifest, spark)

    if not skip_pred_delta_write:
        logger.info("Saved prediction Delta to %s", pred_delta_path)
    logger.info("Saved prediction manifest to %s", pred_pkl_path)
    logger.info("Saved prediction metrics JSON to %s", pred_json_path)
    for key, title in (
        ("holdout_overall", "holdout_overall"),
        ("holdout_val", "val"),
        ("holdout_test", "test"),
        ("holdout_oot", "oot"),
    ):
        _log_metric_block(title, metrics.get(key, _empty_metrics()))

    return {
        "model_manifest_path": model_manifest_path,
        "prediction_manifest_path": pred_pkl_path,
        "prediction_delta_path": pred_delta_path,
        "X_val_test_oot": paths["X_val_test_oot"],
    }


def _load_predict_context(
    spark,
    run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, str], str, dict[str, bool], float, list[str], str]:
    manifest, manifest_path = load_training_manifest(
        spark,
        run_id,
        args.model_date,
        paths=paths,
        feature_set=args.feature_set,
        model_type=args.model_type,
    )
    per_label_paths = manifest.get("per_label_model_paths")
    if not per_label_paths:
        raise ValueError(f"Training manifest at {manifest_path} missing per_label_model_paths")

    feature_set = manifest.get("feature_set", args.feature_set)
    if feature_set != args.feature_set:
        logger.warning(
            "Using feature_set=%r from manifest (CLI had %r)",
            feature_set,
            args.feature_set,
        )
    components = _feature_components(feature_set)
    threshold = args.multilabel_threshold
    if manifest.get("multilabel_threshold") is not None and threshold == 0.5:
        threshold = float(manifest["multilabel_threshold"])

    label_list = list(per_label_paths.keys())
    model_type = resolve_model_type_for_run(spark, run_id, per_label_paths, args.model_type)
    manifest_model_type = manifest.get("model_type")
    if manifest_model_type and manifest_model_type != model_type:
        logger.warning(
            "Manifest model_type=%r differs from resolved %r; using resolved type for scoring",
            manifest_model_type,
            model_type,
        )
    return (
        manifest,
        manifest_path,
        per_label_paths,
        feature_set,
        components,
        threshold,
        label_list,
        model_type,
    )


def _build_holdout_from_features(
    spark,
    paths: dict[str, str],
    feature_set: str,
    components: dict[str, bool],
    *,
    holdout_splits: list[str] | None,
    limit: int | None,
) -> DataFrame:
    dcw_vocab = load_dcw_vocab(spark, paths) if components["dcw"] else None
    _, holdout_features, labels, embeddings_df = load_features(
        spark, paths, feature_set, include_holdout=True
    )
    holdout_df = prepare_holdout_data(holdout_features, labels, embeddings_df)

    if holdout_splits:
        holdout_df = holdout_df.filter(F.col(SPLIT_COL).isin(holdout_splits))
        logger.info("Filtered holdout to splits: %s", holdout_splits)

    if limit:
        holdout_df = holdout_df.limit(limit)
        logger.info("Smoke test: limited holdout to %s rows", f"{limit:,}")

    return build_feature_column(holdout_df, feature_set, dcw_vocab=dcw_vocab)


def _load_saved_holdout_x(
    spark,
    paths: dict[str, str],
    *,
    holdout_splits: list[str] | None,
    limit: int | None,
) -> DataFrame:
    x_path = paths["X_val_test_oot"]
    logger.info("Loading saved holdout X from %s", x_path)
    holdout_df = _read_delta(spark, x_path, "X_val_test_oot")
    required = {DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL, FEATURES_COL}
    missing = required - set(holdout_df.columns)
    if missing:
        raise ValueError(
            f"X_val_test_oot at {x_path} missing columns {missing}. "
            "Run --predict-stage features first."
        )

    if holdout_splits:
        holdout_df = holdout_df.filter(F.col(SPLIT_COL).isin(holdout_splits))
        logger.info("Filtered saved holdout X to splits: %s", holdout_splits)

    if limit:
        holdout_df = holdout_df.limit(limit)
        logger.info("Smoke test: limited saved holdout X to %s rows", f"{limit:,}")

    return holdout_df


def run_predict_stage_features(
    spark,
    run_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> None:
    components = _feature_components(args.feature_set)
    logger.info(
        "Predict stage=features: assembling holdout X (feature_set=%s)",
        args.feature_set,
    )
    holdout_df = _build_holdout_from_features(
        spark,
        paths,
        args.feature_set,
        components,
        holdout_splits=args.holdout_splits,
        limit=args.limit,
    )

    if args.dry_run:
        n_rows = holdout_df.count()
        logger.info("DRY RUN: would save %s holdout rows to %s", f"{n_rows:,}", paths["X_val_test_oot"])
        return

    save_holdout_features(holdout_df, paths)
    logger.info("Predict stage=features complete: %s", paths["X_val_test_oot"])


def run_predict_stage_predict(
    spark,
    run_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> None:
    manifest, manifest_path, per_label_paths, feature_set, components, threshold, label_list, model_type = (
        _load_predict_context(spark, run_id, paths, args)
    )
    logger.info(
        "Predict stage=predict: scoring %s labels from saved holdout X",
        f"{len(label_list):,}",
    )
    holdout_df = _load_saved_holdout_x(
        spark,
        paths,
        holdout_splits=args.holdout_splits,
        limit=args.limit,
    )

    predictions = predict_multilabel(
        holdout_df,
        label_list,
        threshold,
        per_label_paths=per_label_paths,
        model_type=model_type,
    )

    if args.dry_run:
        metrics = evaluate_multilabel(predictions, label_list)
        print_dry_run_summary(
            run_id,
            args,
            components,
            None,
            holdout_df,
            label_list,
            metrics,
            predictions,
            manifest.get("hyperparameters", _default_hyperparameters(model_type)),
        )
        return

    pred_delta_path = prediction_delta_path(args.prediction_date, args.prediction_suffix)
    predictions, prediction_ts, pred_delta_path = checkpoint_predictions(
        spark,
        predictions,
        run_id,
        feature_run_id,
        pred_delta_path=pred_delta_path,
        multilabel_threshold=threshold,
    )
    logger.info(
        "Predict stage=predict complete: %s rows checkpointed to %s (ts=%s)",
        f"{predictions.count():,}",
        pred_delta_path,
        prediction_ts,
    )


def run_predict_stage_metrics(
    spark,
    run_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> None:
    manifest, manifest_path, _, feature_set, components, threshold, label_list, model_type = (
        _load_predict_context(spark, run_id, paths, args)
    )
    pred_delta_path = resolve_prediction_delta_path(
        spark, args.prediction_date, args.prediction_suffix
    )
    logger.info("Predict stage=metrics: loading predictions from %s", pred_delta_path)
    predictions = spark.read.format("delta").load(pred_delta_path)
    required = {DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL, PREDICTED_LABELS_COL}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction Delta at {pred_delta_path} missing columns {missing}")

    if args.holdout_splits:
        predictions = predictions.filter(F.col(SPLIT_COL).isin(args.holdout_splits))
        logger.info("Filtered predictions to splits: %s", args.holdout_splits)

    if args.limit:
        predictions = predictions.limit(args.limit)
        logger.info("Smoke test: limited predictions to %s rows", f"{args.limit:,}")

    if args.dry_run:
        metrics = evaluate_multilabel(predictions, label_list)
        print_dry_run_summary(
            run_id,
            args,
            components,
            None,
            None,
            label_list,
            metrics,
            predictions,
            manifest.get("hyperparameters", _default_hyperparameters(model_type)),
        )
        return

    prediction_ts_rows = predictions.select("prediction_ts").distinct().limit(2).collect()
    if len(prediction_ts_rows) != 1:
        raise ValueError(
            f"Expected one prediction_ts in {pred_delta_path}, found {len(prediction_ts_rows)}"
        )
    prediction_ts = prediction_ts_rows[0].prediction_ts

    logger.info("Computing metrics on checkpointed predictions")
    metrics = evaluate_multilabel(predictions, label_list)

    save_evaluation_outputs(
        spark,
        run_id,
        feature_run_id,
        paths,
        manifest_path,
        label_list,
        metrics,
        predictions,
        threshold,
        prediction_ts,
        pred_delta_path,
        skip_x_write=True,
        skip_pred_delta_write=True,
    )
    logger.info("Predict stage=metrics complete")


def run_predict_all(
    spark,
    run_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> None:
    manifest, manifest_path, per_label_paths, feature_set, components, threshold, label_list, model_type = (
        _load_predict_context(spark, run_id, paths, args)
    )
    logger.info("Scoring %s labels on val/test/oot", f"{len(label_list):,}")

    holdout_df = _build_holdout_from_features(
        spark,
        paths,
        feature_set,
        components,
        holdout_splits=args.holdout_splits,
        limit=args.limit,
    )

    if not args.dry_run:
        save_holdout_features(holdout_df, paths)

    predictions = predict_multilabel(
        holdout_df,
        label_list,
        threshold,
        per_label_paths=per_label_paths,
        model_type=model_type,
    )

    if args.dry_run:
        metrics = evaluate_multilabel(predictions, label_list)
        print_dry_run_summary(
            run_id,
            args,
            components,
            None,
            holdout_df,
            label_list,
            metrics,
            predictions,
            manifest.get("hyperparameters", _default_hyperparameters(model_type)),
        )
        return

    pred_delta_path = prediction_delta_path(args.prediction_date, args.prediction_suffix)
    predictions, prediction_ts, pred_delta_path = checkpoint_predictions(
        spark,
        predictions,
        run_id,
        feature_run_id,
        pred_delta_path=pred_delta_path,
        multilabel_threshold=threshold,
    )
    logger.info("Computing metrics on checkpointed predictions (slim Delta reload)")
    metrics = evaluate_multilabel(predictions, label_list)

    save_evaluation_outputs(
        spark,
        run_id,
        feature_run_id,
        paths,
        manifest_path,
        label_list,
        metrics,
        predictions,
        threshold,
        prediction_ts,
        pred_delta_path,
        skip_x_write=True,
        skip_pred_delta_write=True,
    )


def run_recover_manifest(
    spark,
    run_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """Write model manifest from existing per_label models (skip retraining)."""
    components = _feature_components(args.feature_set)
    model_params = resolve_model_params(args)
    reconstructed = _reconstruct_manifest_from_per_label_models(
        spark, run_id, paths, args.feature_set, model_type=args.model_type
    )
    label_list = list(reconstructed["per_label_model_paths"].keys())
    model_paths = reconstructed["per_label_model_paths"]
    try:
        n_train = spark.read.format("delta").load(paths["X_train"]).count()
    except Exception as exc:
        logger.warning("Could not count X_train (%s); using 0", exc)
        n_train = 0

    train_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    model_manifest_path = model_bank_model_manifest_path(run_id, args.model_type, train_date)
    prediction_ts = datetime.now(timezone.utc).isoformat()
    hyperparameters = {
        **model_params,
        "model_type": args.model_type,
        "multilabel_threshold": args.multilabel_threshold,
        "max_labels": args.max_labels,
        "train_only": True,
        "recovered_manifest": True,
    }
    metadata = {
        "run_id": run_id,
        "feature_run_id": feature_run_id,
        "timestamp": prediction_ts,
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "uses_tfidf": components["tfidf"],
        "uses_log_tfidf": components["log_tfidf"],
        "uses_dcw": components["dcw"],
        "uses_embeddings": components["embeddings"],
        "train_only": True,
        "multilabel_strategy": MULTILABEL_STRATEGY,
        "multilabel_threshold": args.multilabel_threshold,
        "max_labels": args.max_labels,
        "label_normalization": LABEL_NORMALIZATION,
        "input_feature_paths": {
            "tfidf_train": paths["tfidf_train"],
            "tfidf_val_test_oot": paths["tfidf_val_test_oot"],
            "dcw_train": paths["dcw_train"],
            "dcw_val_test_oot": paths["dcw_val_test_oot"],
            "embeddings": paths["embeddings"],
            "tfidf_pkl": paths["tfidf_pkl"],
            "dcw_pkl": paths["dcw_pkl"],
        },
        "labels_path": paths["labels"],
        "assembled_dataset_paths": {
            "X_train": paths["X_train"],
            "X_val_test_oot": paths["X_val_test_oot"],
            "X_unlabelled": paths["X_unlabelled"],
        },
        "per_label_model_paths": model_paths,
        "probability_columns": label_prob_column_map(label_list),
        "hyperparameters": hyperparameters,
        "metrics": {},
        "row_counts": {
            "train_documents": n_train,
            "holdout_documents": 0,
            "val_documents": 0,
            "test_documents": 0,
            "oot_documents": 0,
        },
        "num_unique_labels": len(label_list),
        "split_column": SPLIT_COL,
        "notes": "Manifest recovered from per_label models without retraining.",
        "recovered_from_per_label": True,
    }
    save_pickle(model_manifest_path, metadata, spark)
    model_manifest_json = model_manifest_path.replace(".pkl", ".json")
    save_json(model_manifest_json, metadata, spark)
    logger.info("Recovered model manifest to %s", model_manifest_path)
    logger.info("Recovered model manifest JSON to %s", model_manifest_json)


def run_predict_only(
    spark,
    run_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> None:
    stage = args.predict_stage
    if stage == "features":
        run_predict_stage_features(spark, run_id, feature_run_id, paths, args)
    elif stage == "predict":
        run_predict_stage_predict(spark, run_id, feature_run_id, paths, args)
    elif stage == "metrics":
        run_predict_stage_metrics(spark, run_id, feature_run_id, paths, args)
    else:
        run_predict_all(spark, run_id, feature_run_id, paths, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a multi-label Spark ML classifier on precomputed Gold features"
    )
    parser.add_argument("--run-id", default=None, help="Training run id (model_bank path). Default: UTC timestamp")
    parser.add_argument(
        "--feature-run-id",
        default=None,
        help="Gold feature run to read TF-IDF/DCW from (e.g. run001). Default: same as --run-id",
    )
    parser.add_argument(
        "--x-run-id",
        default=None,
        help="Gold run for assembled X_train / X_val_test_oot (default: same as --feature-run-id).",
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
        "--recover-manifest",
        action="store_true",
        help="Write model manifest JSON/pkl from existing per_label models (no retraining).",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Load saved per-label models; run holdout pipeline (see --predict-stage).",
    )
    parser.add_argument(
        "--predict-stage",
        choices=PREDICT_STAGES,
        default="all",
        help=(
            "With --predict-only: features=save X_val_test_oot; predict=score and checkpoint Delta; "
            "metrics=metrics from checkpoint; all=full pipeline (default)"
        ),
    )
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="Prediction batch YYYYMMDD for predict/metrics stages (default: today or latest Delta).",
    )
    parser.add_argument(
        "--prediction-suffix",
        default=None,
        help="Optional tag appended to prediction batch name, e.g. LR → prediction_20260613_LR",
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
    if args.train_only and args.predict_only:
        parser.error("--train-only and --predict-only are mutually exclusive")
    if args.recover_manifest and (args.train_only or args.predict_only):
        parser.error("--recover-manifest cannot be combined with --train-only or --predict-only")
    if args.predict_stage != "all" and not args.predict_only:
        parser.error("--predict-stage requires --predict-only")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_id = args.run_id or default_run_id()
    feature_run_id = args.feature_run_id or run_id
    x_run_id = args.x_run_id or feature_run_id
    paths = load_schema_paths(feature_run_id, x_run_id=x_run_id)
    components = _feature_components(args.feature_set)
    model_params = resolve_model_params(args)

    logger.info("Training run ID: %s | Feature run ID: %s | X run ID: %s", run_id, feature_run_id, x_run_id)
    logger.info("Model: %s | Feature set: %s", args.model_type, args.feature_set)
    if args.train_only:
        logger.info("Train-only mode: holdout load, scoring, and predictions will be skipped")
    if args.predict_only:
        logger.info("Predict-only mode: stage=%s", args.predict_stage)
    if args.dry_run:
        logger.info("DRY RUN enabled: skipping all writes to R2/S3A/model_bank.")

    spark = create_spark_session("gold-model-training")

    if args.recover_manifest:
        run_recover_manifest(spark, run_id, feature_run_id, paths, args)
        logger.info("Manifest recovery complete for run_id=%s", run_id)
        return

    if args.predict_only:
        run_predict_only(spark, run_id, feature_run_id, paths, args)
        logger.info("Holdout pipeline complete for run_id=%s (stage=%s)", run_id, args.predict_stage)
        return

    include_holdout = not args.train_only
    dcw_vocab = load_dcw_vocab(spark, paths) if components["dcw"] else None
    train_features, holdout_features, labels, embeddings_df = load_features(
        spark, paths, args.feature_set, include_holdout=include_holdout
    )
    train_df, holdout_df = prepare_training_data(
        train_features, holdout_features, labels, embeddings_df
    )

    if args.limit:
        train_df = train_df.limit(args.limit)
        if holdout_df is not None:
            holdout_df = holdout_df.limit(args.limit)
        logger.info("Smoke test: limited to %s rows per split after joins", f"{args.limit:,}")

    train_df = build_feature_column(train_df, args.feature_set, dcw_vocab=dcw_vocab)
    if holdout_df is not None:
        holdout_df = build_feature_column(holdout_df, args.feature_set, dcw_vocab=dcw_vocab)

    models, label_list = train_multilabel_model(
        train_df, args.max_labels, args.model_type, model_params
    )

    predictions: DataFrame | None = None
    metrics: dict[str, dict[str, float]] | None = None
    if not args.train_only:
        assert holdout_df is not None
        predictions = predict_multilabel(
            holdout_df,
            label_list,
            args.multilabel_threshold,
            models=models,
            model_type=args.model_type,
        )
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
            model_params,
        )
    else:
        hyperparameters = {
            **model_params,
            "model_type": args.model_type,
            "multilabel_threshold": args.multilabel_threshold,
            "max_labels": args.max_labels,
            "train_only": args.train_only,
        }
        save_outputs(
            spark,
            run_id,
            feature_run_id,
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
