"""Shared multi-label Spark ML primitives: features, scoring, metrics, I/O."""
from __future__ import annotations

import argparse
import itertools
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

_PIPELINE_DIR = Path(__file__).resolve().parent
_INCLUDE_DIR = _PIPELINE_DIR.parent
_GOLD_DIR = _INCLUDE_DIR / "gold"
_PROJECT_ROOT = _INCLUDE_DIR.parent
for _path in (_PROJECT_ROOT, _INCLUDE_DIR, _GOLD_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from gold_io import bootstrap_paths, load_bytes_from_path, load_pickle, save_json, save_pickle, write_delta
from run_paths import (
    default_feature_run_id,
    gold_run_table_path,
    load_schema,
    model_bank_experiment_subdir,
    model_bank_feature_importance_path,
    model_bank_holdout_metrics_path,
    model_bank_model_manifest_json_path,
    model_bank_model_manifest_path,
    model_bank_prediction_metrics_manifest_path,
    model_bank_threshold_sweep_path,
    normalize_prediction_suffix,
    prediction_batch_name,
    prediction_delta_path,
    resolve_experiment_paths,
    resolve_training_paths,
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
SOURCE_LABEL_COL = "label"
SOURCE_LABEL_FALLBACK_COL = "labels"  # gold/label_store column name on R2
EMBEDDING_COL = "embedding"  # column name written by legal_embeddings.py / wiki_embeddings.py
HOLDOUT_SPLITS = ("val", "test", "oot")
PREDICT_STAGES = (
    "features",
    "predict",
    "metrics",
    "threshold_sweep",
    "feature_importance",
    "eval",
    "all",
)
PREDICT_STAGE_ALL_DEPRECATED_MSG = (
    "--predict-stage all is deprecated. Prefer separate jobs: "
    "features → predict → eval (avoids OOM and matches batch_evaluate_experiments.py)."
)
DEFAULT_THRESHOLD_SWEEP = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
)
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

RF_PARAMS = {"numTrees": 50, "maxDepth": 10, "maxBins": 32}
LR_PARAMS = {"maxIter": 100, "regParam": 0.0, "elasticNetParam": 0.0}
DEFAULT_LR_PARAM_GRID: dict[str, list[Any]] = {
    "regParam": [0.0, 0.001, 0.01, 0.1, 1.0],
    "elasticNetParam": [0.0, 0.5, 1.0],
    "maxIter": [100],
}
DEFAULT_RF_PARAM_GRID: dict[str, list[Any]] = {
    "numTrees": [50, 100],
    "maxDepth": [8, 12, 16],
    "maxBins": [32, 64],
}
GRID_SEARCH_METRIC_CHOICES = ("micro_f1", "macro_f1", "exact_match_ratio")
DEFAULT_FEATURE_IMPORTANCE_TOP_K = 50

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


def load_schema_paths(
    feature_run_id: str,
    gold_run_id: str | None = None,
    *,
    x_run_id: str | None = None,
) -> dict[str, str]:
    """Resolve gold paths: TF-IDF/DCW from feature_run_id; X matrices from gold_run_id."""
    assembled_id = gold_run_id or x_run_id
    paths = resolve_training_paths(feature_run_id, gold_run_id=assembled_id)
    return {
        **paths,
        "tfidf_features_train": paths["tfidf_train"],
        "tfidf_features_val_test_oot": paths["tfidf_val_test_oot"],
        "dcw_features_train": paths["dcw_train"],
        "dcw_features_val_test_oot": paths["dcw_val_test_oot"],
    }


def resolve_prediction_exp_id(args: argparse.Namespace) -> str | None:
    if getattr(args, "exp_id", None):
        return args.exp_id
    if args.prediction_suffix:
        return normalize_prediction_suffix(args.prediction_suffix) or args.prediction_suffix.strip()
    if args.run_id:
        return args.run_id
    return None


def resolve_exp_id(args: argparse.Namespace) -> str:
    """Experiment id for model_bank/experiments/{exp_id}/."""
    exp = resolve_prediction_exp_id(args)
    if exp:
        return exp
    return default_run_id()


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


def load_tfidf_vocab(spark, paths: dict[str, str]) -> list[str]:
    """Frozen n-gram order from tfidf.pkl (matches tfidf / log_tfidf vector indices)."""
    pkl_path = paths["tfidf_pkl"]
    logger.info("Loading TF-IDF vocabulary from %s", pkl_path)
    bundle = load_pickle(pkl_path, spark)
    vocab = bundle["artifact"]["vocab"]
    if not vocab:
        raise ValueError(f"No vocabulary found in TF-IDF artifact at {pkl_path}")
    logger.info("TF-IDF vocabulary size: %s", f"{len(vocab):,}")
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
    exp_id: str | None = None,
    prediction_suffix: str | None = None,
) -> str:
    """Return prediction Delta path; default is latest under model_predictions/prediction_date=.../."""
    if prediction_date or exp_id or prediction_suffix:
        path = prediction_delta_path(
            prediction_date,
            exp_id,
            prediction_suffix=prediction_suffix,
        )
        logger.info("Using prediction Delta path: %s", path)
        return path

    schema = load_schema()
    gold_path = schema["gold"]["path"]
    mp = schema["gold"]["model_predictions"]
    base = f"{gold_path}/{mp['base']}"
    date_prefix = mp["date_partition_prefix"]
    date_dirs = sorted(
        name for name in _list_hadoop_child_names(spark, base) if name.startswith(date_prefix)
    )
    if not date_dirs:
        raise FileNotFoundError(
            f"No prediction Delta found under {base}. Run --predict-stage predict first."
        )
    latest_date_dir = f"{base}/{date_dirs[-1]}"
    exp_dirs = sorted(
        name
        for name in _list_hadoop_child_names(spark, latest_date_dir)
        if not name.endswith((".pkl", ".json"))
    )
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment predictions under {latest_date_dir}")
    path = f"{latest_date_dir}/{exp_dirs[-1]}"
    logger.info("Resolved latest prediction Delta: %s", path)
    return path


def _prediction_exp_id_from_delta_path(pred_delta_path: str) -> str | None:
    parts = pred_delta_path.rstrip("/").split("/")
    if not parts:
        return None
    return parts[-1] if parts[-1] and not parts[-1].endswith((".pkl", ".json")) else None


def _prediction_batch_date_from_delta_path(pred_delta_path: str) -> str | None:
    date_prefix = load_schema()["gold"]["model_predictions"]["date_partition_prefix"]
    for part in pred_delta_path.split("/"):
        if part.startswith(date_prefix):
            return part[len(date_prefix) :]
    # Legacy v1: prediction_YYYYMMDD_suffix
    legacy_prefix = "prediction_"
    for part in pred_delta_path.split("/"):
        if part.startswith(legacy_prefix):
            body = part[len(legacy_prefix) :]
            date_raw = body.split("_", 1)[0]
            if date_raw.isdigit() and len(date_raw) == 8:
                return f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
    return None


def resolve_model_manifest_path(
    spark,
    exp_id: str,
    model_date: str | None = None,
    model_type: str | None = None,
) -> str:
    model_name = model_type or load_schema()["model_bank"]["experiments"].get(
        "default_model_name", DEFAULT_MODEL_TYPE
    )
    if model_date:
        return model_bank_model_manifest_path(exp_id, model_name, model_date)
    manifest_dir = model_bank_experiment_subdir(exp_id, "manifest_dir")
    candidates = sorted(
        name
        for name in _list_hadoop_child_names(spark, manifest_dir)
        if name.startswith(f"{model_name}_") and name.endswith(".pkl")
    )
    if not candidates:
        raise FileNotFoundError(f"No model manifest found under {manifest_dir}")
    manifest_path = f"{manifest_dir}/{candidates[-1]}"
    logger.info("Resolved latest model manifest: %s", manifest_path)
    return manifest_path


def load_training_manifest(
    spark,
    exp_id: str,
    model_date: str | None = None,
    paths: dict[str, str] | None = None,
    feature_set: str = "tfidf_dcw_embeddings",
    model_type: str = DEFAULT_MODEL_TYPE,
) -> tuple[dict[str, Any], str]:
    manifest_path = resolve_model_manifest_path(spark, exp_id, model_date, model_type=model_type)
    logger.info("Loading training manifest from %s", manifest_path)
    manifest = load_pickle(manifest_path, spark)
    return manifest, manifest_path


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
    if args.max_bins is not None:
        params["maxBins"] = args.max_bins
    return params


def build_assembled_feature_names(
    feature_set: str,
    *,
    tfidf_vocab: list[str] | None = None,
    dcw_vocab: list[str] | None = None,
    embedding_dim: int | None = None,
) -> list[str]:
    """Human-readable names in VectorAssembler order for coefficient interpretation."""
    components = _feature_components(feature_set)
    names: list[str] = []

    if components["tfidf"]:
        if not tfidf_vocab:
            raise ValueError("tfidf_vocab is required for feature importance with tfidf features")
        names.extend(f"tfidf:{term}" for term in tfidf_vocab)

    if components["log_tfidf"]:
        if not tfidf_vocab:
            raise ValueError("tfidf_vocab is required for feature importance with log_tfidf features")
        names.extend(f"log_tfidf:{term}" for term in tfidf_vocab)

    if components["dcw"]:
        if not dcw_vocab:
            raise ValueError("dcw_vocab is required for feature importance with dcw features")
        names.extend(f"dcw:{lemma}" for lemma in dcw_vocab)

    if components["embeddings"]:
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError("embedding_dim is required for feature importance with embeddings")
        names.extend(f"embedding:{idx}" for idx in range(embedding_dim))

    if not names:
        raise ValueError(f"No feature names resolved for feature_set={feature_set!r}")
    return names


def _spark_ml_vector_to_list(vector: Vector | Any | None) -> list[float]:
    if vector is None:
        return []
    if hasattr(vector, "toArray"):
        return [float(v) for v in vector.toArray()]
    return [float(v) for v in vector]


def _extract_model_feature_scores(model: Any, model_type: str) -> list[float]:
    if model_type == "logistic_regression":
        return _spark_ml_vector_to_list(model.coefficients)
    if model_type == "random_forest":
        return _spark_ml_vector_to_list(model.featureImportances)
    raise ValueError(f"Unsupported model_type for feature importance: {model_type!r}")


def _per_label_fi_row(model_type: str, feature: str, raw_score: float) -> dict[str, Any]:
    if model_type == "logistic_regression":
        return {
            "rank": 0,
            "feature": feature,
            "coefficient": raw_score,
            "abs_coefficient": abs(raw_score),
        }
    return {"rank": 0, "feature": feature, "importance": raw_score}


def _per_label_fi_sort_key(model_type: str, row: dict[str, Any]) -> tuple[float, str]:
    if model_type == "logistic_regression":
        return (-row["abs_coefficient"], row["feature"])
    return (-row["importance"], row["feature"])


def _global_fi_row(model_type: str, feature: str, mean_score: float) -> dict[str, Any]:
    if model_type == "logistic_regression":
        return {"rank": 0, "feature": feature, "mean_abs_coefficient": mean_score}
    return {"rank": 0, "feature": feature, "mean_importance": mean_score}


def _global_fi_sort_key(model_type: str, row: dict[str, Any]) -> tuple[float, str]:
    if model_type == "logistic_regression":
        return (-row["mean_abs_coefficient"], row["feature"])
    return (-row["mean_importance"], row["feature"])


def _fi_method_name(model_type: str) -> str:
    if model_type == "logistic_regression":
        return "abs_logistic_regression_coefficient"
    if model_type == "random_forest":
        return "random_forest_gini_importance"
    raise ValueError(f"Unsupported model_type for feature importance: {model_type!r}")


def compute_feature_importance(
    models: dict[str, Any],
    label_list: list[str],
    feature_names: list[str],
    *,
    model_type: str,
    top_k: int = DEFAULT_FEATURE_IMPORTANCE_TOP_K,
) -> dict[str, Any]:
    """Rank features per label and globally (LR: |coefficient|; RF: Gini importance)."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    per_label: dict[str, list[dict[str, Any]]] = {}
    global_sum = [0.0] * len(feature_names)
    labels_with_scores = 0
    use_abs_for_global = model_type == "logistic_regression"

    for label in label_list:
        model = models.get(label)
        if model is None:
            continue
        scores = _extract_model_feature_scores(model, model_type)
        if len(scores) != len(feature_names):
            raise ValueError(
                f"Feature score length {len(scores)} != feature name length {len(feature_names)} "
                f"for label={label!r}"
            )
        ranked = [
            _per_label_fi_row(model_type, feature_names[idx], scores[idx])
            for idx in range(len(feature_names))
        ]
        ranked.sort(key=lambda row: _per_label_fi_sort_key(model_type, row))
        for rank, row in enumerate(ranked[:top_k], start=1):
            row["rank"] = rank
        per_label[label] = ranked[:top_k]

        labels_with_scores += 1
        for idx, value in enumerate(scores):
            global_sum[idx] += abs(value) if use_abs_for_global else value

    if labels_with_scores == 0:
        if model_type == "logistic_regression":
            raise ValueError("No logistic regression coefficients found for feature importance")
        raise ValueError("No random forest feature importances found")

    global_mean = [value / labels_with_scores for value in global_sum]
    global_ranked = [
        _global_fi_row(model_type, feature_names[idx], global_mean[idx])
        for idx in range(len(feature_names))
    ]
    global_ranked.sort(key=lambda row: _global_fi_sort_key(model_type, row))
    for rank, row in enumerate(global_ranked[:top_k], start=1):
        row["rank"] = rank

    return {
        "method": _fi_method_name(model_type),
        "top_k": top_k,
        "num_labels": labels_with_scores,
        "num_features": len(feature_names),
        "global_top": global_ranked[:top_k],
        "per_label": per_label,
    }


def _resolve_embedding_dim(
    spark,
    paths: dict[str, str],
    *,
    embedding_dim_source_df: DataFrame | None = None,
) -> int:
    if embedding_dim_source_df is not None:
        sample = (
            embedding_dim_source_df.select(EMBEDDING_VECTOR_COL)
            .filter(F.col(EMBEDDING_VECTOR_COL).isNotNull())
            .limit(1)
            .collect()
        )
        if not sample:
            raise ValueError("Cannot resolve embedding dimension from training/holdout features")
        return int(sample[0][EMBEDDING_VECTOR_COL].size)

    embeddings_raw = _read_delta(spark, paths["embeddings"], "embeddings")
    sample = (
        embeddings_raw.transform(_ensure_embedding_vector)
        .select(EMBEDDING_VECTOR_COL)
        .filter(F.col(EMBEDDING_VECTOR_COL).isNotNull())
        .limit(1)
        .collect()
    )
    if not sample:
        raise ValueError("Cannot resolve embedding dimension for feature importance")
    return int(sample[0][EMBEDDING_VECTOR_COL].size)


def _resolve_feature_names_for_importance(
    spark,
    paths: dict[str, str],
    feature_set: str,
    components: dict[str, bool],
    *,
    tfidf_vocab: list[str] | None = None,
    dcw_vocab: list[str] | None = None,
    embedding_dim_source_df: DataFrame | None = None,
) -> list[str]:
    if (components["tfidf"] or components["log_tfidf"]) and tfidf_vocab is None:
        tfidf_vocab = load_tfidf_vocab(spark, paths)
    if components["dcw"] and dcw_vocab is None:
        dcw_vocab = load_dcw_vocab(spark, paths)

    embedding_dim: int | None = None
    if components["embeddings"]:
        embedding_dim = _resolve_embedding_dim(
            spark, paths, embedding_dim_source_df=embedding_dim_source_df
        )

    return build_assembled_feature_names(
        feature_set,
        tfidf_vocab=tfidf_vocab,
        dcw_vocab=dcw_vocab,
        embedding_dim=embedding_dim,
    )


def _compute_feature_importance_for_run(
    spark,
    model_type: str,
    models: dict[str, Any],
    label_list: list[str],
    paths: dict[str, str],
    feature_set: str,
    components: dict[str, bool],
    *,
    top_k: int,
    tfidf_vocab: list[str] | None = None,
    dcw_vocab: list[str] | None = None,
    embedding_dim_source_df: DataFrame | None = None,
) -> dict[str, Any]:
    feature_names = _resolve_feature_names_for_importance(
        spark,
        paths,
        feature_set,
        components,
        tfidf_vocab=tfidf_vocab,
        dcw_vocab=dcw_vocab,
        embedding_dim_source_df=embedding_dim_source_df,
    )
    return compute_feature_importance(
        models,
        label_list,
        feature_names,
        model_type=model_type,
        top_k=top_k,
    )


def _global_top_score(feature_importance: dict[str, Any]) -> tuple[str, float]:
    """Return (feature_name, score) from the top global feature row."""
    row = feature_importance["global_top"][0]
    for key in ("mean_abs_coefficient", "mean_importance"):
        if key in row:
            return row["feature"], float(row[key])
    raise ValueError("global_top row missing mean_abs_coefficient or mean_importance")


def _iter_param_combos(
    keys: list[str],
    param_grid: dict[str, list[Any]],
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    value_lists = [param_grid.get(key, defaults[key]) for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


def _default_param_grid(model_type: str) -> dict[str, list[Any]]:
    if model_type == "logistic_regression":
        return DEFAULT_LR_PARAM_GRID
    if model_type == "random_forest":
        return DEFAULT_RF_PARAM_GRID
    raise ValueError(f"Unsupported model_type for grid search: {model_type!r}")


def _default_model_params(model_type: str) -> dict[str, Any]:
    if model_type == "logistic_regression":
        return dict(LR_PARAMS)
    if model_type == "random_forest":
        return dict(RF_PARAMS)
    raise ValueError(f"Unsupported model_type for grid search: {model_type!r}")


def _iter_param_combos(
    keys: list[str],
    param_grid: dict[str, list[Any]],
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    value_lists = [param_grid.get(key, defaults[key]) for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


def _default_param_grid(model_type: str) -> dict[str, list[Any]]:
    if model_type == "logistic_regression":
        return DEFAULT_LR_PARAM_GRID
    if model_type == "random_forest":
        return DEFAULT_RF_PARAM_GRID
    raise ValueError(f"Unsupported model_type for grid search: {model_type!r}")


def _default_model_params(model_type: str) -> dict[str, Any]:
    if model_type == "logistic_regression":
        return dict(LR_PARAMS)
    if model_type == "random_forest":
        return dict(RF_PARAMS)
    raise ValueError(f"Unsupported model_type for grid search: {model_type!r}")


def grid_search_hyperparameters(
    model_type: str,
    train_df: DataFrame,
    val_df: DataFrame,
    label_list: list[str],
    threshold: float,
    *,
    metric: str = "micro_f1",
    param_grid: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit one binary classifier per label on train for each grid point; pick params by val score."""
    if metric not in GRID_SEARCH_METRIC_CHOICES:
        raise ValueError(f"Unsupported grid-search metric: {metric!r}")
    if model_type not in MODEL_TYPE_CHOICES:
        raise ValueError(f"Unsupported model_type for grid search: {model_type!r}")

    grid = param_grid or _default_param_grid(model_type)
    keys = list(_default_model_params(model_type).keys())
    combos = _iter_param_combos(keys, grid, _default_model_params(model_type))
    model_short = "LR" if model_type == "logistic_regression" else "RF"
    logger.info(
        "%s grid search: %s combinations on val split (%s rows, metric=%s)",
        model_short,
        f"{len(combos):,}",
        f"{val_df.count():,}",
        metric,
    )

    best_score = float("-inf")
    best_params = _default_model_params(model_type)
    trial_results: list[dict[str, Any]] = []

    for trial_idx, params in enumerate(combos, start=1):
        logger.info("Grid search trial %s/%s: %s", trial_idx, len(combos), params)
        trial_models, _ = train_multilabel_model(
            train_df,
            max_labels=None,
            model_type=model_type,
            model_params=params,
            label_list=label_list,
        )
        val_predictions = predict_multilabel(
            val_df,
            label_list,
            threshold,
            models=trial_models,
            model_type=model_type,
        )
        val_metrics = _compute_multilabel_metrics(val_predictions, label_list)
        score = float(val_metrics[metric])
        trial_results.append({**params, metric: score, "val_documents": val_metrics["documents"]})
        logger.info("Grid search trial %s/%s %s=%.4f", trial_idx, len(combos), metric, score)
        if score > best_score:
            best_score = score
            best_params = dict(params)

    logger.info(
        "Grid search best params: %s (%s=%.4f)",
        best_params,
        metric,
        best_score,
    )
    return best_params, trial_results


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
    *,
    label_list: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if model_type == "random_forest":
        logger.warning(
            "RandomForestClassifier may be slow or memory-heavy on high-dimensional sparse text features."
        )
    logger.info("%s hyperparameters: %s", model_type, model_params)

    label_list = label_list or _collect_training_labels(train_df, max_labels)
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


def parse_threshold_sweep_values(raw: str | None) -> tuple[float, ...]:
    if not raw:
        return DEFAULT_THRESHOLD_SWEEP
    values: list[float] = []
    for part in raw.split(","):
        cleaned = part.strip()
        if cleaned:
            values.append(float(cleaned))
    if not values:
        raise ValueError("No thresholds parsed from --threshold-sweep")
    return tuple(values)


def compute_threshold_sweep(
    predictions: DataFrame,
    label_list: list[str],
    thresholds: tuple[float, ...] | list[float],
    *,
    prob_column_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate holdout metrics at multiple probability thresholds using saved prob_* columns."""
    prob_cols = prob_columns_in_df(predictions)
    if not prob_cols:
        raise ValueError(
            "Prediction Delta has no prob_* columns. Re-run --predict-stage predict after rebuilding."
        )

    prob_map = prob_column_map or {
        label: col
        for label, col in label_prob_column_map(label_list).items()
        if col in prob_cols
    }
    if not prob_map:
        raise ValueError("Could not map any labels to prob_* columns on the prediction Delta")

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        scored = apply_multilabel_threshold(
            predictions,
            label_list,
            float(threshold),
            prob_column_map=prob_map,
        )
        metrics = evaluate_multilabel(scored, label_list)
        for split, values in metrics.items():
            if isinstance(values, dict):
                rows.append({"threshold": float(threshold), "split": split, **values})

    return {
        "thresholds": [float(t) for t in thresholds],
        "num_probability_columns": len(prob_cols),
        "rows": rows,
    }


def save_threshold_sweep_outputs(
    spark,
    exp_id: str,
    batch_date: str | None,
    sweep: dict[str, Any],
    *,
    prediction_delta_path: str,
    prediction_ts: str | None = None,
) -> str:
    pkl_path = model_bank_threshold_sweep_path(exp_id, batch_date)
    json_path = pkl_path.replace(".pkl", ".json")
    payload = {
        "exp_id": exp_id,
        "prediction_delta_path": prediction_delta_path,
        "prediction_ts": prediction_ts,
        "thresholds": sweep["thresholds"],
        "num_probability_columns": sweep["num_probability_columns"],
        "rows": sweep["rows"],
    }
    save_pickle(pkl_path, payload, spark)
    save_json(json_path, payload, spark)
    logger.info("Saved threshold sweep to %s", pkl_path)
    logger.info("Saved threshold sweep JSON to %s", json_path)
    return pkl_path


def _hadoop_path_exists(spark, path: str) -> bool:
    sc = spark.sparkContext
    URI = sc._jvm.java.net.URI
    Path = sc._jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI(path), sc._jsc.hadoopConfiguration())
    return fs.exists(Path(path))


def _feature_importance_json_exists(spark, exp_id: str) -> str | None:
    for subdir_key in ("feature_importance_dir", "model_dir"):
        base = model_bank_experiment_subdir(exp_id, subdir_key)
        matches = [
            name
            for name in _list_hadoop_child_names(spark, base)
            if name.startswith("feature_importance_") and name.endswith(".json")
        ]
        if matches:
            return f"{base}/{sorted(matches)[-1]}"
    return None


def _load_checkpointed_predictions(
    spark,
    args: argparse.Namespace,
    label_list: list[str],
) -> tuple[DataFrame, str, str]:
    pred_delta_path = resolve_prediction_delta_path(
        spark,
        args.prediction_date,
        resolve_prediction_exp_id(args),
        prediction_suffix=args.prediction_suffix,
    )
    logger.info("Loading predictions from %s", pred_delta_path)
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

    prediction_ts_rows = predictions.select("prediction_ts").distinct().limit(2).collect()
    if len(prediction_ts_rows) != 1:
        raise ValueError(
            f"Expected one prediction_ts in {pred_delta_path}, found {len(prediction_ts_rows)}"
        )
    prediction_ts = prediction_ts_rows[0].prediction_ts
    return predictions, pred_delta_path, prediction_ts


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
    pred_exp_id = _prediction_exp_id_from_delta_path(pred_delta_path) or run_id
    batch_date = _prediction_batch_date_from_delta_path(pred_delta_path)
    pred_pkl_path = model_bank_prediction_metrics_manifest_path(pred_exp_id, batch_date)
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




def create_pipeline_spark_session(app_name: str):
    return create_spark_session(app_name)
