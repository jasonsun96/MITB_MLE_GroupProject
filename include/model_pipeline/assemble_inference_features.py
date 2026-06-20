"""Assemble inference-ready X_unlabelled from Gold corpus tables.

This applies frozen feature artifacts from model_bank/features/{feature_run_id}
to documents marked with category='inference' in gold/label_store. It does not
refit TF-IDF or DCW statistics.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.ml.linalg import Vector, Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, MapType, StringType

_PIPELINE_DIR = Path(__file__).resolve().parent
_INCLUDE_DIR = _PIPELINE_DIR.parent
_GOLD_DIR = _INCLUDE_DIR / "gold"
_PROJECT_ROOT = _INCLUDE_DIR.parent
for _path in (_PROJECT_ROOT, _INCLUDE_DIR, _GOLD_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from gold_io import load_pickle
from model_pipeline.multilabel_core import (
    DCW_FEATURES_COL,
    DOCUMENT_ID_COL,
    EMBEDDING_COL,
    EMBEDDING_VECTOR_COL,
    FEATURES_COL,
    SPLIT_COL,
    _ensure_embedding_vector,
    _feature_components,
    build_feature_column,
    create_pipeline_spark_session,
    load_dcw_vocab,
    load_schema_paths,
)
from tfidf_processing import add_tfidf_column, load_tfidf_artifact

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_RUN_ID = "run004"
DEFAULT_GOLD_RUN_ID = "run004"
DEFAULT_FEATURE_SET = "tfidf_dcw"
DEFAULT_INFERENCE_CATEGORY = "inference"


def _dcw_score_from_artifact(spark, paths: dict[str, str]) -> dict[str, float]:
    """Load frozen DCW scores from dcw.pkl, falling back to dcw_score Delta."""
    try:
        bundle = load_pickle(paths["dcw_pkl"], spark)
        score = bundle.get("score")
        if isinstance(score, dict) and score:
            logger.info("Loaded DCW score map from %s", paths["dcw_pkl"])
            return {str(k): float(v) for k, v in score.items()}
    except Exception:
        logger.exception("Could not load DCW pickle from %s; falling back to dcw_score", paths["dcw_pkl"])

    rows = (
        spark.read.format("delta")
        .load(paths["dcw_score_path"])
        .select("lemma", "score")
        .filter(F.col("lemma").isNotNull() & F.col("score").isNotNull())
        .collect()
    )
    if not rows:
        raise ValueError(f"No DCW scores found at {paths['dcw_score_path']}")
    logger.info("Loaded DCW score map from %s", paths["dcw_score_path"])
    return {row.lemma: float(row.score) for row in rows}


def add_dcw_features_column(df: DataFrame, score: dict[str, float]) -> DataFrame:
    """Add sparse dcw_features map: lemma -> count(lemma in doc) * frozen score."""
    score_bc = df.sparkSession.sparkContext.broadcast(score)

    @udf(MapType(StringType(), DoubleType()))
    def compute_dcw_map(pos_counts: dict | None) -> dict[str, float]:
        if not pos_counts:
            return {}
        scores = score_bc.value
        out: dict[str, float] = {}
        for pos in ("NOUN", "PROPN"):
            values = pos_counts.get(pos) or {}
            for lemma, count in values.items():
                weight = scores.get(lemma)
                if weight is not None:
                    out[lemma] = out.get(lemma, 0.0) + float(count) * float(weight)
        return out

    return df.withColumn(DCW_FEATURES_COL, compute_dcw_map(F.col("pos_counts")))


def _load_inference_ids(
    spark,
    labels_path: str,
    category: str,
    batch_id: str | None,
) -> DataFrame:
    labels = spark.read.format("delta").load(labels_path)
    required = {DOCUMENT_ID_COL, SPLIT_COL}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels table missing required columns: {sorted(missing)}")

    df = labels.filter(F.col(SPLIT_COL) == category)
    if batch_id and "batch_id" in df.columns:
        df = df.filter(F.col("batch_id") == batch_id)

    select_cols = [DOCUMENT_ID_COL, SPLIT_COL]
    if "snapshot_date" in df.columns:
        select_cols.append("snapshot_date")
    if "batch_id" in df.columns:
        select_cols.append("batch_id")
    elif batch_id:
        df = df.withColumn("batch_id", F.lit(batch_id))
        select_cols.append("batch_id")

    ids = df.select(*select_cols).dropDuplicates([DOCUMENT_ID_COL])
    count = ids.count()
    if count == 0:
        batch_msg = f" and batch_id={batch_id!r}" if batch_id else ""
        raise ValueError(f"No documents found with category={category!r}{batch_msg}")
    logger.info("Loaded %s inference document ids", f"{count:,}")
    return ids


def _load_embeddings(spark, paths: dict[str, str]) -> DataFrame:
    raw = spark.read.format("delta").load(paths["embeddings"])
    if EMBEDDING_COL not in raw.columns:
        raise ValueError(f"embeddings table missing {EMBEDDING_COL!r}")
    return (
        raw.select(DOCUMENT_ID_COL, EMBEDDING_COL)
        .dropDuplicates([DOCUMENT_ID_COL])
        .transform(_ensure_embedding_vector)
        .select(DOCUMENT_ID_COL, EMBEDDING_VECTOR_COL)
    )


def assemble_inference_features(
    spark,
    paths: dict[str, str],
    *,
    feature_set: str,
    category: str,
    batch_id: str | None,
    limit: int | None,
) -> DataFrame:
    components = _feature_components(feature_set)
    ids = _load_inference_ids(spark, paths["labels"], category, batch_id)
    if limit:
        ids = ids.limit(limit)
        logger.info("Smoke test: limited inference ids to %s rows", f"{limit:,}")

    assembled = ids

    if components["tfidf"] or components["log_tfidf"]:
        ngrams = spark.read.format("delta").load(paths["ngrams"])
        tfidf_artifact = load_tfidf_artifact(paths["tfidf_pkl"], spark)
        tfidf_df = add_tfidf_column(ngrams.join(ids.select(DOCUMENT_ID_COL), DOCUMENT_ID_COL, "inner"), tfidf_artifact)
        keep = [DOCUMENT_ID_COL]
        if components["tfidf"]:
            keep.append("tfidf")
        if components["log_tfidf"]:
            keep.append("log_tfidf")
        assembled = assembled.join(tfidf_df.select(*keep), DOCUMENT_ID_COL, "inner")
        logger.info("Joined frozen TF-IDF features")

    dcw_vocab: list[str] | None = None
    if components["dcw"]:
        pos = spark.read.format("delta").load(paths["pos_tags"])
        dcw_score = _dcw_score_from_artifact(spark, paths)
        dcw_vocab = load_dcw_vocab(spark, paths)
        dcw_df = add_dcw_features_column(pos.join(ids.select(DOCUMENT_ID_COL), DOCUMENT_ID_COL, "inner"), dcw_score)
        assembled = assembled.join(dcw_df.select(DOCUMENT_ID_COL, DCW_FEATURES_COL), DOCUMENT_ID_COL, "inner")
        logger.info("Joined frozen DCW features")

    if components["embeddings"]:
        embeddings = _load_embeddings(spark, paths)
        assembled = assembled.join(embeddings, DOCUMENT_ID_COL, "inner")
        logger.info("Joined embeddings")

    before_features = assembled.count()
    if before_features == 0:
        raise ValueError("No inference rows remain after joining requested feature tables")

    with_features = build_feature_column(assembled, feature_set, dcw_vocab=dcw_vocab)
    output = (
        with_features
        .withColumn("feature_run_id", F.lit(paths["feature_run_id"]))
        .withColumn("feature_set", F.lit(feature_set))
        .withColumn("assembled_at", F.lit(datetime.now(timezone.utc).isoformat()))
        .select(
            DOCUMENT_ID_COL,
            *([SPLIT_COL] if SPLIT_COL in with_features.columns else []),
            *([F.col("snapshot_date")] if "snapshot_date" in with_features.columns else []),
            *([F.col("batch_id")] if "batch_id" in with_features.columns else []),
            FEATURES_COL,
            *([DCW_FEATURES_COL] if DCW_FEATURES_COL in with_features.columns else []),
            "feature_run_id",
            "feature_set",
            "assembled_at",
        )
    )
    logger.info("Assembled %s inference feature rows", f"{output.count():,}")
    return output


def write_x_unlabelled(df: DataFrame, path: str, batch_id: str | None) -> None:
    writer = df.write.format("delta").option("mergeSchema", "true")
    if batch_id and "batch_id" in df.columns:
        (
            writer.mode("overwrite")
            .partitionBy("batch_id")
            .option("replaceWhere", f"batch_id = '{batch_id}'")
            .save(path)
        )
    else:
        writer.mode("overwrite").save(path)
    logger.info("Wrote X_unlabelled to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble inference X_unlabelled from Gold corpus features")
    parser.add_argument("--feature-run-id", default=DEFAULT_FEATURE_RUN_ID)
    parser.add_argument("--gold-run-id", default=DEFAULT_GOLD_RUN_ID)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--category", default=DEFAULT_INFERENCE_CATEGORY)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    paths = load_schema_paths(args.feature_run_id, gold_run_id=args.gold_run_id)
    logger.info("Feature run: %s", args.feature_run_id)
    logger.info("Gold run: %s", args.gold_run_id)
    logger.info("Output X_unlabelled: %s", paths["X_unlabelled"])

    spark = create_pipeline_spark_session("assemble-inference-features")
    try:
        output = assemble_inference_features(
            spark,
            paths,
            feature_set=args.feature_set,
            category=args.category,
            batch_id=args.batch_id,
            limit=args.limit,
        )
        write_x_unlabelled(output, paths["X_unlabelled"], args.batch_id)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
