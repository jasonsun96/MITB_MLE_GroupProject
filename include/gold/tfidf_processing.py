"""
TF-IDF feature freezing for the legal corpus.

Reads precomputed Gold n-gram counts, joins the Gold labels/split table, fits
vocabulary and IDF on train documents only, applies the frozen artifact to all
splits, and writes TF-IDF + log-TF-IDF feature tables plus model_bank artefacts.

Log-TF-IDF weighting: log(tf) × (1 + log(idf)) using frozen train IDF values.

Upstream:  gold/ngrams        (run ngram_processing.py first)
Split:      gold/labels        (document_id, category)
Outputs:    gold/runs/{run_id}/tfidf_train
            gold/runs/{run_id}/tfidf_val_test_oot
            model_bank/runs/{run_id}/feature_extractors/tfidf.pkl

notes:
  - vocabulary and IDF are learned from category == "train" only
  - val/test/oot never influence vocab selection or IDF
  - --no-split treats all ngram_count rows as train (smoke tests only)
"""

import argparse
import logging
import math
from datetime import datetime, timezone

from pyspark.ml.linalg import Vector, Vectors, VectorUDT
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from utils.spark_session import create_spark_session

from gold_io import (PARTITION_COL, bootstrap_paths, columns_with_snapshot,
                     load_pickle, save_json, save_pickle, write_delta)
from run_paths import default_feature_run_id, resolve_feature_run_paths

bootstrap_paths()

logger = logging.getLogger(__name__)

MIN_N = 1
MAX_N = 3
MAX_FEATURES = 50_000
MIN_DOC_FREQ = 2
TRAIN_CATEGORY = "train"
LOG_TFIDF_FORMULA = "log(tf) * (1 + log(idf))"

TFIDF_OUTPUT_COLUMNS = [
    "doc_index",
    "document_id",
    "tfidf",
    "log_tfidf",
    "silver_ingest_ts",
    "silver_source",
]


def load_ngram_counts(spark, ngram_path: str, limit: int | None = None):
    raw = spark.read.format("delta").load(ngram_path)

    required = {
        "document_id",
        "labels",
        "tokens",
        "token_count",
        "ngram_counts",
        "text_source",
        "silver_ingest_ts",
        "silver_source",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Required column(s) missing from ngram_count table: {sorted(missing)}")

    df = raw
    if limit:
        df = df.limit(limit)
        logger.info("Smoke test mode: limited ngram_count to %s rows", f"{limit:,}")

    count = df.count()
    logger.info("Loaded %s rows from ngram_count at %s", f"{count:,}", ngram_path)
    return df


def load_split_labels(spark, labels_path: str):
    labels = spark.read.format("delta").load(labels_path)

    required = {"document_id", "category"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Required column(s) missing from labels table: {sorted(missing)}")

    split = labels.select("document_id", "category").dropDuplicates(["document_id"])
    count = split.count()
    logger.info("Loaded %s label/split rows from %s", f"{count:,}", labels_path)
    return split


def join_ngrams_with_labels(ngrams, labels):
    joined = ngrams.join(labels, on="document_id", how="inner")
    n_in = ngrams.count()
    n_out = joined.count()
    logger.info(
        "Joined ngram_count with labels: %s -> %s rows (%s unmatched)",
        f"{n_in:,}",
        f"{n_out:,}",
        f"{n_in - n_out:,}",
    )
    return joined


def _build_vocab_from_train(train_df, max_features: int, min_doc_freq: int) -> tuple[list[str], dict[str, int]]:
    doc_ngrams = train_df.select("document_id", F.explode(F.map_keys(F.col("ngram_counts"))).alias("ngram")).dropDuplicates(["document_id", "ngram"])

    vocab_df = doc_ngrams.groupBy("ngram").agg(F.countDistinct("document_id").alias("doc_freq")).filter(F.col("doc_freq") >= min_doc_freq).orderBy(F.desc("doc_freq"), "ngram").limit(max_features)

    vocab = [row.ngram for row in vocab_df.collect()]
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(vocab)}
    logger.info(
        "Built train-only vocabulary: %s features (min_doc_freq=%s, max_features=%s)",
        f"{len(vocab):,}",
        min_doc_freq,
        max_features,
    )
    return vocab, ngram_to_idx


def _fit_idf_values(train_df, vocab: list[str], n_train: int) -> list[float]:
    """Smoothed IDF matching Spark ML defaults: log((m + 1) / (d + 1))."""
    if not vocab:
        return []

    doc_freq_rows = (
        train_df.select("document_id", F.explode(F.map_keys(F.col("ngram_counts"))).alias("ngram"))
        .filter(F.col("ngram").isin(vocab))
        .dropDuplicates(["document_id", "ngram"])
        .groupBy("ngram")
        .agg(F.countDistinct("document_id").alias("doc_freq"))
        .collect()
    )
    doc_freq = {row.ngram: row.doc_freq for row in doc_freq_rows}
    return [math.log((n_train + 1) / (doc_freq.get(ngram, 0) + 1)) for ngram in vocab]


def _make_map_to_vector_udf(ngram_to_idx: dict[str, int], vocab_size: int):
    idx_map = ngram_to_idx

    @udf(VectorUDT())
    def map_to_count_vector(ngram_map: dict | None) -> Vector:
        if not ngram_map:
            return Vectors.sparse(vocab_size, [], [])

        pairs: list[tuple[int, float]] = []
        for ngram, count in ngram_map.items():
            idx = idx_map.get(ngram)
            if idx is not None:
                pairs.append((idx, float(count)))

        pairs.sort(key=lambda item: item[0])
        indices = [idx for idx, _ in pairs]
        values = [value for _, value in pairs]
        return Vectors.sparse(vocab_size, indices, values)

    return map_to_count_vector


def _make_tfidf_udf(idf_values: list[float], vocab_size: int):
    idf_arr = idf_values

    @udf(VectorUDT())
    def counts_to_tfidf(count_vector: Vector | None) -> Vector:
        if count_vector is None or count_vector.numNonzeros() == 0:
            return Vectors.sparse(vocab_size, [], [])

        indices: list[int] = []
        values: list[float] = []
        for idx, tf in zip(count_vector.indices, count_vector.values):
            weight = float(tf) * idf_arr[int(idx)]
            if weight != 0.0:
                indices.append(int(idx))
                values.append(weight)

        return Vectors.sparse(vocab_size, indices, values)

    return counts_to_tfidf


def _make_log_tfidf_udf(idf_values: list[float], vocab_size: int):
    idf_arr = idf_values

    @udf(VectorUDT())
    def counts_to_log_tfidf(count_vector: Vector | None) -> Vector:
        if count_vector is None or count_vector.numNonzeros() == 0:
            return Vectors.sparse(vocab_size, [], [])

        out_indices: list[int] = []
        out_values: list[float] = []
        for idx, tf in zip(count_vector.indices, count_vector.values):
            tf = float(tf)
            if tf <= 0:
                continue
            idf = idf_arr[int(idx)]
            if idf <= 0:
                continue
            weight = math.log(tf) * (1.0 + math.log(idf))
            if weight != 0.0:
                out_indices.append(int(idx))
                out_values.append(weight)

        return Vectors.sparse(vocab_size, out_indices, out_values)

    return counts_to_log_tfidf


def build_tfidf_artifact(train_df, max_features: int = MAX_FEATURES, min_doc_freq: int = MIN_DOC_FREQ) -> dict:
    n_train = train_df.count()
    if n_train == 0:
        raise ValueError("Cannot fit TF-IDF artifact: train split is empty")

    vocab, ngram_to_idx = _build_vocab_from_train(train_df, max_features, min_doc_freq)
    vocab_size = len(vocab)
    if vocab_size == 0:
        raise ValueError("Cannot fit TF-IDF artifact: train vocabulary is empty")

    idf_values = _fit_idf_values(train_df, vocab, n_train)
    train_document_ids = [row.document_id for row in train_df.select("document_id").distinct().collect()]

    logger.info("Fitted IDF on %s train documents", f"{n_train:,}")

    return {
        "vocab": vocab,
        "ngram_to_idx": ngram_to_idx,
        "vocab_size": vocab_size,
        "idf_values": idf_values,
        "train_document_ids": train_document_ids,
        "max_features": max_features,
        "min_doc_freq": min_doc_freq,
        "n_train_documents": n_train,
    }


def save_tfidf_artifact(
    artifact: dict,
    paths: dict[str, str],
    ngrams_path: str,
    no_split: bool,
    spark,
) -> None:
    meta = {
        "gold_ingest_ts": datetime.now(timezone.utc).isoformat(),
        "run_id": paths["run_id"],
        "ngrams_source": ngrams_path,
        "n_train_documents": artifact["n_train_documents"],
        "n_features": artifact["vocab_size"],
        "ngram_range": [MIN_N, MAX_N],
        "max_features": artifact["max_features"],
        "min_doc_freq": artifact["min_doc_freq"],
        "train_category": TRAIN_CATEGORY,
        "no_split": no_split,
        "ngrams_table": paths["ngrams"],
        "labels_table": paths["labels"],
        "tfidf_train_table": paths["tfidf_train"],
        "tfidf_val_test_oot_table": paths["tfidf_val_test_oot"],
        "counts_column": "ngram_counts",
        "tfidf_column": "tfidf",
        "log_tfidf_column": "log_tfidf",
        "log_tfidf_formula": LOG_TFIDF_FORMULA,
        "log_base": "natural",
        "final_feature_columns": ["tfidf", "log_tfidf"],
        "partition_col": PARTITION_COL,
    }
    save_pickle(paths["tfidf_pkl"], {"artifact": artifact, "meta": meta}, spark)
    logger.info("Saved TF-IDF extractor to %s", paths["tfidf_pkl"])
    _save_tfidf_json_exports(artifact, meta, paths, spark)


def _save_tfidf_json_exports(artifact: dict, meta: dict, paths: dict[str, str], spark) -> None:
    """Human-readable JSON mirrors (monitoring / audit) alongside tfidf.pkl."""
    json_dir = paths["tfidf_json_dir"]
    suffix = f"{MIN_N}_{MAX_N}"
    exports = {
        f"gold_meta_{suffix}.json": meta,
        f"gold_vocab_{suffix}.json": artifact["vocab"],
        f"gold_ngram_index_{suffix}.json": artifact["ngram_to_idx"],
        f"gold_idf_{suffix}.json": artifact["idf_values"],
        "gold_train_document_ids.json": artifact["train_document_ids"],
    }
    for filename, payload in exports.items():
        path = f"{json_dir}/{filename}"
        save_json(path, payload, spark)
        logger.info("Saved TF-IDF JSON export to %s", path)


def load_tfidf_artifact(pkl_path: str, spark) -> dict:
    bundle = load_pickle(pkl_path, spark)
    artifact = bundle["artifact"]
    return {
        "vocab": artifact["vocab"],
        "ngram_to_idx": artifact["ngram_to_idx"],
        "vocab_size": artifact["vocab_size"],
        "idf_values": artifact["idf_values"],
        "train_document_ids": artifact["train_document_ids"],
    }


def add_tfidf_column(df, artifact: dict):
    vocab_size = artifact["vocab_size"]
    map_udf = _make_map_to_vector_udf(artifact["ngram_to_idx"], vocab_size)
    tfidf_udf = _make_tfidf_udf(artifact["idf_values"], vocab_size)
    log_tfidf_udf = _make_log_tfidf_udf(artifact["idf_values"], vocab_size)
    return (
        df.withColumn("_count_vector", map_udf(F.col("ngram_counts")))
        .withColumn("tfidf", tfidf_udf(F.col("_count_vector")))
        .withColumn("log_tfidf", log_tfidf_udf(F.col("_count_vector")))
        .drop("_count_vector")
    )


def select_tfidf_output(df):
    return df.select(*columns_with_snapshot(df, TFIDF_OUTPUT_COLUMNS))


def main():
    parser = argparse.ArgumentParser(description="Gold layer: freeze TF-IDF and log(tf)×(1+log(idf)) from ngram_count")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Feature run id (e.g. run001). Default: schema gold.runs.default_run_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit ngrams rows for smoke testing. Omit for full corpus.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Smoke test only: fit and score on all ngram_count rows (no labels join).",
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

    run_id = args.run_id or default_feature_run_id()
    paths = resolve_feature_run_paths(run_id)

    logger.info("Run ID              : %s", run_id)
    logger.info("Input  (ngrams)     : %s", paths["ngrams"])
    logger.info("Input  (labels)     : %s", paths["labels"])
    logger.info("Output (train)      : %s", paths["tfidf_train"])
    logger.info("Output (holdout)    : %s", paths["tfidf_val_test_oot"])
    logger.info("TF-IDF extractor    : %s", paths["tfidf_pkl"])

    spark = create_spark_session("gold-tfidf")
    ngrams = load_ngram_counts(spark, paths["ngrams"], args.limit)

    if args.no_split:
        logger.warning("--no-split enabled: using all ngram_count rows as train (smoke test only)")
        train_df = ngrams
        holdout_df = ngrams.limit(0)
    else:
        labels = load_split_labels(spark, paths["labels"])
        joined = join_ngrams_with_labels(ngrams, labels)
        train_df = joined.filter(F.col("category") == TRAIN_CATEGORY)
        holdout_df = joined.filter(F.col("category") != TRAIN_CATEGORY)

        n_train = train_df.count()
        n_holdout = holdout_df.count()
        logger.info("Split: %s train | %s val/test/oot", f"{n_train:,}", f"{n_holdout:,}")

    min_doc_freq = MIN_DOC_FREQ
    if args.no_split and args.limit:
        min_doc_freq = 1
        logger.warning(
            "Smoke test: using min_doc_freq=1 (production default is %s)",
            MIN_DOC_FREQ,
        )

    artifact = build_tfidf_artifact(train_df, MAX_FEATURES, min_doc_freq)
    save_tfidf_artifact(artifact, paths, paths["ngrams"], args.no_split, spark)

    partition_col = PARTITION_COL if PARTITION_COL in train_df.columns else None

    train_features = select_tfidf_output(add_tfidf_column(train_df, artifact))
    holdout_features = select_tfidf_output(add_tfidf_column(holdout_df, artifact))

    write_delta(train_features, paths["tfidf_train"], partition_col)
    write_delta(holdout_features, paths["tfidf_val_test_oot"], partition_col)

    train_count = spark.read.format("delta").load(paths["tfidf_train"]).count()
    holdout_count = spark.read.format("delta").load(paths["tfidf_val_test_oot"]).count()
    logger.info("Wrote %s rows to %s", f"{train_count:,}", paths["tfidf_train"])
    logger.info("Wrote %s rows to %s", f"{holdout_count:,}", paths["tfidf_val_test_oot"])
    logger.info("TF-IDF + log-TF-IDF feature freezing complete")


if __name__ == "__main__":
    main()
