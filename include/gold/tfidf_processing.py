"""
TF-IDF feature freezing for the legal corpus.

Reads precomputed Gold n-gram counts, joins the Gold labels/split table, fits
vocabulary and IDF on train documents only, applies the frozen artifact to all
splits, and writes TF-IDF + log-TF-IDF feature tables plus model_bank artefacts.

Log-TF-IDF weighting: log(tf) × (1 + log(idf)) using frozen train IDF values.

Upstream:  gold/ngram_count   (run ngram_processing.py first)
Split:      gold/labels        (document_id, category)
Outputs:    gold/tfidf_features_train
            gold/tfidf_features_val_test_oot
            model_bank/features_extractor/tfidf/

notes:
  - vocabulary and IDF are learned from category == "train" only
  - val/test/oot never influence vocab selection or IDF
  - --no-split treats all ngram_count rows as train (smoke tests only)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pyspark.ml.feature import IDF
from pyspark.ml.linalg import Vector, Vectors, VectorUDT
from pyspark.sql import functions as F
from pyspark.sql.functions import udf

from gold_io import (
    PARTITION_COL,
    PROJECT_ROOT,
    bootstrap_paths,
    columns_with_snapshot,
    save_bytes_to_path,
    write_delta,
)
from utils.spark_session import create_spark_session

bootstrap_paths()

logger = logging.getLogger(__name__)

MIN_N = 1
MAX_N = 3
MAX_FEATURES = 50_000
MIN_DOC_FREQ = 2
TRAIN_CATEGORY = "train"
LOG_TFIDF_FORMULA = "log(tf) * (1 + log(idf))"

NGRAM_INDEX_FILE = "gold_ngram_index_1_3.json"
IDF_FILE = "gold_idf_1_3.json"
TRAIN_DOC_IDS_FILE = "gold_train_document_ids.json"

TFIDF_OUTPUT_COLUMNS = [
    "doc_index",
    "document_id",
    "tfidf",
    "log_tfidf",
    "silver_ingest_ts",
    "silver_source",
]


def _load_json_from_path(path: str, spark) -> Any:
    jvm = spark.sparkContext._jvm
    uri = jvm.java.net.URI(path)
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, spark.sparkContext._jsc.hadoopConfiguration())
    stream = fs.open(hadoop_path)
    reader = jvm.java.io.BufferedReader(jvm.java.io.InputStreamReader(stream))
    lines = []
    line = reader.readLine()
    while line is not None:
        lines.append(line)
        line = reader.readLine()
    reader.close()
    return json.loads("".join(lines))


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
    doc_ngrams = (
        train_df.select("document_id", F.explode(F.map_keys(F.col("ngram_counts"))).alias("ngram"))
        .dropDuplicates(["document_id", "ngram"])
    )

    vocab_df = (
        doc_ngrams.groupBy("ngram")
        .agg(F.countDistinct("document_id").alias("doc_freq"))
        .filter(F.col("doc_freq") >= min_doc_freq)
        .orderBy(F.desc("doc_freq"), "ngram")
        .limit(max_features)
    )

    vocab = [row.ngram for row in vocab_df.collect()]
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(vocab)}
    logger.info(
        "Built train-only vocabulary: %s features (min_doc_freq=%s, max_features=%s)",
        f"{len(vocab):,}",
        min_doc_freq,
        max_features,
    )
    return vocab, ngram_to_idx


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

    map_udf = _make_map_to_vector_udf(ngram_to_idx, vocab_size)
    train_vectors = train_df.withColumn("count_vector", map_udf(F.col("ngram_counts")))

    idf_model = IDF(inputCol="count_vector", outputCol="tfidf").fit(train_vectors)
    idf_values = [float(x) for x in idf_model.idf.toArray()]
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
    artifact_paths: dict[str, str],
    ngram_count_path: str,
    no_split: bool,
    spark,
) -> None:
    save_bytes_to_path(artifact_paths["vocab"], json.dumps(artifact["vocab"]).encode("utf-8"), spark)
    save_bytes_to_path(
        artifact_paths["ngram_index"],
        json.dumps(artifact["ngram_to_idx"]).encode("utf-8"),
        spark,
    )
    save_bytes_to_path(artifact_paths["idf"], json.dumps(artifact["idf_values"]).encode("utf-8"), spark)
    save_bytes_to_path(
        artifact_paths["train_doc_ids"],
        json.dumps(artifact["train_document_ids"]).encode("utf-8"),
        spark,
    )

    meta = {
        "gold_ingest_ts": datetime.now(timezone.utc).isoformat(),
        "ngram_count_source": ngram_count_path,
        "n_train_documents": artifact["n_train_documents"],
        "n_features": artifact["vocab_size"],
        "ngram_range": [MIN_N, MAX_N],
        "max_features": artifact["max_features"],
        "min_doc_freq": artifact["min_doc_freq"],
        "train_category": TRAIN_CATEGORY,
        "no_split": no_split,
        "ngram_count_table": paths["ngram_count"],
        "labels_table": paths["labels"],
        "tfidf_features_train_table": paths["tfidf_features_train"],
        "tfidf_features_val_test_oot_table": paths["tfidf_features_val_test_oot"],
        "model_bank": paths["model_bank"],
        "vocab_path": artifact_paths["vocab"],
        "ngram_index_path": artifact_paths["ngram_index"],
        "idf_path": artifact_paths["idf"],
        "train_document_ids_path": artifact_paths["train_doc_ids"],
        "meta_path": artifact_paths["meta"],
        "vocab_file": Path(artifact_paths["vocab"]).name,
        "counts_column": "ngram_counts",
        "tfidf_column": "tfidf",
        "log_tfidf_column": "log_tfidf",
        "log_tfidf_formula": LOG_TFIDF_FORMULA,
        "log_base": "natural",
        "final_feature_columns": ["tfidf", "log_tfidf"],
        "partition_col": PARTITION_COL,
    }
    save_bytes_to_path(artifact_paths["meta"], json.dumps(meta, indent=2).encode("utf-8"), spark)

    logger.info("Saved TF-IDF model_bank artifacts under %s", paths["model_bank"])
    logger.info("  vocab: %s", artifact_paths["vocab"])
    logger.info("  ngram_index: %s", artifact_paths["ngram_index"])
    logger.info("  idf: %s", artifact_paths["idf"])
    logger.info("  train_document_ids: %s", artifact_paths["train_doc_ids"])
    logger.info("  meta: %s", artifact_paths["meta"])


def load_tfidf_artifact(artifact_paths: dict[str, str], spark) -> dict:
    vocab = _load_json_from_path(artifact_paths["vocab"], spark)
    ngram_to_idx = _load_json_from_path(artifact_paths["ngram_index"], spark)
    idf_values = _load_json_from_path(artifact_paths["idf"], spark)
    train_document_ids = _load_json_from_path(artifact_paths["train_doc_ids"], spark)

    return {
        "vocab": vocab,
        "ngram_to_idx": ngram_to_idx,
        "vocab_size": len(vocab),
        "idf_values": idf_values,
        "train_document_ids": train_document_ids,
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
    parser = argparse.ArgumentParser(
        description="Gold layer: freeze TF-IDF and log(tf)×(1+log(idf)) from ngram_count"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit ngram_count rows for smoke testing. Omit for full corpus.",
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

    with open(PROJECT_ROOT / "schema.yaml") as f:
        schema = yaml.safe_load(f)

    gold_path = schema["gold"]["path"]
    gold_tables = schema["gold"]["tables"]
    tfidf_extractor = schema["model_bank"]["features_extractor"]["tfidf"]
    extractor_base = f"{schema['model_bank']['path']}/{tfidf_extractor['path']}"

    paths = {
        "ngram_count": f"{gold_path}/{gold_tables['ngram_count']['path']}",
        "labels": f"{gold_path}/{gold_tables['labels']['path']}",
        "tfidf_features_train": f"{gold_path}/{gold_tables['tfidf_features_train']['path']}",
        "tfidf_features_val_test_oot": f"{gold_path}/{gold_tables['tfidf_features_val_test_oot']['path']}",
        "model_bank": extractor_base,
    }
    artifact_paths = {
        "vocab": f"{extractor_base}/{tfidf_extractor['vocab_file']}",
        "meta": f"{extractor_base}/{tfidf_extractor['meta_file']}",
        "ngram_index": f"{extractor_base}/{NGRAM_INDEX_FILE}",
        "idf": f"{extractor_base}/{IDF_FILE}",
        "train_doc_ids": f"{extractor_base}/{TRAIN_DOC_IDS_FILE}",
    }

    logger.info("Input  (ngram_count): %s", paths["ngram_count"])
    logger.info("Input  (labels)     : %s", paths["labels"])
    logger.info("Output (train)      : %s", paths["tfidf_features_train"])
    logger.info("Output (holdout)    : %s", paths["tfidf_features_val_test_oot"])
    logger.info("Model bank          : %s", paths["model_bank"])

    spark = create_spark_session("gold-tfidf")
    ngrams = load_ngram_counts(spark, paths["ngram_count"], args.limit)

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

    artifact = build_tfidf_artifact(train_df, MAX_FEATURES, MIN_DOC_FREQ)
    save_tfidf_artifact(
        artifact,
        paths,
        artifact_paths,
        paths["ngram_count"],
        args.no_split,
        spark,
    )

    partition_col = PARTITION_COL if PARTITION_COL in train_df.columns else None

    train_features = select_tfidf_output(add_tfidf_column(train_df, artifact))
    holdout_features = select_tfidf_output(add_tfidf_column(holdout_df, artifact))

    write_delta(train_features, paths["tfidf_features_train"], partition_col)
    write_delta(holdout_features, paths["tfidf_features_val_test_oot"], partition_col)

    train_count = spark.read.format("delta").load(paths["tfidf_features_train"]).count()
    holdout_count = spark.read.format("delta").load(paths["tfidf_features_val_test_oot"]).count()
    logger.info("Wrote %s rows to %s", f"{train_count:,}", paths["tfidf_features_train"])
    logger.info("Wrote %s rows to %s", f"{holdout_count:,}", paths["tfidf_features_val_test_oot"])
    logger.info("TF-IDF + log-TF-IDF feature freezing complete")


if __name__ == "__main__":
    main()
