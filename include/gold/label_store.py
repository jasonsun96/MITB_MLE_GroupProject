from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from gold_io import PARTITION_COL, bootstrap_paths, write_delta
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pyspark.sql import functions as F
from pyspark.sql import types as T
from sklearn.preprocessing import MultiLabelBinarizer

from utils.spark_session import create_spark_session

bootstrap_paths()


LEGAL_SILVER_TABLE = "legal_docs_processed"
LABEL_STORE_TABLE = "label_store"
ID_COLUMN = "CELEX"
LABEL_COLUMN = "labels"
SPLIT_COLUMN = "category"
TEST_START_YEAR = 2003
VALIDATION_FRACTION_OF_PRE_TEST = 2 / 9
DEFAULT_RANDOM_SEED = 42
MIN_STRATIFY_LABEL_COUNT = 2

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.yaml"

logger = logging.getLogger(__name__)


def build_label_store(df):
    required_columns = {ID_COLUMN, PARTITION_COL, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Required column(s) missing from silver input: {sorted(missing_columns)}")

    return (
        df.select(
            F.trim(F.col(ID_COLUMN)).alias("document_id"),
            F.col(PARTITION_COL),
            F.trim(F.col(LABEL_COLUMN)).alias(LABEL_COLUMN),
        )
        .filter(F.col("document_id").isNotNull())
        .filter(F.length(F.col("document_id")) > 0)
        .filter(F.col(LABEL_COLUMN).isNotNull())
        .filter(F.length(F.col(LABEL_COLUMN)) > 0)
    )


def parse_labels(raw_labels: str) -> list[str]:
    """Parse semicolon-delimited labels without splitting commas in label names."""
    return sorted({label.strip().lower() for label in raw_labels.split(";") if label.strip()})


def split_pre_test_documents(rows, random_seed: int):
    document_ids = [row.document_id for row in rows]
    parsed_labels = [parse_labels(row.labels) for row in rows]

    label_counts = Counter(label for labels in parsed_labels for label in labels)
    stratify_labels = {label for label, count in label_counts.items() if count >= MIN_STRATIFY_LABEL_COUNT}
    stratify_targets = [[label for label in labels if label in stratify_labels] for labels in parsed_labels]

    if not stratify_labels:
        raise ValueError("No labels occur often enough for multilabel stratification")

    encoder = MultiLabelBinarizer(classes=sorted(stratify_labels))
    targets = encoder.fit_transform(stratify_targets)
    row_indexes = np.arange(len(rows)).reshape(-1, 1)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION_OF_PRE_TEST,
        random_state=random_seed,
    )
    train_indexes, validation_indexes = next(splitter.split(row_indexes, targets))

    assignments = {document_ids[index]: "train" for index in train_indexes}
    assignments.update({document_ids[index]: "val" for index in validation_indexes})
    return assignments, len(stratify_labels)


def assign_splits(label_store, spark, random_seed: int):
    duplicate_ids = label_store.groupBy("document_id").count().filter(F.col("count") > 1).select("document_id").limit(10).collect()
    if duplicate_ids:
        sample = [row.document_id for row in duplicate_ids]
        raise ValueError(f"Duplicate document_id values would leak across splits: {sample}")

    snapshot_year = F.year(F.to_date(F.col(PARTITION_COL)))
    invalid_dates = label_store.filter(snapshot_year.isNull()).limit(10).collect()
    if invalid_dates:
        sample = [row.snapshot_date for row in invalid_dates]
        raise ValueError(f"Invalid snapshot_date values: {sample}")

    pre_test = label_store.filter(snapshot_year < TEST_START_YEAR)
    test = label_store.filter(snapshot_year >= TEST_START_YEAR).withColumn(SPLIT_COLUMN, F.lit("test"))

    pre_test_rows = pre_test.select("document_id", LABEL_COLUMN).orderBy("document_id").collect()
    if not pre_test_rows:
        raise ValueError("No pre-test documents available for train/validation splitting")

    assignments, stratified_label_count = split_pre_test_documents(pre_test_rows, random_seed=random_seed)
    assignment_schema = T.StructType(
        [
            T.StructField("document_id", T.StringType(), nullable=False),
            T.StructField(SPLIT_COLUMN, T.StringType(), nullable=False),
        ]
    )
    assignment_df = spark.createDataFrame(assignments.items(), schema=assignment_schema)
    train_validation = pre_test.join(assignment_df, on="document_id", how="inner")

    logger.info(
        "Stratified %s labels across %s pre-test documents",
        f"{stratified_label_count:,}",
        f"{len(pre_test_rows):,}",
    )
    return train_validation.unionByName(test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold layer: legal document label store")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for the train/validation multilabel split",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)

    silver = schema["silver"]
    gold = schema["gold"]
    input_path = f"{silver['path']}/{silver['tables'][LEGAL_SILVER_TABLE]['path']}"
    output_path = f"{gold['path']}/{gold['tables'][LABEL_STORE_TABLE]['path']}"

    logger.info("Input  (silver): %s", input_path)
    logger.info("Output (gold)  : %s", output_path)

    spark = create_spark_session("gold-label-store")
    silver_df = spark.read.format("delta").load(input_path)
    label_store = build_label_store(silver_df)
    label_store = assign_splits(label_store, spark, random_seed=args.random_seed)

    try:
        write_delta(label_store, output_path, partition_col=PARTITION_COL)
    except Exception:
        logger.exception("Failed to write Gold label store to %s", output_path)
        raise

    output_count = spark.read.format("delta").load(output_path).count()
    logger.info("Wrote %s rows to %s", f"{output_count:,}", output_path)
    label_store.groupBy(SPLIT_COLUMN).count().orderBy(SPLIT_COLUMN).show(truncate=False)
    logger.info("Gold label store extraction complete")
