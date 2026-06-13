from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml
from pyspark.sql import functions as F

from utils.spark_session import create_spark_session

LEGAL_DOCS_TABLE = "legal_docs_raw"
WIKI_DOCS_TABLE = "wiki_docs_raw"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.yaml"

DEFAULT_CSV_READ_OPTIONS = {
    "header": True,
    "inferSchema": False,
    "multiLine": True,
    "maxCharsPerColumn": 10_000_000,
    "maxColumns": 100_000,
}

LEGAL_DOCS_CSV_READ_OPTIONS = {
    **DEFAULT_CSV_READ_OPTIONS,
    "multiLine": False,
    "quote": '"',
    "escape": '"',
    "mode": "FAILFAST",
}

WIKI_DOCS_CSV_READ_OPTIONS = {
    **DEFAULT_CSV_READ_OPTIONS,
    "quote": '"',
    "escape": '"',
    "mode": "FAILFAST",
}

MERGED_CSV_RECORD_PATTERN = r"[\r\n]+\s*\"?[0-9][0-9A-Z()]{7,20},"

logger = logging.getLogger(__name__)


def get_dates(spark, source_path: str, file_prefix: str, start_date: str, end_date: str) -> list[str]:
    sc = spark.sparkContext
    uri_class = sc._jvm.java.net.URI
    path_class = sc._jvm.org.apache.hadoop.fs.Path
    file_system_class = sc._jvm.org.apache.hadoop.fs.FileSystem

    file_system = file_system_class.get(
        uri_class(source_path),
        sc._jsc.hadoopConfiguration(),
    )
    statuses = file_system.listStatus(path_class(source_path))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    for status in statuses:
        filename = status.getPath().getName()
        if not filename.startswith(file_prefix) or not filename.endswith(".csv"):
            continue

        date_part = filename[len(file_prefix) : -4]
        try:
            snapshot_date = datetime.strptime(date_part, "%Y%m%d")
        except ValueError:
            continue

        if start <= snapshot_date <= end:
            dates.append(snapshot_date.strftime("%Y-%m-%d"))

    return sorted(dates)


def write_delta_table(df, output_path: str, partition_col: str | None = None, snapshot_date: str | None = None) -> None:
    writer = df.write.format("delta").option("mergeSchema", "true")
    if snapshot_date is not None:
        writer.partitionBy(partition_col).mode("overwrite").option(
            "replaceWhere",
            f"snapshot_date = '{snapshot_date}'",
        ).save(output_path)
        return

    writer.mode("overwrite").save(output_path)


def ingest_legal_docs(spark, landing: dict, bronze: dict, start_date: str, end_date: str) -> None:
    dataset_config = landing["legal_docs"]
    table_config = {
        **bronze["tables"][LEGAL_DOCS_TABLE],
        **dataset_config["tables"][LEGAL_DOCS_TABLE],
    }
    source_path = f"{landing['path']}/{dataset_config['folder']}"
    output_path = f"{bronze['path']}/{table_config['path']}"
    file_prefix = dataset_config["file_prefix"]

    dates = get_dates(spark, source_path, file_prefix, start_date, end_date)
    logger.info("[legal_docs] Discovered %s snapshots in R2", len(dates))

    for snapshot_date in dates:
        date_nodash = snapshot_date.replace("-", "")
        source_file_path = f"{source_path}/{file_prefix}{date_nodash}.csv"

        try:
            df = spark.read.options(**LEGAL_DOCS_CSV_READ_OPTIONS).csv(source_file_path)
            merged_records = df.filter(F.col("act_raw_text").rlike(MERGED_CSV_RECORD_PATTERN))
            sample_ids = [row["CELEX"] for row in merged_records.select("CELEX").limit(10).collect()]
            if sample_ids:
                raise ValueError(f"Detected merged CSV records in {source_file_path}; " f"affected CELEX sample: {sample_ids}")

            df = df.withColumn("snapshot_date", F.lit(snapshot_date))
            write_delta_table(
                df,
                output_path,
                partition_col=table_config["partition_col"],
                snapshot_date=snapshot_date,
            )
            logger.info(
                "Processed %s for %s. Bronze table written to %s",
                LEGAL_DOCS_TABLE,
                snapshot_date,
                output_path,
            )
        except Exception:
            logger.exception(
                "Failed to process %s from %s",
                LEGAL_DOCS_TABLE,
                source_file_path,
            )
            raise


def ingest_wiki_docs(spark, landing: dict, bronze: dict) -> None:
    dataset_config = landing["wiki_docs"]
    table_config = {
        **bronze["tables"][WIKI_DOCS_TABLE],
        **dataset_config["tables"][WIKI_DOCS_TABLE],
    }
    source_path = f"{landing['path']}/{dataset_config['folder']}"
    source_file_path = f"{source_path}/{dataset_config['filename']}"
    output_path = f"{bronze['path']}/{table_config['path']}"

    try:
        df = spark.read.options(**WIKI_DOCS_CSV_READ_OPTIONS).csv(source_file_path)
        malformed_rows = df.filter(F.col("id").isNull() | ~F.col("id").rlike(r"^\d+$") | F.col("url").isNull() | ~F.col("url").rlike(r"^https?://"))
        sample_rows = [row.asDict() for row in malformed_rows.select("id", "url", "title").limit(10).collect()]
        if sample_rows:
            raise ValueError(f"Detected malformed wiki CSV rows in {source_file_path}; " f"sample rows: {sample_rows}")

        write_delta_table(df, output_path)
        logger.info(
            "Processed %s. Bronze table written to %s",
            WIKI_DOCS_TABLE,
            output_path,
        )
    except Exception:
        logger.exception(
            "Failed to process %s from %s",
            WIKI_DOCS_TABLE,
            source_file_path,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Bronze layer ingestion pipeline")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Backfill start date (YYYY-MM-DD), overrides schema.yaml",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Backfill end date (YYYY-MM-DD), overrides schema.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(SCHEMA_PATH) as schema_file:
        schema = yaml.safe_load(schema_file)

    landing = schema["landing"]
    bronze = schema["bronze"]
    start_date = args.start_date or schema["backfill"]["start_date"]
    end_date = args.end_date or schema["backfill"]["end_date"]

    logger.info("Bronze ingest started | %s to %s", start_date, end_date)

    spark = create_spark_session("ingest-bronze")
    ingest_legal_docs(spark, landing, bronze, start_date, end_date)
    ingest_wiki_docs(spark, landing, bronze)

    logger.info("Bronze ingest complete")


if __name__ == "__main__":
    main()
