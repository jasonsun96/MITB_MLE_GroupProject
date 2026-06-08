import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml
from pyspark.sql import functions as F

from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Bronze layer ingestion pipeline")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
parser.add_argument("--start-date", default=None, help="Backfill start date (YYYY-MM-DD), overrides schema.yaml")
parser.add_argument("--end-date", default=None, help="Backfill end date (YYYY-MM-DD), overrides schema.yaml")
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

LANDING = schema["landing"]
BRONZE = schema["bronze"]
BRONZE_TABLES = BRONZE["tables"]
LANDING_PATH = LANDING["path"]
BRONZE_PATH = BRONZE["path"]
START_DATE = args.start_date or schema["backfill"]["start_date"]
END_DATE = args.end_date or schema["backfill"]["end_date"]

spark = create_spark_session("ingest_bronze")

DEFAULT_CSV_READ_OPTIONS = {
    "header": True,
    "inferSchema": False,
    "multiLine": True,
    "maxCharsPerColumn": 10000000,
    "maxColumns": 100000,
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


def get_dates(spark, source_path, file_prefix, start_date_str, end_date_str):
    sc = spark.sparkContext
    URI = sc._jvm.java.net.URI
    Path = sc._jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._jvm.org.apache.hadoop.fs.FileSystem

    fs = FileSystem.get(URI(source_path), sc._jsc.hadoopConfiguration())
    statuses = fs.listStatus(Path(source_path))

    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")

    dates = []
    for status in statuses:
        fname = status.getPath().getName()
        if fname.startswith(file_prefix) and fname.endswith(".csv"):
            date_part = fname[len(file_prefix) : -4]
            try:
                date = datetime.strptime(date_part, "%Y%m%d")
                if start <= date <= end:
                    dates.append(date.strftime("%Y-%m-%d"))
            except ValueError:
                pass

    return sorted(dates)


def write_delta_table(df, output_path, partition_col=None, snapshot_date_str=None):
    writer = df.write.format("delta").option("mergeSchema", "true")
    if snapshot_date_str is not None:
        writer.partitionBy(partition_col).mode("overwrite").option("replaceWhere", f"snapshot_date = '{snapshot_date_str}'").save(output_path)
    else:
        writer.mode("overwrite").save(output_path)


logger.info(f"Bronze ingest started | {START_DATE} to {END_DATE}")


def ingest_legal_docs(spark):
    dataset_config = LANDING["legal_docs"]
    table_name = "legal_docs_raw"
    table_config = {**BRONZE_TABLES[table_name], **dataset_config["tables"][table_name]}
    source_path = f"{LANDING_PATH}/{dataset_config['folder']}"
    output_path = os.path.join(BRONZE_PATH, table_config["path"])
    file_prefix = dataset_config["file_prefix"]

    dates = get_dates(spark, source_path, file_prefix, START_DATE, END_DATE)
    logger.info("[legal_docs] Discovered %s snapshots in R2", len(dates))

    for snapshot_date_str in dates:
        date_nodash = snapshot_date_str.replace("-", "")
        source_file_path = os.path.join(source_path, f"{file_prefix}{date_nodash}.csv")

        try:
            df = spark.read.options(**LEGAL_DOCS_CSV_READ_OPTIONS).csv(source_file_path)
            merged_records = df.filter(F.col("act_raw_text").rlike(MERGED_CSV_RECORD_PATTERN))
            sample_ids = [row["CELEX"] for row in merged_records.select("CELEX").limit(10).collect()]
            if sample_ids:
                raise ValueError(f"Detected merged CSV records in {source_file_path}; " f"affected CELEX sample: {sample_ids}")

            df = df.withColumn("snapshot_date", F.lit(snapshot_date_str))

            write_delta_table(df, output_path, table_config["partition_col"], snapshot_date_str)
            logger.info("Processed %s for %s. Bronze table written to %s", table_name, snapshot_date_str, output_path)

        except Exception:
            logger.exception("Failed to process %s from %s", table_name, source_file_path)
            raise


def ingest_wiki_docs(spark):
    dataset_config = LANDING["wiki_docs"]
    table_name = "wiki_docs_raw"
    table_config = {**BRONZE_TABLES[table_name], **dataset_config["tables"][table_name]}
    source_path = f"{LANDING_PATH}/{dataset_config['folder']}"
    source_file_path = os.path.join(source_path, dataset_config["filename"])
    output_path = os.path.join(BRONZE_PATH, table_config["path"])

    try:
        df = spark.read.options(**WIKI_DOCS_CSV_READ_OPTIONS).csv(source_file_path)
        malformed_rows = df.filter(F.col("id").isNull() | ~F.col("id").rlike(r"^\d+$") | F.col("url").isNull() | ~F.col("url").rlike(r"^https?://"))
        sample_rows = [row.asDict() for row in malformed_rows.select("id", "url", "title").limit(10).collect()]
        if sample_rows:
            raise ValueError(f"Detected malformed wiki CSV rows in {source_file_path}; sample rows: {sample_rows}")

        write_delta_table(df, output_path)
        logger.info("Processed %s. Bronze table written to %s", table_name, output_path)

    except Exception:
        logger.exception("Failed to process %s from %s", table_name, source_file_path)
        raise


ingest_legal_docs(spark)
ingest_wiki_docs(spark)


logger.info("Bronze ingest complete")
