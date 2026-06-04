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

with open(Path(__file__).parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

LANDING = schema["landing"]
BRONZE = schema["bronze"]
BRONZE_TABLES = BRONZE["tables"]
LANDING_PATH = LANDING["path"]
BRONZE_PATH = BRONZE["path"]
START_DATE = args.start_date or schema["backfill"]["start_date"]
END_DATE = args.end_date or schema["backfill"]["end_date"]

spark = create_spark_session("ingest_bronze")


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


logger.info(f"Bronze ingest started | {START_DATE} to {END_DATE}")


def process_bronze_table(table_name, table_config, source_path, bronze_path, spark, snapshot_date_str=None, file_prefix=None, filename=None):
    try:
        if file_prefix and snapshot_date_str:
            date_nodash = snapshot_date_str.replace("-", "")
            resolved_filename = f"{file_prefix}{date_nodash}.csv"
        elif filename:
            resolved_filename = filename
        else:
            resolved_filename = f"{table_name}.csv"
        source_file_path = os.path.join(source_path, resolved_filename)
        df = spark.read.csv(source_file_path, header=True, inferSchema=False, multiLine=True, maxCharsPerColumn=10000000, maxColumns=100000)

        if snapshot_date_str is not None:
            df = df.withColumn("snapshot_date", F.lit(snapshot_date_str))

        if snapshot_date_str is not None and not file_prefix:
            df = df.filter(F.col("snapshot_date") == snapshot_date_str)

        partition_col = table_config["partition_col"]
        output_path = os.path.join(bronze_path, table_config["path"])

        writer = df.write.format("delta").option("mergeSchema", "true")
        if snapshot_date_str is not None:
            writer.partitionBy(partition_col).mode("overwrite").option("replaceWhere", f"snapshot_date = '{snapshot_date_str}'").save(output_path)
        else:
            writer.mode("overwrite").save(output_path)

        logger.info(f"Processed {table_name} for {snapshot_date_str}. Bronze table written to {output_path}")

    except Exception as e:
        logger.error(e)


for dataset_name in LANDING:
    if dataset_name == "path":
        continue
    source_path = f"{LANDING_PATH}/{LANDING[dataset_name]['folder']}"
    file_prefix = LANDING[dataset_name].get("file_prefix")
    filename = LANDING[dataset_name].get("filename")
    bronze_path = BRONZE_PATH
    landing_tables = LANDING[dataset_name]["tables"]

    if file_prefix:
        dates = get_dates(spark, source_path, file_prefix, START_DATE, END_DATE)
        logger.info(f"[{dataset_name}] Discovered {len(dates)} snapshots in R2")
    else:
        dates = [None]

    for date_str in dates:
        for table_name, landing_table_config in landing_tables.items():
            table_config = {**BRONZE_TABLES[table_name], **landing_table_config}
            logger.debug(f"Bronze [{dataset_name}]: {table_name} @ {date_str}")
            process_bronze_table(table_name, table_config, source_path, bronze_path, spark, date_str, file_prefix, filename)


logger.info("Bronze ingest complete")
