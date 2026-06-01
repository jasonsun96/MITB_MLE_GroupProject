import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import pyspark
import yaml
from delta import configure_spark_with_delta_pip
from dotenv import load_dotenv

from utils.data_processing_bronze import process_bronze_table

load_dotenv(Path(__file__).parent / ".env")

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

with open("schema.yaml") as f:
    schema = yaml.safe_load(f)

LANDING = schema["landing"]
BRONZE = schema["bronze"]
LANDING_PATH = LANDING["path"]
BRONZE_PATH = BRONZE["path"]
START_DATE = args.start_date or schema["backfill"]["start_date"]
END_DATE = args.end_date or schema["backfill"]["end_date"]

ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

builder = (
    pyspark.sql.SparkSession.builder.appName("mle-bronze-ingest")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.endpoint", R2_ENDPOINT)
    .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY_ID)
    .config("spark.hadoop.fs.s3a.secret.key", SECRET_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
)
spark = configure_spark_with_delta_pip(
    builder,
    extra_packages=[
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    ],
).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


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

for dataset_name in LANDING:
    if dataset_name == "path":
        continue
    source_path = f"{LANDING_PATH}/{LANDING[dataset_name]['folder']}"
    file_prefix = LANDING[dataset_name].get("file_prefix")
    filename = LANDING[dataset_name].get("filename")
    bronze_path = BRONZE_PATH
    bronze_tables = BRONZE[dataset_name]["tables"]

    if file_prefix:
        dates = get_dates(spark, source_path, file_prefix, START_DATE, END_DATE)
        logger.info(f"[{dataset_name}] Discovered {len(dates)} snapshots in R2")
    else:
        dates = [None]

    for date_str in dates:
        for table_name, table_config in bronze_tables.items():
            logger.debug(f"Bronze [{dataset_name}]: {table_name} @ {date_str}")
            process_bronze_table(table_name, table_config, source_path, bronze_path, spark, date_str, file_prefix, filename)

logger.info("Bronze ingest complete")
