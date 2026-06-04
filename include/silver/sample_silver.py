import argparse
import logging
from pathlib import Path

import yaml

from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Bronze layer ingestion pipeline")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

BRONZE = schema["bronze"]
BRONZE_PATH = BRONZE["path"]
BRONZE_TABLES = BRONZE["tables"]
SILVER = schema["silver"]
SILVER_PATH = SILVER["path"]
SILVER_TABLES = SILVER["tables"]

spark = create_spark_session("sample-silver")


def sample_processing_function(bronze_table_path, silver_table_path, spark):
    df = spark.read.format("delta").load(bronze_table_path)

    try:
        writer = df.write.format("delta").option("mergeSchema", "true")
        writer.mode("overwrite").save(silver_table_path)

    except Exception as e:
        logger.error(e)


bronze_table_path = f"{BRONZE_PATH}/{BRONZE_TABLES['legal_docs_raw']['path']}"
silver_table_path = f"{SILVER_PATH}/{SILVER_TABLES['legal_docs_processed']['path']}"
sample_processing_function(bronze_table_path, silver_table_path, spark)


logger.info("Silver processing complete")
