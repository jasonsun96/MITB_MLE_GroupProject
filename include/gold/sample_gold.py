import argparse
import logging
from pathlib import Path

import yaml
from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Gold layer processing pipeline")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

SILVER = schema["silver"]
SILVER_PATH = SILVER["path"]
SILVER_TABLES = SILVER["tables"]
GOLD = schema["gold"]
GOLD_PATH = GOLD["path"]
GOLD_TABLES = GOLD["tables"]

spark = create_spark_session("sample-gold")


def sample_processing_function(silver_table_path, gold_table_path, spark):
    df = spark.read.format("delta").load(silver_table_path)

    try:
        writer = df.write.format("delta").option("mergeSchema", "true")
        writer.mode("overwrite").save(gold_table_path)

    except Exception:
        logger.exception("Failed to process gold table from %s", silver_table_path)
        raise


silver_table_path = f"{SILVER_PATH}/{SILVER_TABLES['legal_docs_processed']['path']}"
gold_table_path = f"{GOLD_PATH}/{GOLD_TABLES['sample_gold']['path']}"
sample_processing_function(silver_table_path, gold_table_path, spark)


logger.info("Gold processing complete")
