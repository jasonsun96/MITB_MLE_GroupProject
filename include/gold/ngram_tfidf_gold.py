import argparse
import logging
from pathlib import Path

import yaml

from utils.ngram_tfidf import (
    MAX_FEATURES,
    MAX_N,
    MIN_N,
    build_gold_features,
    prepare_silver_data,
    save_gold_artifacts,
)
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

with open(Path(__file__).parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

SILVER = schema["silver"]
SILVER_PATH = SILVER["path"]
SILVER_TABLES = SILVER["tables"]
GOLD = schema["gold"]
GOLD_PATH = GOLD["path"]
GOLD_TABLES = GOLD["tables"]

spark = create_spark_session("sample-gold")


def process_ngram_tfidf_gold(silver_table_path, gold_table_path, spark):
    try:
        silver_df = spark.read.format("delta").load(silver_table_path)
        prepared_silver = prepare_silver_data(silver_df, silver_table_path)
        df_gold, vocab, n_docs, n_features = build_gold_features(
            prepared_silver,
            MIN_N,
            MAX_N,
            MAX_FEATURES,
        )
        save_gold_artifacts(
            gold_table_path,
            df_gold,
            vocab,
            silver_table_path,
            MIN_N,
            MAX_N,
            MAX_FEATURES,
            n_docs,
            n_features,
            spark,
        )
    except Exception as e:
        logger.error(e)


silver_table_path = f"{SILVER_PATH}/{SILVER_TABLES['legal_docs_processed']['path']}"
gold_table_path = f"{GOLD_PATH}/{GOLD_TABLES['features_ngram_tfidf']['path']}"
process_ngram_tfidf_gold(silver_table_path, gold_table_path, spark)


logger.info("Gold processing complete")
