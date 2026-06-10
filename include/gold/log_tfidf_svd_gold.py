import argparse
import logging
from pathlib import Path

import yaml

from utils.log_tfidf_svd import (
    M_VALUES,
    build_log_tfidf_svd_features,
    save_log_tfidf_svd_artifacts,
)
from utils.ngram_tfidf import MAX_FEATURES, MAX_N, MIN_N, prepare_silver_data
from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Gold layer: log-TF-IDF + SVD features")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
parser.add_argument(
    "--m-values",
    default=",".join(str(m) for m in M_VALUES),
    help="Comma-separated SVD dimensions, e.g. 50,100,200,500",
)
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

m_values = [int(m.strip()) for m in args.m_values.split(",") if m.strip()]

with open(Path(__file__).resolve().parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

SILVER = schema["silver"]
SILVER_PATH = SILVER["path"]
SILVER_TABLES = SILVER["tables"]
GOLD = schema["gold"]
GOLD_PATH = GOLD["path"]
GOLD_TABLES = GOLD["tables"]

spark = create_spark_session("log-tfidf-svd-gold")


def process_log_tfidf_svd_gold(silver_table_path, gold_table_path, spark, m_values):
    silver_df = spark.read.format("delta").load(silver_table_path)
    prepared_silver = prepare_silver_data(silver_df, silver_table_path)
    df_gold, vocab, svd_model, n_docs, n_features, m_values = build_log_tfidf_svd_features(
        prepared_silver,
        MIN_N,
        MAX_N,
        MAX_FEATURES,
        m_values,
    )
    save_log_tfidf_svd_artifacts(
        gold_table_path,
        df_gold,
        vocab,
        svd_model,
        silver_table_path,
        MIN_N,
        MAX_N,
        MAX_FEATURES,
        n_docs,
        n_features,
        m_values,
        spark,
    )


silver_table_path = f"{SILVER_PATH}/{SILVER_TABLES['legal_docs_processed']['path']}"
gold_table_path = f"{GOLD_PATH}/{GOLD_TABLES['features_log_tfidf_svd']['path']}"

try:
    process_log_tfidf_svd_gold(silver_table_path, gold_table_path, spark, m_values)
except Exception:
    logger.exception("Log-TF-IDF + SVD gold job failed")
    raise

logger.info("Log-TF-IDF + SVD gold processing complete")
