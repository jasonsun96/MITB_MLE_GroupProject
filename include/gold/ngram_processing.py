"""
Gold n-gram counts for the legal corpus.

Reads legal silver, tokenizes with NLTK (stopwords + lemmatization), and writes
per-document raw n-gram count maps to gold. No corpus vocabulary is fitted here;
TF-IDF vocabulary/IDF fitting happens later in a separate model-bank job on
train data only.

schema (gold/ngrams):
  document_id, labels, snapshot_date
  tokens, token_count
  ngram_counts: {ngram_string: count}   e.g. {"sample": 2, "sample text": 1}
  text_source, silver_ingest_ts, silver_source

notes:
  - regular UDF (not pandas_udf) for Python worker compatibility
  - NLTK loaded once per Python worker (module-level singleton)
  - text capped at 500k chars at Spark level before tokenization
  - repartition by snapshot_date so partitionBy on write lines up with shuffle
"""
from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from gold_io import bootstrap_paths

bootstrap_paths()

from utils.spark_session import create_spark_session

MIN_N = 1
MAX_N = 3
MAX_TEXT_CHARS = 500_000
MIN_TEXT_CHARS = 100
REPARTITION_N = 200
PARTITION_COL = "snapshot_date"

TEXT_COLUMN = "act_raw_text"
LABEL_COLUMN = "labels"
ID_COLUMN = "CELEX"
LEGAL_SILVER_TABLE = "legal_docs_processed"
NGRAM_CORPUS_TABLE = "ngrams"

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.yaml"

logger = logging.getLogger(__name__)

_STOPWORDS: set[str] | None = None
_LEMMATIZER = None

UDF_RETURN_TYPE = StructType(
    [
        StructField("tokens", StringType(), nullable=False),
        StructField("token_count", IntegerType(), nullable=False),
        StructField("ngram_counts", MapType(StringType(), IntegerType()), nullable=False),
    ]
)


def ensure_nltk_data() -> None:
    import nltk

    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            else:
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download(resource, quiet=True)


def _stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_data()
        from nltk.corpus import stopwords

        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        ensure_nltk_data()
        from nltk.stem import WordNetLemmatizer

        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def _preprocess_tokens(text: str) -> list[str]:
    stops = _stopwords()
    lemmatizer = _lemmatizer()
    tokens: list[str] = []

    for token in re.findall(r"[a-z0-9]+", str(text).lower()):
        if token in stops:
            continue
        token = lemmatizer.lemmatize(token)
        if len(token) == 1 and token.isalpha():
            continue
        if token.isdigit():
            if len(token) == 4:
                tokens.append(token)
            continue
        tokens.append(token)

    return tokens


def _count_ngrams(tokens: list[str], min_n: int, max_n: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    n_tokens = len(tokens)

    for n in range(min_n, max_n + 1):
        if n_tokens < n:
            continue
        for i in range(n_tokens - n + 1):
            ngram = " ".join(tokens[i : i + n])
            counts[ngram] = counts.get(ngram, 0) + 1

    return counts


def _extract_ngram_counts(text: str | None) -> dict:
    empty = {"tokens": "", "token_count": 0, "ngram_counts": {}}

    if not text:
        return empty

    try:
        tokens = _preprocess_tokens(text)
        if not tokens:
            return empty

        return {
            "tokens": " ".join(tokens),
            "token_count": len(tokens),
            "ngram_counts": _count_ngrams(tokens, MIN_N, MAX_N),
        }
    except Exception:
        return empty


extract_ngram_counts_udf = F.udf(_extract_ngram_counts, returnType=UDF_RETURN_TYPE)


def main():
    parser = argparse.ArgumentParser(description="Gold layer: per-document n-gram counts")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N rows for smoke testing. Omit for full corpus.",
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

    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)

    silver = schema["silver"]
    gold = schema["gold"]
    INPUT_PATH = f"{silver['path']}/{silver['tables'][LEGAL_SILVER_TABLE]['path']}"
    OUTPUT_PATH = f"{gold['path']}/{gold['corpus'][NGRAM_CORPUS_TABLE]['path']}"

    logger.info("Input  (silver): %s", INPUT_PATH)
    logger.info("Output (gold)  : %s", OUTPUT_PATH)

    spark = create_spark_session("gold-ngram-counts")
    raw = spark.read.format("delta").load(INPUT_PATH)

    if ID_COLUMN not in raw.columns:
        raise ValueError(f"Required column missing from silver input: {ID_COLUMN}")
    if TEXT_COLUMN not in raw.columns:
        raise ValueError(f"Required column missing from silver input: {TEXT_COLUMN}")
    if LABEL_COLUMN not in raw.columns:
        raise ValueError(f"Required column missing from silver input: {LABEL_COLUMN}")

    select_exprs = [
        F.col(ID_COLUMN).alias("document_id"),
        F.col(LABEL_COLUMN).alias("labels"),
        F.substring(F.col(TEXT_COLUMN), 1, MAX_TEXT_CHARS).alias("_text"),
    ]
    if PARTITION_COL in raw.columns:
        select_exprs.insert(2, F.col(PARTITION_COL).alias(PARTITION_COL))

    label_filter_raw = F.col(LABEL_COLUMN).isNotNull() & (F.length(F.col(LABEL_COLUMN)) > 0)
    label_filter_selected = F.col("labels").isNotNull() & (F.length(F.col("labels")) > 0)
    text_filter = F.col("_text").isNotNull() & (F.length(F.trim(F.col("_text"))) > MIN_TEXT_CHARS)

    if args.limit:
        # Pick document IDs without reading act_raw_text — dropDuplicates + limit
        # must not run after loading multi-MB text columns or the JVM OOMs.
        id_exprs = [
            F.col(ID_COLUMN).alias("document_id"),
            F.col(LABEL_COLUMN).alias("labels"),
        ]
        if PARTITION_COL in raw.columns:
            id_exprs.append(F.col(PARTITION_COL).alias(PARTITION_COL))

        doc_ids = [
            row.document_id
            for row in (
                raw.select(*id_exprs)
                .filter(label_filter_raw)
                .dropDuplicates(["document_id"])
                .limit(args.limit)
                .collect()
            )
        ]
        if not doc_ids:
            raise ValueError("No documents matched smoke-test filters")

        logger.info("Smoke test mode: loading text for %s documents", f"{len(doc_ids):,}")

        df = (
            raw.filter(F.col(ID_COLUMN).isin(doc_ids))
            .select(*select_exprs)
            .filter(text_filter)
            .filter(label_filter_selected)
        )
        input_count = len(doc_ids)
    else:
        df = (
            raw.filter(label_filter_raw)
            .select(*select_exprs)
            .filter(text_filter)
            .dropDuplicates(["document_id"])
        )
        if PARTITION_COL in df.columns:
            df = df.repartition(REPARTITION_N, PARTITION_COL)
        input_count = df.count()

    logger.info(
        "Processing %s documents across %s partitions",
        f"{input_count:,}",
        df.rdd.getNumPartitions(),
    )

    silver_ingest_ts = datetime.now(timezone.utc).isoformat()

    result = (
        df.withColumn("_ngrams", extract_ngram_counts_udf(F.col("_text")))
        .filter(F.col("_ngrams.token_count") > 0)
        .select(
            F.col("document_id"),
            F.col("labels"),
            *([F.col(PARTITION_COL)] if PARTITION_COL in df.columns else []),
            F.col("_ngrams.tokens").alias("tokens"),
            F.col("_ngrams.token_count").alias("token_count"),
            F.col("_ngrams.ngram_counts").alias("ngram_counts"),
            F.lit(TEXT_COLUMN).alias("text_source"),
            F.lit(silver_ingest_ts).alias("silver_ingest_ts"),
            F.lit(INPUT_PATH).alias("silver_source"),
        )
    )

    writer = result.write.format("delta").mode("overwrite").option("mergeSchema", "true")
    if PARTITION_COL in result.columns:
        writer = writer.partitionBy(PARTITION_COL)
    writer.save(OUTPUT_PATH)

    output_count = spark.read.format("delta").load(OUTPUT_PATH).count()
    logger.info("Wrote %s rows to %s", f"{output_count:,}", OUTPUT_PATH)
    logger.info("Gold n-gram counts extraction complete")


if __name__ == "__main__":
    main()
