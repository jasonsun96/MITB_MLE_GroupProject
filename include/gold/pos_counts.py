"""
Gold layer: POS-tagged lemma counts for legal documents.

Reads cleaned legal text, runs spaCy POS tagging, and writes a Delta table
with per-document lemma counts grouped by POS tag. Downstream consumers
(Yuhui's DP/DC, noun-only feature pipelines, anyone wanting verbs or
adjectives) read this single table and filter to the POS tags they need.

Output schema:
    document_id      string                                         CELEX id
    labels           string                                         raw multi-label string
    snapshot_date    string                                         partition key
    pos_counts       map<string, map<string, int>>                  pos_tag -> {lemma: count}
    n_unique_tokens  int                                            distinct (lemma, pos) pairs
    n_total_tokens   int                                            sum of all token occurrences

Example access pattern (noun extraction for DP/DC):
    nouns = {**row.pos_counts.get("NOUN", {}),
             **row.pos_counts.get("PROPN", {})}

Implementation notes:
    - Uses a regular Python UDF (NOT pandas_udf / mapInPandas) to sidestep
      Apache Arrow + Java 17 compatibility issues.
    - spaCy is lazy-loaded once per Python worker via a module-level singleton,
      then reused across every row that worker processes.
    - We disable spaCy's parser and ner components since we only need POS.
    - Text is truncated to 500k chars before tagging to bound per-row memory.
"""
import argparse
import logging
from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from utils.spark_session import create_spark_session

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Gold layer: POS-tagged lemma counts")
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit to N rows for smoke testing. Omit for full corpus.",
)
parser.add_argument(
    "--input-layer",
    default="bronze",
    choices=["bronze", "silver"],
    help="Source layer to read from. Switch to 'silver' once it exists.",
)
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

if args.input_layer == "bronze":
    INPUT_PATH = f"{schema['bronze']['path']}/{schema['bronze']['tables']['legal_docs_raw']['path']}"
    TEXT_COL = "act_raw_text"
else:
    INPUT_PATH = f"{schema['silver']['path']}/{schema['silver']['tables']['legal_docs_processed']['path']}"
    TEXT_COL = "text"

OUTPUT_PATH = f"{schema['gold']['path']}/{schema['gold']['tables']['pos_counts']['path']}"

logger.info(f"Input  ({args.input_layer}): {INPUT_PATH}")
logger.info(f"Output (gold)             : {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# UDF output type: pos_counts (nested map) + two convenience counts
# ---------------------------------------------------------------------------
UDF_RETURN_TYPE = StructType(
    [
        StructField(
            "pos_counts",
            MapType(StringType(), MapType(StringType(), IntegerType())),
            nullable=False,
        ),
        StructField("n_unique_tokens", IntegerType(), nullable=False),
        StructField("n_total_tokens", IntegerType(), nullable=False),
    ]
)


# Lazy module-level spaCy singleton. Loaded once per Python worker process.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        # Default 1M chars is too small for the largest EurLex docs.
        # Since parser/ner are disabled, bumping is safe.
        _NLP.max_length = 5_000_000
    return _NLP


def _extract_pos_counts(text):
    """UDF body. Returns the struct defined by UDF_RETURN_TYPE."""
    empty = {"pos_counts": {}, "n_unique_tokens": 0, "n_total_tokens": 0}

    if not text:
        return empty

    nlp = _get_nlp()

    # Defensive: cap at spaCy's max_length in case Spark-level truncation missed.
    if len(text) > nlp.max_length:
        text = text[: nlp.max_length]

    try:
        doc = nlp(text)

        # Group lemma counts by POS tag.
        pos_counts = {}
        for token in doc:
            if not token.is_alpha:
                continue
            pos = token.pos_
            lemma = token.lemma_.lower()
            bucket = pos_counts.setdefault(pos, {})
            bucket[lemma] = bucket.get(lemma, 0) + 1

        n_unique = sum(len(b) for b in pos_counts.values())
        n_total = sum(sum(b.values()) for b in pos_counts.values())
    except Exception:
        # Any per-document failure: empty result, don't fail the whole job.
        return empty

    return {
        "pos_counts": pos_counts,
        "n_unique_tokens": n_unique,
        "n_total_tokens": int(n_total),
    }


extract_pos_counts_udf = F.udf(_extract_pos_counts, returnType=UDF_RETURN_TYPE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    spark = create_spark_session("gold-pos-counts")

    raw = spark.read.format("delta").load(INPUT_PATH)

    # Cap text length at Spark level to bound per-row memory cost.
    # ~75k words covers everything past the p95 of the corpus.
    MAX_TEXT_CHARS = 500_000

    df = raw.select(
        F.col("CELEX").alias("document_id"),
        F.substring(F.col(TEXT_COL), 1, MAX_TEXT_CHARS).alias("text"),
        F.col("labels").alias("labels"),
        F.col("snapshot_date").alias("snapshot_date"),
    ).filter(F.col("text").isNotNull() & (F.length("text") > 100))

    if args.limit:
        df = df.limit(args.limit)
        logger.info(f"Smoke test mode: limited to {args.limit:,} rows")

    # Repartition by snapshot_date so the shuffle aligns with partitionBy
    # on write, keeping output file count manageable.
    df = df.repartition(200, "snapshot_date")

    input_count = df.count()
    logger.info(
        f"Processing {input_count:,} documents across "
        f"{df.rdd.getNumPartitions()} partitions"
    )

    result = (
        df.withColumn("_pos", extract_pos_counts_udf(F.col("text")))
          .select(
              F.col("document_id"),
              F.col("labels"),
              F.col("snapshot_date"),
              F.col("_pos.pos_counts").alias("pos_counts"),
              F.col("_pos.n_unique_tokens").alias("n_unique_tokens"),
              F.col("_pos.n_total_tokens").alias("n_total_tokens"),
          )
    )

    (
        result.write.format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .partitionBy("snapshot_date")
        .save(OUTPUT_PATH)
    )

    output_count = spark.read.format("delta").load(OUTPUT_PATH).count()
    logger.info(f"Wrote {output_count:,} rows to {OUTPUT_PATH}")
    logger.info("Gold POS-counts extraction complete")


if __name__ == "__main__":
    main()
