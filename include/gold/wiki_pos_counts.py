"""
gold POS counts for the wiki (non-law baseline) corpus.

same idea as pos_counts.py but reading from wiki bronze instead. wiki has
a different schema so a few column tweaks:
  - id column is `id` (legal uses CELEX)
  - text column is `text` (legal uses act_raw_text)
  - no labels column yet, Cheewei is adding them. for now we write null
  - no snapshot_date so no partitionBy on write

output schema kept parallel to the legal gold table so downstream code can
treat them the same. once wiki labels exist in bronze, re-run this script
and the labels column will populate. until then filter WHERE labels IS NOT
NULL before doing anything supervised.
"""
import argparse
import logging
from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Gold layer: POS-tagged lemma counts (wiki)")
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

# load schema config
with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

if args.input_layer == "bronze":
    INPUT_PATH = f"{schema['bronze']['path']}/{schema['bronze']['tables']['wiki_docs_raw']['path']}"
else:
    INPUT_PATH = f"{schema['silver']['path']}/{schema['silver']['tables']['wiki_docs_processed']['path']}"

OUTPUT_PATH = f"{schema['gold']['path']}/{schema['gold']['tables']['pos_counts_wiki']['path']}"

logger.info(f"Input  ({args.input_layer}): {INPUT_PATH}")
logger.info(f"Output (gold)             : {OUTPUT_PATH}")

# UDF output: pos_counts nested map + two count fields
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


# lazy spacy singleton, loaded once per python worker
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        _NLP.max_length = 5_000_000  # default 1M is too small, bumping is safe with parser/ner off
    return _NLP


def _extract_pos_counts(text):
    """UDF body. Returns the struct defined by UDF_RETURN_TYPE."""
    empty = {"pos_counts": {}, "n_unique_tokens": 0, "n_total_tokens": 0}

    if not text:
        return empty

    nlp = _get_nlp()

    if len(text) > nlp.max_length:
        text = text[: nlp.max_length]

    try:
        doc = nlp(text)

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
        # one bad doc shouldn't kill the whole job
        return empty

    return {
        "pos_counts": pos_counts,
        "n_unique_tokens": n_unique,
        "n_total_tokens": int(n_total),
    }


extract_pos_counts_udf = F.udf(_extract_pos_counts, returnType=UDF_RETURN_TYPE)


def main():
    spark = create_spark_session("gold-pos-counts-wiki")

    raw = spark.read.format("delta").load(INPUT_PATH)

    # cap text at 500k chars to bound python worker memory (same reason as legal)
    MAX_TEXT_CHARS = 500_000

    # wiki bronze columns are id / text only. no labels, no snapshot_date yet.
    # we still write labels as a null column so the schema matches legal gold,
    # makes downstream code easier when reading both
    df = raw.select(
        F.col("id").alias("document_id"),
        F.substring(F.col("text"), 1, MAX_TEXT_CHARS).alias("text"),
        F.lit(None).cast(StringType()).alias("labels"),
    ).filter(F.col("text").isNotNull() & (F.length("text") > 100))

    if args.limit:
        df = df.limit(args.limit)
        logger.info(f"Smoke test mode: limited to {args.limit:,} rows")

    # no snapshot_date on wiki, so just flat repartition
    df = df.repartition(200)

    input_count = df.count()
    logger.info(
        f"Processing {input_count:,} wiki documents across "
        f"{df.rdd.getNumPartitions()} partitions"
    )

    result = (
        df.withColumn("_pos", extract_pos_counts_udf(F.col("text")))
          .select(
              F.col("document_id"),
              F.col("labels"),
              F.col("_pos.pos_counts").alias("pos_counts"),
              F.col("_pos.n_unique_tokens").alias("n_unique_tokens"),
              F.col("_pos.n_total_tokens").alias("n_total_tokens"),
          )
    )

    (
        result.write.format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .save(OUTPUT_PATH)
    )

    output_count = spark.read.format("delta").load(OUTPUT_PATH).count()
    logger.info(f"Wrote {output_count:,} rows to {OUTPUT_PATH}")
    logger.info("Gold POS-counts extraction (wiki) complete")


if __name__ == "__main__":
    main()
