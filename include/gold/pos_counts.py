"""
gold POS counts for the legal corpus.

reads legal silver, runs spaCy POS tagging, writes lemma counts grouped by
POS tag to gold. anyone downstream (Yuhui's DP/DC, noun-only stuff, whatever)
just reads this and filters to the POS tags they need.

schema:
  document_id, labels, snapshot_date
  pos_counts: {pos_tag: {lemma: count}}    e.g. {"NOUN": {"sample": 29, ...}, "VERB": {...}}
  n_unique_tokens, n_total_tokens

to pull nouns + proper nouns (the DP/DC use case):
  nouns = {**row.pos_counts.get("NOUN", {}), **row.pos_counts.get("PROPN", {})}

notes:
  - regular UDF instead of pandas_udf, otherwise arrow blows up on java 17
  - spacy loaded once per python worker (module-level singleton)
  - parser and ner disabled, we only need POS
  - text capped at 500k chars before tagging to keep python worker memory bounded
"""
import argparse
import logging
from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from utils.spark_session import create_spark_session

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

SILVER = schema["silver"]
SILVER_PATH = SILVER["path"]
SILVER_TABLES = SILVER["tables"]
GOLD = schema["gold"]
GOLD_PATH = GOLD["path"]
GOLD_TABLES = GOLD["tables"]

INPUT_PATH  = f"{SILVER_PATH}/{SILVER_TABLES['legal_docs_processed']['path']}"
OUTPUT_PATH = f"{GOLD_PATH}/{GOLD_TABLES['pos_counts']['path']}"
# silver passes through bronze's column names, so still act_raw_text not text
TEXT_COL    = "act_raw_text"

logger.info(f"Input  (silver): {INPUT_PATH}")
logger.info(f"Output (gold)  : {OUTPUT_PATH}")

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
        # default is 1M chars, some eurlex docs are bigger so bump it.
        # safe to do since parser/ner are off
        _NLP.max_length = 5_000_000
    return _NLP


def _extract_pos_counts(text):
    """UDF body. Returns the struct defined by UDF_RETURN_TYPE."""
    empty = {"pos_counts": {}, "n_unique_tokens": 0, "n_total_tokens": 0}

    if not text:
        return empty

    nlp = _get_nlp()

    # belt and braces: cap again in case spark-level truncation missed something
    if len(text) > nlp.max_length:
        text = text[: nlp.max_length]

    try:
        doc = nlp(text)

        # group lemma counts by POS tag
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
    spark = create_spark_session("gold-pos-counts")

    raw = spark.read.format("delta").load(INPUT_PATH)

    # truncate doc text at spark level. 500k chars is well past p95 of the
    # corpus, so we lose ~nothing from real docs but cap the python worker
    # memory on outliers (some eurlex docs are 1.5M+ chars)
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

    # repartition by snapshot_date so the shuffle lines up with partitionBy
    # on write. otherwise you'd end up with way too many small files in R2
    df = df.repartition(200, "snapshot_date")

    input_count = df.count()
    logger.info(
        f"Processing {input_count:,} documents across "
        f"{df.rdd.getNumPartitions()} partitions"
    )

    # run the udf, then flatten the struct cols back to top-level
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
