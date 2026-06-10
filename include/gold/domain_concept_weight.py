import argparse
import logging
from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, LongType, MapType, StringType, StructField, StructType

from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Gold layer: domain concept weighting")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
parser.add_argument("--no-split", action="store_true", help="Skip labels.pkl split — use all legal data as train. For smoke testing only.")
parser.add_argument("--limit", type=int, default=None, help="Limit to N rows for smoke testing. Omit for full corpus.")
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

GOLD = schema["gold"]
GOLD_PATH = GOLD["path"]
GOLD_TABLES = GOLD["tables"]

LEGAL_POS_PATH        = f"{GOLD_PATH}/{GOLD_TABLES['pos_counts']['path']}"
WIKI_POS_PATH         = f"{GOLD_PATH}/{GOLD_TABLES['pos_counts_wiki']['path']}"
DCW_TRAIN_PATH        = f"{GOLD_PATH}/{GOLD_TABLES['dcw_features_train']['path']}"
DCW_VAL_TEST_OOT_PATH = f"{GOLD_PATH}/{GOLD_TABLES['dcw_features_val_test_oot']['path']}"
LABELS_PATH           = f"{GOLD_PATH}/{GOLD_TABLES['labels']['path']}"

MODEL_BANK            = schema["model_bank"]
MODEL_BANK_PATH       = MODEL_BANK["path"]
MODEL_BANK_FE         = MODEL_BANK["features_extractor"]
DCW_SCORE_PATH        = f"{MODEL_BANK_PATH}/{MODEL_BANK_FE['dcw_score']}"
DCW_TRAIN_DOC_IDS_PATH = f"{MODEL_BANK_PATH}/{MODEL_BANK_FE['dcw_train_doc_ids']}"

spark = create_spark_session("domain-concept-weight")

POS_TAGS  = ["NOUN", "PROPN"]
ALPHA     = 0.5
BETA      = 0.5

# schema for the lemma extraction UDF output
_LEMMA_SCHEMA = ArrayType(StructType([
    StructField("lemma", StringType()),
    StructField("count", IntegerType()),
]))

# schema for the DC contribution UDF output
_DC_CONTRIB_SCHEMA = ArrayType(StructType([
    StructField("lemma", StringType()),
    StructField("contrib", DoubleType()),
]))

def save_dcw_artifact(filtered_vocab, dp, dc, score, train_document_ids):
    """
    Save the DCW scoring artefact as two Delta tables.
    dcw_score: one row per lemma with corpus_count, dp, dc, score.
    dcw_train_doc_ids: one row per train document_id.
    """
    score_rows = [
        {"lemma": lemma, "corpus_count": int(filtered_vocab[lemma]),
         "dp": dp[lemma], "dc": dc[lemma], "score": score[lemma]}
        for lemma in score
    ]
    spark.createDataFrame(score_rows) \
        .write.format("delta").mode("overwrite").save(DCW_SCORE_PATH)
    logger.info(f"DCW score table saved to {DCW_SCORE_PATH}")

    spark.createDataFrame([{"document_id": d} for d in train_document_ids]) \
        .write.format("delta").mode("overwrite").save(DCW_TRAIN_DOC_IDS_PATH)
    logger.info(f"DCW train doc IDs saved to {DCW_TRAIN_DOC_IDS_PATH}")


def load_dcw_artifact():
    """Load the frozen DCW artifact from Delta tables. Returns a dict."""
    rows = spark.read.format("delta").load(DCW_SCORE_PATH).collect()
    filtered_vocab    = {r["lemma"]: r["corpus_count"] for r in rows}
    dp                = {r["lemma"]: r["dp"]           for r in rows}
    dc                = {r["lemma"]: r["dc"]           for r in rows}
    score             = {r["lemma"]: r["score"]        for r in rows}

    train_document_ids = {
        r["document_id"]
        for r in spark.read.format("delta").load(DCW_TRAIN_DOC_IDS_PATH).collect()
    }
    logger.info(f"DCW artifact loaded: {len(score):,} terms, {len(train_document_ids):,} train docs")
    return {"filtered_vocab": filtered_vocab, "dp": dp, "dc": dc,
            "score": score, "train_document_ids": train_document_ids}


def load_split_labels():
    """Load gold/labels Delta table. Returns a Spark DataFrame with document_id and category columns.
    category values: train / val / test / oot
    """
    df = spark.read.format("delta").load(LABELS_PATH)
    logger.info(f"Split labels loaded: {df.count():,} rows")
    return df


def add_n_nouns_propn(df, pos_tags=POS_TAGS):
    """Add n_nouns_propn: total NOUN+PROPN token count per document."""
    def count_nouns(pc):
        if not pc:
            return 0
        return sum(sum((pc.get(pos) or {}).values()) for pos in pos_tags)

    udf_fn = F.udf(count_nouns, LongType())
    return df.withColumn("n_nouns_propn", udf_fn(F.col("pos_counts")))


def build_filtered_vocab(train_df, pos_tags=POS_TAGS, min_freq=50, min_len=1):
    """
    Aggregate corpus-level lemma counts from train, apply frequency + length filter.
    Returns a plain Python dict {lemma: corpus_count} — frozen artefact for all splits.
    Collected to driver.
    """
    def extract_lemmas(pc):
        if not pc:
            return []
        return [
            (lemma, cnt)
            for pos in pos_tags
            for lemma, cnt in (pc.get(pos) or {}).items()
        ]

    extract_udf = F.udf(extract_lemmas, _LEMMA_SCHEMA)

    rows = (
        train_df
        .withColumn("lc", F.explode(extract_udf(F.col("pos_counts"))))
        .groupBy(F.col("lc.lemma").alias("lemma"))
        .agg(F.sum("lc.count").alias("total_count"))
        .filter(
            (F.length(F.col("lemma")) > min_len) &
            (F.col("total_count") > min_freq)
        )
        .collect()
    )

    return {row["lemma"]: row["total_count"] for row in rows}


def compute_score(dp, dc, alpha=ALPHA, beta=BETA):
    """
    Combine DP and DC into a single domain score per term (Eq. 3 in paper).
    Score(t) = alpha * (DP(t) / max_DP) + beta * (DC(t) / max_DC)
    Returns dict {lemma: score}. Computed on driver.
    """
    max_dp = max(dp.values())
    max_dc = max(dc.values())

    common = dp.keys() & dc.keys()
    return {
        term: alpha * (dp[term] / max_dp) + beta * (dc[term] / max_dc)
        for term in common
    }


def compute_dp(filtered_vocab, wiki_df, pos_tags=POS_TAGS):
    """
    Compute Domain Pertinence for each term in filtered_vocab.
    DP(t) = freq(t in legal train) / max(freq(t in wiki), 1)
    Returns dict {lemma: dp_score}. Collected to driver.
    """
    vocab_bc = spark.sparkContext.broadcast(set(filtered_vocab.keys()))

    def extract_lemmas_in_vocab(pc):
        if not pc:
            return []
        vocab = vocab_bc.value
        return [
            (lemma, cnt)
            for pos in pos_tags
            for lemma, cnt in (pc.get(pos) or {}).items()
            if lemma in vocab
        ]

    extract_udf = F.udf(extract_lemmas_in_vocab, _LEMMA_SCHEMA)

    wiki_counts = (
        wiki_df
        .withColumn("lc", F.explode(extract_udf(F.col("pos_counts"))))
        .groupBy(F.col("lc.lemma").alias("lemma"))
        .agg(F.sum("lc.count").alias("wiki_count"))
        .collect()
    )

    wiki_count_dict = {row["lemma"]: row["wiki_count"] for row in wiki_counts}

    return {
        lemma: legal_count / max(wiki_count_dict.get(lemma, 0), 1)
        for lemma, legal_count in filtered_vocab.items()
    }


def compute_dc(train_df, filtered_vocab, pos_tags=POS_TAGS):
    """
    Compute Domain Consensus for each term in filtered_vocab.
    DC(t) = -sum_{dk} nfreq(t/dk) * log(nfreq(t/dk))
    where nfreq(t/dk) = count(t in dk) / total NOUN+PROPN count in dk.
    Requires n_nouns_propn column to already exist on train_df.
    Returns dict {lemma: dc_score}. Collected to driver.
    """
    import math

    vocab_bc = spark.sparkContext.broadcast(set(filtered_vocab.keys()))

    def extract_dc_contribs(pc, n_nouns):
        if not pc or not n_nouns or n_nouns == 0:
            return []
        vocab = vocab_bc.value
        result = []
        for pos in pos_tags:
            for lemma, cnt in (pc.get(pos) or {}).items():
                if lemma in vocab:
                    nfreq = cnt / n_nouns
                    result.append((lemma, nfreq * math.log(nfreq)))
        return result

    extract_udf = F.udf(extract_dc_contribs, _DC_CONTRIB_SCHEMA)

    rows = (
        train_df
        .withColumn("dc_contrib", F.explode(extract_udf(F.col("pos_counts"), F.col("n_nouns_propn"))))
        .groupBy(F.col("dc_contrib.lemma").alias("lemma"))
        .agg((-F.sum("dc_contrib.contrib")).alias("dc"))
        .collect()
    )

    return {row["lemma"]: row["dc"] for row in rows}


def add_n_nouns_propn_filtered(df, filtered_vocab, pos_tags=POS_TAGS):
    """Add n_nouns_propn_filtered: count of NOUN+PROPN tokens whose lemma is in the filtered vocab."""
    vocab_bc = spark.sparkContext.broadcast(set(filtered_vocab.keys()))

    def count_filtered(pc):
        if not pc:
            return 0
        vocab = vocab_bc.value
        return sum(
            cnt
            for pos in pos_tags
            for lemma, cnt in (pc.get(pos) or {}).items()
            if lemma in vocab
        )

    udf_fn = F.udf(count_filtered, LongType())
    return df.withColumn("n_nouns_propn_filtered", udf_fn(F.col("pos_counts")))


def add_dcw_columns(df, score, pos_tags=POS_TAGS):
    """
    Add one column per lemma in score: dcw_{lemma} = freq(t in doc) * score[t].
    Lemmas absent from a document get 0.0.
    Computed via a single UDF pass to avoid O(vocab) plan nodes.
    """
    score_bc = spark.sparkContext.broadcast(score)

    def compute_dcw_map(pc):
        if not pc:
            return {}
        s = score_bc.value
        result = {}
        for pos in pos_tags:
            for lemma, cnt in (pc.get(pos) or {}).items():
                if lemma in s:
                    result[lemma] = result.get(lemma, 0.0) + cnt * s[lemma]
        return result

    map_udf = F.udf(compute_dcw_map, MapType(StringType(), DoubleType()))
    df = df.withColumn("_dcw_map", map_udf(F.col("pos_counts")))

    existing = [F.col(c) for c in df.columns if c != "_dcw_map"]
    dcw_cols = [
        F.coalesce(F.col("_dcw_map")[lemma], F.lit(0.0)).alias(f"dcw_{lemma}")
        for lemma in score
    ]
    return df.select(existing + dcw_cols)


def main():
    legal = spark.read.format("delta").load(LEGAL_POS_PATH)
    wiki  = spark.read.format("delta").load(WIKI_POS_PATH)

    if args.limit:
        legal = legal.limit(args.limit)
        wiki  = wiki.limit(args.limit)
        logger.info(f"Smoke test mode: limited to {args.limit:,} rows")

    logger.info(f"Loaded legal: {legal.count():,} rows")
    logger.info(f"Loaded wiki : {wiki.count():,} rows")

    if args.no_split: # REMOVE WHEN DONE
        logger.warning("--no-split: using all legal data as train, skipping val/test/oot. Smoke test only.")
        train        = legal
        val_test_oot = None
    else:
        labels_sdf = load_split_labels().select("document_id", "category")
        legal = legal.join(labels_sdf, on="document_id", how="inner")
        train        = legal.filter(F.col("category") == "train").drop("category")
        val_test_oot = legal.filter(F.col("category") != "train").drop("category")
        logger.info(f"Val/test/oot size: {val_test_oot.count():,} documents")

    train_document_ids = {row["document_id"] for row in train.select("document_id").collect()}
    logger.info(f"Train size: {len(train_document_ids):,} documents")

    train = add_n_nouns_propn(train)
    n_nouns_propn_total = train.agg(F.sum("n_nouns_propn")).collect()[0][0]
    logger.info(f"n_nouns_propn added — corpus total: {n_nouns_propn_total:,}")

    filtered_vocab = build_filtered_vocab(train)
    logger.info(f"Filtered vocab size: {len(filtered_vocab):,} lemmas")

    train = add_n_nouns_propn_filtered(train, filtered_vocab)
    n_nouns_propn_filtered_total = train.agg(F.sum("n_nouns_propn_filtered")).collect()[0][0]
    logger.info(f"n_nouns_propn_filtered added — corpus total: {n_nouns_propn_filtered_total:,}")

    dp = compute_dp(filtered_vocab, wiki)
    logger.info(f"DP computed for {len(dp):,} terms")

    dc = compute_dc(train, filtered_vocab)
    logger.info(f"DC computed for {len(dc):,} terms")

    score = compute_score(dp, dc)
    logger.info(f"Domain score computed for {len(score):,} terms")

    # ARTIFACT
    save_dcw_artifact(filtered_vocab, dp, dc, score, train_document_ids)
  

    train_dcw = add_dcw_columns(train, score)
    logger.info(f"DCW columns added to train ({len(score):,} lemma columns)")

    (
        train_dcw.drop("pos_counts", "n_unique_tokens", "n_total_tokens", "labels")
        .write.format("delta")
        .mode("overwrite")
        .partitionBy("snapshot_date")
        .option("mergeSchema", "true")
        .save(DCW_TRAIN_PATH)
    )
    output_count = spark.read.format("delta").load(DCW_TRAIN_PATH).count()
    logger.info(f"Wrote {output_count:,} rows to {DCW_TRAIN_PATH}")

    if val_test_oot is not None: # REMOVE WHEN DONE
        artifact = load_dcw_artifact()
        val_test_oot = add_n_nouns_propn(val_test_oot)
        val_test_oot = add_n_nouns_propn_filtered(val_test_oot, artifact["filtered_vocab"])
        val_test_oot_dcw = add_dcw_columns(val_test_oot, artifact["score"])
        (
            val_test_oot_dcw.drop("pos_counts", "n_unique_tokens", "n_total_tokens", "labels")
            .write.format("delta")
            .mode("overwrite")
            .partitionBy("snapshot_date")
            .option("mergeSchema", "true")
            .save(DCW_VAL_TEST_OOT_PATH)
        )
        output_count_vto = spark.read.format("delta").load(DCW_VAL_TEST_OOT_PATH).count()
        logger.info(f"Wrote {output_count_vto:,} rows to {DCW_VAL_TEST_OOT_PATH}")

    logger.info("Domain concept weighting complete")


if __name__ == "__main__":
    main()