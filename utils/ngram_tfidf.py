import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from utils.text_preprocess import filter_token_noise, preprocess_tokens_base

logger = logging.getLogger(__name__)

# unigrams config
MIN_N = 1
MAX_N = 3
MAX_FEATURES = 50_000

TEXT_COLUMN = "act_raw_text"
LABEL_COLUMN = "labels"
ID_COLUMN = "CELEX"
TOKENS_COLUMN = "tokens"
DATE_COLUMNS = ("Date_document", "Date_publication")

DROP_COLUMNS = (
    # "bronze_ingest_ts",
    # "bronze_source_key",
    "Cites_links",
    "Ammends_links",
    "Eurlex_link",
    "ELI_link",
    "Proposal_link",
    "Oeil_link",
)

def _preprocess_tokens_for_gold(text: str) -> list[str]:
    return filter_token_noise(preprocess_tokens_base(text))


preprocess_tokens_udf = F.udf(_preprocess_tokens_for_gold, ArrayType(StringType()))


def _save_bytes_to_path(path: str, data: bytes, spark) -> None:
    jvm = spark.sparkContext._jvm
    uri = jvm.java.net.URI(path)
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, spark.sparkContext._jsc.hadoopConfiguration())
    out = fs.create(hadoop_path, True)
    out.write(bytearray(data))
    out.close()


def _clean_string_columns(df):
    for col_name, dtype in df.dtypes:
        if dtype != "string" or col_name in DROP_COLUMNS:
            continue
        df = df.withColumn(
            col_name,
            F.when(F.length(F.trim(F.coalesce(F.col(col_name), F.lit("")))) == 0, F.lit(None)).otherwise(
                F.trim(F.regexp_replace(F.coalesce(F.col(col_name), F.lit("")), r"\s+", " "))
            ),
        )
    return df


def prepare_silver_data(df, source_path: str):
    n_in = df.count()

    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    if drop_cols:
        df = df.drop(*drop_cols)

    missing = [col for col in (ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN) if col not in df.columns]
    if missing:
        raise ValueError(f"Required column missing from silver input: {missing}")

    df = _clean_string_columns(df)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df = df.withColumn(col, F.date_format(F.to_date(F.col(col)), "yyyy-MM-dd"))

    df = (
        df.filter(F.col(LABEL_COLUMN).isNotNull() & (F.length(F.col(LABEL_COLUMN)) > 0))
        .dropDuplicates([ID_COLUMN])
        .withColumn(
            "text_source",
            F.when(F.length(F.trim(F.coalesce(F.col(TEXT_COLUMN), F.lit("")))) > 0, F.lit(TEXT_COLUMN))
            .when(
                F.col(TOKENS_COLUMN).isNotNull() & (F.length(F.trim(F.col(TOKENS_COLUMN))) > 0),
                F.lit(TOKENS_COLUMN),
            )
            .otherwise(F.lit(TEXT_COLUMN)),
        )
        .withColumn(
            "token_array",
            F.when(
                F.length(F.trim(F.coalesce(F.col(TEXT_COLUMN), F.lit("")))) > 0,
                preprocess_tokens_udf(F.col(TEXT_COLUMN)),
            )
            .when(
                F.col(TOKENS_COLUMN).isNotNull() & (F.length(F.trim(F.col(TOKENS_COLUMN))) > 0),
                F.split(F.trim(F.col(TOKENS_COLUMN)), r"\s+"),
            )
            .otherwise(F.array().cast(ArrayType(StringType()))),
        )
        .withColumn(TOKENS_COLUMN, F.array_join(F.col("token_array"), " "))
        .withColumn("token_count", F.size(F.col("token_array")))
        .filter(F.size(F.col("token_array")) > 0)
        .withColumn("silver_ingest_ts", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn("silver_source", F.lit(source_path))
        .withColumn("doc_index", F.monotonically_increasing_id())
    )

    n_out = df.count()
    logger.info(
        "Silver preparation: %s input rows -> %s cleaned rows (%s dropped)",
        f"{n_in:,}",
        f"{n_out:,}",
        f"{n_in - n_out:,}",
    )
    return df


def build_gold_features(df, min_n: int, max_n: int, max_features: int):
    count_vectorizer = CountVectorizer(
        inputCol="token_array",
        outputCol="ngram_counts",
        vocabSize=max_features,
        minDF=2.0,
        ngramRange=(min_n, max_n),
    )
    count_model = count_vectorizer.fit(df)
    df_counts = count_model.transform(df)

    idf = IDF(inputCol="ngram_counts", outputCol="tfidf")
    idf_model = idf.fit(df_counts)
    df_gold = idf_model.transform(df_counts)

    vocab = count_model.vocabulary
    n_docs = df_gold.count()
    n_features = len(vocab)

    text_source = df_gold.select("text_source").first()[0]
    logger.info("Text source: %s", text_source)
    logger.info("N-gram range: (%s, %s)", min_n, max_n)
    logger.info("Documents: %s | Features: %s", f"{n_docs:,}", f"{n_features:,}")

    return df_gold, vocab, n_docs, n_features


def save_gold_artifacts(
    gold_table_path: str,
    df_gold,
    vocab,
    source_path: str,
    min_n: int,
    max_n: int,
    max_features: int,
    n_docs: int,
    n_features: int,
    spark,
) -> None:
    tag = f"{min_n}_{max_n}"
    base_path = gold_table_path.rstrip("/")
    vocab_path = f"{base_path}/gold_vocab_{tag}.json"
    meta_path = f"{base_path}/gold_meta_{tag}.json"
    docs_path = f"{base_path}/gold_documents_{tag}"

    gold_df = df_gold.select(
        "doc_index",
        ID_COLUMN,
        LABEL_COLUMN,
        "text_source",
        TOKENS_COLUMN,
        "token_count",
        "ngram_counts",
        "tfidf",
        "silver_ingest_ts",
        "silver_source",
    )

    gold_df.write.format("delta").option("mergeSchema", "true").mode("overwrite").save(gold_table_path)

    df_gold.select("doc_index", ID_COLUMN, LABEL_COLUMN, "text_source").write.format("delta").mode("overwrite").save(
        docs_path
    )

    _save_bytes_to_path(vocab_path, json.dumps(list(vocab)).encode("utf-8"), spark)

    meta = {
        "gold_ingest_ts": datetime.now(timezone.utc).isoformat(),
        "silver_source": source_path,
        "n_documents": n_docs,
        "n_features": n_features,
        "ngram_range": [min_n, max_n],
        "max_features": max_features,
        "preprocessing": [
            "lowercase",
            "nltk_stopword_removal",
            "nltk_lemmatization",
            "drop_single_char_alpha",
            "drop_digits_except_4digit",
        ],
        "gold_table": gold_table_path,
        "documents_table": docs_path,
        "vocab_file": Path(vocab_path).name,
        "counts_column": "ngram_counts",
        "tfidf_column": "tfidf",
    }
    _save_bytes_to_path(meta_path, json.dumps(meta, indent=2).encode("utf-8"), spark)

    logger.info("Saved gold artifacts to %s", base_path)
    logger.info("  Delta table: %s", Path(gold_table_path).name)
    logger.info("  %s (document metadata)", Path(docs_path).name)
    logger.info("  %s", Path(vocab_path).name)
    logger.info("  %s", Path(meta_path).name)
