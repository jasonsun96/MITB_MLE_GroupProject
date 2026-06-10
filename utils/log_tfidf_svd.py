import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from pyspark.ml.feature import CountVectorizer, IDF, TruncatedSVD
from pyspark.ml.linalg import Vector, Vectors, VectorUDT
from pyspark.sql import functions as F
from pyspark.sql.functions import udf

from utils.ngram_tfidf import (
    ID_COLUMN,
    LABEL_COLUMN,
    MAX_FEATURES,
    MAX_N,
    MIN_N,
    TOKENS_COLUMN,
    _save_bytes_to_path,
    prepare_silver_data,
)

logger = logging.getLogger(__name__)

M_VALUES = [50, 100, 200, 500]

LOG_TFIDF_FORMULA = "log(tf) * (1 + log(idf))"


def _make_log_tfidf_udf(idf_vector):
    idf_arr = [float(x) for x in idf_vector.toArray()]
    vocab_size = len(idf_arr)

    @udf(VectorUDT())
    def log_tfidf(counts: Vector) -> Vector:
        if counts is None or counts.numNonzeros() == 0:
            return Vectors.sparse(vocab_size, [], [])

        out_indices = []
        out_values = []
        for idx, tf in zip(counts.indices, counts.values):
            tf = float(tf)
            if tf <= 0:
                continue
            idf = idf_arr[int(idx)]
            if idf <= 0:
                continue
            weight = math.log(tf) * (1.0 + math.log(idf))
            if weight != 0.0:
                out_indices.append(int(idx))
                out_values.append(weight)

        return Vectors.sparse(vocab_size, out_indices, out_values)

    return log_tfidf


def _make_truncate_svd_udf(m: int):
    @udf(VectorUDT())
    def truncate_svd(v: Vector) -> Vector:
        if v is None:
            return Vectors.dense([0.0] * m)
        return Vectors.dense([float(x) for x in v.toArray()[:m]])

    return truncate_svd


def _explained_variance_summary(svd_model, m_values: list[int]) -> dict:
    per_component = [float(x) for x in svd_model.explainedVariance.toArray()]
    cumulative = []
    running = 0.0
    for value in per_component:
        running += value
        cumulative.append(running)

    return {
        "per_component": per_component,
        "cumulative_at_m": {str(m): cumulative[m - 1] for m in m_values if m <= len(cumulative)},
    }


def build_log_tfidf_svd_features(
    df,
    min_n: int,
    max_n: int,
    max_features: int,
    m_values: list[int] | None = None,
):
    if not m_values:
        m_values = M_VALUES
    if min(m_values) < 1:
        raise ValueError("All SVD dimensions m must be >= 1")

    max_m = max(m_values)

    count_vectorizer = CountVectorizer(
        inputCol="token_array",
        outputCol="ngram_counts",
        vocabSize=max_features,
        minDF=2.0,
        ngramRange=(min_n, max_n),
    )
    count_model = count_vectorizer.fit(df)
    df_counts = count_model.transform(df)

    idf_model = IDF(inputCol="ngram_counts", outputCol="tfidf_std").fit(df_counts)
    log_tfidf_udf = _make_log_tfidf_udf(idf_model.idf)
    df_log = df_counts.withColumn("log_tfidf", log_tfidf_udf(F.col("ngram_counts")))

    svd_model = TruncatedSVD(k=max_m, inputCol="log_tfidf", outputCol="svd_full").fit(df_log)
    df_svd = svd_model.transform(df_log)

    for m in m_values:
        df_svd = df_svd.withColumn(f"svd_m{m}", _make_truncate_svd_udf(m)(F.col("svd_full")))

    df_gold = df_svd.drop("ngram_counts", "tfidf_std", "svd_full")

    vocab = count_model.vocabulary
    n_docs = df_gold.count()
    n_features = len(vocab)

    text_source = df_gold.select("text_source").first()[0]
    logger.info("Text source: %s", text_source)
    logger.info("Weighting: %s", LOG_TFIDF_FORMULA)
    logger.info("N-gram range: (%s, %s)", min_n, max_n)
    logger.info("SVD dimensions: %s", m_values)
    logger.info("Documents: %s | Features: %s", f"{n_docs:,}", f"{n_features:,}")

    return df_gold, vocab, svd_model, n_docs, n_features, m_values


def save_log_tfidf_svd_artifacts(
    gold_table_path: str,
    df_gold,
    vocab,
    svd_model,
    source_path: str,
    min_n: int,
    max_n: int,
    max_features: int,
    n_docs: int,
    n_features: int,
    m_values: list[int],
    spark,
) -> None:
    tag = f"{min_n}_{max_n}"
    base_path = gold_table_path.rstrip("/")
    vocab_path = f"{base_path}/gold_vocab_{tag}.json"
    meta_path = f"{base_path}/gold_meta_{tag}.json"
    docs_path = f"{base_path}/gold_documents_{tag}"

    svd_columns = [f"svd_m{m}" for m in m_values]
    gold_columns = [
        "doc_index",
        ID_COLUMN,
        LABEL_COLUMN,
        "text_source",
        TOKENS_COLUMN,
        "token_count",
        "log_tfidf",
        *svd_columns,
        "silver_ingest_ts",
        "silver_source",
    ]

    gold_df = df_gold.select(*gold_columns)
    gold_df.write.format("delta").option("mergeSchema", "true").mode("overwrite").save(gold_table_path)

    df_gold.select("doc_index", ID_COLUMN, LABEL_COLUMN, "text_source").write.format("delta").mode(
        "overwrite"
    ).save(docs_path)

    _save_bytes_to_path(vocab_path, json.dumps(list(vocab)).encode("utf-8"), spark)

    meta = {
        "gold_ingest_ts": datetime.now(timezone.utc).isoformat(),
        "silver_source": source_path,
        "n_documents": n_docs,
        "n_features": n_features,
        "ngram_range": [min_n, max_n],
        "max_features": max_features,
        "weighting_formula": LOG_TFIDF_FORMULA,
        "log_base": "natural",
        "m_values": m_values,
        "svd_k_fitted": max(m_values),
        "svd_columns": svd_columns,
        "explained_variance": _explained_variance_summary(svd_model, m_values),
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
        "log_tfidf_column": "log_tfidf",
        "training_feature_columns": svd_columns,
    }
    _save_bytes_to_path(meta_path, json.dumps(meta, indent=2).encode("utf-8"), spark)

    logger.info("Saved log-TF-IDF + SVD gold artifacts to %s", base_path)
    logger.info("  Delta table: %s", Path(gold_table_path).name)
    logger.info("  %s (document metadata)", Path(docs_path).name)
    logger.info("  %s", Path(vocab_path).name)
    logger.info("  %s", Path(meta_path).name)
