"""
gold Legal-BERT embeddings for the wiki (non-law baseline) corpus.

mirror of legal_embeddings.py but reads from wiki bronze. wiki has a different
schema:
  - id column is `id` (legal uses CELEX)
  - text column is `text` (legal uses act_raw_text)
  - no labels column yet (Cheewei is adding them, null until then)
  - no snapshot_date (no partitionBy on write)

same model and chunk-and-pool strategy as legal so the two gold tables stay
comparable downstream.

wiki docs are short (median ~100 words), so most will produce just one chunk
and run much faster than legal.
"""

import argparse
import logging
from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark.sql.types import (ArrayType, FloatType, IntegerType, StringType,
                               StructField, StructType)
from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Gold layer: Legal-BERT embeddings (wiki)")
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit to N rows for smoke testing",
)
parser.add_argument(
    "--input-layer",
    default="silver",
    choices=["bronze", "silver"],
)
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

if args.input_layer == "bronze":
    INPUT_PATH = f"{schema['bronze']['path']}/{schema['bronze']['tables']['wiki_docs_raw']['path']}"
else:
    INPUT_PATH = f"{schema['silver']['path']}/{schema['silver']['tables']['wiki_docs_processed']['path']}"

OUTPUT_PATH = f"{schema['gold']['path']}/{schema['gold']['corpus']['embeddings_wiki']['path']}"

logger.info(f"Input  ({args.input_layer}): {INPUT_PATH}")
logger.info(f"Output (gold)             : {OUTPUT_PATH}")

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_DIM = 768
CHUNK_TOKENS = 510
MAX_CHUNKS = 5

UDF_RETURN_TYPE = StructType(
    [
        StructField("embedding", ArrayType(FloatType()), nullable=False),
        StructField("embedding_model", StringType(), nullable=False),
        StructField("n_chunks", IntegerType(), nullable=False),
    ]
)


_TOKENIZER = None
_MODEL = None
_DEVICE = None


def _get_model():
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        # use GPU when available, otherwise CPU (same code path for everyone)
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(MODEL_NAME).to(_DEVICE)
        _MODEL.eval()
        torch.set_grad_enabled(False)
    return _TOKENIZER, _MODEL, _DEVICE


def _embed_document(text):
    empty = {"embedding": [0.0] * EMBEDDING_DIM, "embedding_model": MODEL_NAME, "n_chunks": 0}

    if not text:
        return empty

    import torch

    tokenizer, model, device = _get_model()

    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not token_ids:
        return empty

    max_len = CHUNK_TOKENS * MAX_CHUNKS
    token_ids = token_ids[:max_len]
    chunks = [token_ids[i : i + CHUNK_TOKENS] for i in range(0, len(token_ids), CHUNK_TOKENS)]

    if not chunks:
        return empty

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    chunk_input_ids = [[cls_id] + c + [sep_id] for c in chunks]
    max_chunk_len = max(len(c) for c in chunk_input_ids)

    input_ids_batch = []
    attention_mask_batch = []
    for ids in chunk_input_ids:
        pad_n = max_chunk_len - len(ids)
        input_ids_batch.append(ids + [pad_id] * pad_n)
        attention_mask_batch.append([1] * len(ids) + [0] * pad_n)

    input_ids = torch.tensor(input_ids_batch, device=device)
    attention_mask = torch.tensor(attention_mask_batch, device=device)

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        chunk_vectors = summed / counts
        doc_vector = chunk_vectors.mean(dim=0).cpu()
        return {
            "embedding": doc_vector.tolist(),
            "embedding_model": MODEL_NAME,
            "n_chunks": len(chunks),
        }
    except Exception:
        return empty


embed_udf = F.udf(_embed_document, returnType=UDF_RETURN_TYPE)


def main():
    spark = create_spark_session("gold-wiki-embeddings")

    raw = spark.read.format("delta").load(INPUT_PATH)

    MAX_TEXT_CHARS = 500_000

    # wiki bronze: id / text only, no labels or snapshot_date.
    # write labels as null so the schema matches legal gold.
    df = raw.select(
        F.col("id").alias("document_id"),
        F.substring(F.col("text"), 1, MAX_TEXT_CHARS).alias("text"),
        F.lit(None).cast(StringType()).alias("labels"),
    ).filter(F.col("text").isNotNull() & (F.length("text") > 100))

    if args.limit:
        df = df.limit(args.limit)
        logger.info(f"Smoke test mode: limited to {args.limit:,} rows")

    df = df.repartition(32)

    input_count = df.count()
    logger.info(f"Processing {input_count:,} wiki documents across " f"{df.rdd.getNumPartitions()} partitions")

    result = df.withColumn("_emb", embed_udf(F.col("text"))).select(
        F.col("document_id"),
        F.col("labels"),
        F.col("_emb.embedding").alias("embedding"),
        F.col("_emb.embedding_model").alias("embedding_model"),
        F.col("_emb.n_chunks").alias("n_chunks"),
    )

    (result.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(OUTPUT_PATH))

    output_count = spark.read.format("delta").load(OUTPUT_PATH).count()
    logger.info(f"Wrote {output_count:,} rows to {OUTPUT_PATH}")
    logger.info("Gold wiki embeddings complete")


if __name__ == "__main__":
    main()
