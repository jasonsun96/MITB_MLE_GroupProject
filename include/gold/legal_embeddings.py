import argparse
import logging
from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark.sql.types import (ArrayType, FloatType, IntegerType, StringType,
                               StructField, StructType)

from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Gold layer: Legal-BERT embeddings (legal)")
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
parser.add_argument(
    "--snapshot-date",
    default=None,
    help="Process and overwrite only this snapshot_date partition (YYYY-MM-DD).",
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
    INPUT_PATH = f"{schema['bronze']['path']}/{schema['bronze']['tables']['legal_docs_raw']['path']}"
else:
    INPUT_PATH = f"{schema['silver']['path']}/{schema['silver']['tables']['legal_docs_processed']['path']}"
# silver passes through bronze's column names, so still act_raw_text for both
TEXT_COL = "act_raw_text"

OUTPUT_PATH = f"{schema['gold']['path']}/{schema['gold']['corpus']['embeddings']['path']}"

logger.info(f"Input  ({args.input_layer}): {INPUT_PATH}")
logger.info(f"Output (gold)             : {OUTPUT_PATH}")

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_DIM = 768
CHUNK_TOKENS = 510  # 512 minus room for [CLS] and [SEP]
MAX_CHUNKS = 5  # cap per doc to bound runtime on the long tail

# UDF output: the embedding vector + the model name (so we can stack multiple
# embedding models in the same table later if we want to compare)
UDF_RETURN_TYPE = StructType(
    [
        StructField("embedding", ArrayType(FloatType()), nullable=False),
        StructField("embedding_model", StringType(), nullable=False),
        StructField("n_chunks", IntegerType(), nullable=False),
    ]
)


# lazy singletons, loaded once per python worker
_TOKENIZER = None
_MODEL = None
_DEVICE = None


def _get_model():
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        # use GPU when the container has one (--gpus / compose device
        # reservation), otherwise plain CPU. teammates without nvidia
        # hardware run the exact same code path.
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(MODEL_NAME).to(_DEVICE)
        _MODEL.eval()
        # disable autograd, we're only doing inference
        torch.set_grad_enabled(False)
    return _TOKENIZER, _MODEL, _DEVICE


def _embed_document(text):
    """tokenize -> chunk -> embed -> mean pool -> mean of chunks"""
    empty = {"embedding": [0.0] * EMBEDDING_DIM, "embedding_model": MODEL_NAME, "n_chunks": 0}

    if not text:
        return empty

    import torch

    tokenizer, model, device = _get_model()

    # tokenize the whole doc, no special tokens (we add them per chunk)
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    if not token_ids:
        return empty

    # split into chunks of CHUNK_TOKENS, cap at MAX_CHUNKS
    max_len = CHUNK_TOKENS * MAX_CHUNKS
    token_ids = token_ids[:max_len]
    chunks = [token_ids[i : i + CHUNK_TOKENS] for i in range(0, len(token_ids), CHUNK_TOKENS)]

    if not chunks:
        return empty

    # build a single batch with [CLS] + chunk + [SEP], pad to max length in batch
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
        # last_hidden_state shape: (n_chunks, seq_len, 768)
        hidden = outputs.last_hidden_state
        # mean pool per chunk, masking out padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        chunk_vectors = summed / counts  # (n_chunks, 768)
        # average across chunks
        doc_vector = chunk_vectors.mean(dim=0).cpu()  # (768,) back to host
        return {
            "embedding": doc_vector.tolist(),
            "embedding_model": MODEL_NAME,
            "n_chunks": len(chunks),
        }
    except Exception:
        # one bad doc shouldn't kill the whole job
        return empty


embed_udf = F.udf(_embed_document, returnType=UDF_RETURN_TYPE)


def main():
    spark = create_spark_session("gold-legal-embeddings")

    raw = spark.read.format("delta").load(INPUT_PATH)
    if args.snapshot_date:
        raw = raw.filter(F.col("snapshot_date") == F.lit(args.snapshot_date))
        logger.info("Scoped legal embeddings extraction to snapshot_date=%s", args.snapshot_date)

    # truncate doc text at spark level. with 5-chunk cap on the python side
    # we'd never use more than ~2500 chars worth of text anyway
    MAX_TEXT_CHARS = 500_000

    select_exprs = [
        F.col("CELEX").alias("document_id"),
        F.substring(F.col(TEXT_COL), 1, MAX_TEXT_CHARS).alias("text"),
        F.col("labels").alias("labels"),
        F.col("snapshot_date").alias("snapshot_date"),
    ]
    if "batch_id" in raw.columns:
        select_exprs.append(F.col("batch_id"))

    df = raw.select(*select_exprs).filter(F.col("text").isNotNull() & (F.length("text") > 100))

    if args.limit:
        df = df.limit(args.limit)
        logger.info(f"Smoke test mode: limited to {args.limit:,} rows")

    # fewer, larger partitions than POS because BERT model load is expensive
    # (440 MB per worker). want to amortise that load across many docs per worker.
    df = df.repartition(32, "snapshot_date")

    input_count = df.count()
    logger.info(f"Processing {input_count:,} documents across " f"{df.rdd.getNumPartitions()} partitions")

    # run the udf, then flatten the struct cols back to top-level
    result = df.withColumn("_emb", embed_udf(F.col("text"))).select(
        F.col("document_id"),
        F.col("labels"),
        F.col("snapshot_date"),
        *([F.col("batch_id")] if "batch_id" in df.columns else []),
        F.col("_emb.embedding").alias("embedding"),
        F.col("_emb.embedding_model").alias("embedding_model"),
        F.col("_emb.n_chunks").alias("n_chunks"),
    )

    writer = result.write.format("delta").mode("overwrite").option("mergeSchema", "true")
    if args.snapshot_date:
        writer = writer.option("replaceWhere", f"snapshot_date = '{args.snapshot_date}'")
    writer.partitionBy("snapshot_date").save(OUTPUT_PATH)

    output_count = spark.read.format("delta").load(OUTPUT_PATH).count()
    logger.info(f"Wrote {output_count:,} rows to {OUTPUT_PATH}")
    logger.info("Gold legal embeddings complete")


if __name__ == "__main__":
    main()
