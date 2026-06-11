# Silver to Gold: N-gram and TF-IDF Pipelines

This document describes how legal document features move from **silver** through **gold** to **model_bank** in the medallion pipeline.

The design separates responsibilities to avoid train/val/test leakage:

1. **N-gram extraction** — tokenize and count n-grams per document (no vocabulary fitting).
2. **Train/val/test/OOT split** — assign each document a `category` (upstream of feature fitting).
3. **TF-IDF freezing** — fit vocabulary and IDF on **train only**, apply frozen weights to all splits.

```text
silver/legal_docs_processed
        │
        ▼  ngram_processing.py
gold/ngram_count ─────────────┬──────────────────────────────┐
        │                     │                              │
        │                     ▼                              │
        │           tfidf_processing.py ◄── join ── gold/labels
        │                     │              (document_id, category)
        │                     │
        │         fit vocab + IDF on category == "train"
        │         apply frozen artifact to all splits
        │                     │
        ├─────────────────────┼──────────────────────────────┐
        ▼                     ▼                              ▼
gold/ngram_count      gold/tfidf_features_train     model_bank/features_extractor/tfidf/
                      gold/tfidf_features_val_test_oot   (vocab, idf, meta)
```

---

## Pipeline context

| Layer | Job script | R2 path (legal docs) |
|-------|------------|----------------------|
| Bronze | `include/bronze/ingest_bronze.py` | `s3a://cs611-project/bronze/legal_docs_raw` |
| Silver | `include/silver/sample_silver.py` | `s3a://cs611-project/silver/legal_docs_processed` |
| Gold (n-grams) | `include/gold/ngram_processing.py` | `s3a://cs611-project/gold/ngram_count` |
| Gold (labels/split) | *(not implemented yet)* | `s3a://cs611-project/gold/labels` |
| Gold (TF-IDF features) | `include/gold/tfidf_processing.py` | `s3a://cs611-project/gold/tfidf_features_train` |
| | | `s3a://cs611-project/gold/tfidf_features_val_test_oot` |
| Model bank | *(written by tfidf_processing.py)* | `s3a://cs611-project/model_bank/features_extractor/tfidf/` |

Paths are defined in `schema.yaml`.

The Airflow DAG (`dags/pipeline.py`) runs bronze → silver → gold in order. Update the gold task to call `ngram_processing.py` and `tfidf_processing.py` (not the legacy `sample_gold.py`).

---

## Job 1: N-gram extraction (`ngram_processing.py`)

**Reads:** `silver/legal_docs_processed`  
**Writes:** `gold/ngram_count`  
**Fits anything?** No — per-document n-gram count maps only.

### Silver input columns

| Column | Purpose |
|--------|---------|
| `CELEX` | Document ID (aliased to `document_id` in gold) |
| `act_raw_text` | Raw legal document text |
| `labels` | Topic label |
| `snapshot_date` | Partition column (optional but expected) |

Silver may also contain `act_clean_text` and quality flags from `sample_silver.py`; the n-gram job currently tokenizes `act_raw_text` only.

### Processing steps

1. Truncate text to `MAX_TEXT_CHARS = 500_000`.
2. Filter null or very short text (length ≤ 100).
3. Filter empty labels; deduplicate by `CELEX`.
4. Tokenize with NLTK inside a Spark UDF:
   - Lowercase alphanumeric tokens
   - English stopword removal
   - Lemmatization (WordNet)
   - Drop single-character alphabetic tokens
   - Keep 4-digit year tokens; drop other pure numbers
5. Build per-document n-gram counts (`MIN_N=1`, `MAX_N=3`) as a **map** — no `CountVectorizer`, no `MAX_FEATURES`, no global vocabulary.
6. Write Delta table partitioned by `snapshot_date` when present.

### Output table: `gold/ngram_count`

| Column | Type | Description |
|--------|------|-------------|
| `document_id` | string | Former `CELEX` |
| `labels` | string | Topic label |
| `snapshot_date` | string | Snapshot partition (if available) |
| `tokens` | string | Space-joined preprocessed tokens |
| `token_count` | int | Number of tokens |
| `ngram_counts` | map | `{ngram_string: count}` e.g. `{"legal act": 3}` |
| `text_source` | string | Source text column used (`act_raw_text`) |
| `silver_ingest_ts` | string | UTC timestamp of this job |
| `silver_source` | string | Upstream silver table path |

### Run

```bash
docker compose run --rm document-topic-tagger python include/gold/ngram_processing.py
docker compose run --rm document-topic-tagger python include/gold/ngram_processing.py --limit 100
```

---

## Job 2: Labels / split table (`gold/labels`)

**Status:** Schema path exists; **no producer script in the repo yet.**

`tfidf_processing.py` expects a Delta table at `gold/labels` with at least:

| Column | Description |
|--------|-------------|
| `document_id` | Join key (matches `ngram_count.document_id`) |
| `category` | Split label: `train`, `val`, `test`, `oot`, etc. |

Split logic used by the TF-IDF job:

| Filter | Use |
|--------|-----|
| `category == "train"` | Fit vocabulary and IDF |
| `category != "train"` | Score only (val / test / OOT grouped as holdout) |

Documents in `ngram_count` without a matching `labels` row are dropped at join time.

---

## Job 3: TF-IDF feature freezing (`tfidf_processing.py`)

**Reads:** `gold/ngram_count` + `gold/labels`  
**Writes:** `gold/tfidf_features_train`, `gold/tfidf_features_val_test_oot`, model_bank artefacts  
**Fits on:** train split only.

### Flow

```text
load_ngram_counts()
    │
    ▼
load_split_labels()  →  join on document_id
    │
    ├─ train   (category == "train")
    └─ holdout (category != "train")
    │
    ▼
build_tfidf_artifact(train)     # vocab from train doc-freq + fit IDF
    │
    ▼
save_tfidf_artifact()           # model_bank JSON artefacts
    │
    ▼
add_tfidf_column(train)         # frozen transform
add_tfidf_column(holdout)
    │
    ▼
write gold/tfidf_features_train
write gold/tfidf_features_val_test_oot
```

### Configuration (defaults)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MIN_N` / `MAX_N` | `1` / `3` | N-gram range (must match ngram job) |
| `MAX_FEATURES` | `50000` | Max vocabulary size (train only) |
| `MIN_DOC_FREQ` | `2` | Term must appear in ≥ 2 train documents |

### Feature columns

Both columns are produced from the **same frozen train artifact** (vocabulary + IDF):

| Column | Formula | Description |
|--------|---------|-------------|
| `tfidf` | `tf × idf` | Standard sparse TF-IDF (Spark IDF weights) |
| `log_tfidf` | `log(tf) × (1 + log(idf))` | Custom log-TF-IDF sparse vector |

For each term \(i\) where \(tf_i > 0\) and \(idf_i > 0\):

```text
log_tfidf_i = log(tf_i) × (1 + log(idf_i))
```

- \(tf_i\) = raw n-gram count from the frozen vocabulary index
- \(idf_i\) = train-fitted IDF weight from the frozen artifact
- `log` = natural logarithm

`log_tfidf` is **not** `log(tf × idf)` and is **not** refit on holdout data.

### Output feature tables

**Train:** `s3a://cs611-project/gold/tfidf_features_train`  
**Holdout (val/test/oot):** `s3a://cs611-project/gold/tfidf_features_val_test_oot`

| Column | Description |
|--------|-------------|
| `document_id` | Document ID |
| `tfidf` | Sparse TF-IDF vector |
| `log_tfidf` | Sparse log-TF-IDF vector |
| `silver_ingest_ts` | Lineage from `ngram_count` |
| `silver_source` | Lineage from `ngram_count` |
| `snapshot_date` | Included when present |
| `doc_index` | Included when present in upstream `ngram_count` |

### Model bank artefacts

Written to `s3a://cs611-project/model_bank/features_extractor/tfidf/`:

| File | Contents |
|------|----------|
| `gold_vocab_1_3.json` | Ordered n-gram terms |
| `gold_ngram_index_1_3.json` | `{ngram: index}` mapping |
| `gold_idf_1_3.json` | Frozen IDF values |
| `gold_train_document_ids.json` | Train document IDs used for fitting |
| `gold_meta_1_3.json` | Run metadata, paths, `final_feature_columns` |

Key metadata fields:

```json
{
  "tfidf_column": "tfidf",
  "log_tfidf_column": "log_tfidf",
  "log_tfidf_formula": "log(tf) * (1 + log(idf))",
  "final_feature_columns": ["tfidf", "log_tfidf"]
}
```

### CLI

```bash
# Full run (requires gold/labels)
docker compose run --rm document-topic-tagger python include/gold/tfidf_processing.py

# Smoke test: skip labels join, fit on all ngram_count rows
docker compose run --rm document-topic-tagger python include/gold/tfidf_processing.py --no-split --limit 100
```

| Flag | Purpose |
|------|---------|
| `--limit N` | Process only N rows from `ngram_count` |
| `--no-split` | Smoke test only — treat all rows as train |
| `--log-level` | Logging verbosity |

---

## Leakage controls

| Step | Trained on full corpus? | Notes |
|------|-------------------------|-------|
| N-gram extraction | N/A (no fitting) | Per-doc maps only |
| Vocabulary selection | **Train only** | Top terms by train document frequency |
| IDF | **Train only** | Spark `IDF.fit()` on train count vectors |
| `log_tfidf` weights | **Train IDF only** | Uses frozen train `idf` values |
| Holdout scoring | No refit | Same vocab + IDF applied to val/test/oot |

Val/test/oot never influence vocabulary or IDF.

---

## Using gold features for training

Gold feature tables are **feature stores**, not trainers. Join labels from `ngram_count` or your labels table for `y`.

| Experiment | Table | Feature column (`X`) |
|------------|-------|----------------------|
| Baseline TF-IDF | `tfidf_features_train` | `tfidf` |
| Log-TF-IDF | `tfidf_features_train` | `log_tfidf` |

```python
from utils.spark_session import create_spark_session

spark = create_spark_session("read-gold")

train = spark.read.format("delta").load("s3a://cs611-project/gold/tfidf_features_train")
holdout = spark.read.format("delta").load("s3a://cs611-project/gold/tfidf_features_val_test_oot")

train.select("document_id", "tfidf").show(5)
```

To map a vector index to an n-gram term, load `gold_vocab_1_3.json` from model_bank. Index `i` in the sparse vector corresponds to `vocab[i]`.

---

## Code layout

```text
include/gold/
  ngram_processing.py    # Job 1: silver → gold/ngram_count
  tfidf_processing.py    # Job 3: ngram_count + labels → TF-IDF features + model_bank
  gold_io.py             # Shared Delta / JSON write helpers

utils/
  spark_session.py       # Spark + Delta + R2 connection

schema.yaml              # Table paths for all layers
docs/ngram_tfidf.md      # This document
```

---

## How to run (full sequence)

```bash
docker compose build document-topic-tagger

# 1. Silver (if not already done)
docker compose run --rm document-topic-tagger python include/silver/sample_silver.py

# 2. Gold n-gram extraction
docker compose run --rm document-topic-tagger python include/gold/ngram_processing.py

# 3. Labels/split table (must exist before step 4)
#    gold/labels with document_id + category

# 4. TF-IDF feature freezing
docker compose run --rm document-topic-tagger python include/gold/tfidf_processing.py
```

Requires R2 credentials (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) in `utils/spark_session.py`.

### Via Airflow

1. `docker compose up -d`
2. Open `http://localhost:8080`, unpause and trigger `medallion_pipeline`.
3. Ensure `dags/pipeline.py` calls the current gold scripts.

---

## Bronze EDA (`notebooks/eda.ipynb`)

Interactive exploratory analysis on bronze `legal_docs_raw`. Section 6 mirrors the tokenization in `ngram_processing.py` (inline Spark UDF copies for worker compatibility).

```bash
docker compose run --rm -p 8888:8888 -v "${PWD}:/app" document-topic-tagger bash -lc \
  "pip install -q jupyter && jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir=/app/notebooks"
```

---

## Silver vs gold: summary

| Aspect | Silver | `gold/ngram_count` | `gold/tfidf_features_*` |
|--------|--------|--------------------|-------------------------|
| Purpose | Cleaned document storage | Per-doc n-gram maps | ML-ready sparse vectors |
| Text | Raw / cleaned columns | Tokenized string | Not stored (vectors only) |
| Fitting | None | None | Vocab + IDF on train |
| Key ID | `CELEX` | `document_id` | `document_id` |
| Features | None | `ngram_counts` map | `tfidf`, `log_tfidf` |

---

## Reading data back

```python
from utils.spark_session import create_spark_session

spark = create_spark_session("read-gold")

ngrams = spark.read.format("delta").load("s3a://cs611-project/gold/ngram_count")
train = spark.read.format("delta").load("s3a://cs611-project/gold/tfidf_features_train")

ngrams.select("document_id", "labels", "token_count").show(5)
train.select("document_id", "tfidf").show(5)
```

Check `model_bank/features_extractor/tfidf/gold_meta_1_3.json` for artefact paths, feature column names, and the log-TF-IDF formula.
