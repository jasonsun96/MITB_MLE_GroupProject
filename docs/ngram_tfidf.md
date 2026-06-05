# Silver to Gold: N-gram TF-IDF Feature Pipeline

This document describes what happens when data moves from the **silver** layer to the **gold** layer in the medallion pipeline, specifically for the `features_ngram_tfidf` gold table.

## Pipeline context

The Airflow DAG (`dags/pipeline.py`) runs three tasks in order:

```text
ingest_bronze  →  process_silver  →  process_gold
```

| Layer  | Job script                         | R2 path (legal docs)                          |
|--------|-------------------------------------|-----------------------------------------------|
| Bronze | `include/bronze/ingest_bronze.py`   | `s3a://cs611-project/bronze/legal_docs_raw`   |
| Silver | `include/silver/sample_silver.py`   | `s3a://cs611-project/silver/legal_docs_processed` |
| Gold   | `include/gold/ngram_tfidf_gold.py`    | `s3a://cs611-project/gold/features_ngram_tfidf`   |

> **Note:** The bronze → silver step is currently a pass-through copy. Most cleaning and feature engineering happens in the gold job described below.

## What silver provides (input)

**Source table:** `s3a://cs611-project/silver/legal_docs_processed`

**Format:** Delta Lake table on Cloudflare R2 (accessed by Spark via the S3-compatible `s3a://` protocol).

**Expected columns used by the gold job:**

| Column          | Purpose                                      |
|-----------------|----------------------------------------------|
| `CELEX`         | Unique document identifier                   |
| `act_raw_text`  | Raw legal document text                      |
| `labels`        | Topic / label for the document               |
| `tokens`        | Optional pre-tokenized text (if already present) |
| `Date_document` | Optional; normalized to `YYYY-MM-DD`         |
| `Date_publication` | Optional; normalized to `YYYY-MM-DD`      |

Silver may also contain bronze metadata columns and link columns that are dropped during gold processing.

## What the gold job does (overview)

The entry script `include/gold/ngram_tfidf_gold.py` is a thin orchestrator. It loads paths from `schema.yaml`, creates a Spark session, and calls three functions from `utils/ngram_tfidf.py`:

```text
silver Delta table
    │
    ▼  prepare_silver_data()     Step 1: clean + tokenize
    │
    ▼  build_gold_features()    Step 2: n-gram counts + TF-IDF
    │
    ▼  save_gold_artifacts()      Step 3: write to R2
    │
gold Delta table + JSON artifacts
```

---

## Step 1: `prepare_silver_data()` — clean and tokenize

**File:** `utils/ngram_tfidf.py`

Reads the silver DataFrame and applies the following transformations.

### Columns dropped

These columns are removed if present:

- `bronze_ingest_ts`, `bronze_source_key`
- `Cites_links`, `Ammends_links`, `Eurlex_link`, `ELI_link`, `Proposal_link`, `Oeil_link`

### String cleaning

- All string columns are trimmed.
- Repeated whitespace is collapsed to a single space.
- Empty strings are set to `null`.

### Row filtering

- Rows with missing or empty `labels` are removed.
- Duplicate documents (same `CELEX`) are deduplicated (first row kept).
- Rows with no tokens after preprocessing are removed.

### Date normalization

If present, `Date_document` and `Date_publication` are parsed and formatted as `YYYY-MM-DD`.

### Tokenization

Text is tokenized using `utils/text_preprocess.py` (`preprocess_tokens`), which:

1. Lowercases text and extracts alphanumeric tokens.
2. Removes English stopwords (NLTK).
3. Lemmatizes tokens (NLTK WordNet).
4. Drops single-character alphabetic tokens.
5. Drops pure numbers except 4-digit years (e.g. keeps `2019`, drops `5`).

Priority for token input:

1. `act_raw_text` — tokenized with NLTK preprocessing.
2. `tokens` — split on whitespace if raw text is empty.
3. Otherwise the row is dropped.

### New columns added

| Column             | Description                                      |
|--------------------|--------------------------------------------------|
| `text_source`      | Which column was used for tokenization           |
| `token_array`      | Array of preprocessed tokens                     |
| `tokens`           | Space-joined token string                        |
| `token_count`      | Number of tokens                                 |
| `silver_ingest_ts` | UTC timestamp of this preparation step           |
| `silver_source`    | Silver table path (lineage)                      |
| `doc_index`        | Monotonic row index for downstream alignment     |

---

## Step 2: `build_gold_features()` — n-gram counts and TF-IDF

**File:** `utils/ngram_tfidf.py`

Uses **Spark MLlib** on the `token_array` column.

### Configuration (defaults)

| Parameter      | Value   | Meaning                          |
|----------------|---------|----------------------------------|
| `MIN_N`        | `1`     | Minimum n-gram size (unigrams)   |
| `MAX_N`        | `3`     | Maximum n-gram size (trigrams)     |
| `MAX_FEATURES` | `50000` | Maximum vocabulary size          |
| `minDF`        | `2.0`   | Term must appear in ≥ 2 documents |

### Processing

1. **CountVectorizer** — builds a sparse n-gram count vector per document (`ngram_counts` column).
2. **IDF** — converts counts to TF-IDF weights (`tfidf` column).

### New columns added

| Column         | Type            | Description                    |
|----------------|-----------------|--------------------------------|
| `ngram_counts` | Spark ML Vector | Sparse n-gram frequency vector |
| `tfidf`        | Spark ML Vector | Sparse TF-IDF weight vector    |

A vocabulary list is also produced in memory (saved in Step 3 as JSON). Each vector index maps to an n-gram term in that vocabulary.

---

## Step 3: `save_gold_artifacts()` — write to R2

**File:** `utils/ngram_tfidf.py`

All output is written under:

```text
s3a://cs611-project/gold/features_ngram_tfidf/
```

### Outputs

| Artifact | Path | Format | Contents |
|----------|------|--------|----------|
| Main gold table | `features_ngram_tfidf/` | Delta Lake (Parquet + `_delta_log`) | Full feature table with all columns |
| Document metadata | `features_ngram_tfidf/gold_documents_1_3/` | Delta Lake | `doc_index`, `CELEX`, `labels`, `text_source` |
| Vocabulary | `features_ngram_tfidf/gold_vocab_1_3.json` | JSON | List of n-gram terms (index-aligned with vectors) |
| Run metadata | `features_ngram_tfidf/gold_meta_1_3.json` | JSON | Document count, feature count, n-gram range, preprocessing steps |

The `1_3` suffix reflects `MIN_N=1` and `MAX_N=3`.

### Main gold table columns

| Column             | Description                    |
|--------------------|--------------------------------|
| `doc_index`        | Row index                      |
| `CELEX`            | Document ID                    |
| `labels`           | Topic labels                   |
| `text_source`      | Source column used for tokens  |
| `tokens`           | Preprocessed token string      |
| `token_count`      | Number of tokens               |
| `ngram_counts`     | N-gram count vector            |
| `tfidf`            | TF-IDF vector                  |
| `silver_ingest_ts` | Preparation timestamp          |
| `silver_source`    | Upstream silver table path     |

Writes use `mode("overwrite")`, so each run replaces the previous gold output.

### How R2 storage works

Spark connects to Cloudflare R2 using S3-compatible credentials (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) configured in `utils/spark_session.py`.

- Delta tables are written via `df.write.format("delta").save(...)`.
- JSON files are written via the Hadoop S3A filesystem API.

---

## Code layout

```text
include/gold/
  ngram_tfidf_gold.py       # Entry point: logging, schema, Spark session, orchestration

utils/
  ngram_tfidf.py            # prepare_silver_data, build_gold_features, save_gold_artifacts
  text_preprocess.py        # NLTK tokenization used by the gold job
  spark_session.py          # Spark + Delta + R2 connection

schema.yaml                 # Table paths for silver and gold layers
```

---

## How to run

### Via Airflow (recommended)

1. Build the Docker image after code changes:

   ```bash
   docker compose build document-topic-tagger
   docker compose up
   ```

2. Open Airflow at `http://localhost:8080` and trigger the `medallion_pipeline` DAG.

3. The `process_gold` task runs the gold job after `process_silver` completes.

> Update `dags/pipeline.py` if the gold script path still points to the old `sample_gold.py` filename. The current gold script is `include/gold/ngram_tfidf_gold.py`.

### Manually (local / Docker)

```bash
python include/gold/ngram_tfidf_gold.py
```

Requires R2 credentials in the environment and an existing silver Delta table at the path defined in `schema.yaml`.

---

## Silver vs gold: summary of changes

| Aspect | Silver | Gold |
|--------|--------|------|
| Purpose | Cleaned document storage | ML-ready feature vectors |
| Text | Raw or lightly processed | Tokenized, stopword-removed, lemmatized |
| Structure | One row per document (flat columns) | Same rows + sparse vector columns |
| Features | None | N-gram counts + TF-IDF weights |
| Artifacts | Delta table only | Delta table + vocab JSON + metadata JSON |
| Link / bronze metadata columns | May be present | Dropped |
| Duplicate documents | May exist | Deduplicated by `CELEX` |
| Empty labels / empty text | May exist | Filtered out |

---

## Reading gold data back

```python
from utils.spark_session import create_spark_session

spark = create_spark_session("read-gold")

gold_df = spark.read.format("delta").load("s3a://cs611-project/gold/features_ngram_tfidf")
docs_df = spark.read.format("delta").load("s3a://cs611-project/gold/features_ngram_tfidf/gold_documents_1_3")

gold_df.select("CELEX", "labels", "token_count").show(5)
```

To map a vector index to an n-gram term, load `gold_vocab_1_3.json` from R2. The term at index `i` in the vocabulary corresponds to position `i` in the `ngram_counts` and `tfidf` vectors.
