# Silver to Gold: Feature Pipelines

This document describes what happens when data moves from the **silver** layer to the **gold** layer in the medallion pipeline. There are two gold feature jobs that share the same silver input and tokenization step:

| Gold table | Job script | Weighting | Output features |
|------------|------------|-----------|-----------------|
| `features_ngram_tfidf` | `include/gold/ngram_tfidf_gold.py` | Standard TF-IDF (`tf × idf`) | `ngram_counts`, `tfidf` (sparse, up to 50k dims) |
| `features_log_tfidf_svd` | `include/gold/log_tfidf_svd_gold.py` | Log-TF-IDF `log(tf) × (1 + log(idf))` + SVD | `log_tfidf` (sparse) + `svd_m50`, `svd_m100`, … (dense) |

Both tables write to R2 under `s3a://cs611-project/gold/` and keep `CELEX` / `doc_index` on every row so features link back to documents.

## Pipeline context

The Airflow DAG (`dags/pipeline.py`) runs three tasks in order:

```text
ingest_bronze  →  process_silver  →  process_gold
```

| Layer  | Job script | R2 path (legal docs) |
|--------|------------|----------------------|
| Bronze | `include/bronze/ingest_bronze.py` | `s3a://cs611-project/bronze/legal_docs_raw` |
| Silver | `include/silver/sample_silver.py` | `s3a://cs611-project/silver/legal_docs_processed` |
| Gold (baseline) | `include/gold/ngram_tfidf_gold.py` | `s3a://cs611-project/gold/features_ngram_tfidf` |
| Gold (log-TF-IDF + SVD) | `include/gold/log_tfidf_svd_gold.py` | `s3a://cs611-project/gold/features_log_tfidf_svd` |

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

## Shared gold pipeline (both jobs)

Both gold jobs start from the same silver table and reuse `prepare_silver_data()` from `utils/ngram_tfidf.py`:

```text
silver Delta table
    │
    ▼  prepare_silver_data()          Step 1: clean + tokenize (shared)
    │
    ├─► build_gold_features()         Pipeline A: standard TF-IDF
    │       └─ save_gold_artifacts()
    │
    └─► build_log_tfidf_svd_features()  Pipeline B: log-TF-IDF + SVD
            └─ save_log_tfidf_svd_artifacts()
```

---

## Pipeline A: standard n-gram TF-IDF

The entry script `include/gold/ngram_tfidf_gold.py` calls three functions from `utils/ngram_tfidf.py`:

```text
prepare_silver_data() → build_gold_features() → save_gold_artifacts()
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

Text is tokenized using `utils/text_preprocess.py` in **two steps**. The gold job chains both inside a Spark UDF (`_preprocess_tokens_for_gold` in `utils/ngram_tfidf.py`).

**Step A — `preprocess_tokens_base(text)`**

1. Lowercases text and extracts alphanumeric tokens (`raw_tokenize`).
2. Removes English stopwords (NLTK).
3. Lemmatizes tokens (NLTK WordNet).

**Step B — `filter_token_noise(tokens)`**

1. Drops single-character alphabetic tokens (e.g. `e`, `p`).
2. Drops pure numbers except 4-digit years (keeps `2019`, drops `5`, `276`).

`preprocess_tokens(text)` composes both steps: `filter_token_noise(preprocess_tokens_base(text))`.

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

## Pipeline A — Step 2: `build_gold_features()` — n-gram counts and TF-IDF

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

## Pipeline A — Step 3: `save_gold_artifacts()` — write to R2

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

## Pipeline B: log-TF-IDF + SVD

The entry script `include/gold/log_tfidf_svd_gold.py` loads paths from `schema.yaml`, creates a Spark session, and calls functions from `utils/log_tfidf_svd.py` (reusing `prepare_silver_data()` from `utils/ngram_tfidf.py`).

```text
prepare_silver_data()
    │
    ▼  CountVectorizer              ngram_counts (TF)
    ▼  IDF (fit for idf weights)    idf vector (not saved as final feature)
    ▼  custom log-TF-IDF UDF       log_tfidf
    ▼  TruncatedSVD (k = max m)     svd_full → sliced to svd_m50, svd_m100, …
    ▼  save_log_tfidf_svd_artifacts()
```

### Weighting formula

For each term \(i\) where \(tf_i > 0\) and \(idf_i > 0\):

```text
log_tfidf_i = log(tf_i) × (1 + log(idf_i))
```

- \(tf_i\) = raw n-gram count from `CountVectorizer`
- \(idf_i\) = Spark ML IDF weight from the fitted `IDF` model (`idf_model.idf`)
- `log` = natural logarithm (`math.log`)

This is **not** the same as `log(tf × idf)`. The IDF model is fit only to obtain per-term `idf` weights; the intermediate standard `tfidf_std` column is dropped before save.

### Configuration (defaults)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MIN_N` / `MAX_N` | `1` / `3` | Same n-gram range as Pipeline A |
| `MAX_FEATURES` | `50000` | Same vocabulary cap |
| `minDF` | `2.0` | Term must appear in ≥ 2 documents |
| `M_VALUES` | `[50, 100, 200, 500]` | SVD output dimensions to save |

SVD is fitted **once** with `k = max(M_VALUES)`, then each `svd_m{m}` column keeps the first `m` components.

### New columns added

| Column | Type | Description |
|--------|------|-------------|
| `log_tfidf` | Spark ML Vector (sparse) | Custom log-TF-IDF weights |
| `svd_m50` | Spark ML Vector (dense) | First 50 SVD components |
| `svd_m100` | Spark ML Vector (dense) | First 100 SVD components |
| `svd_m200` | Spark ML Vector (dense) | First 200 SVD components |
| `svd_m500` | Spark ML Vector (dense) | First 500 SVD components |

Intermediate columns (`ngram_counts`, `tfidf_std`, `svd_full`) are dropped before write.

### Outputs

All output is written under:

```text
s3a://cs611-project/gold/features_log_tfidf_svd/
```

| Artifact | Path | Format | Contents |
|----------|------|--------|----------|
| Main gold table | `features_log_tfidf_svd/` | Delta Lake | `log_tfidf` + `svd_m*` columns + document metadata |
| Document metadata | `features_log_tfidf_svd/gold_documents_1_3/` | Delta Lake | `doc_index`, `CELEX`, `labels`, `text_source` |
| Vocabulary | `features_log_tfidf_svd/gold_vocab_1_3.json` | JSON | N-gram terms (index-aligned with `log_tfidf`) |
| Run metadata | `features_log_tfidf_svd/gold_meta_1_3.json` | JSON | Formula, `m_values`, explained variance, training column names |

### Main gold table columns

| Column | Description |
|--------|-------------|
| `doc_index` | Row index |
| `CELEX` | Document ID |
| `labels` | Topic labels |
| `text_source` | Source column used for tokens |
| `tokens` | Preprocessed token string |
| `token_count` | Number of tokens |
| `log_tfidf` | Sparse log-TF-IDF vector |
| `svd_m50`, `svd_m100`, … | Dense SVD embeddings at each `m` |
| `silver_ingest_ts` | Preparation timestamp |
| `silver_source` | Upstream silver table path |

### How to run Pipeline B

```bash
docker compose build document-topic-tagger
docker compose run --rm document-topic-tagger python include/gold/log_tfidf_svd_gold.py
```

Custom SVD dimensions:

```bash
docker compose run --rm document-topic-tagger python include/gold/log_tfidf_svd_gold.py --m-values 50,100,300
```

---

## Using gold features for training

Both gold tables are **feature stores**, not trainers. A separate training step reads one table, picks a feature column as `X`, and uses `labels` as `y`.

| Experiment | Gold table | Feature column (`X`) | Typical use |
|------------|------------|----------------------|-------------|
| Baseline | `features_ngram_tfidf` | `tfidf` | Sparse linear models |
| Log-TF-IDF + SVD | `features_log_tfidf_svd` | `svd_m100` (or other `m`) | Dense, lower-dimensional models |

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Pipeline B example — compare different m values
df = spark.read.format("delta").load("s3a://cs611-project/gold/features_log_tfidf_svd")
train, test = df.randomSplit([0.8, 0.2], seed=42)

pipeline = Pipeline(stages=[
    StringIndexer(inputCol="labels", outputCol="label"),
    LogisticRegression(featuresCol="svd_m100", labelCol="label", maxIter=20),
])
model = pipeline.fit(train)
```

Use the same train/test split (`seed=42`) when comparing `tfidf` vs `svd_m50` vs `svd_m100` across tables. For strict ML evaluation, fit vectorizer / IDF / SVD on the training split only (see note below).

> **Data leakage note:** These gold jobs fit IDF and SVD on the full corpus before save. That is fine for feature caching and quick experiments, but for rigorous evaluation you should fit transformations on train data only.

---

## Code layout

```text
include/gold/
  ngram_tfidf_gold.py       # Pipeline A entry point
  log_tfidf_svd_gold.py     # Pipeline B entry point

utils/
  ngram_tfidf.py            # prepare_silver_data, build_gold_features, save_gold_artifacts
  log_tfidf_svd.py          # build_log_tfidf_svd_features, save_log_tfidf_svd_artifacts
  text_preprocess.py        # preprocess_tokens_base, filter_token_noise, preprocess_tokens
  spark_session.py          # Spark + Delta + R2 connection

notebooks/
  eda.ipynb                 # Bronze EDA (quality checks + token frequency analysis)

schema.yaml                 # Table paths for silver and gold layers
docs/ngram_tfidf.md         # This document
```

---

## Bronze EDA (`notebooks/eda.ipynb`)

Interactive exploratory analysis on bronze `legal_docs_raw` (and a preview of `wiki_docs_raw`). Run via Docker Jupyter:

```bash
docker compose run --rm -p 8888:8888 -v "${PWD}:/app" document-topic-tagger bash -lc \
  "pip install -q jupyter && jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir=/app/notebooks"
```

| Section | What it checks |
|---------|----------------|
| 1 | Missing `CELEX`, `act_raw_text`, or `labels` |
| 2 | Duplicate `CELEX` values |
| 3 | Label distribution |
| 4 | Document length (characters / words) |
| 5 | Top raw tokens (lowercase alphanumeric, no stopword removal) |
| 6a | Top tokens after `preprocess_tokens_base` (stopwords removed + lemmatized) |
| 6b | Top tokens after `filter_token_noise` (1-char letters and non-year digits removed) |

Sections 6a and 6b use inline Spark UDF copies of `utils/text_preprocess.py` (workers cannot import the local `utils` package). Section 6b matches the tokenization used by the gold job.

Token frequency cells sample 10% of non-empty documents to avoid Spark OOM on full-text tokenization.

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

Pipeline A (standard TF-IDF):

```bash
docker compose run --rm document-topic-tagger python include/gold/ngram_tfidf_gold.py
```

Pipeline B (log-TF-IDF + SVD):

```bash
docker compose run --rm document-topic-tagger python include/gold/log_tfidf_svd_gold.py
docker compose run --rm document-topic-tagger python include/gold/log_tfidf_svd_gold.py --m-values 50,100,300
```

Requires R2 credentials in the environment and an existing silver Delta table at the path defined in `schema.yaml`.

---

## Silver vs gold: summary of changes

| Aspect | Silver | Gold |
|--------|--------|------|
| Purpose | Cleaned document storage | ML-ready feature vectors |
| Text | Raw or lightly processed | Tokenized (stopword-removed, lemmatized, noise-filtered) |
| Structure | One row per document (flat columns) | Same rows + sparse vector columns |
| Features | None | TF-IDF and/or log-TF-IDF + SVD vectors |
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

To map a vector index to an n-gram term, load `gold_vocab_1_3.json` from R2. The term at index `i` in the vocabulary corresponds to position `i` in the sparse vectors (`ngram_counts`, `tfidf`, or `log_tfidf`).

Pipeline B example:

```python
svd_df = spark.read.format("delta").load("s3a://cs611-project/gold/features_log_tfidf_svd")
svd_df.select("CELEX", "labels", "svd_m100").show(5)
```

Check `gold_meta_1_3.json` for `m_values`, `weighting_formula`, and `explained_variance.cumulative_at_m` when choosing `m`.
