# Model training & prediction guide

End-to-end reference for multi-label topic classification using precomputed Gold features, Spark ML binary relevance, and run-versioned storage on R2.

---

## 1. High-level pipeline

```
Upstream (run separately)
  ngrams → TF-IDF (tfidf_processing.py) → gold/runs/{id}/tfidf_*
  ngrams → DCW (domain_concept_weight.py)  → gold/runs/{id}/dcw_*
  corpus → embeddings                      → gold/embeddings
  label_store                              → gold/label_store

Training (model_training.py)
  train-only  → fit per-label models + X_train
  predict     → score val/test/oot → predictions Delta + metrics

Evaluation
  predict_results.ipynb → metrics, threshold sweep from saved prob_* columns
```

**Important:** `model_training.py` does **not** refit TF-IDF, DCW, or embeddings. It only joins precomputed tables and trains/scores classifiers.

---

## 2. Run IDs (three different concepts)

| Flag | Example | Purpose |
|------|---------|---------|
| `--run-id` | `run003_tfidf_dcw` | **Model experiment** — where trained models and manifest are saved under `model_bank/runs/{run_id}/` |
| `--feature-run-id` | `run001` | **Upstream gold features** — read TF-IDF, DCW, pickles from this run |
| `--x-run-id` | `run003` | **Assembled matrices** — where `X_train` / `X_val_test_oot` are written and read |

### Typical pattern (reuse gold, isolate experiments)

```powershell
--run-id run003_tfidf_dcw      # models live here
--feature-run-id run001        # read tfidf/dcw from first gold run
--x-run-id run003              # assembled X for this feature-set experiment
```

If `--x-run-id` is omitted, assembled X uses the same path as `--feature-run-id`.

### What is **not** copied between runs

Reusing `--feature-run-id run001` does **not** duplicate TF-IDF/DCW tables. Only models (under `--run-id`) and optionally assembled X (under `--x-run-id`) are new.

---

## 3. R2 layout

### Shared corpus (not run-scoped)

```
s3a://cs611-project/gold/
  ngrams/
  embeddings/
  label_store/          # split in column `category`: train | val | test | oot
  pos_tags/
```

### Per feature run (e.g. `run001`)

```
gold/runs/run001/
  tfidf_train
  tfidf_val_test_oot
  dcw_train
  dcw_val_test_oot
  X_train                 # assembled features vector (optional location via --x-run-id)
  X_val_test_oot

model_bank/runs/run001/feature_extractors/
  tfidf.pkl
  dcw.pkl
  dcw_score               # DCW vocabulary (8k lemmas)
```

### Per model run (e.g. `run003_tfidf_dcw`)

```
model_bank/runs/run003_tfidf_dcw/model/
  random_forest_YYYYMMDD.pkl       # manifest (metadata + metrics)
  random_forest_YYYYMMDD.json      # human-readable manifest
  logistic_regression_YYYYMMDD.*   # if LR was trained
  per_label/
    {safe_label_name}/             # Spark ML model directory per label
```

### Predictions (date + optional suffix)

```
gold/model_predictions/
  prediction_20260612/             # Delta (no suffix)
  prediction_20260612_RF/           # RF batch with probabilities
  prediction_20260613_LR/           # LR batch
  prediction_20260614_tfidf_dcw/   # tfidf_dcw experiment
  prediction_20260614_tfidf_dcw.pkl  # metrics manifest (metrics stage)
  prediction_20260614_tfidf_dcw.json
```

Predictions are keyed by **date + suffix**, not by `--run-id`. Each Delta row includes `run_id` and `feature_run_id` columns for traceability.

---

## 4. Feature sets

| `--feature-set` | Contents |
|-----------------|----------|
| `tfidf_dcw_embeddings` | **Default** — TF-IDF + DCW + Legal-BERT embeddings |
| `tfidf_dcw` | TF-IDF + DCW only (no embeddings) |
| `log_tfidf_dcw` | Log-TF-IDF + DCW |
| `embeddings` | Embeddings only |
| `dcw` | DCW only |
| `all` | Log-TF-IDF + DCW + embeddings |

**Rule:** Train, features, and predict must use the **same** `--feature-set`. The assembled `features` column dimension and content must match what the model was trained on.

---

## 5. Model types & hyperparameters

### Binary relevance

One binary classifier per label (15 labels in current setup). Multi-label output is built by thresholding each label's positive-class probability.

### Random Forest (default)

| Param | Default | CLI override |
|-------|---------|--------------|
| `numTrees` | 50 | `--num-trees` |
| `maxDepth` | 10 | `--max-depth` |

### Logistic Regression

| Param | Default | CLI override |
|-------|---------|--------------|
| `maxIter` | 100 | `--max-iter` |
| `regParam` | 0.0 | `--reg-param` |
| `elasticNetParam` | 0.0 | `--elastic-net-param` |

```powershell
--model-type random_forest          # default
--model-type logistic_regression
```

**Critical:** `--model-type` at predict time must match the saved `per_label/` models. Mismatch causes:

```
Error loading metadata: Expected RandomForestClassificationModel but found LogisticRegressionModel
```

After rebuild, the code auto-detects model type from Spark metadata when CLI and disk disagree.

### Prediction threshold

| Param | Default | Notes |
|-------|---------|-------|
| `--multilabel-threshold` | 0.5 | Applied at predict time; label included if `prob >= threshold` |

Saved prediction Delta includes `prob_*` columns so you can sweep thresholds in the notebook without re-scoring.

---

## 6. Training flow

### 6.1 Train only (recommended first step)

Fits models on **train split only** (18,640 docs). Skips holdout scoring.

```powershell
docker compose build document-topic-tagger
$env:SPARK_MASTER = "local[2]"
$env:SPARK_DRIVER_MEMORY = "10g"

docker compose run --rm document-topic-tagger python include/training/model_training.py `
  --run-id run003_tfidf_dcw `
  --feature-run-id run001 `
  --x-run-id run003 `
  --feature-set tfidf_dcw `
  --train-only
```

**Writes:**

- `model_bank/runs/run003_tfidf_dcw/model/per_label/*`
- `model_bank/runs/run003_tfidf_dcw/model/random_forest_YYYYMMDD.pkl` + `.json`
- `gold/runs/run003/X_train` (if `--x-run-id run003`)

**Does not write:** predictions or holdout metrics.

### 6.2 Recover manifest (if `.pkl` save failed but models exist)

Same issue as early LR run — training completed but manifest upload failed:

```powershell
docker compose run --rm document-topic-tagger python include/training/model_training.py `
  --run-id run003_tfidf_dcw `
  --feature-run-id run001 `
  --feature-set tfidf_dcw `
  --model-type logistic_regression `
  --recover-manifest
```

---

## 7. Prediction flow (three stages)

Splitting stages avoids OOM and lets you rerun metrics without re-scoring.

| Stage | CLI | Needs trained models? | Output |
|-------|-----|----------------------|--------|
| `features` | `--predict-stage features` | **No** | `gold/runs/{x-run-id}/X_val_test_oot` |
| `predict` | `--predict-stage predict` | **Yes** | `gold/model_predictions/prediction_{date}_{suffix}/` |
| `metrics` | `--predict-stage metrics` | No (reads Delta) | `.pkl` + `.json` manifests |
| `all` | `--predict-stage all` | Yes | Everything in one job |

### 7.1 Assemble holdout features

Only needed once per `(feature-set, x-run-id)` combination, or when changing feature set.

```powershell
docker compose run --rm document-topic-tagger python include/training/model_training.py `
  --run-id run003_tfidf_dcw `
  --feature-run-id run001 `
  --x-run-id run003 `
  --feature-set tfidf_dcw `
  --predict-only --predict-stage features
```

- Reads **holdout** TF-IDF/DCW from `run001` (not train).
- Does **not** need model manifest (after latest code fix).
- **No date** required for this stage.

### 7.2 Score holdout

```powershell
docker compose run --rm document-topic-tagger python include/training/model_training.py `
  --run-id run003_tfidf_dcw `
  --feature-run-id run001 `
  --x-run-id run003 `
  --feature-set tfidf_dcw `
  --model-type random_forest `
  --predict-only --predict-stage predict `
  --prediction-date 20260614 `
  --prediction-suffix tfidf_dcw
```

- Reads saved `X_val_test_oot` from `run003`.
- Loads `per_label/` models from `run003_tfidf_dcw`.
- **Does not** read train features.

**Delta columns:** `document_id`, `category`, `target_labels`, `predicted_labels`, `prob_{label}`, `multilabel_threshold`, `run_id`, `feature_run_id`, `prediction_ts`

### 7.3 Compute metrics

```powershell
docker compose run --rm document-topic-tagger python include/training/model_training.py `
  --run-id run003_tfidf_dcw `
  --feature-run-id run001 `
  --feature-set tfidf_dcw `
  --predict-only --predict-stage metrics `
  --prediction-date 20260614 `
  --prediction-suffix tfidf_dcw
```

Or compute in **`notebooks/predict_results.ipynb`** from the Delta (no metrics stage required).

---

## 8. Evaluation metrics

| Metric | Meaning |
|--------|---------|
| **micro_f1** | Overall label decision quality (weighted by label frequency) |
| **macro_f1** | Average F1 across labels (sensitive to rare labels) |
| **accuracy** | Fraction of documents where **all** labels match exactly (subset accuracy) |
| **micro_precision / micro_recall** | Precision/recall across all label decisions |
| **hamming_loss** | Fraction of wrong label slots per document |

Reported per split: `holdout_val`, `holdout_test`, `holdout_oot`, `holdout_overall`.

### Threshold sweep (notebook)

Cell 3 in `predict_results.ipynb` tries thresholds 0.30–0.60 using saved `prob_*` columns — no Docker re-run needed.

```python
EVAL_DATE = "20260614"
EVAL_SUFFIX = "tfidf_dcw"
```

---

## 9. Docker notes

### One-off jobs (training / predict)

```powershell
docker compose build document-topic-tagger   # after code changes
docker compose run --rm document-topic-tagger python include/training/model_training.py ...
```

- `run --rm` removes the container when done.
- **No** `docker compose up` or `down` required for these jobs.
- Rebuild after code changes (image has no live volume mount for `include/`).

### Long-running services

```powershell
docker compose up -d jupyter    # notebooks at localhost:8888
docker compose down             # stop jupyter/airflow when done
```

### Suggested Spark env (PowerShell)

```powershell
$env:SPARK_MASTER = "local[2]"
$env:SPARK_DRIVER_MEMORY = "10g"
```

---

## 10. Common pitfalls

### Wrong feature set on holdout X

If `X_val_test_oot` was built with `tfidf_dcw_embeddings` but you predict with `tfidf_dcw`, feature dimensions won't match. Re-run `--predict-stage features` with the correct `--feature-set`.

### Overwriting `X_val_test_oot` on run001

Without `--x-run-id`, assembled X writes to `gold/runs/run001/X_val_test_oot` and overwrites prior experiments. Use `--x-run-id run003` (or similar) to isolate.

### Overwriting predictions

Same `--prediction-date` + suffix **overwrites** that Delta folder. Use different dates or suffixes to keep RF vs LR vs tfidf_dcw batches:

```powershell
--prediction-suffix RF
--prediction-suffix LR
--prediction-suffix tfidf_dcw
```

### Model manifest missing

Training can succeed but `.pkl` manifest upload fails. Models in `per_label/` are still valid. Run `--recover-manifest` or rely on per_label fallback + auto model-type detection.

### Model type mismatch

Always pass `--model-type` matching training, or rely on auto-detection after rebuild.

### Features stage used to require manifest

Fixed: `--predict-stage features` no longer loads model manifest.

### `--feature-run-id` is not "train features for predict"

It only points to where **upstream** TF-IDF/DCW tables live. Predict reads **holdout** assembled X, not train.

---

## 11. Example experiment matrix

| Experiment | `--run-id` | `--feature-set` | `--model-type` | `--prediction-suffix` |
|------------|------------|-----------------|----------------|------------------------|
| RF full features | `run001` | `tfidf_dcw_embeddings` | `random_forest` | `RF` |
| LR full features | `run002_lr` | `tfidf_dcw_embeddings` | `logistic_regression` | `LR` |
| RF tfidf+dcw only | `run003_tfidf_dcw` | `tfidf_dcw` | `random_forest` | `tfidf_dcw` |

Shared gold features: `--feature-run-id run001` for all unless you rebuild TF-IDF/DCW under a new run.

---

## 12. Quick command cheat sheet

```powershell
# Train
--run-id <model_run> --feature-run-id run001 --x-run-id <x_run> `
  --feature-set tfidf_dcw --train-only

# Recover manifest
--run-id <model_run> --feature-run-id run001 --recover-manifest

# Holdout features
--run-id <model_run> --feature-run-id run001 --x-run-id <x_run> `
  --feature-set tfidf_dcw --predict-only --predict-stage features

# Predict
--run-id <model_run> --feature-run-id run001 --x-run-id <x_run> `
  --feature-set tfidf_dcw --model-type random_forest `
  --predict-only --predict-stage predict `
  --prediction-suffix tfidf_dcw

# Metrics
--predict-only --predict-stage metrics --prediction-date YYYYMMDD --prediction-suffix tfidf_dcw
```

---

## 13. Related files

| File | Role |
|------|------|
| `include/training/model_training.py` | Train, predict, metrics CLI |
| `include/gold/tfidf_processing.py` | Fit TF-IDF on train, apply to holdout |
| `include/gold/domain_concept_weight.py` | DCW features |
| `include/gold/run_paths.py` | Path helpers |
| `include/gold/gold_io.py` | Delta / pickle I/O to R2 |
| `schema.yaml` | Bucket and table layout |
| `notebooks/predict_results.ipynb` | Load metrics, threshold sweep |
| `docs/ngram_tfidf.md` | Upstream TF-IDF feature job |

---

## 14. Data splits (reference)

| Split | Approx. docs | Purpose |
|-------|----------------|---------|
| train | 18,640 | Fit models |
| val | 5,338 | Tuning / early stopping (not used in default pipeline) |
| test | 2,668 | Evaluation |
| oot | 1,687 | Out-of-time generalization |
| **holdout total** | **9,693** | val + test + oot scored at predict time |

Labels: 15 practice-area labels (binary relevance). Default threshold 0.5.
