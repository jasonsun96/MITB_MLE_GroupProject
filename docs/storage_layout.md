# Storage layout v2

Top-level bucket layout. Paths are defined in `schema.yaml`.

```
landing/
bronze/
silver/

gold/
  embeddings/
  ngrams/
  pos_tags/
  label_store/
  runs/
    run001/          # tfidf_*, dcw_*, X_*
    run002/          # X_* only
    run003/          # X_* only
    run004/          # X_* only
  model_predictions/
    prediction_date=2026-06-15/
      exp004_LR_tfidf_dcw_gs/

model_bank/
  features/
    run001/ … run004/    # tfidf.pkl, dcw.pkl, tfidf/, dcw_score/
  experiments/
    exp001_RF_emb_tfidf_dcw/
    exp002_LR_emb_tfidf_dcw/
    exp003_LR_tfidf_dcw/
    exp004_LR_tfidf_dcw_gs/
      model/per_label/
      manifest/
      metrics/
      feature_importance/
```

## ID rules

| ID | `model_bank/features/` | `gold/runs/` |
|----|------------------------|--------------|
| `run001` | tfidf + dcw pickles | tfidf + dcw tables + X |
| `run002`–`run004` | pickles (copy until refit) | **X only** (assembled matrices) |

Experiments use:

- `--feature-run-id run001` — read upstream tfidf/dcw tables + pickles
- `--gold-run-id run003` — read/write `X_train` / `X_val_test_oot` for that experiment

## CLI

```powershell
docker compose run --rm document-topic-tagger python include/training/model_training.py `
  --exp-id exp004_LR_tfidf_dcw_gs `
  --feature-run-id run001 `
  --gold-run-id run004 `
  --feature-set tfidf_dcw `
  --model-type logistic_regression `
  --predict-only --predict-stage predict `
  --prediction-date 2026-06-15
```

## Migrate from v1 layout

```powershell
docker compose run --rm document-topic-tagger python scripts/migrate_storage_layout.py --dry-run
docker compose run --rm document-topic-tagger python scripts/migrate_storage_layout.py --execute
```

| Legacy | New |
|--------|-----|
| `model_bank/runs/run001/feature_extractors/` | `model_bank/features/run001/` |
| `model_bank/runs/run004/model/` | `model_bank/experiments/exp004_LR_tfidf_dcw_gs/model/` |
| `gold/model_predictions/prediction_20260612_RF` | `prediction_date=2026-06-12/exp001_RF_emb_tfidf_dcw/` |
| `gold/model_predictions/prediction_20260613_LR` | `prediction_date=2026-06-13/exp002_LR_emb_tfidf_dcw/` |

Gold `runs/run003` and `runs/run004` keep the same names (no feat* rename).

See `run_registry.yaml` for the full experiment map.
