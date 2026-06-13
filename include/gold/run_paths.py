"""Central path helpers for versioned gold runs and model_bank layout."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gold_io import PROJECT_ROOT

_SCHEMA: dict[str, Any] | None = None


def load_schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        with open(PROJECT_ROOT / "schema.yaml") as f:
            _SCHEMA = yaml.safe_load(f)
    return _SCHEMA


def corpus_table_path(table_key: str) -> str:
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    rel = schema["gold"]["corpus"][table_key]["path"]
    return f"{gold_path}/{rel}"


def legacy_gold_table_path(table_key: str) -> str:
    """Pre-migration flat gold path (e.g. ngram_count, dcw_features_train)."""
    schema = load_schema()
    legacy = schema["gold"].get("legacy", {})
    if table_key not in legacy:
        raise KeyError(f"Unknown legacy gold table: {table_key}")
    return f"{schema['gold']['path']}/{legacy[table_key]}"


def default_feature_run_id() -> str:
    return load_schema()["gold"]["runs"].get("default_run_id", "run001")


def gold_run_table_path(run_id: str, table_key: str) -> str:
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    runs = schema["gold"]["runs"]
    rel = runs[table_key]
    return f"{gold_path}/{runs['base']}/{run_id}/{rel}"


def gold_run_root(run_id: str) -> str:
    schema = load_schema()
    return f"{schema['gold']['path']}/{schema['gold']['runs']['base']}/{run_id}"


def model_bank_run_root(run_id: str) -> str:
    schema = load_schema()
    mb = schema["model_bank"]
    return f"{mb['path']}/{mb['runs']['base']}/{run_id}"


def model_bank_feature_extractor_dir(run_id: str) -> str:
    schema = load_schema()
    fe = schema["model_bank"]["runs"]["feature_extractors"]
    return f"{model_bank_run_root(run_id)}/{fe}"


def model_bank_feature_extractor_path(run_id: str, filename: str) -> str:
    return f"{model_bank_feature_extractor_dir(run_id)}/{filename}"


def model_bank_tfidf_pkl_path(run_id: str) -> str:
    return model_bank_feature_extractor_path(run_id, load_schema()["model_bank"]["runs"]["tfidf_pkl"])


def model_bank_tfidf_svd_pkl_path(run_id: str) -> str:
    return model_bank_feature_extractor_path(run_id, load_schema()["model_bank"]["runs"]["tfidf_svd_pkl"])


def model_bank_tfidf_svd_model_dir(run_id: str) -> str:
    return f"{model_bank_feature_extractor_dir(run_id)}/tfidf_svd_model"


def model_bank_dcw_pkl_path(run_id: str) -> str:
    return model_bank_feature_extractor_path(run_id, load_schema()["model_bank"]["runs"]["dcw_pkl"])


def model_bank_tfidf_json_dir(run_id: str) -> str:
    return f"{model_bank_feature_extractor_dir(run_id)}/tfidf"


def model_bank_dcw_score_path(run_id: str) -> str:
    return f"{model_bank_feature_extractor_dir(run_id)}/dcw_score"


def model_bank_dcw_train_doc_ids_path(run_id: str) -> str:
    return f"{model_bank_feature_extractor_dir(run_id)}/dcw_train_doc_ids"


def model_bank_model_manifest_path(run_id: str, model_name: str, train_date: str | None = None) -> str:
    schema = load_schema()
    model_dir = schema["model_bank"]["runs"]["model_dir"]
    date = train_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{model_bank_run_root(run_id)}/{model_dir}/{model_name}_{date}.pkl"


def model_bank_per_label_models_dir(run_id: str) -> str:
    schema = load_schema()
    model_dir = schema["model_bank"]["runs"]["model_dir"]
    return f"{model_bank_run_root(run_id)}/{model_dir}/per_label"


def normalize_prediction_suffix(suffix: str | None) -> str:
    """Return a path-safe suffix such as '_LR' (empty string when suffix is None/blank)."""
    if not suffix:
        return ""
    cleaned = suffix.strip().lstrip("_")
    return f"_{cleaned}" if cleaned else ""


def prediction_batch_name(batch_date: str | None = None, suffix: str | None = None) -> str:
    schema = load_schema()
    mp = schema["gold"]["model_predictions"]
    date = batch_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{mp['file_prefix']}{date}{normalize_prediction_suffix(suffix)}"


def prediction_manifest_path(batch_date: str | None = None, suffix: str | None = None) -> str:
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    mp = schema["gold"]["model_predictions"]
    return f"{gold_path}/{mp['base']}/{prediction_batch_name(batch_date, suffix)}.pkl"


def prediction_delta_path(batch_date: str | None = None, suffix: str | None = None) -> str:
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    mp = schema["gold"]["model_predictions"]
    return f"{gold_path}/{mp['base']}/{prediction_batch_name(batch_date, suffix)}"


def prediction_manifest_path_for_batch(batch_name: str) -> str:
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    mp = schema["gold"]["model_predictions"]
    return f"{gold_path}/{mp['base']}/{batch_name}.pkl"


def resolve_feature_run_paths(run_id: str, dcw_run_id: str | None = None) -> dict[str, str]:
    """Paths for one gold/model_bank feature run (TF-IDF, DCW, X matrices)."""
    schema = load_schema()
    mb = schema["model_bank"]["runs"]
    dcw_id = dcw_run_id or run_id
    return {
        "run_id": run_id,
        "dcw_run_id": dcw_id,
        "gold_run_root": gold_run_root(run_id),
        "ngrams": corpus_table_path("ngrams"),
        "labels": corpus_table_path("labels"),
        "label_store": corpus_table_path("label_store"),
        "embeddings": corpus_table_path("embeddings"),
        "pos_tags": corpus_table_path("pos_tags"),
        "tfidf_train": gold_run_table_path(run_id, "tfidf_train"),
        "tfidf_val_test_oot": gold_run_table_path(run_id, "tfidf_val_test_oot"),
        "tfidf_svd_train": gold_run_table_path(run_id, "tfidf_svd_train"),
        "tfidf_svd_val_test_oot": gold_run_table_path(run_id, "tfidf_svd_val_test_oot"),
        "dcw_train": gold_run_table_path(dcw_id, "dcw_train"),
        "dcw_val_test_oot": gold_run_table_path(dcw_id, "dcw_val_test_oot"),
        "X_train": gold_run_table_path(run_id, "X_train"),
        "X_val_test_oot": gold_run_table_path(run_id, "X_val_test_oot"),
        "X_unlabelled": gold_run_table_path(run_id, "X_unlabelled"),
        "assembled_run_id": run_id,
        "tfidf_pkl": model_bank_tfidf_pkl_path(run_id),
        "tfidf_svd_pkl": model_bank_tfidf_svd_pkl_path(run_id),
        "tfidf_svd_model_dir": model_bank_tfidf_svd_model_dir(run_id),
        "tfidf_json_dir": model_bank_tfidf_json_dir(run_id),
        "dcw_pkl": model_bank_dcw_pkl_path(dcw_id),
        "dcw_score_path": model_bank_dcw_score_path(dcw_id),
        "dcw_train_doc_ids_path": model_bank_dcw_train_doc_ids_path(dcw_id),
        "model_bank_run_root": model_bank_run_root(run_id),
        "per_label_models_dir": model_bank_per_label_models_dir(run_id),
        "default_model_name": mb.get("default_model_name", "random_forest"),
    }
