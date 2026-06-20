"""Central path helpers for gold runs, model_bank features, and experiments."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gold_io import PROJECT_ROOT

_SCHEMA: dict[str, Any] | None = None
_RUN_REGISTRY: dict[str, Any] | None = None


def load_schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        with open(PROJECT_ROOT / "schema.yaml") as f:
            _SCHEMA = yaml.safe_load(f)
    return _SCHEMA


def load_run_registry() -> dict[str, Any]:
    global _RUN_REGISTRY
    if _RUN_REGISTRY is None:
        registry_path = PROJECT_ROOT / "run_registry.yaml"
        if registry_path.exists():
            with open(registry_path) as f:
                _RUN_REGISTRY = yaml.safe_load(f)
        else:
            _RUN_REGISTRY = {}
    return _RUN_REGISTRY


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


def legacy_storage_root(key: str) -> str:
    schema = load_schema()
    legacy = schema.get("legacy_storage", {})
    if key not in legacy:
        raise KeyError(f"Unknown legacy_storage key: {key}")
    return legacy[key]


def default_feature_run_id() -> str:
    return load_schema()["gold"]["runs"].get("default_run_id", "run001")


def default_gold_run_id() -> str:
    return load_schema()["gold"]["runs"].get("default_gold_run_id", default_feature_run_id())


def gold_run_table_path(gold_run_id: str, table_key: str) -> str:
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    runs = schema["gold"]["runs"]
    rel = runs[table_key]
    return f"{gold_path}/{runs['base']}/{gold_run_id}/{rel}"


def gold_run_root(gold_run_id: str) -> str:
    schema = load_schema()
    return f"{schema['gold']['path']}/{schema['gold']['runs']['base']}/{gold_run_id}"


# ---------------------------------------------------------------------------
# model_bank / features  (frozen TF-IDF + DCW artefacts)
# ---------------------------------------------------------------------------


def model_bank_features_root(feature_run_id: str) -> str:
    schema = load_schema()
    mb = schema["model_bank"]
    return f"{mb['path']}/{mb['features']['base']}/{feature_run_id}"


def model_bank_feature_path(feature_run_id: str, *parts: str) -> str:
    base = model_bank_features_root(feature_run_id)
    return "/".join([base, *parts]) if parts else base


def model_bank_tfidf_pkl_path(feature_run_id: str) -> str:
    name = load_schema()["model_bank"]["features"]["tfidf_pkl"]
    return model_bank_feature_path(feature_run_id, name)


def model_bank_tfidf_svd_pkl_path(feature_run_id: str) -> str:
    name = load_schema()["model_bank"]["features"]["tfidf_svd_pkl"]
    return model_bank_feature_path(feature_run_id, name)


def model_bank_tfidf_svd_model_dir(feature_run_id: str) -> str:
    return model_bank_feature_path(feature_run_id, "tfidf_svd_model")


def model_bank_dcw_pkl_path(feature_run_id: str) -> str:
    name = load_schema()["model_bank"]["features"]["dcw_pkl"]
    return model_bank_feature_path(feature_run_id, name)


def model_bank_tfidf_json_dir(feature_run_id: str) -> str:
    name = load_schema()["model_bank"]["features"]["tfidf_json_dir"]
    return model_bank_feature_path(feature_run_id, name)


def model_bank_dcw_score_path(feature_run_id: str) -> str:
    name = load_schema()["model_bank"]["features"]["dcw_score"]
    return model_bank_feature_path(feature_run_id, name)


def model_bank_dcw_train_doc_ids_path(feature_run_id: str) -> str:
    name = load_schema()["model_bank"]["features"]["dcw_train_doc_ids"]
    return model_bank_feature_path(feature_run_id, name)


# ---------------------------------------------------------------------------
# model_bank / experiments  (trained classifiers + manifests)
# ---------------------------------------------------------------------------


def model_bank_experiment_root(exp_id: str) -> str:
    schema = load_schema()
    mb = schema["model_bank"]
    return f"{mb['path']}/{mb['experiments']['base']}/{exp_id}"


def model_bank_experiment_subdir(exp_id: str, subdir_key: str) -> str:
    schema = load_schema()
    rel = schema["model_bank"]["experiments"][subdir_key]
    return f"{model_bank_experiment_root(exp_id)}/{rel}"


def model_bank_per_label_models_dir(exp_id: str) -> str:
    schema = load_schema()
    per_label = schema["model_bank"]["experiments"]["per_label_subdir"]
    model_dir = schema["model_bank"]["experiments"]["model_dir"]
    return f"{model_bank_experiment_subdir(exp_id, 'model_dir')}/{per_label}"


def model_bank_model_manifest_path(
    exp_id: str,
    model_name: str,
    train_date: str | None = None,
) -> str:
    date = train_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    manifest_dir = model_bank_experiment_subdir(exp_id, "manifest_dir")
    return f"{manifest_dir}/{model_name}_{date}.pkl"


def model_bank_model_manifest_json_path(
    exp_id: str,
    model_name: str,
    train_date: str | None = None,
) -> str:
    return model_bank_model_manifest_path(exp_id, model_name, train_date).replace(".pkl", ".json")


def model_bank_feature_importance_path(exp_id: str, train_date: str | None = None) -> str:
    date = train_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    fi_dir = model_bank_experiment_subdir(exp_id, "feature_importance_dir")
    return f"{fi_dir}/feature_importance_{date}.json"


def model_bank_holdout_metrics_path(exp_id: str, batch_date: str | None = None) -> str:
    date = _normalize_prediction_date(batch_date).replace("-", "")
    metrics_dir = model_bank_experiment_subdir(exp_id, "metrics_dir")
    return f"{metrics_dir}/holdout_metrics_{date}.json"


def model_bank_prediction_metrics_manifest_path(exp_id: str, batch_date: str | None = None) -> str:
    date = _normalize_prediction_date(batch_date).replace("-", "")
    metrics_dir = model_bank_experiment_subdir(exp_id, "metrics_dir")
    return f"{metrics_dir}/prediction_{date}.pkl"


def model_bank_threshold_sweep_path(exp_id: str, batch_date: str | None = None) -> str:
    date = _normalize_prediction_date(batch_date).replace("-", "")
    metrics_dir = model_bank_experiment_subdir(exp_id, "metrics_dir")
    return f"{metrics_dir}/threshold_sweep_{date}.pkl"


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated v1 layout under model_bank/runs/)
# ---------------------------------------------------------------------------


def model_bank_run_root(run_id: str) -> str:
    """Deprecated: v1 path model_bank/runs/{id}. Use experiment or features root."""
    schema = load_schema()
    return f"{schema['model_bank']['path']}/runs/{run_id}"


def model_bank_feature_extractor_dir(feature_run_id: str) -> str:
    """Deprecated alias → model_bank_features_root."""
    return model_bank_features_root(feature_run_id)


# ---------------------------------------------------------------------------
# Predictions (Hive-style date partition + experiment id)
# ---------------------------------------------------------------------------


def _normalize_prediction_date(batch_date: str | None) -> str:
    if batch_date is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cleaned = batch_date.strip()
    if re.fullmatch(r"\d{8}", cleaned):
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    return cleaned


def prediction_batch_name(
    batch_date: str | None = None,
    exp_id: str | None = None,
    *,
    prediction_suffix: str | None = None,
) -> str:
    """Return experiment folder name under prediction_date= partition."""
    if exp_id:
        return exp_id
    if prediction_suffix:
        return prediction_suffix.strip().lstrip("_")
    schema = load_schema()
    default_exp = schema["gold"]["model_predictions"].get("default_exp_id")
    if default_exp:
        return default_exp
    raise ValueError("prediction path requires --exp-id or --prediction-suffix")


def normalize_prediction_suffix(suffix: str | None) -> str:
    """Backward compat: map old suffix style to exp_id folder name."""
    if not suffix:
        return ""
    return suffix.strip().lstrip("_")


def prediction_delta_path(
    batch_date: str | None = None,
    exp_id: str | None = None,
    *,
    prediction_suffix: str | None = None,
) -> str:
    schema = load_schema()
    gold_path = schema["gold"]["path"]
    mp = schema["gold"]["model_predictions"]
    date_part = _normalize_prediction_date(batch_date)
    exp_folder = prediction_batch_name(batch_date, exp_id, prediction_suffix=prediction_suffix)
    prefix = mp["date_partition_prefix"]
    return f"{gold_path}/{mp['base']}/{prefix}{date_part}/{exp_folder}"


def prediction_manifest_path(
    batch_date: str | None = None,
    exp_id: str | None = None,
    *,
    prediction_suffix: str | None = None,
) -> str:
    """Metrics sidecar for a prediction batch (stored under experiment metrics/)."""
    resolved_exp = exp_id or normalize_prediction_suffix(prediction_suffix)
    if not resolved_exp:
        raise ValueError("prediction manifest requires exp_id or prediction_suffix")
    return model_bank_prediction_metrics_manifest_path(resolved_exp, batch_date)


def prediction_manifest_path_for_batch(batch_name: str, batch_date: str | None = None) -> str:
    return model_bank_prediction_metrics_manifest_path(batch_name, batch_date)


def resolve_training_paths(
    feature_run_id: str,
    gold_run_id: str | None = None,
    dcw_run_id: str | None = None,
) -> dict[str, str]:
    """Resolve gold tables + feature artefacts for training/predict."""
    schema = load_schema()
    mb_features = schema["model_bank"]["features"]
    dcw_id = dcw_run_id or feature_run_id
    assembled_id = gold_run_id or feature_run_id
    return {
        "feature_run_id": feature_run_id,
        "gold_run_id": assembled_id,
        "dcw_run_id": dcw_id,
        "run_id": feature_run_id,  # backward compat
        "assembled_run_id": assembled_id,
        "gold_run_root": gold_run_root(assembled_id),
        "ngrams": corpus_table_path("ngrams"),
        "labels": corpus_table_path("labels"),
        "label_store": corpus_table_path("label_store"),
        "embeddings": corpus_table_path("embeddings"),
        "pos_tags": corpus_table_path("pos_tags"),
        "tfidf_train": gold_run_table_path(feature_run_id, "tfidf_train"),
        "tfidf_val_test_oot": gold_run_table_path(feature_run_id, "tfidf_val_test_oot"),
        "tfidf_svd_train": gold_run_table_path(feature_run_id, "tfidf_svd_train"),
        "tfidf_svd_val_test_oot": gold_run_table_path(feature_run_id, "tfidf_svd_val_test_oot"),
        "dcw_train": gold_run_table_path(dcw_id, "dcw_train"),
        "dcw_val_test_oot": gold_run_table_path(dcw_id, "dcw_val_test_oot"),
        "X_train": gold_run_table_path(assembled_id, "X_train"),
        "X_val_test_oot": gold_run_table_path(assembled_id, "X_val_test_oot"),
        "X_unlabelled": gold_run_table_path(assembled_id, "X_unlabelled"),
        "tfidf_pkl": model_bank_tfidf_pkl_path(feature_run_id),
        "tfidf_svd_pkl": model_bank_tfidf_pkl_path(feature_run_id),
        "tfidf_svd_model_dir": model_bank_tfidf_svd_model_dir(feature_run_id),
        "tfidf_json_dir": model_bank_tfidf_json_dir(feature_run_id),
        "dcw_pkl": model_bank_dcw_pkl_path(dcw_id),
        "dcw_score_path": model_bank_dcw_score_path(dcw_id),
        "dcw_train_doc_ids_path": model_bank_dcw_train_doc_ids_path(dcw_id),
        "features_root": model_bank_features_root(feature_run_id),
        "default_model_name": schema["model_bank"]["experiments"].get(
            "default_model_name", "random_forest"
        ),
    }


def resolve_feature_run_paths(run_id: str, dcw_run_id: str | None = None) -> dict[str, str]:
    """Backward-compatible wrapper (feature + gold run share the same id)."""
    paths = resolve_training_paths(run_id, gold_run_id=run_id, dcw_run_id=dcw_run_id)
    paths["model_bank_run_root"] = model_bank_features_root(run_id)
    paths["per_label_models_dir"] = None  # requires exp_id; set by caller
    return paths


def resolve_experiment_paths(exp_id: str) -> dict[str, str]:
    return {
        "exp_id": exp_id,
        "experiment_root": model_bank_experiment_root(exp_id),
        "per_label_models_dir": model_bank_per_label_models_dir(exp_id),
        "manifest_dir": model_bank_experiment_subdir(exp_id, "manifest_dir"),
        "metrics_dir": model_bank_experiment_subdir(exp_id, "metrics_dir"),
        "feature_importance_dir": model_bank_experiment_subdir(exp_id, "feature_importance_dir"),
    }
