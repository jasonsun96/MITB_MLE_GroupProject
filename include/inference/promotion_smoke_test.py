import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "include" / "gold"
for _path in (PROJECT_ROOT, GOLD_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from include.inference.model_registry import (
    _clean_optional,
    _resolve_per_label_model_paths,
    get_alias,
    load_registry_paths,
    validate_model_artifact,
    write_deployment_history_event,
)
from include.model_pipeline.multilabel_core import (
    DOCUMENT_ID_COL,
    FEATURES_COL,
    load_trained_models,
    load_training_manifest,
)
from utils.spark_session import create_spark_session

logger = logging.getLogger(__name__)


def smoke_test_alias(spark, alias: str, sample_size: int) -> dict:
    paths = load_registry_paths()
    registry_alias = get_alias(spark, paths, alias)
    if registry_alias is None:
        raise ValueError(f"Registry alias {alias!r} is not set")

    exp_id = _clean_optional(registry_alias.get("exp_id"))
    model_type = _clean_optional(registry_alias.get("model_type"))
    model_date = _clean_optional(registry_alias.get("model_date"))
    if not exp_id or not model_type:
        raise ValueError(f"Registry alias {alias!r} must define exp_id and model_type")

    validation = validate_model_artifact(
        spark,
        paths,
        exp_id=exp_id,
        model_type=model_type,
        model_date=model_date,
        prediction_threshold=registry_alias.get("prediction_threshold"),
    )
    manifest, manifest_path = load_training_manifest(
        spark,
        exp_id,
        model_date,
        model_type=model_type,
    )
    x_train_path = (manifest.get("assembled_dataset_paths") or {}).get("X_train")
    if not x_train_path:
        raise ValueError(f"Model manifest at {manifest_path} missing assembled_dataset_paths.X_train")

    sample = spark.read.format("delta").load(x_train_path).select(DOCUMENT_ID_COL, FEATURES_COL).limit(sample_size)
    row_count = sample.count()
    if row_count == 0:
        raise ValueError(f"No smoke-test rows found at {x_train_path}")

    per_label_paths = _resolve_per_label_model_paths(
        spark,
        exp_id,
        manifest_path,
        manifest["per_label_model_paths"],
    )
    models = load_trained_models(per_label_paths, model_type)
    for label, model in models.items():
        scored_count = model.transform(sample).select(DOCUMENT_ID_COL, "probability").count()
        if scored_count != row_count:
            raise ValueError(f"Smoke test row-count mismatch for label={label!r}: {scored_count} != {row_count}")

    result = {
        "alias": alias,
        "exp_id": exp_id,
        "model_type": model_type,
        "model_date": model_date,
        "model_manifest_path": validation["model_manifest_path"],
        "feature_run_id": validation["feature_run_id"],
        "feature_path": validation["feature_path"],
        "sample_path": x_train_path,
        "sample_rows": row_count,
        "labels_scored": len(models),
        "status": "passed",
        "smoke_tested_at": datetime.now(timezone.utc).isoformat(),
    }
    event = write_deployment_history_event(
        spark,
        paths,
        {
            "event_type": "promotion_smoke_test",
            "alias": alias,
            "exp_id": exp_id,
            "model_type": model_type,
            "model_date": model_date,
            "status": "passed",
            "sample_rows": row_count,
            "labels_scored": len(models),
            "model_manifest_path": validation["model_manifest_path"],
            "feature_run_id": validation["feature_run_id"],
            "feature_path": validation["feature_path"],
        },
    )
    result["history_event_id"] = event["event_id"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a model registry alias")
    parser.add_argument("--alias", default="production", choices=("production", "shadow"))
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    spark = create_spark_session("promotion-smoke-test")
    try:
        print(json.dumps(smoke_test_alias(spark, args.alias, args.sample_size), indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
