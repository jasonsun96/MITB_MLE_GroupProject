from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from include.inference.batch_inference import (load_paths, publish_predictions,
                                               read_delta_version,
                                               score_cohort,
                                               validate_model_run)
from include.inference.model_registry import hadoop_path_exists, read_json, write_json

logger = logging.getLogger(__name__)


def load_rollback_context(spark, batch_id: str) -> tuple[dict, dict, str]:
    if not batch_id.strip():
        raise ValueError("batch_id is required")

    batch_root = f"{load_paths()['output']}/{batch_id}"
    manifest_path = f"{batch_root}/manifest.json"
    rollback_path = f"{batch_root}/rollback.json"

    if not hadoop_path_exists(spark, manifest_path):
        raise FileNotFoundError(f"Batch manifest does not exist: {manifest_path}")

    manifest = read_json(spark, manifest_path)
    validation_path = manifest.get("validation_path", f"{batch_root}/validation.json")
    if not hadoop_path_exists(spark, validation_path):
        raise FileNotFoundError(f"Batch validation does not exist: {validation_path}")

    validation = read_json(spark, validation_path)
    if validation.get("status") != "passed":
        raise ValueError("Batch validation must have status='passed' before rollback")
    if not manifest.get("candidate_run_id") or validation.get("candidate_count", 0) <= 0:
        raise ValueError(f"Batch {batch_id} did not use a candidate model")

    return manifest, validation, rollback_path


def validate_replacements(replacements, expected_count: int, production_run_id: str) -> None:
    from pyspark.sql import functions as F

    actual_count = replacements.count()
    unique_count = replacements.select("document_id").distinct().count()
    if actual_count != expected_count or unique_count != expected_count:
        raise ValueError(f"Rollback coverage failed: expected={expected_count}, rows={actual_count}, " f"unique_documents={unique_count}")
    if replacements.filter(F.col("model_run_id") != production_run_id).limit(1).count():
        raise ValueError("Rollback replacements use an unexpected model run")
    if replacements.filter(F.col("deployment_group") != "production").limit(1).count():
        raise ValueError("Rollback replacements must use deployment_group='production'")


def rollback_batch(spark, batch_id: str) -> dict:
    from pyspark.sql import functions as F

    manifest, validation, rollback_path = load_rollback_context(spark, batch_id)
    if hadoop_path_exists(spark, rollback_path):
        existing = read_json(spark, rollback_path)
        if existing.get("status") == "completed":
            logger.info("Rollback already completed for batch %s", batch_id)
            return existing

    staged_predictions = spark.read.format("delta").load(manifest["predictions_path"])
    candidate_ids = staged_predictions.filter(F.col("deployment_group") == "candidate").select("document_id")

    expected_count = int(validation["candidate_count"])
    if candidate_ids.distinct().count() != expected_count:
        raise ValueError("Staged candidate assignments do not match validation.json")

    if "input_version" not in manifest:
        logger.warning(
            "Manifest for batch %s has no input_version; reading the latest Delta version",
            batch_id,
        )
    source = read_delta_version(
        spark,
        manifest["input_path"],
        manifest.get("input_version"),
    )
    if "batch_id" in source.columns:
        source = source.filter(F.col("batch_id") == batch_id)
    rollback_input = source.join(candidate_ids, on="document_id", how="inner")
    production_run_id = manifest["production_run_id"]
    production_contract = validate_model_run(
        spark,
        production_run_id,
        manifest.get("production_model_config"),
    )
    scored = score_cohort(rollback_input, production_contract)
    replacements = scored.select(
        F.lit(batch_id).alias("batch_id"),
        "document_id",
        F.lit(production_run_id).alias("model_run_id"),
        F.lit("production").alias("deployment_group"),
        "predicted_labels",
        F.current_timestamp().alias("prediction_timestamp"),
    )
    validate_replacements(replacements, expected_count, production_run_id)

    published_path = manifest.get(
        "published_predictions_path",
        load_paths()["published_predictions"],
    )
    from delta.tables import DeltaTable

    if not DeltaTable.isDeltaTable(spark, published_path):
        raise FileNotFoundError(f"Published predictions Delta table does not exist: {published_path}")
    publish_predictions(spark, replacements, published_path)

    result = {
        "batch_id": batch_id,
        "status": "completed",
        "replaced_count": expected_count,
        "from_run_id": manifest["candidate_run_id"],
        "to_run_id": production_run_id,
        "published_predictions_path": published_path,
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(spark, rollback_path, result, overwrite=True)
    logger.info("Rolled back %s candidate predictions for batch %s", expected_count, batch_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll back candidate batch predictions")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from utils.spark_session import create_spark_session

    spark = create_spark_session("rollback-canary-inference")
    try:
        print(json.dumps(rollback_batch(spark, args.batch_id), indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
