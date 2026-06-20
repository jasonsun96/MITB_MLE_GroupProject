from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from include.inference.model_registry import (get_alias, hadoop_path_exists,
                                              load_registry_paths, read_json,
                                              run_path, write_json)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def load_paths() -> dict[str, str]:
    import yaml

    with (PROJECT_ROOT / "schema.yaml").open() as schema_file:
        schema = yaml.safe_load(schema_file)

    gold = schema["gold"]
    base = gold["path"].rstrip("/")
    tables = gold.get("tables") or {}
    inference_features = (
        tables.get("inference_features", {}).get("path")
        or f"{gold['runs']['base']}/{gold['runs'].get('default_gold_run_id', gold['runs']['default_run_id'])}/{gold['runs']['X_unlabelled']}"
    )
    batch_inference = tables.get("batch_inference", {}).get("path") or "batch_inference"
    published_predictions = tables.get("published_predictions", {}).get("path") or "published_predictions"
    return {
        "input": f"{base}/{inference_features}",
        "output": f"{base}/{batch_inference}",
        "published_predictions": f"{base}/{published_predictions}",
    }


def canary_document_count(total_documents: int, canary_percentage: float) -> int:
    if not 0 <= canary_percentage < 100:
        raise ValueError("canary_percentage must be between 0 and 100")
    if canary_percentage == 0:
        return 0
    if total_documents < 2:
        raise ValueError("Batch canary inference requires at least two documents")

    requested = round(total_documents * canary_percentage / 100)
    return min(max(requested, 1), total_documents - 1)


def get_delta_version(spark, path: str) -> int:
    from delta.tables import DeltaTable

    if not DeltaTable.isDeltaTable(spark, path):
        raise ValueError(f"Inference input is not a Delta table: {path}")
    return int(DeltaTable.forPath(spark, path).history(1).select("version").first()["version"])


def read_delta_version(spark, path: str, version: int | None = None):
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", version)
    return reader.load(path)


def load_or_create_manifest(spark, batch_id: str, canary_percentage: float, input_path: str | None) -> dict:
    paths = load_paths()
    manifest_path = f"{paths['output']}/{batch_id}/manifest.json"
    if hadoop_path_exists(spark, manifest_path):
        logger.info("Reusing manifest at %s", manifest_path)
        return read_json(spark, manifest_path)

    registry_paths = load_registry_paths()
    production = get_alias(spark, registry_paths, "production")
    candidate = get_alias(spark, registry_paths, "candidate")
    if not production:
        raise ValueError("Production alias must be set")
    if not 0 <= canary_percentage < 100:
        raise ValueError("canary_percentage must be between 0 and 100")

    candidate_run_id = candidate["run_id"] if candidate else None
    use_canary = canary_percentage > 0 and candidate_run_id is not None and candidate_run_id != production["run_id"]
    if canary_percentage > 0 and not use_canary:
        logger.info("No distinct candidate model is set; using production for the full batch")

    resolved_input_path = input_path or paths["input"]
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": resolved_input_path,
        "input_version": get_delta_version(spark, resolved_input_path),
        "predictions_path": f"{paths['output']}/{batch_id}/predictions",
        "validation_path": f"{paths['output']}/{batch_id}/validation.json",
        "published_predictions_path": paths["published_predictions"],
        "production_run_id": production["run_id"],
        "candidate_run_id": candidate_run_id if use_canary else None,
        "canary_percentage": canary_percentage if use_canary else 0,
    }
    write_json(spark, manifest_path, manifest, overwrite=False)
    return manifest


def assign_cohorts(df, canary_percentage: float):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    candidate_count = canary_document_count(df.count(), canary_percentage)
    if candidate_count == 0:
        return df.withColumn("deployment_group", F.lit("production"))

    stable_order = Window.orderBy(F.xxhash64("document_id"), F.col("document_id"))
    return (
        df.withColumn("_canary_rank", F.row_number().over(stable_order))
        .withColumn(
            "deployment_group",
            F.when(F.col("_canary_rank") <= candidate_count, "candidate").otherwise("production"),
        )
        .drop("_canary_rank")
    )


def validate_input(df) -> int:
    from pyspark.sql import functions as F

    required = {"document_id", "features"}
    if missing := required - set(df.columns):
        raise ValueError(f"Inference input is missing columns: {sorted(missing)}")
    if df.filter(F.col("document_id").isNull()).limit(1).count():
        raise ValueError("Inference input contains null document_id values")

    input_count = df.count()
    if input_count == 0:
        raise ValueError("Inference input is empty")
    if df.select("document_id").distinct().count() != input_count:
        raise ValueError("Inference input contains duplicate document_id values")
    return input_count


def validate_model_run(spark, registry_paths: dict[str, str], run_id: str) -> dict:
    from pyspark.ml.classification import RandomForestClassificationModel

    root = run_path(registry_paths, run_id)
    metadata_path = f"{root}/metadata.json"
    label_mapping_path = f"{root}/label_mapping.json"
    for path in (root, metadata_path, label_mapping_path):
        if not hadoop_path_exists(spark, path):
            raise FileNotFoundError(f"Required model artifact does not exist: {path}")

    metadata = read_json(spark, metadata_path)
    label_mapping = read_json(spark, label_mapping_path)
    if metadata.get("run_id") != run_id:
        raise ValueError(f"Model metadata run_id does not match manifest run: {run_id}")
    if not metadata.get("feature_set"):
        raise ValueError(f"Model run {run_id} has no feature_set in metadata")

    try:
        threshold = float(metadata["multilabel_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Model run {run_id} has an invalid multilabel_threshold") from exc
    if not 0 <= threshold <= 1:
        raise ValueError(f"Model run {run_id} threshold must be between 0 and 1")

    labels = label_mapping.get("labels")
    model_paths = label_mapping.get("model_paths")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Model run {run_id} must contain a nonempty labels list")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Model run {run_id} contains duplicate labels")
    if not isinstance(model_paths, dict):
        raise ValueError(f"Model run {run_id} has no model_paths mapping")
    if label_mapping.get("num_labels") != len(labels):
        raise ValueError(f"Model run {run_id} num_labels does not match labels")

    models = {}
    for label in labels:
        model_path = model_paths.get(label)
        if not model_path:
            raise ValueError(f"Model run {run_id} has no model path for label {label!r}")
        if not hadoop_path_exists(spark, model_path):
            raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
        try:
            models[label] = RandomForestClassificationModel.load(model_path)
        except Exception as exc:
            raise ValueError(f"Could not load model for label {label!r} from {model_path}") from exc

    return {
        "run_id": run_id,
        "feature_set": metadata["feature_set"],
        "threshold": threshold,
        "labels": labels,
        "models": models,
    }


def validate_model_contracts(spark, manifest: dict, registry_paths: dict[str, str]) -> dict:
    contracts = {"production": validate_model_run(spark, registry_paths, manifest["production_run_id"])}
    if manifest.get("candidate_run_id"):
        contracts["candidate"] = validate_model_run(spark, registry_paths, manifest["candidate_run_id"])
        production_features = contracts["production"]["feature_set"]
        candidate_features = contracts["candidate"]["feature_set"]
        if candidate_features != production_features:
            raise ValueError("Production and candidate models require different feature sets: " f"{production_features!r} != {candidate_features!r}")
    return contracts


def score_cohort(df, contract: dict):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType

    @udf(DoubleType())
    def positive_probability(probability):
        values = probability.toArray()
        return float(values[1]) if len(values) > 1 else float(values[0])

    scored = df
    probability_columns = []
    for index, label in enumerate(contract["labels"]):
        column = f"_probability_{index}"
        model = contract["models"][label]
        probabilities = model.transform(scored).select(
            "document_id",
            positive_probability("probability").alias(column),
        )
        scored = scored.join(probabilities, on="document_id")
        probability_columns.append(column)

    predictions = [F.when(F.col(column) >= contract["threshold"], label) for label, column in zip(contract["labels"], probability_columns)]
    predictions = scored.withColumn(
        "predicted_labels",
        F.array_compact(F.array(*predictions)),
    ).drop(*probability_columns)
    return predictions


def validate_predictions(source, predictions, manifest: dict, input_count: int, allowed_labels: dict[str, set[str]]) -> dict:
    from pyspark.sql import functions as F

    required = {
        "batch_id",
        "document_id",
        "model_run_id",
        "deployment_group",
        "predicted_labels",
        "prediction_timestamp",
    }
    if missing := required - set(predictions.columns):
        raise ValueError(f"Prediction output is missing columns: {sorted(missing)}")

    output_types = dict(predictions.dtypes)
    expected_types = {
        "batch_id": "string",
        "document_id": "string",
        "model_run_id": "string",
        "deployment_group": "string",
        "predicted_labels": "array<string>",
        "prediction_timestamp": "timestamp",
    }
    type_errors = {column: {"expected": expected, "actual": output_types[column]} for column, expected in expected_types.items() if output_types[column] != expected}
    if type_errors:
        raise ValueError(f"Prediction output has invalid column types: {type_errors}")

    non_nullable = [
        "batch_id",
        "document_id",
        "model_run_id",
        "deployment_group",
        "predicted_labels",
        "prediction_timestamp",
    ]
    null_condition = F.lit(False)
    for column in non_nullable:
        null_condition = null_condition | F.col(column).isNull()
    if predictions.filter(null_condition).limit(1).count():
        raise ValueError("Prediction output contains null required values")
    if predictions.filter(F.col("batch_id") != manifest["batch_id"]).limit(1).count():
        raise ValueError("Prediction output contains an unexpected batch_id")

    prediction_count = predictions.count()
    unique_count = predictions.select("document_id").distinct().count()
    if prediction_count != input_count or unique_count != input_count:
        raise ValueError(f"Prediction coverage failed: input={input_count}, rows={prediction_count}, " f"unique_documents={unique_count}")
    if source.select("document_id").join(predictions.select("document_id"), "document_id", "left_anti").limit(1).count():
        raise ValueError("One or more input documents are missing predictions")
    if predictions.select("document_id").join(source.select("document_id"), "document_id", "left_anti").limit(1).count():
        raise ValueError("Prediction output contains unexpected documents")

    expected_runs = {
        "production": manifest["production_run_id"],
        "candidate": manifest["candidate_run_id"],
    }
    counts = {}
    for group, run_id in expected_runs.items():
        if run_id is None:
            counts[group] = 0
            continue
        group_df = predictions.filter(F.col("deployment_group") == group)
        counts[group] = group_df.count()
        if group_df.filter(F.col("model_run_id") != run_id).limit(1).count():
            raise ValueError(f"{group} predictions use an unexpected model run")

        valid_labels = sorted(allowed_labels[group])
        invalid_labels = group_df.select(F.explode_outer("predicted_labels").alias("label")).filter(F.col("label").isNotNull() & ~F.col("label").isin(valid_labels))
        if invalid_labels.limit(1).count():
            raise ValueError(f"{group} predictions contain labels outside the model mapping")

    if predictions.filter(~F.col("deployment_group").isin([group for group, run_id in expected_runs.items() if run_id])).limit(1).count():
        raise ValueError("Prediction output contains an unexpected deployment_group")

    expected_candidate = canary_document_count(input_count, manifest["canary_percentage"])
    if counts["candidate"] != expected_candidate:
        raise ValueError(f"Candidate count mismatch: expected={expected_candidate}, " f"actual={counts['candidate']}")
    if counts["production"] != input_count - expected_candidate:
        raise ValueError("Production count does not match the expected cohort size")

    return {
        "batch_id": manifest["batch_id"],
        "status": "passed",
        "input_count": input_count,
        "prediction_count": prediction_count,
        "production_count": counts["production"],
        "candidate_count": counts["candidate"],
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def publish_predictions(spark, predictions, published_path: str) -> None:
    from delta.tables import DeltaTable

    if not DeltaTable.isDeltaTable(spark, published_path):
        (predictions.write.format("delta").mode("errorifexists").partitionBy("batch_id").save(published_path))
        logger.info("Created published predictions table at %s", published_path)
        return

    target = DeltaTable.forPath(spark, published_path)
    (
        target.alias("target")
        .merge(
            predictions.alias("source"),
            "target.batch_id = source.batch_id " "AND target.document_id = source.document_id",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info("Merged batch predictions into %s", published_path)


def run_batch(spark, batch_id: str, canary_percentage: float, input_path: str | None = None) -> str:
    from pyspark.sql import functions as F

    manifest = load_or_create_manifest(spark, batch_id, canary_percentage, input_path)
    registry_paths = load_registry_paths()
    contracts = validate_model_contracts(spark, manifest, registry_paths)

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
        source = source.filter(F.col("batch_id") == manifest["batch_id"])
    input_count = validate_input(source)

    assigned = assign_cohorts(source, manifest["canary_percentage"])
    groups = [("production", manifest["production_run_id"])]
    if manifest["candidate_run_id"]:
        groups.append(("candidate", manifest["candidate_run_id"]))

    outputs = []
    allowed_labels = {}
    for group, run_id in groups:
        cohort = assigned.filter(F.col("deployment_group") == group)
        scored = score_cohort(cohort, contracts[group])
        allowed_labels[group] = set(contracts[group]["labels"])
        outputs.append(
            scored.select(
                F.lit(batch_id).alias("batch_id"),
                "document_id",
                F.lit(run_id).alias("model_run_id"),
                "deployment_group",
                "predicted_labels",
                F.current_timestamp().alias("prediction_timestamp"),
            )
        )

    predictions = outputs[0]
    for output in outputs[1:]:
        predictions = predictions.unionByName(output)

    validation = validate_predictions(
        source,
        predictions,
        manifest,
        input_count,
        allowed_labels,
    )
    predictions.write.format("delta").mode("overwrite").save(manifest["predictions_path"])
    published_path = manifest.get(
        "published_predictions_path",
        load_paths()["published_predictions"],
    )
    publish_predictions(spark, predictions, published_path)
    validation_path = manifest.get(
        "validation_path",
        f"{load_paths()['output']}/{batch_id}/validation.json",
    )
    write_json(spark, validation_path, validation, overwrite=True)
    logger.info("Wrote predictions to %s", manifest["predictions_path"])
    logger.info("Published predictions to %s", published_path)
    logger.info("Wrote validation result to %s", validation_path)
    return published_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch canary inference")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--canary-percentage", type=float, default=10.0)
    parser.add_argument("--input-path")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from utils.spark_session import create_spark_session

    spark = create_spark_session("batch-canary-inference")
    try:
        published_path = run_batch(
            spark,
            args.batch_id,
            args.canary_percentage,
            args.input_path,
        )
        print(json.dumps({"published_predictions_path": published_path}, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
