from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "include" / "gold"
for _path in (PROJECT_ROOT, GOLD_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from include.inference.model_registry import (hadoop_path_exists, read_json,
                                              write_json)
from include.model_pipeline.multilabel_core import (load_trained_models,
                                                    load_training_manifest)
from run_paths import model_bank_per_label_models_dir

logger = logging.getLogger(__name__)
DEFAULT_FEATURE_CONFIG_PATH = PROJECT_ROOT / "config" / "batch_inference.yaml"


def _join_storage_path(base: str, path: str) -> str:
    if "://" in path:
        return path
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def load_feature_config(config_path: str | Path | None) -> dict:
    import yaml

    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Feature config not found: {path}")
    with path.open() as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Feature config must be a YAML mapping: {path}")
    return config


def load_paths(feature_config_path: str | Path | None = DEFAULT_FEATURE_CONFIG_PATH) -> dict[str, Any]:
    import yaml

    with (PROJECT_ROOT / "schema.yaml").open() as schema_file:
        schema = yaml.safe_load(schema_file)

    gold = schema["gold"]
    base = gold["path"].rstrip("/")
    tables = gold.get("tables") or {}
    feature_config = load_feature_config(feature_config_path)
    feature_paths = feature_config.get("features") or {}
    if not isinstance(feature_paths, dict):
        raise ValueError("Feature config field 'features' must be a mapping")

    configured_input = feature_paths.get("input_path")
    configured_inputs = feature_paths.get("input_paths") or {}
    if configured_inputs and not isinstance(configured_inputs, dict):
        raise ValueError("Feature config field 'features.input_paths' must be a mapping")
    configured_gold_run_id = feature_paths.get("gold_run_id")
    default_gold_run_id = gold["runs"].get("default_gold_run_id", gold["runs"]["default_run_id"])
    gold_run_id = str(configured_gold_run_id or default_gold_run_id).strip()
    inference_features = configured_input or tables.get("inference_features", {}).get("path") or f"{gold['runs']['base']}/{gold_run_id}/{gold['runs']['X_unlabelled']}"
    batch_inference = tables.get("batch_inference", {}).get("path") or "batch_inference"
    published_predictions = tables.get("published_predictions", {}).get("path") or "published_predictions"
    return {
        "input": _join_storage_path(base, str(inference_features)),
        "input_configured": bool(configured_input or tables.get("inference_features", {}).get("path")),
        "feature_input_paths": {
            key: _join_storage_path(base, str(value))
            for key, value in configured_inputs.items()
            if value
        },
        "output": _join_storage_path(base, str(batch_inference)),
        "published_predictions": _join_storage_path(base, str(published_predictions)),
    }


def _clean_optional(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_prediction_threshold(value, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"Feature config {field_name} must be between 0 and 1")
    return threshold


def _deployment_alias_from_config(config: dict, name: str) -> dict | None:
    deployment = config.get("deployment") or {}
    if not isinstance(deployment, dict):
        raise ValueError("Feature config field 'deployment' must be a mapping")

    entry = deployment.get(name)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise ValueError(f"Feature config field 'deployment.{name}' must be a mapping")

    if name == "shadow" and entry.get("enabled") is False:
        return None

    run_id = _clean_optional(entry.get("run_id")) or _clean_optional(entry.get("exp_id"))
    if not run_id:
        if name == "production" or entry.get("enabled") is True:
            raise ValueError(f"Feature config deployment.{name} must define run_id")
        return None
    exp_id = _clean_optional(entry.get("exp_id"))
    model_type = _clean_optional(entry.get("model_type"))
    if not exp_id:
        raise ValueError(f"Feature config deployment.{name} must define exp_id")
    if not model_type:
        raise ValueError(f"Feature config deployment.{name} must define model_type")

    return {
        "alias": name,
        "run_id": run_id,
        "exp_id": exp_id,
        "model_type": model_type,
        "model_date": _clean_optional(entry.get("model_date")),
        "prediction_threshold": _parse_prediction_threshold(
            entry.get("prediction_threshold"),
            f"deployment.{name}.prediction_threshold",
        ),
        "source": "github_config",
    }


def load_deployment_aliases(feature_config_path: str | Path | None) -> tuple[dict, dict | None]:
    config = load_feature_config(feature_config_path)
    production = _deployment_alias_from_config(config, "production")
    shadow = _deployment_alias_from_config(config, "shadow")
    if not production:
        raise ValueError("Feature config must define deployment.production.run_id")
    logger.info("Using Git-configured production model run: %s", production["run_id"])
    if shadow:
        logger.info("Using Git-configured shadow model run: %s", shadow["run_id"])
    return production, shadow


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


def _default_feature_input_path(paths: dict[str, str], batch_id: str, group: str) -> str:
    return f"{paths['output']}/{batch_id}/features/{group}"


def _resolve_feature_input_paths(
    paths: dict[str, str],
    batch_id: str,
    input_path: str | None,
    use_shadow: bool,
) -> dict[str, str]:
    configured = paths.get("feature_input_paths") or {}
    configured_production_input = paths["input"] if paths.get("input_configured") else None
    resolved = {
        "production": input_path or configured.get("production") or configured_production_input or _default_feature_input_path(paths, batch_id, "production"),
    }
    # When no legacy/custom input is set, default production to the deployment-specific
    # table emitted by assemble_inference_features.py.
    if not input_path and not configured.get("production") and not configured_production_input:
        resolved["production"] = _default_feature_input_path(paths, batch_id, "production")
    if use_shadow:
        if input_path and not configured.get("shadow"):
            raise ValueError("Shadow deployment requires features.input_paths.shadow when --input-path overrides production features")
        resolved["shadow"] = configured.get("shadow") or _default_feature_input_path(paths, batch_id, "shadow")
    return resolved


def load_or_create_manifest(
    spark,
    batch_id: str,
    input_path: str | None,
    feature_config_path: str | Path | None = DEFAULT_FEATURE_CONFIG_PATH,
) -> dict:
    paths = load_paths(feature_config_path)
    manifest_path = f"{paths['output']}/{batch_id}/manifest.json"
    if hadoop_path_exists(spark, manifest_path):
        logger.info("Reusing manifest at %s", manifest_path)
        manifest = read_json(spark, manifest_path)
        production, shadow = load_deployment_aliases(feature_config_path)
        if (
            manifest.get("production_model_config", {}).get("prediction_threshold")
            != production.get("prediction_threshold")
            or (manifest.get("shadow_model_config") or {}).get("prediction_threshold")
            != ((shadow or {}).get("prediction_threshold"))
        ):
            manifest["production_model_config"] = production
            if manifest.get("shadow_run_id"):
                manifest["shadow_model_config"] = shadow
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(spark, manifest_path, manifest, overwrite=True)
        feature_input_paths = manifest.get("feature_input_paths") or {"production": manifest["input_path"]}
        current_versions = {
            group: get_delta_version(spark, path)
            for group, path in feature_input_paths.items()
        }
        if current_versions != (manifest.get("input_versions") or {}):
            logger.info(
                "Feature input versions changed for batch %s; refreshing manifest input_versions",
                batch_id,
            )
            manifest["input_versions"] = current_versions
            manifest["input_version"] = current_versions.get("production")
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(spark, manifest_path, manifest, overwrite=True)
        return manifest

    production, shadow = load_deployment_aliases(feature_config_path)
    shadow_run_id = shadow["run_id"] if shadow else None
    use_shadow = shadow_run_id is not None and shadow_run_id != production["run_id"]
    if shadow_run_id and not use_shadow:
        logger.info("Shadow model matches production; scoring production only")

    feature_input_paths = _resolve_feature_input_paths(paths, batch_id, input_path, use_shadow)
    input_versions = {
        group: get_delta_version(spark, path)
        for group, path in feature_input_paths.items()
    }
    resolved_input_path = feature_input_paths["production"]
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": resolved_input_path,
        "input_version": input_versions["production"],
        "feature_input_paths": feature_input_paths,
        "input_versions": input_versions,
        "predictions_path": f"{paths['output']}/{batch_id}/predictions",
        "validation_path": f"{paths['output']}/{batch_id}/validation.json",
        "published_predictions_path": paths["published_predictions"],
        "production_run_id": production["run_id"],
        "shadow_run_id": shadow_run_id if use_shadow else None,
        "deployment_source": production["source"],
        "production_model_config": production,
        "shadow_model_config": shadow if use_shadow else None,
    }
    write_json(spark, manifest_path, manifest, overwrite=False)
    return manifest


def assign_deployment_group(df, group: str):
    from pyspark.sql import functions as F

    return df.withColumn("deployment_group", F.lit(group))


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


def load_group_input(spark, manifest: dict, group: str):
    from pyspark.sql import functions as F

    feature_input_paths = manifest.get("feature_input_paths") or {"production": manifest["input_path"]}
    input_versions = manifest.get("input_versions") or {"production": manifest.get("input_version")}
    if group not in feature_input_paths:
        raise ValueError(f"Batch manifest missing feature input path for {group}")
    source = read_delta_version(
        spark,
        feature_input_paths[group],
        input_versions.get(group),
    )
    if "batch_id" in source.columns:
        source = source.filter(F.col("batch_id") == manifest["batch_id"])
    return source


def validate_feature_run(source, contract: dict, group: str) -> None:
    from pyspark.sql import functions as F

    expected_feature_run_id = contract.get("feature_run_id")
    if expected_feature_run_id and "feature_run_id" in source.columns:
        unexpected_feature_runs = (
            source
            .select("feature_run_id")
            .distinct()
            .filter(F.col("feature_run_id").isNull() | (F.col("feature_run_id") != expected_feature_run_id))
        )
        if unexpected_feature_runs.limit(1).count():
            raise ValueError(
                f"{group} inference input was assembled with a different feature_run_id "
                f"than the model expects: {expected_feature_run_id!r}"
            )


def _safe_label_name(label: str) -> str:
    return re.sub(r"[^\w.-]", "_", label)[:128]


def _model_metadata_path(model_path: str) -> str:
    return f"{model_path.rstrip('/')}/metadata"


def _current_per_label_model_path(exp_id: str, label: str, original_path: str) -> str:
    current_base = model_bank_per_label_models_dir(exp_id).rstrip("/")
    original_leaf = original_path.rstrip("/").rsplit("/", 1)[-1]
    safe_leaf = _safe_label_name(label)
    return f"{current_base}/{original_leaf or safe_leaf}"


def _resolve_per_label_model_paths(spark, exp_id: str, manifest_path: str, per_label_paths: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[tuple[str, str]] = []

    for label, path in per_label_paths.items():
        metadata_path = _model_metadata_path(path)
        if hadoop_path_exists(spark, metadata_path):
            resolved[label] = path
            continue

        current_path = _current_per_label_model_path(exp_id, label, path)
        current_metadata_path = _model_metadata_path(current_path)
        if current_path != path and hadoop_path_exists(spark, current_metadata_path):
            logger.warning(
                "Model manifest at %s points label=%r to missing legacy path %s; using %s",
                manifest_path,
                label,
                path,
                current_path,
            )
            resolved[label] = current_path
            continue

        missing.append((label, metadata_path))

    if missing:
        preview = "\n".join(f"- {label}: {path}" for label, path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Model manifest at {manifest_path} references missing Spark model metadata paths:\n"
            f"{preview}{suffix}\n"
            "Retrain the experiment or update the manifest to point at existing per-label models."
        )

    return resolved


def _load_experiment_contract(spark, run_id: str, model_config: dict) -> dict:
    exp_id = _clean_optional(model_config.get("exp_id"))
    model_type = _clean_optional(model_config.get("model_type"))
    model_date = _clean_optional(model_config.get("model_date"))
    if not exp_id:
        raise ValueError(f"Model config for run {run_id} must define exp_id")
    if not model_type:
        raise ValueError(f"Model config for run {run_id} must define model_type")

    manifest, manifest_path = load_training_manifest(
        spark,
        exp_id,
        model_date,
        model_type=model_type,
    )
    per_label_paths = manifest.get("per_label_model_paths")
    if not isinstance(per_label_paths, dict) or not per_label_paths:
        raise ValueError(f"Model manifest at {manifest_path} missing per_label_model_paths")
    if not manifest.get("feature_set"):
        raise ValueError(f"Model manifest at {manifest_path} missing feature_set")

    threshold = model_config.get("prediction_threshold")
    threshold = float(threshold if threshold is not None else manifest.get("multilabel_threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError(f"Model config for run {run_id} has invalid prediction_threshold")

    labels = list(per_label_paths.keys())
    per_label_paths = _resolve_per_label_model_paths(spark, exp_id, manifest_path, per_label_paths)
    models = load_trained_models(per_label_paths, model_type)
    return {
        "run_id": run_id,
        "exp_id": exp_id,
        "model_type": model_type,
        "model_manifest_path": manifest_path,
        "feature_set": manifest["feature_set"],
        "feature_run_id": manifest.get("feature_run_id"),
        "threshold": threshold,
        "labels": labels,
        "models": models,
    }


def validate_model_run(spark, run_id: str, model_config: dict) -> dict:
    if not isinstance(model_config, dict):
        raise ValueError(f"Batch manifest for run {run_id} missing model config; recreate the batch manifest from config YAML")
    return _load_experiment_contract(spark, run_id, model_config)


def validate_model_contracts(spark, manifest: dict) -> dict:
    contracts = {
        "production": validate_model_run(
            spark,
            manifest["production_run_id"],
            manifest.get("production_model_config"),
        )
    }
    if manifest.get("shadow_run_id"):
        contracts["shadow"] = validate_model_run(
            spark,
            manifest["shadow_run_id"],
            manifest.get("shadow_model_config"),
        )
        production_labels = set(contracts["production"]["labels"])
        shadow_labels = set(contracts["shadow"]["labels"])
        if shadow_labels != production_labels:
            raise ValueError("Production and shadow models require the same label set")
    return contracts


def score_cohort(df, contract: dict):
    from pyspark.sql import functions as F
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType

    @udf(DoubleType())
    def positive_probability(probability):
        values = probability.toArray()
        return float(values[1]) if len(values) > 1 else float(values[0])

    features = df.select("document_id", "features")
    scored = df.select("document_id", "deployment_group")
    probability_columns = []
    for index, label in enumerate(contract["labels"]):
        column = f"_probability_{index}"
        model = contract["models"][label]
        probabilities = model.transform(features).select(
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
    expected_prediction_count = input_count * (2 if manifest.get("shadow_run_id") else 1)
    if prediction_count != expected_prediction_count or unique_count != input_count:
        raise ValueError(f"Prediction coverage failed: input={input_count}, rows={prediction_count}, " f"unique_documents={unique_count}")
    if source.select("document_id").join(predictions.select("document_id"), "document_id", "left_anti").limit(1).count():
        raise ValueError("One or more input documents are missing predictions")
    if predictions.select("document_id").join(source.select("document_id"), "document_id", "left_anti").limit(1).count():
        raise ValueError("Prediction output contains unexpected documents")

    expected_runs = {
        "production": manifest["production_run_id"],
        "shadow": manifest.get("shadow_run_id"),
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

    if counts["production"] != input_count:
        raise ValueError("Production count does not match the input size")
    expected_shadow = input_count if manifest.get("shadow_run_id") else 0
    if counts["shadow"] != expected_shadow:
        raise ValueError(f"Shadow count mismatch: expected={expected_shadow}, actual={counts['shadow']}")

    return {
        "batch_id": manifest["batch_id"],
        "status": "passed",
        "input_count": input_count,
        "prediction_count": prediction_count,
        "production_count": counts["production"],
        "shadow_count": counts["shadow"],
        "prediction_thresholds": manifest.get("prediction_thresholds", {}),
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def publish_predictions(spark, predictions, published_path: str) -> None:
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F

    production_predictions = predictions.filter(F.col("deployment_group") == "production")

    if not DeltaTable.isDeltaTable(spark, published_path):
        (production_predictions.write.format("delta").mode("errorifexists").partitionBy("batch_id").save(published_path))
        logger.info("Created published predictions table at %s", published_path)
        return

    target = DeltaTable.forPath(spark, published_path)
    (
        target.alias("target")
        .merge(
            production_predictions.alias("source"),
            "target.batch_id = source.batch_id " "AND target.document_id = source.document_id",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info("Merged batch predictions into %s", published_path)


def run_batch(
    spark,
    batch_id: str,
    input_path: str | None = None,
    feature_config_path: str | Path | None = DEFAULT_FEATURE_CONFIG_PATH,
) -> str:
    from pyspark.sql import functions as F

    manifest = load_or_create_manifest(spark, batch_id, input_path, feature_config_path)
    contracts = validate_model_contracts(spark, manifest)
    manifest["prediction_thresholds"] = {
        group: contract["threshold"]
        for group, contract in contracts.items()
    }

    if "feature_input_paths" not in manifest and "input_version" not in manifest:
        logger.warning(
            "Manifest for batch %s has no input_version; reading the latest Delta version",
            batch_id,
        )
    groups = [("production", manifest["production_run_id"])]
    if manifest.get("shadow_run_id"):
        groups.append(("shadow", manifest["shadow_run_id"]))

    sources = {}
    counts = {}
    for group, _ in groups:
        source = load_group_input(spark, manifest, group)
        counts[group] = validate_input(source)
        validate_feature_run(source, contracts[group], group)
        sources[group] = source

    input_count = counts["production"]
    production_ids = sources["production"].select("document_id")
    for group, _ in groups[1:]:
        if counts[group] != input_count:
            raise ValueError(f"{group} feature input row count does not match production")
        if production_ids.join(sources[group].select("document_id"), "document_id", "left_anti").limit(1).count():
            raise ValueError(f"{group} feature input is missing production document IDs")
        if sources[group].select("document_id").join(production_ids, "document_id", "left_anti").limit(1).count():
            raise ValueError(f"{group} feature input contains documents absent from production")

    outputs = []
    allowed_labels = {}
    for group, run_id in groups:
        cohort = assign_deployment_group(sources[group], group)
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

    predictions.write.format("delta").mode("overwrite").save(manifest["predictions_path"])
    predictions = spark.read.format("delta").load(manifest["predictions_path"])

    validation = validate_predictions(
        sources["production"],
        predictions,
        manifest,
        input_count,
        allowed_labels,
    )
    published_path = manifest.get(
        "published_predictions_path",
        load_paths(feature_config_path)["published_predictions"],
    )
    publish_predictions(spark, predictions, published_path)
    validation_path = manifest.get(
        "validation_path",
        f"{load_paths(feature_config_path)['output']}/{batch_id}/validation.json",
    )
    write_json(spark, validation_path, validation, overwrite=True)
    logger.info("Wrote predictions to %s", manifest["predictions_path"])
    logger.info("Published predictions to %s", published_path)
    logger.info("Wrote validation result to %s", validation_path)
    return published_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch shadow inference")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--feature-config", default=str(DEFAULT_FEATURE_CONFIG_PATH))
    parser.add_argument("--input-path")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from utils.spark_session import create_spark_session

    spark = create_spark_session("batch-shadow-inference")
    try:
        published_path = run_batch(
            spark,
            args.batch_id,
            args.input_path,
            args.feature_config,
        )
        print(json.dumps({"published_predictions_path": published_path}, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
