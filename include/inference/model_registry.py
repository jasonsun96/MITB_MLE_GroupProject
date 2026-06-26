import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALIAS_NAMES = ("production", "shadow")
SCHEMA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXP_ID_RE = _RUN_ID_RE


def _clean_optional(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_registry_paths(schema_path: Path = PROJECT_ROOT / "schema.yaml") -> dict[str, str]:
    import yaml

    with schema_path.open() as schema_file:
        model_bank = yaml.safe_load(schema_file)["model_bank"]

    base = model_bank["path"].rstrip("/")
    runs_path = model_bank.get("runs", {}).get("path", "runs")
    features_path = model_bank.get("features", {}).get("base", "features")
    experiments_path = model_bank.get("experiments", {}).get("base", "experiments")
    aliases_path = model_bank.get("aliases", {}).get("path", "aliases")
    deployment_history_path = model_bank.get("deployment_history", {}).get("path", "deployment_history")
    return {
        "runs": f"{base}/{runs_path}",
        "features": f"{base}/{features_path}",
        "experiments": f"{base}/{experiments_path}",
        "aliases": f"{base}/{aliases_path}",
        "deployment_history": f"{base}/{deployment_history_path}",
    }


def _hadoop_path_and_fs(spark, path: str):
    jvm = spark.sparkContext._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    configuration = spark.sparkContext._jsc.hadoopConfiguration()
    return hadoop_path, hadoop_path.getFileSystem(configuration)


def hadoop_path_exists(spark, path: str) -> bool:
    hadoop_path, fs = _hadoop_path_and_fs(spark, path)
    return bool(fs.exists(hadoop_path))


def read_json(spark, path: str) -> dict:
    hadoop_path, fs = _hadoop_path_and_fs(spark, path)
    stream = fs.open(hadoop_path)
    try:
        data = bytes(spark.sparkContext._jvm.org.apache.commons.io.IOUtils.toByteArray(stream))
    finally:
        stream.close()
    return json.loads(data.decode("utf-8"))


def write_json(spark, path: str, payload: dict, *, overwrite: bool) -> None:
    hadoop_path, fs = _hadoop_path_and_fs(spark, path)
    stream = fs.create(hadoop_path, overwrite)
    try:
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        stream.write(bytearray(data))
    finally:
        stream.close()


def alias_path(paths: dict[str, str], alias: str) -> str:
    _validate_alias(alias)
    return f"{paths['aliases']}/{alias}.json"


def run_path(paths: dict[str, str], run_id: str) -> str:
    _validate_run_id(run_id)
    return f"{paths['runs']}/{run_id}"


def experiment_path(paths: dict[str, str], exp_id: str) -> str:
    _validate_exp_id(exp_id)
    return f"{paths['experiments']}/{exp_id}"


def get_alias(spark, paths: dict[str, str], alias: str) -> dict | None:
    path = alias_path(paths, alias)
    if not hadoop_path_exists(spark, path):
        return None
    return read_json(spark, path)


def write_deployment_history_event(spark, paths: dict[str, str], event: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    event_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
    payload = {
        "event_id": event_id,
        "occurred_at": now.isoformat(),
        "schema_version": SCHEMA_VERSION,
        **event,
    }
    write_json(spark, f"{paths['deployment_history']}/{event_id}.json", payload, overwrite=False)
    return payload


def _safe_label_name(label: str) -> str:
    return re.sub(r"[^\w.-]", "_", label)[:128]


def _model_metadata_path(model_path: str) -> str:
    return f"{model_path.rstrip('/')}/metadata"


def _current_per_label_model_path(exp_id: str, label: str, original_path: str) -> str:
    from run_paths import model_bank_per_label_models_dir

    current_base = model_bank_per_label_models_dir(exp_id).rstrip("/")
    original_leaf = original_path.rstrip("/").rsplit("/", 1)[-1]
    safe_leaf = _safe_label_name(label)
    return f"{current_base}/{original_leaf or safe_leaf}"


def _resolve_per_label_model_paths(
    spark,
    exp_id: str,
    manifest_path: str,
    per_label_paths: dict[str, str],
) -> dict[str, str]:
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
            resolved[label] = current_path
            continue

        missing.append((label, metadata_path))

    if missing:
        preview = "\n".join(f"- {label}: {path}" for label, path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Model manifest at {manifest_path} references missing Spark model metadata paths:\n"
            f"{preview}{suffix}"
        )

    return resolved


def validate_model_artifact(
    spark,
    paths: dict[str, str],
    *,
    exp_id: str,
    model_type: str,
    model_date: str | None = None,
    prediction_threshold: float | None = None,
) -> dict[str, Any]:
    from include.model_pipeline.multilabel_core import load_training_manifest, load_trained_models

    if prediction_threshold is not None and not 0 <= prediction_threshold <= 1:
        raise ValueError("prediction_threshold must be between 0 and 1")

    manifest, manifest_path = load_training_manifest(
        spark,
        exp_id,
        model_date,
        model_type=model_type,
    )
    per_label_paths = manifest.get("per_label_model_paths")
    if not isinstance(per_label_paths, dict) or not per_label_paths:
        raise ValueError(f"Model manifest at {manifest_path} missing per_label_model_paths")

    resolved_paths = _resolve_per_label_model_paths(spark, exp_id, manifest_path, per_label_paths)
    load_trained_models(resolved_paths, model_type)
    feature_run_id = str(manifest.get("feature_run_id") or "").strip()
    if not feature_run_id:
        raise ValueError(f"Model manifest at {manifest_path} missing feature_run_id")
    feature_path = f"{paths['features']}/{feature_run_id}"
    if not hadoop_path_exists(spark, feature_path):
        raise FileNotFoundError(f"Feature run does not exist: {feature_path}")
    return {
        "exp_id": exp_id,
        "feature_run_id": feature_run_id,
        "feature_path": feature_path,
        "model_type": model_type,
        "model_date": model_date,
        "model_manifest_path": manifest_path,
        "labels": sorted(resolved_paths),
        "label_count": len(resolved_paths),
    }


def set_alias(
    spark,
    paths: dict[str, str],
    alias: str,
    run_id: str | None = None,
    *,
    exp_id: str | None = None,
    model_type: str | None = None,
    model_date: str | None = None,
    prediction_threshold: float | None = None,
    actor: str | None = None,
    reason: str | None = None,
) -> dict:
    _validate_alias(alias)
    if run_id:
        raise ValueError("--run-id is no longer supported for experiment aliases; use --exp-id")
    if prediction_threshold is not None and not 0 <= prediction_threshold <= 1:
        raise ValueError("prediction_threshold must be between 0 and 1")
    if not exp_id:
        raise ValueError("exp_id is required")
    if not model_type:
        raise ValueError("model_type is required")
    model_path = experiment_path(paths, exp_id)
    if not hadoop_path_exists(spark, model_path):
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    validation = validate_model_artifact(
        spark,
        paths,
        exp_id=exp_id,
        model_type=model_type,
        model_date=model_date,
        prediction_threshold=prediction_threshold,
    )

    previous = get_alias(spark, paths, alias)
    now = datetime.now(timezone.utc)
    model_alias = {
        "alias": alias,
        "updated_at": now.isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    if exp_id:
        model_alias["exp_id"] = exp_id
    if model_type:
        model_alias["model_type"] = model_type
    if model_date:
        model_alias["model_date"] = model_date
    if prediction_threshold is not None:
        model_alias["prediction_threshold"] = prediction_threshold
    if validation is not None:
        model_alias["feature_run_id"] = validation.get("feature_run_id")
        model_alias["validation"] = {
            "model_manifest_path": validation["model_manifest_path"],
            "feature_run_id": validation.get("feature_run_id"),
            "feature_path": validation.get("feature_path"),
            "label_count": validation["label_count"],
            "validated_at": now.isoformat(),
        }
    write_json(spark, alias_path(paths, alias), model_alias, overwrite=True)

    write_deployment_history_event(spark, paths, {
        "event_type": "alias_updated",
        "alias": alias,
        "previous_exp_id": previous.get("exp_id") if previous else None,
        "exp_id": exp_id,
        "feature_run_id": validation.get("feature_run_id"),
        "model_type": model_type,
        "model_date": model_date,
        "prediction_threshold": prediction_threshold,
        "validation": model_alias.get("validation"),
        "actor": actor,
        "reason": reason,
    })
    return model_alias


def _validate_alias(alias: str) -> None:
    if alias not in ALIAS_NAMES:
        raise ValueError(f"Alias must be one of {ALIAS_NAMES}; got {alias!r}")


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must start with an alphanumeric character and contain only " "letters, digits, '.', '_' or '-' (maximum 128 characters)")


def _validate_exp_id(exp_id: str) -> None:
    if not _EXP_ID_RE.fullmatch(exp_id):
        raise ValueError("exp_id must start with an alphanumeric character and contain only letters, digits, '.', '_' or '-' (maximum 128 characters)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage model deployment aliases in R2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="Print an alias as JSON")
    get_parser.add_argument("--alias", choices=ALIAS_NAMES, required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a model candidate without updating an alias")
    validate_parser.add_argument("--exp-id", required=True)
    validate_parser.add_argument("--model-type", choices=("logistic_regression", "random_forest"), required=True)
    validate_parser.add_argument("--model-date", help="Training manifest date, e.g. 20260615")
    validate_parser.add_argument("--prediction-threshold", type=float)

    set_parser = subparsers.add_parser("set", help="Point an alias at an immutable model run")
    set_parser.add_argument("--alias", choices=ALIAS_NAMES, required=True)
    set_parser.add_argument("--run-id", help=argparse.SUPPRESS)
    set_parser.add_argument("--exp-id", help="Experiment id under model_bank/experiments/{exp_id}")
    set_parser.add_argument("--model-type", choices=("logistic_regression", "random_forest"))
    set_parser.add_argument("--model-date", help="Training manifest date, e.g. 20260615")
    set_parser.add_argument("--prediction-threshold", type=float)
    set_parser.add_argument("--actor")
    set_parser.add_argument("--reason")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    from utils.spark_session import create_spark_session

    spark = create_spark_session("model-registry")
    paths = load_registry_paths()
    try:
        if args.command == "get":
            model_alias = get_alias(spark, paths, args.alias)
            if model_alias is None:
                raise SystemExit(f"Alias is not set: {args.alias}")
        elif args.command == "validate":
            model_alias = validate_model_artifact(
                spark,
                paths,
                exp_id=args.exp_id,
                model_type=args.model_type,
                model_date=args.model_date,
                prediction_threshold=args.prediction_threshold,
            )
        else:
            model_alias = set_alias(
                spark,
                paths,
                args.alias,
                args.run_id,
                exp_id=args.exp_id,
                model_type=args.model_type,
                model_date=args.model_date,
                prediction_threshold=args.prediction_threshold,
                actor=args.actor,
                reason=args.reason,
            )
        print(json.dumps(model_alias, indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
