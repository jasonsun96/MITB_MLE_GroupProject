"""Manage model deployment aliases stored in R2 through Spark/Hadoop."""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALIAS_NAMES = ("production", "shadow")
SCHEMA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def load_registry_paths(schema_path: Path = PROJECT_ROOT / "schema.yaml") -> dict[str, str]:
    import yaml

    with schema_path.open() as schema_file:
        model_bank = yaml.safe_load(schema_file)["model_bank"]

    base = model_bank["path"].rstrip("/")
    runs_path = model_bank.get("runs", {}).get("path", "runs")
    aliases_path = model_bank.get("aliases", {}).get("path", "aliases")
    deployment_history_path = model_bank.get("deployment_history", {}).get("path", "deployment_history")
    return {
        "runs": f"{base}/{runs_path}",
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


def get_alias(spark, paths: dict[str, str], alias: str) -> dict | None:
    path = alias_path(paths, alias)
    if not hadoop_path_exists(spark, path):
        return None
    return read_json(spark, path)


def set_alias(
    spark,
    paths: dict[str, str],
    alias: str,
    run_id: str,
    *,
    actor: str | None = None,
    reason: str | None = None,
) -> dict:
    _validate_alias(alias)
    _validate_run_id(run_id)
    model_path = run_path(paths, run_id)
    if not hadoop_path_exists(spark, model_path):
        raise FileNotFoundError(f"Model run does not exist: {model_path}")

    previous = get_alias(spark, paths, alias)
    now = datetime.now(timezone.utc)
    model_alias = {
        "alias": alias,
        "run_id": run_id,
        "updated_at": now.isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    write_json(spark, alias_path(paths, alias), model_alias, overwrite=True)

    event_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
    event = {
        "event_id": event_id,
        "event_type": "alias_updated",
        "alias": alias,
        "previous_run_id": previous["run_id"] if previous else None,
        "run_id": run_id,
        "occurred_at": now.isoformat(),
        "actor": actor,
        "reason": reason,
        "schema_version": SCHEMA_VERSION,
    }
    event_path = f"{paths['deployment_history']}/{event_id}.json"
    write_json(spark, event_path, event, overwrite=False)
    return model_alias


def _validate_alias(alias: str) -> None:
    if alias not in ALIAS_NAMES:
        raise ValueError(f"Alias must be one of {ALIAS_NAMES}; got {alias!r}")


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must start with an alphanumeric character and contain only " "letters, digits, '.', '_' or '-' (maximum 128 characters)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage model deployment aliases in R2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="Print an alias as JSON")
    get_parser.add_argument("--alias", choices=ALIAS_NAMES, required=True)

    set_parser = subparsers.add_parser("set", help="Point an alias at an immutable model run")
    set_parser.add_argument("--alias", choices=ALIAS_NAMES, required=True)
    set_parser.add_argument("--run-id", required=True)
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
        else:
            model_alias = set_alias(
                spark,
                paths,
                args.alias,
                args.run_id,
                actor=args.actor,
                reason=args.reason,
            )
        print(json.dumps(model_alias, indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
