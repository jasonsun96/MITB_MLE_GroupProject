import json
import pickle
import sys
from pathlib import Path
from typing import Any

GOLD_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = GOLD_DIR.parents[1]


PARTITION_COL = "snapshot_date"


def bootstrap_paths() -> None:
    for path in (PROJECT_ROOT, GOLD_DIR):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def columns_with_snapshot(df, columns: list[str]) -> list[str]:
    cols = [col for col in columns if col in df.columns]
    if PARTITION_COL in df.columns and PARTITION_COL not in cols:
        cols.append(PARTITION_COL)

    return cols


def write_delta(df, path: str, partition_col: str | None = None, replace_partition_value: str | None = None) -> None:
    writer = df.write.format("delta").mode("overwrite").option("mergeSchema", "true")
    if partition_col and replace_partition_value is not None:
        writer = writer.option("replaceWhere", f"{partition_col} = '{replace_partition_value}'")
    if partition_col and partition_col in df.columns:
        writer = writer.partitionBy(partition_col)

    writer.save(path)


def _hadoop_fs(spark, path: str):
    jvm = spark.sparkContext._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())

    return jvm, fs, hadoop_path


def save_bytes_to_path(path: str, data: bytes, spark) -> None:
    jvm, fs, hadoop_path = _hadoop_fs(spark, path)
    out = fs.create(hadoop_path, True)

    try:
        if data:
            out.write(bytearray(data))
    finally:
        out.close()


def load_bytes_from_path(path: str, spark) -> bytes:
    jvm, fs, hadoop_path = _hadoop_fs(spark, path)
    if not fs.exists(hadoop_path):
        raise FileNotFoundError(path)
    in_stream = fs.open(hadoop_path)

    try:
        jbytes = jvm.org.apache.hadoop.io.IOUtils.readFullyToByteArray(in_stream)
        return bytes(jbytes)
    finally:
        in_stream.close()


def save_pickle(path: str, obj: Any, spark) -> None:
    save_bytes_to_path(path, pickle.dumps(obj), spark)


def save_json(path: str, obj: Any, spark) -> None:
    save_bytes_to_path(
        path,
        json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"),
        spark,
    )


def load_pickle(path: str, spark) -> Any:
    return pickle.loads(load_bytes_from_path(path, spark))
