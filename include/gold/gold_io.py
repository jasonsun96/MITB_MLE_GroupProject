"""Shared Delta I/O helpers for gold and model-bank jobs."""
from __future__ import annotations

import sys
from pathlib import Path

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


def write_delta(df, path: str, partition_col: str | None = None) -> None:
    writer = df.write.format("delta").mode("overwrite").option("mergeSchema", "true")
    if partition_col and partition_col in df.columns:
        writer = writer.partitionBy(partition_col)
    writer.save(path)


def save_bytes_to_path(path: str, data: bytes, spark) -> None:
    jvm = spark.sparkContext._jvm
    uri = jvm.java.net.URI(path)
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, spark.sparkContext._jsc.hadoopConfiguration())
    out = fs.create(hadoop_path, True)
    out.write(bytearray(data))
    out.close()
