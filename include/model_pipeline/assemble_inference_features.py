"""Assemble deployment-specific inference feature tables from Gold corpus tables.

This applies frozen feature artifacts from model_bank/features/{feature_run_id}
to documents marked with category='inference' in gold/label_store. It does not
refit TF-IDF or DCW statistics.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PIPELINE_DIR = Path(__file__).resolve().parent
_INCLUDE_DIR = _PIPELINE_DIR.parent
_GOLD_DIR = _INCLUDE_DIR / "gold"
_PROJECT_ROOT = _INCLUDE_DIR.parent
for _path in (_PROJECT_ROOT, _INCLUDE_DIR, _GOLD_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import yaml
from gold_io import load_pickle
from model_pipeline.multilabel_core import (DCW_FEATURES_COL, DOCUMENT_ID_COL,
                                            EMBEDDING_COL,
                                            EMBEDDING_VECTOR_COL, FEATURES_COL,
                                            SPLIT_COL,
                                            _ensure_embedding_vector,
                                            _feature_components,
                                            build_feature_column,
                                            create_pipeline_spark_session,
                                            load_dcw_vocab, load_schema_paths,
                                            load_training_manifest)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, MapType, StringType
from tfidf_processing import add_tfidf_column, load_tfidf_artifact


logger = logging.getLogger(__name__)

DEFAULT_INFERENCE_CATEGORY = "inference"
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "batch_inference.yaml"


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_assembly_config(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Feature assembly config not found: {path}")

    with path.open() as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Feature assembly config must be a YAML mapping: {path}")

    logger.info("Loaded feature assembly config from %s", path)
    return config


def _manifest_value(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    if value is None:
        return None
    return str(value).strip() or None


def _deployment_model_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    deployment_config = config.get("deployment") or {}
    if not isinstance(deployment_config, dict):
        raise ValueError("Feature assembly config field 'deployment' must be a mapping")
    production_config = deployment_config.get("production") or {}
    if not isinstance(production_config, dict) or not production_config:
        raise ValueError("Feature assembly config must define deployment.production")
    contexts = [("production", production_config)]

    shadow_config = deployment_config.get("shadow")
    if shadow_config is not None:
        if not isinstance(shadow_config, dict):
            raise ValueError("Feature assembly config field 'deployment.shadow' must be a mapping")
        if shadow_config.get("enabled") is not False:
            shadow_run_id = _clean_optional(shadow_config.get("run_id")) or _clean_optional(shadow_config.get("exp_id"))
            production_run_id = _clean_optional(production_config.get("run_id")) or _clean_optional(production_config.get("exp_id"))
            if not shadow_run_id and shadow_config.get("enabled") is True:
                raise ValueError("Feature assembly config deployment.shadow must define run_id when enabled")
            if shadow_run_id and shadow_run_id != production_run_id:
                contexts.append(("shadow", shadow_config))
    return contexts


def _resolve_model_context(
    spark,
    alias: str,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    category: str | None,
) -> dict[str, Any]:
    """Resolve model-selected feature inputs for one deployment alias."""
    run_id = _clean_optional(model_config.get("run_id")) or _clean_optional(model_config.get("exp_id"))
    exp_id = _clean_optional(model_config.get("exp_id"))
    model_type = _clean_optional(model_config.get("model_type"))
    model_date = _clean_optional(model_config.get("model_date"))

    manifest: dict[str, Any] | None = None
    manifest_path: str | None = None
    if not exp_id:
        raise ValueError(f"Feature assembly config deployment.{alias} must define exp_id")
    if not model_type:
        raise ValueError(f"Feature assembly config deployment.{alias} must define model_type")
    manifest, manifest_path = load_training_manifest(spark, exp_id, model_date, model_type=model_type)

    feature_run_id = _manifest_value(manifest or {}, "feature_run_id")
    feature_set = _manifest_value(manifest or {}, "feature_set")
    if not feature_run_id:
        raise ValueError(f"Model manifest for {exp_id!r} must define feature_run_id")
    if not feature_set:
        raise ValueError(f"Model manifest for {exp_id!r} must define feature_set")

    gold_run_ids = feature_config.get("gold_run_ids") or {}
    if gold_run_ids and not isinstance(gold_run_ids, dict):
        raise ValueError("Feature assembly config field 'features.gold_run_ids' must be a mapping")
    gold_run_id = (
        _clean_optional(gold_run_ids.get(alias))
        or _clean_optional(feature_config.get("gold_run_id"))
        or _manifest_value(manifest or {}, "gold_run_id")
        or feature_run_id
    )

    return {
        "alias": alias,
        "run_id": run_id,
        "exp_id": exp_id,
        "model_type": model_type or _manifest_value(manifest or {}, "model_type"),
        "model_date": model_date,
        "model_manifest_path": manifest_path,
        "feature_run_id": feature_run_id,
        "gold_run_id": gold_run_id,
        "feature_set": feature_set,
        "category": category,
    }


def resolve_feature_assembly_contexts(spark, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve model-selected feature inputs for production and optional shadow."""
    config = load_assembly_config(args.config)
    feature_config = config.get("features") or {}
    if not isinstance(feature_config, dict):
        raise ValueError("Feature assembly config field 'features' must be a mapping")

    categories = feature_config.get("categories")
    if categories is not None:
        if not isinstance(categories, list) or not all(_clean_optional(category) for category in categories):
            raise ValueError("Feature assembly config field 'features.categories' must be a non-empty list of category names")
        resolved_categories = [_clean_optional(category) for category in categories]
    else:
        resolved_categories = [_clean_optional(feature_config.get("category")) or _clean_optional(args.category) or DEFAULT_INFERENCE_CATEGORY]

    contexts = [
        _resolve_model_context(spark, alias, model_config, feature_config, category)
        for alias, model_config in _deployment_model_configs(config)
        for category in resolved_categories
    ]
    return contexts


def load_batch_feature_base(config_path: str | Path | None, batch_id: str) -> str:
    config = load_assembly_config(config_path)
    feature_config = config.get("features") or {}
    if not isinstance(feature_config, dict):
        raise ValueError("Feature assembly config field 'features' must be a mapping")
    output_base = feature_config.get("output_base")
    if output_base:
        return str(output_base).rstrip("/")

    with (_PROJECT_ROOT / "schema.yaml").open() as schema_file:
        schema = yaml.safe_load(schema_file)
    gold = schema["gold"]
    gold_base = gold["path"].rstrip("/")
    tables = gold.get("tables") or {}
    batch_inference = tables.get("batch_inference", {}).get("path") or "batch_inference"
    return f"{gold_base}/{batch_inference}/{batch_id}/features"


def _dcw_score_from_artifact(spark, paths: dict[str, str]) -> dict[str, float]:
    """Load frozen DCW scores from dcw.pkl, falling back to dcw_score Delta."""
    try:
        bundle = load_pickle(paths["dcw_pkl"], spark)
        score = bundle.get("score")
        if isinstance(score, dict) and score:
            logger.info("Loaded DCW score map from %s", paths["dcw_pkl"])
            return {str(k): float(v) for k, v in score.items()}
    except Exception:
        logger.exception("Could not load DCW pickle from %s; falling back to dcw_score", paths["dcw_pkl"])

    rows = spark.read.format("delta").load(paths["dcw_score_path"]).select("lemma", "score").filter(F.col("lemma").isNotNull() & F.col("score").isNotNull()).collect()
    if not rows:
        raise ValueError(f"No DCW scores found at {paths['dcw_score_path']}")
    logger.info("Loaded DCW score map from %s", paths["dcw_score_path"])
    return {row.lemma: float(row.score) for row in rows}


def add_dcw_features_column(df: DataFrame, score: dict[str, float]) -> DataFrame:
    """Add sparse dcw_features map: lemma -> count(lemma in doc) * frozen score."""
    score_bc = df.sparkSession.sparkContext.broadcast(score)

    @udf(MapType(StringType(), DoubleType()))
    def compute_dcw_map(pos_counts: dict | None) -> dict[str, float]:
        if not pos_counts:
            return {}
        scores = score_bc.value
        out: dict[str, float] = {}
        for pos in ("NOUN", "PROPN"):
            values = pos_counts.get(pos) or {}
            for lemma, count in values.items():
                weight = scores.get(lemma)
                if weight is not None:
                    out[lemma] = out.get(lemma, 0.0) + float(count) * float(weight)
        return out

    return df.withColumn(DCW_FEATURES_COL, compute_dcw_map(F.col("pos_counts")))


def _snapshot_date_from_batch_id(batch_id: str) -> str:
    try:
        return datetime.strptime(batch_id, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            "labels table has no batch_id column, so --batch-id must be YYYYMMDD "
            f"to filter by snapshot_date; got {batch_id!r}"
        ) from exc


def _load_inference_ids(spark, labels_path: str, category: str, batch_id: str) -> DataFrame:
    labels = spark.read.format("delta").load(labels_path)
    required = {DOCUMENT_ID_COL, SPLIT_COL}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels table missing required columns: {sorted(missing)}")

    df = labels.filter(F.col(SPLIT_COL) == category)
    if "batch_id" in labels.columns:
        df = df.filter(F.col("batch_id") == batch_id)
        batch_filter = f"batch_id={batch_id!r}"
    elif "snapshot_date" in labels.columns:
        snapshot_date = _snapshot_date_from_batch_id(batch_id)
        df = df.filter(F.col("snapshot_date") == snapshot_date)
        batch_filter = f"snapshot_date={snapshot_date!r}"
        logger.info("labels table has no batch_id column; filtering inference ids by %s", batch_filter)
    else:
        raise ValueError("labels table must contain either batch_id or snapshot_date")

    select_cols = [DOCUMENT_ID_COL, SPLIT_COL]
    if "snapshot_date" in df.columns:
        select_cols.append("snapshot_date")
    if "batch_id" in df.columns:
        select_cols.append("batch_id")

    ids = df.select(*select_cols).dropDuplicates([DOCUMENT_ID_COL])
    count = ids.count()
    if count == 0:
        raise ValueError(f"No documents found with category={category!r} and {batch_filter}")
    logger.info("Loaded %s inference document ids", f"{count:,}")
    return ids


def _load_embeddings(spark, paths: dict[str, str]) -> DataFrame:
    raw = spark.read.format("delta").load(paths["embeddings"])
    if EMBEDDING_COL not in raw.columns:
        raise ValueError(f"embeddings table missing {EMBEDDING_COL!r}")
    return raw.select(DOCUMENT_ID_COL, EMBEDDING_COL).dropDuplicates([DOCUMENT_ID_COL]).transform(_ensure_embedding_vector).select(DOCUMENT_ID_COL, EMBEDDING_VECTOR_COL)


def assemble_inference_features(spark, paths: dict[str, str], *, feature_set: str, category: str, batch_id: str, limit: int | None, model_context: dict[str, Any] | None = None) -> DataFrame:
    components = _feature_components(feature_set)
    ids = _load_inference_ids(spark, paths["labels"], category, batch_id)
    if limit:
        ids = ids.limit(limit)
        logger.info("Smoke test: limited inference ids to %s rows", f"{limit:,}")

    assembled = ids

    if components["tfidf"] or components["log_tfidf"]:
        ngrams = spark.read.format("delta").load(paths["ngrams"])
        tfidf_artifact = load_tfidf_artifact(paths["tfidf_pkl"], spark)
        tfidf_df = add_tfidf_column(ngrams.join(ids.select(DOCUMENT_ID_COL), DOCUMENT_ID_COL, "inner"), tfidf_artifact)
        keep = [DOCUMENT_ID_COL]
        if components["tfidf"]:
            keep.append("tfidf")
        if components["log_tfidf"]:
            keep.append("log_tfidf")
        assembled = assembled.join(tfidf_df.select(*keep), DOCUMENT_ID_COL, "inner")
        logger.info("Joined frozen TF-IDF features")

    dcw_vocab: list[str] | None = None
    if components["dcw"]:
        pos = spark.read.format("delta").load(paths["pos_tags"])
        dcw_score = _dcw_score_from_artifact(spark, paths)
        dcw_vocab = load_dcw_vocab(spark, paths)
        dcw_df = add_dcw_features_column(pos.join(ids.select(DOCUMENT_ID_COL), DOCUMENT_ID_COL, "inner"), dcw_score)
        assembled = assembled.join(dcw_df.select(DOCUMENT_ID_COL, DCW_FEATURES_COL), DOCUMENT_ID_COL, "inner")
        logger.info("Joined frozen DCW features")

    if components["embeddings"]:
        embeddings = _load_embeddings(spark, paths)
        assembled = assembled.join(embeddings, DOCUMENT_ID_COL, "inner")
        logger.info("Joined embeddings")

    before_features = assembled.count()
    if before_features == 0:
        raise ValueError("No inference rows remain after joining requested feature tables")

    with_features = build_feature_column(assembled, feature_set, dcw_vocab=dcw_vocab)
    model_context = model_context or {}
    output = (
        with_features.withColumn("feature_run_id", F.lit(paths["feature_run_id"]))
        .withColumn("feature_set", F.lit(feature_set))
        .withColumn("model_exp_id", F.lit(model_context.get("exp_id")).cast("string"))
        .withColumn("model_type", F.lit(model_context.get("model_type")).cast("string"))
        .withColumn("model_manifest_path", F.lit(model_context.get("model_manifest_path")).cast("string"))
        .withColumn("assembled_at", F.lit(datetime.now(timezone.utc).isoformat()))
        .select(
            DOCUMENT_ID_COL,
            *([SPLIT_COL] if SPLIT_COL in with_features.columns else []),
            *([F.col("snapshot_date")] if "snapshot_date" in with_features.columns else []),
            *([F.col("batch_id")] if "batch_id" in with_features.columns else []),
            FEATURES_COL,
            *([DCW_FEATURES_COL] if DCW_FEATURES_COL in with_features.columns else []),
            "feature_run_id",
            "feature_set",
            "model_exp_id",
            "model_type",
            "model_manifest_path",
            "assembled_at",
        )
    )
    logger.info("Assembled %s inference feature rows", f"{output.count():,}")
    return output


def write_inference_features(df: DataFrame, path: str, batch_id: str | None, mode: str = "overwrite") -> None:
    writer = df.write.format("delta").option("mergeSchema", "true")
    if mode == "overwrite" and batch_id and "batch_id" in df.columns:
        (writer.mode("overwrite").partitionBy("batch_id").option("replaceWhere", f"batch_id = '{batch_id}'").save(path))
    elif mode == "overwrite":
        writer.mode("overwrite").save(path)
    else:
        writer.mode(mode).save(path)
    logger.info("Wrote inference features to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble deployment-specific inference features from Gold corpus features")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--category", default=DEFAULT_INFERENCE_CATEGORY)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    spark = create_pipeline_spark_session("assemble-inference-features")
    try:
        contexts = resolve_feature_assembly_contexts(spark, args)
        feature_base = load_batch_feature_base(args.config, args.batch_id)
        contexts_by_alias: dict[str, list[dict[str, Any]]] = {}
        for context in contexts:
            contexts_by_alias.setdefault(context["alias"], []).append(context)

        for alias, alias_contexts in contexts_by_alias.items():
            output_path = f"{feature_base}/{alias}"
            for index, context in enumerate(alias_contexts):
                paths = load_schema_paths(context["feature_run_id"], gold_run_id=context["gold_run_id"])
                logger.info("Deployment group: %s", alias)
                logger.info("Model exp_id: %s", context.get("exp_id") or "<manual>")
                logger.info("Model manifest: %s", context.get("model_manifest_path") or "<none>")
                logger.info("Feature run: %s", context["feature_run_id"])
                logger.info("Feature set: %s", context["feature_set"])
                logger.info("Gold run: %s", context["gold_run_id"])
                logger.info("Label category: %s", context["category"])
                logger.info("Output inference features: %s", output_path)

                output = (
                    assemble_inference_features(
                        spark,
                        paths,
                        feature_set=context["feature_set"],
                        category=context["category"],
                        batch_id=args.batch_id,
                        limit=args.limit,
                        model_context=context,
                    ).withColumn("deployment_group", F.lit(alias))
                )
                write_mode = "overwrite" if index == 0 else "append"
                write_inference_features(output, output_path, args.batch_id, mode=write_mode)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
