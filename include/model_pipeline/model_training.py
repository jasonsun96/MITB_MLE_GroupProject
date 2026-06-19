"""Train multi-label Spark ML classifiers on precomputed Gold features."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import DataFrame

from model_pipeline import cli as pipeline_cli
from model_pipeline.multilabel_core import (
    DEFAULT_FEATURE_IMPORTANCE_TOP_K,
    DOCUMENT_ID_COL,
    FEATURES_COL,
    GRID_SEARCH_METRIC_CHOICES,
    LABEL_NORMALIZATION,
    MODEL_TYPE_CHOICES,
    MULTILABEL_STRATEGY,
    SPLIT_COL,
    TARGET_LABELS_COL,
    _collect_training_labels,
    _compute_feature_importance_for_run,
    _compute_multilabel_metrics,
    _default_param_grid,
    _default_model_params,
    _feature_components,
    _global_top_score,
    _iter_param_combos,
    _safe_label_name,
    _split_row_counts,
    build_feature_column,
    create_pipeline_spark_session,
    default_feature_run_id,
    evaluate_multilabel,
    format_prediction_delta_df,
    label_prob_column_map,
    load_dcw_vocab,
    load_features,
    load_schema_paths,
    load_tfidf_vocab,
    logger,
    model_bank_experiment_subdir,
    model_bank_feature_importance_path,
    model_bank_model_manifest_path,
    model_bank_per_label_models_dir,
    model_bank_prediction_metrics_manifest_path,
    predict_multilabel,
    prediction_delta_path,
    prepare_training_data,
    print_dry_run_summary,
    resolve_exp_id,
    resolve_model_params,
    resolve_prediction_exp_id,
    save_json,
    save_pickle,
    write_delta,
)
from model_pipeline.multilabel_core import F

def grid_search_hyperparameters(
    model_type: str,
    train_df: DataFrame,
    val_df: DataFrame,
    label_list: list[str],
    threshold: float,
    *,
    metric: str = "micro_f1",
    param_grid: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit one binary classifier per label on train for each grid point; pick params by val score."""
    if metric not in GRID_SEARCH_METRIC_CHOICES:
        raise ValueError(f"Unsupported grid-search metric: {metric!r}")
    if model_type not in MODEL_TYPE_CHOICES:
        raise ValueError(f"Unsupported model_type for grid search: {model_type!r}")

    grid = param_grid or _default_param_grid(model_type)
    keys = list(_default_model_params(model_type).keys())
    combos = _iter_param_combos(keys, grid, _default_model_params(model_type))
    model_short = "LR" if model_type == "logistic_regression" else "RF"
    logger.info(
        "%s grid search: %s combinations on val split (%s rows, metric=%s)",
        model_short,
        f"{len(combos):,}",
        f"{val_df.count():,}",
        metric,
    )

    best_score = float("-inf")
    best_params = _default_model_params(model_type)
    trial_results: list[dict[str, Any]] = []

    for trial_idx, params in enumerate(combos, start=1):
        logger.info("Grid search trial %s/%s: %s", trial_idx, len(combos), params)
        trial_models, _ = train_multilabel_model(
            train_df,
            max_labels=None,
            model_type=model_type,
            model_params=params,
            label_list=label_list,
        )
        val_predictions = predict_multilabel(
            val_df,
            label_list,
            threshold,
            models=trial_models,
            model_type=model_type,
        )
        val_metrics = _compute_multilabel_metrics(val_predictions, label_list)
        score = float(val_metrics[metric])
        trial_results.append({**params, metric: score, "val_documents": val_metrics["documents"]})
        logger.info("Grid search trial %s/%s %s=%.4f", trial_idx, len(combos), metric, score)
        if score > best_score:
            best_score = score
            best_params = dict(params)

    logger.info(
        "Grid search best params: %s (%s=%.4f)",
        best_params,
        metric,
        best_score,
    )
    return best_params, trial_results


def _build_binary_classifier(model_type: str, model_params: dict[str, Any]) -> Any:
    if model_type == "logistic_regression":
        return LogisticRegression(
            featuresCol=FEATURES_COL,
            labelCol="binary_label",
            family="binomial",
            maxIter=int(model_params["maxIter"]),
            regParam=float(model_params["regParam"]),
            elasticNetParam=float(model_params["elasticNetParam"]),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            featuresCol=FEATURES_COL,
            labelCol="binary_label",
            numTrees=int(model_params["numTrees"]),
            maxDepth=int(model_params["maxDepth"]),
            maxBins=int(model_params["maxBins"]),
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")


def train_multilabel_model(
    train_df: DataFrame,
    max_labels: int | None,
    model_type: str,
    model_params: dict[str, Any],
    *,
    label_list: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if model_type == "random_forest":
        logger.warning(
            "RandomForestClassifier may be slow or memory-heavy on high-dimensional sparse text features."
        )
    logger.info("%s hyperparameters: %s", model_type, model_params)

    label_list = label_list or _collect_training_labels(train_df, max_labels)
    models: dict[str, Any] = {}

    for label in label_list:
        binary_train = train_df.withColumn(
            "binary_label",
            F.when(F.array_contains(F.col(TARGET_LABELS_COL), label), 1.0).otherwise(0.0),
        )
        classifier = _build_binary_classifier(model_type, model_params)
        models[label] = classifier.fit(binary_train)
        logger.debug("Trained binary model for label=%s", label)

    logger.info(
        "Binary relevance training complete: %s per-label models",
        f"{len(models):,}",
    )
    return models, label_list


def save_outputs(
    spark,
    exp_id: str,
    feature_run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
    components: dict[str, bool],
    train_df: DataFrame,
    holdout_df: DataFrame | None,
    models: dict[str, Any],
    label_list: list[str],
    metrics: dict[str, dict[str, float]] | None,
    predictions: DataFrame | None,
    hyperparameters: dict[str, Any],
    *,
    feature_importance: dict[str, Any] | None = None,
    grid_search_results: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    train_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    model_name = args.model_type
    per_label_dir = model_bank_per_label_models_dir(exp_id)
    model_manifest_path = model_bank_model_manifest_path(exp_id, model_name, train_date)
    pred_exp_id = resolve_prediction_exp_id(args) or exp_id
    pred_delta_path = prediction_delta_path(
        args.prediction_date,
        pred_exp_id,
        prediction_suffix=args.prediction_suffix,
    )
    pred_metrics_path = model_bank_prediction_metrics_manifest_path(pred_exp_id, args.prediction_date)
    feature_importance_path: str | None = None

    model_paths: dict[str, str] = {}
    for label, model in models.items():
        safe = _safe_label_name(label)
        model_path = f"{per_label_dir}/{safe}"
        model.write().overwrite().save(model_path)
        model_paths[label] = model_path

    assembled_cols = [DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL, FEATURES_COL]
    write_delta(train_df.select(*assembled_cols), paths["X_train"])
    logger.info("Saved X_train to %s", paths["X_train"])
    if holdout_df is not None and not args.train_only:
        write_delta(holdout_df.select(*assembled_cols), paths["X_val_test_oot"])
        logger.info("Saved X_val_test_oot to %s", paths["X_val_test_oot"])
    else:
        logger.info("Skipped X_val_test_oot (train-only mode)")

    holdout_counts = _split_row_counts(holdout_df)
    n_train = train_df.count()
    prediction_ts = datetime.now(timezone.utc).isoformat()
    metrics = metrics or {}

    metadata = {
        "exp_id": exp_id,
        "run_id": exp_id,
        "feature_run_id": feature_run_id,
        "gold_run_id": paths.get("gold_run_id", paths.get("assembled_run_id")),
        "timestamp": prediction_ts,
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "uses_tfidf": components["tfidf"],
        "uses_log_tfidf": components["log_tfidf"],
        "uses_dcw": components["dcw"],
        "uses_embeddings": components["embeddings"],
        "train_only": args.train_only,
        "multilabel_strategy": MULTILABEL_STRATEGY,
        "multilabel_threshold": args.multilabel_threshold,
        "max_labels": args.max_labels,
        "label_normalization": LABEL_NORMALIZATION,
        "input_feature_paths": {
            "tfidf_train": paths["tfidf_train"],
            "tfidf_val_test_oot": paths["tfidf_val_test_oot"],
            "dcw_train": paths["dcw_train"],
            "dcw_val_test_oot": paths["dcw_val_test_oot"],
            "embeddings": paths["embeddings"],
            "tfidf_pkl": paths["tfidf_pkl"],
            "dcw_pkl": paths["dcw_pkl"],
        },
        "labels_path": paths["labels"],
        "assembled_dataset_paths": {
            "X_train": paths["X_train"],
            "X_val_test_oot": paths["X_val_test_oot"],
            "X_unlabelled": paths["X_unlabelled"],
        },
        "per_label_model_paths": model_paths,
        "probability_columns": label_prob_column_map(label_list),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "row_counts": {
            "train_documents": n_train,
            "holdout_documents": holdout_counts["holdout"],
            "val_documents": holdout_counts["val"],
            "test_documents": holdout_counts["test"],
            "oot_documents": holdout_counts["oot"],
        },
        "num_unique_labels": len(label_list),
        "split_column": SPLIT_COL,
        "notes": (
            "TF-IDF, DCW, and embeddings were precomputed in upstream Gold jobs "
            "and were not refit in this training script. "
            "Multi-label training uses binary relevance (one binary classifier per label). "
            "Spark ML models live under model/per_label/; this .pkl holds metadata and metrics."
            + (" Holdout scoring deferred (--train-only)." if args.train_only else "")
        ),
    }
    if grid_search_results is not None:
        metadata["grid_search"] = {
            "metric": args.grid_search_metric,
            "param_grid": _default_param_grid(args.model_type),
            "trials": grid_search_results,
        }
    if feature_importance is not None:
        feature_importance_path = model_bank_feature_importance_path(exp_id, train_date)
        save_json(feature_importance_path, feature_importance, spark)
        metadata["feature_importance_path"] = feature_importance_path
        metadata["feature_importance_top_k"] = feature_importance.get("top_k")
        logger.info("Saved feature importance to %s", feature_importance_path)
        top_feature, top_score = _global_top_score(feature_importance)
        logger.info("Global top feature: %s (score=%.6f)", top_feature, top_score)
    save_pickle(model_manifest_path, metadata, spark)
    model_manifest_json = model_manifest_path.replace(".pkl", ".json")
    save_json(model_manifest_json, metadata, spark)
    logger.info("Saved model manifest JSON to %s", model_manifest_json)

    if (
        not args.train_only
        and predictions is not None
        and predictions.limit(1).count() > 0
    ):
        pred_out = format_prediction_delta_df(
            predictions,
            exp_id,
            feature_run_id,
            prediction_ts,
            multilabel_threshold=args.multilabel_threshold,
        )
        write_delta(pred_out, pred_delta_path)
        save_pickle(
            pred_metrics_path,
            {
                "exp_id": exp_id,
                "run_id": exp_id,
                "feature_run_id": feature_run_id,
                "prediction_ts": prediction_ts,
                "prediction_delta_path": pred_delta_path,
                "row_count": holdout_counts["holdout"],
                "metrics": metrics,
            },
            spark,
        )
    elif args.train_only:
        logger.info("Train-only mode: skipped holdout predictions for exp_id=%s", exp_id)
    else:
        logger.warning("No holdout predictions to write for exp_id=%s", exp_id)

    logger.info("Saved per-label Spark models under %s", per_label_dir)
    logger.info("Saved model manifest to %s", model_manifest_path)
    if not args.train_only:
        logger.info("Saved prediction metrics manifest to %s", pred_metrics_path)

    out = {
        "per_label_models_dir": per_label_dir,
        "model_manifest_path": model_manifest_path,
    }
    if feature_importance_path is not None:
        out["feature_importance_path"] = feature_importance_path
    if not args.train_only:
        out["prediction_manifest_path"] = pred_metrics_path
        out["prediction_delta_path"] = pred_delta_path
    return out


def main() -> None:
    parser = pipeline_cli.build_parser(
        description="Train a multi-label Spark ML classifier on precomputed Gold features"
    )
    args = parser.parse_args()
    pipeline_cli.validate_training_args(parser, args)
    pipeline_cli.configure_logging(args)

    exp_id, feature_run_id, gold_run_id, paths, components = pipeline_cli.resolve_run_context(args)
    pipeline_cli.log_run_banner(args, exp_id, feature_run_id, gold_run_id, mode="train")

    spark = create_pipeline_spark_session("gold-model-training")

    include_holdout = (not args.train_only) or args.grid_search
    dcw_vocab = load_dcw_vocab(spark, paths) if components["dcw"] else None
    tfidf_vocab: list[str] | None = None
    if components["tfidf"] or components["log_tfidf"]:
        tfidf_vocab = load_tfidf_vocab(spark, paths)
    train_features, holdout_features, labels, embeddings_df = load_features(
        spark, paths, args.feature_set, include_holdout=include_holdout
    )
    train_df, holdout_df = prepare_training_data(
        train_features, holdout_features, labels, embeddings_df
    )

    if args.limit:
        train_df = train_df.limit(args.limit)
        if holdout_df is not None:
            holdout_df = holdout_df.limit(args.limit)
        logger.info("Smoke test: limited to %s rows per split after joins", f"{args.limit:,}")

    train_df = build_feature_column(train_df, args.feature_set, dcw_vocab=dcw_vocab)
    if holdout_df is not None:
        holdout_df = build_feature_column(holdout_df, args.feature_set, dcw_vocab=dcw_vocab)

    label_list = _collect_training_labels(train_df, args.max_labels)
    grid_search_results: list[dict[str, Any]] | None = None
    if args.grid_search:
        assert holdout_df is not None
        val_df = holdout_df.filter(F.col(SPLIT_COL) == "val")
        if val_df.limit(1).count() == 0:
            raise ValueError("Grid search requires non-empty val split in holdout features")
        model_params, grid_search_results = grid_search_hyperparameters(
            args.model_type,
            train_df,
            val_df,
            label_list,
            args.multilabel_threshold,
            metric=args.grid_search_metric,
        )
    else:
        model_params = resolve_model_params(args)

    models, label_list = train_multilabel_model(
        train_df,
        args.max_labels,
        args.model_type,
        model_params,
        label_list=label_list,
    )

    feature_importance: dict[str, Any] | None = None
    if args.feature_importance_top_k > 0:
        feature_importance = _compute_feature_importance_for_run(
            spark,
            args.model_type,
            models,
            label_list,
            paths,
            args.feature_set,
            components,
            top_k=args.feature_importance_top_k,
            tfidf_vocab=tfidf_vocab,
            dcw_vocab=dcw_vocab,
            embedding_dim_source_df=train_df if components["embeddings"] else None,
        )
        logger.info(
            "Computed top-%s feature importance across %s labels (%s)",
            args.feature_importance_top_k,
            len(label_list),
            args.model_type,
        )

    predictions: DataFrame | None = None
    metrics: dict[str, dict[str, float]] | None = None
    if not args.train_only:
        assert holdout_df is not None
        predictions = predict_multilabel(
            holdout_df,
            label_list,
            args.multilabel_threshold,
            models=models,
            model_type=args.model_type,
        )
        metrics = evaluate_multilabel(predictions, label_list)

    if args.dry_run:
        print_dry_run_summary(
            exp_id,
            args,
            components,
            train_df,
            holdout_df,
            label_list,
            metrics,
            predictions,
            model_params,
        )
        if feature_importance is not None:
            logger.info("Dry-run global top-10 features:")
            for row in feature_importance["global_top"][:10]:
                score = row.get("mean_abs_coefficient", row.get("mean_importance"))
                logger.info("  #%s %s score=%.6f", row["rank"], row["feature"], score)
    else:
        hyperparameters = {
            **model_params,
            "model_type": args.model_type,
            "multilabel_threshold": args.multilabel_threshold,
            "max_labels": args.max_labels,
            "train_only": args.train_only,
            "grid_search": args.grid_search,
        }
        save_outputs(
            spark,
            exp_id,
            feature_run_id,
            paths,
            args,
            components,
            train_df,
            holdout_df,
            models,
            label_list,
            metrics,
            predictions,
            hyperparameters,
            feature_importance=feature_importance,
            grid_search_results=grid_search_results,
        )

    logger.info("Model training complete for exp_id=%s", exp_id)


if __name__ == "__main__":
    main()

