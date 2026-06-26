import argparse
from datetime import datetime, timezone
from typing import Any

from model_pipeline import cli as pipeline_cli
from model_pipeline.multilabel_core import (
    DOCUMENT_ID_COL, FEATURES_COL, PREDICT_STAGE_ALL_DEPRECATED_MSG, SPLIT_COL,
    TARGET_LABELS_COL, F, _compute_feature_importance_for_run,
    _default_hyperparameters, _feature_components,
    _feature_importance_json_exists, _hadoop_path_exists,
    _load_checkpointed_predictions, _read_delta, build_feature_column,
    checkpoint_predictions, compute_threshold_sweep,
    create_pipeline_spark_session, evaluate_multilabel, load_dcw_vocab,
    load_features, load_pickle, load_trained_models, load_training_manifest,
    logger, model_bank_feature_importance_path, parse_threshold_sweep_values,
    predict_multilabel, prediction_delta_path, prepare_holdout_data,
    print_dry_run_summary, prob_columns_in_df, resolve_model_type_for_run,
    resolve_prediction_exp_id, save_evaluation_outputs, save_holdout_features,
    save_json, save_pickle, save_threshold_sweep_outputs)
from pandas import DataFrame


def _load_predict_context(
    spark,
    run_id: str,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, str], str, dict[str, bool], float, list[str], str]:
    manifest, manifest_path = load_training_manifest(
        spark,
        run_id,
        args.model_date,
        paths=paths,
        feature_set=args.feature_set,
        model_type=args.model_type,
    )
    per_label_paths = manifest.get("per_label_model_paths")
    if not per_label_paths:
        raise ValueError(f"Training manifest at {manifest_path} missing per_label_model_paths")

    feature_set = manifest.get("feature_set", args.feature_set)
    if feature_set != args.feature_set:
        logger.warning(
            "Using feature_set=%r from manifest (CLI had %r)",
            feature_set,
            args.feature_set,
        )
    components = _feature_components(feature_set)
    threshold = args.multilabel_threshold
    if manifest.get("multilabel_threshold") is not None and threshold == 0.5:
        threshold = float(manifest["multilabel_threshold"])

    label_list = list(per_label_paths.keys())
    model_type = resolve_model_type_for_run(spark, run_id, per_label_paths, args.model_type)
    manifest_model_type = manifest.get("model_type")
    if manifest_model_type and manifest_model_type != model_type:
        logger.warning(
            "Manifest model_type=%r differs from resolved %r; using resolved type for scoring",
            manifest_model_type,
            model_type,
        )
    return (
        manifest,
        manifest_path,
        per_label_paths,
        feature_set,
        components,
        threshold,
        label_list,
        model_type,
    )


def _build_holdout_from_features(spark, paths: dict[str, str], feature_set: str, components: dict[str, bool], *, holdout_splits: list[str] | None, limit: int | None) -> DataFrame:
    dcw_vocab = load_dcw_vocab(spark, paths) if components["dcw"] else None
    _, holdout_features, labels, embeddings_df = load_features(spark, paths, feature_set, include_holdout=True)
    holdout_df = prepare_holdout_data(holdout_features, labels, embeddings_df)

    if holdout_splits:
        holdout_df = holdout_df.filter(F.col(SPLIT_COL).isin(holdout_splits))
        logger.info("Filtered holdout to splits: %s", holdout_splits)

    if limit:
        holdout_df = holdout_df.limit(limit)
        logger.info("Smoke test: limited holdout to %s rows", f"{limit:,}")

    return build_feature_column(holdout_df, feature_set, dcw_vocab=dcw_vocab)


def _load_saved_holdout_x(spark, paths: dict[str, str], *, holdout_splits: list[str] | None, limit: int | None) -> DataFrame:
    x_path = paths["X_val_test_oot"]
    logger.info("Loading saved holdout X from %s", x_path)
    holdout_df = _read_delta(spark, x_path, "X_val_test_oot")
    required = {DOCUMENT_ID_COL, SPLIT_COL, TARGET_LABELS_COL, FEATURES_COL}
    missing = required - set(holdout_df.columns)
    if missing:
        raise ValueError(f"X_val_test_oot at {x_path} missing columns {missing}. " "Run --predict-stage features first.")

    if holdout_splits:
        holdout_df = holdout_df.filter(F.col(SPLIT_COL).isin(holdout_splits))
        logger.info("Filtered saved holdout X to splits: %s", holdout_splits)

    if limit:
        holdout_df = holdout_df.limit(limit)
        logger.info("Smoke test: limited saved holdout X to %s rows", f"{limit:,}")

    return holdout_df


def _dry_run_holdout_eval_summary(
    run_id: str, args: argparse.Namespace, components: dict[str, bool], label_list: list[str], predictions: DataFrame, manifest: dict[str, Any], model_type: str, *, holdout_df: DataFrame | None = None
) -> None:
    metrics = evaluate_multilabel(predictions, label_list)
    print_dry_run_summary(
        run_id,
        args,
        components,
        None,
        holdout_df,
        label_list,
        metrics,
        predictions,
        manifest.get("hyperparameters", _default_hyperparameters(model_type)),
    )


def run_predict_stage_features(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    components = _feature_components(args.feature_set)
    logger.info(
        "Predict stage=features: assembling holdout X (feature_set=%s)",
        args.feature_set,
    )
    holdout_df = _build_holdout_from_features(
        spark,
        paths,
        args.feature_set,
        components,
        holdout_splits=args.holdout_splits,
        limit=args.limit,
    )

    if args.dry_run:
        n_rows = holdout_df.count()
        logger.info("DRY RUN: would save %s holdout rows to %s", f"{n_rows:,}", paths["X_val_test_oot"])
        return

    save_holdout_features(holdout_df, paths)
    logger.info("Predict stage=features complete: %s", paths["X_val_test_oot"])


def run_predict_stage_predict(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    manifest, manifest_path, per_label_paths, feature_set, components, threshold, label_list, model_type = _load_predict_context(spark, run_id, paths, args)
    logger.info(
        "Predict stage=predict: scoring %s labels from saved holdout X",
        f"{len(label_list):,}",
    )
    holdout_df = _load_saved_holdout_x(
        spark,
        paths,
        holdout_splits=args.holdout_splits,
        limit=args.limit,
    )

    predictions = predict_multilabel(
        holdout_df,
        label_list,
        threshold,
        per_label_paths=per_label_paths,
        model_type=model_type,
    )

    if args.dry_run:
        _dry_run_holdout_eval_summary(
            run_id,
            args,
            components,
            label_list,
            predictions,
            manifest,
            model_type,
            holdout_df=holdout_df,
        )
        return

    pred_exp_id = resolve_prediction_exp_id(args) or run_id
    pred_delta_path = prediction_delta_path(
        args.prediction_date,
        pred_exp_id,
        prediction_suffix=args.prediction_suffix,
    )
    predictions, prediction_ts, pred_delta_path = checkpoint_predictions(
        spark,
        predictions,
        pred_exp_id,
        feature_run_id,
        pred_delta_path=pred_delta_path,
        multilabel_threshold=threshold,
    )
    logger.info(
        "Predict stage=predict complete: %s rows checkpointed to %s (ts=%s)",
        f"{predictions.count():,}",
        pred_delta_path,
        prediction_ts,
    )


def run_predict_stage_metrics(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    manifest, manifest_path, _, feature_set, components, threshold, label_list, model_type = _load_predict_context(spark, run_id, paths, args)
    predictions, pred_delta_path, prediction_ts = _load_checkpointed_predictions(spark, args, label_list)

    if args.dry_run:
        _dry_run_holdout_eval_summary(
            run_id,
            args,
            components,
            label_list,
            predictions,
            manifest,
            model_type,
        )
        return

    logger.info("Computing metrics on checkpointed predictions")
    metrics = evaluate_multilabel(predictions, label_list)

    save_evaluation_outputs(
        spark,
        run_id,
        feature_run_id,
        paths,
        manifest_path,
        label_list,
        metrics,
        predictions,
        threshold,
        prediction_ts,
        pred_delta_path,
        skip_x_write=True,
        skip_pred_delta_write=True,
    )
    logger.info("Predict stage=metrics complete")


def run_predict_stage_threshold_sweep(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    manifest, _, _, feature_set, components, _, label_list, model_type = _load_predict_context(spark, run_id, paths, args)
    predictions, pred_delta_path, prediction_ts = _load_checkpointed_predictions(spark, args, label_list)
    thresholds = parse_threshold_sweep_values(args.threshold_sweep)

    if args.dry_run:
        sweep = compute_threshold_sweep(predictions, label_list, thresholds)
        logger.info(
            "DRY RUN: threshold sweep would write %s rows for %s thresholds",
            len(sweep["rows"]),
            len(thresholds),
        )
        return

    if not prob_columns_in_df(predictions):
        logger.warning(
            "Skipping threshold sweep for %s: no prob_* columns on %s",
            run_id,
            pred_delta_path,
        )
        return

    logger.info(
        "Computing threshold sweep for %s thresholds on %s",
        len(thresholds),
        pred_delta_path,
    )
    sweep = compute_threshold_sweep(predictions, label_list, thresholds)
    save_threshold_sweep_outputs(
        spark,
        run_id,
        args.prediction_date,
        sweep,
        prediction_delta_path=pred_delta_path,
        prediction_ts=prediction_ts,
    )
    logger.info("Predict stage=threshold_sweep complete")


def run_backfill_feature_importance(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    existing_path = _feature_importance_json_exists(spark, run_id)
    if existing_path and not args.force_feature_importance:
        logger.info("Feature importance already exists: %s (use --force-feature-importance to overwrite)", existing_path)
        return

    if args.feature_importance_top_k <= 0:
        logger.info("Skipping feature importance backfill (--feature-importance-top-k 0)")
        return

    manifest, manifest_path, per_label_paths, feature_set, components, _, label_list, model_type = _load_predict_context(spark, run_id, paths, args)
    models = load_trained_models(per_label_paths, model_type)

    feature_importance = _compute_feature_importance_for_run(
        spark,
        model_type,
        models,
        label_list,
        paths,
        feature_set,
        components,
        top_k=args.feature_importance_top_k,
    )

    if args.dry_run:
        logger.info(
            "DRY RUN: would save feature importance (%s labels, top_k=%s)",
            len(label_list),
            args.feature_importance_top_k,
        )
        return

    train_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    fi_path = model_bank_feature_importance_path(run_id, train_date)
    save_json(fi_path, feature_importance, spark)
    logger.info("Backfilled feature importance to %s", fi_path)

    if _hadoop_path_exists(spark, manifest_path):
        try:
            manifest_payload = load_pickle(manifest_path, spark)
            manifest_payload["feature_importance_path"] = fi_path
            manifest_payload["feature_importance_top_k"] = feature_importance.get("top_k")
            save_pickle(manifest_path, manifest_payload, spark)
            save_json(manifest_path.replace(".pkl", ".json"), manifest_payload, spark)
            logger.info("Updated training manifest with feature_importance_path")
        except Exception as exc:
            logger.warning("Could not update training manifest with FI path: %s", exc)


def run_predict_stage_eval(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    """Save holdout metrics manifest, threshold sweep, and feature importance (if missing)."""
    logger.info("Predict stage=eval: metrics + threshold_sweep + feature_importance")
    run_predict_stage_metrics(spark, run_id, feature_run_id, paths, args)
    run_predict_stage_threshold_sweep(spark, run_id, feature_run_id, paths, args)
    run_backfill_feature_importance(spark, run_id, feature_run_id, paths, args)
    logger.info("Predict stage=eval complete for exp_id=%s", run_id)


def run_predict_all(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    """Deprecated one-shot holdout pipeline. Delegates to features → predict → eval."""
    logger.warning(PREDICT_STAGE_ALL_DEPRECATED_MSG)
    run_predict_stage_features(spark, run_id, feature_run_id, paths, args)
    run_predict_stage_predict(spark, run_id, feature_run_id, paths, args)
    if args.dry_run:
        logger.info("DRY RUN: skipping eval (predict stage did not checkpoint a prediction Delta)")
        return
    run_predict_stage_eval(spark, run_id, feature_run_id, paths, args)


def run_predict_only(spark, run_id: str, feature_run_id: str, paths: dict[str, str], args: argparse.Namespace) -> None:
    stage = args.predict_stage
    if stage == "features":
        run_predict_stage_features(spark, run_id, feature_run_id, paths, args)
    elif stage == "predict":
        run_predict_stage_predict(spark, run_id, feature_run_id, paths, args)
    elif stage == "metrics":
        run_predict_stage_metrics(spark, run_id, feature_run_id, paths, args)
    elif stage == "threshold_sweep":
        run_predict_stage_threshold_sweep(spark, run_id, feature_run_id, paths, args)
    elif stage == "feature_importance":
        run_backfill_feature_importance(spark, run_id, feature_run_id, paths, args)
    elif stage == "eval":
        run_predict_stage_eval(spark, run_id, feature_run_id, paths, args)
    else:
        run_predict_all(spark, run_id, feature_run_id, paths, args)


def main() -> None:
    parser = pipeline_cli.build_parser(description="Load saved models and evaluate val/test/oot holdout splits")
    args = parser.parse_args()
    args.predict_only = True
    pipeline_cli.validate_inference_args(parser, args)
    pipeline_cli.configure_logging(args)

    exp_id, feature_run_id, gold_run_id, paths, components = pipeline_cli.resolve_run_context(args)
    pipeline_cli.log_run_banner(args, exp_id, feature_run_id, gold_run_id, mode="inference")

    spark = create_pipeline_spark_session("gold-model-inference")
    run_predict_only(spark, exp_id, feature_run_id, paths, args)
    logger.info("Holdout pipeline complete for exp_id=%s (stage=%s)", exp_id, args.predict_stage)


if __name__ == "__main__":
    main()
