"""
Airflow DAG: retrain a champion experiment and run the staged predict/eval pipeline.

See docs/model_training.md §12.
"""

from __future__ import annotations

import datetime

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG

from docker_common import docker_operator_kwargs

CHAMPION_EXPERIMENT = {
    "exp_id": "exp004_LR_tfidf_dcw_gs",
    "feature_run_id": "run001",
    "gold_run_id": "run004",
    "feature_set": "tfidf_dcw",
    "model_type": "logistic_regression",
    "grid_search": True,
}

_SPARK = docker_operator_kwargs(spark=True)
_EXP = CHAMPION_EXPERIMENT
_GRID = (
    "--grid-search --grid-search-metric micro_f1 "
    if _EXP["grid_search"]
    else ""
)
_TRAIN_CMD = (
    "python include/model_pipeline/model_training.py "
    f"--exp-id {_EXP['exp_id']} "
    f"--feature-run-id {_EXP['feature_run_id']} "
    f"--gold-run-id {_EXP['gold_run_id']} "
    f"--feature-set {_EXP['feature_set']} "
    f"--model-type {_EXP['model_type']} "
    f"{_GRID}"
    "--train-only"
)
_PREDICT_FLAGS = (
    f"--exp-id {_EXP['exp_id']} "
    f"--feature-run-id {_EXP['feature_run_id']} "
    f"--gold-run-id {_EXP['gold_run_id']} "
    f"--feature-set {_EXP['feature_set']} "
    f"--model-type {_EXP['model_type']} "
    "--predict-only"
)

with DAG(
    dag_id="model_retrain_champion",
    description="Train champion model on R2 features, score holdout, write metrics/sweep/FI",
    start_date=datetime.datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["ml", "training", "champion"],
    doc_md=__doc__,
) as dag:
    train = DockerOperator(
        task_id="train_model",
        command=_TRAIN_CMD,
        **_SPARK,
    )

    assemble_holdout_x = DockerOperator(
        task_id="predict_stage_features",
        command=f"python include/model_pipeline/model_inference.py {_PREDICT_FLAGS} --predict-stage features",
        **_SPARK,
    )

    score_holdout = DockerOperator(
        task_id="predict_stage_predict",
        command=(
            f"python include/model_pipeline/model_inference.py {_PREDICT_FLAGS} "
            "--predict-stage predict --prediction-date {{ ds }}"
        ),
        **_SPARK,
    )

    evaluate = DockerOperator(
        task_id="predict_stage_eval",
        command=(
            f"python include/model_pipeline/model_inference.py {_PREDICT_FLAGS} "
            "--predict-stage eval --prediction-date {{ ds }}"
        ),
        **_SPARK,
    )

    train >> assemble_holdout_x >> score_holdout >> evaluate
