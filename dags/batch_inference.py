import datetime
import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG

IMAGE_NAME = "document_topic_tagger"

DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": datetime.timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": datetime.timedelta(minutes=30),
}

R2_ENV = {
    "R2_ACCOUNT_ID": os.environ.get("R2_ACCOUNT_ID", ""),
    "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
    "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    "PYTHONPATH": "/app",
}

COMMON = dict(
    image=IMAGE_NAME,
    environment=R2_ENV,
    auto_remove="force",
    docker_url="unix://var/run/docker.sock",
)

with DAG(
    dag_id="batch_inference_pipeline",
    start_date=datetime.datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    params={
        "canary_percentage": 5.0,
        "feature_run_id": "run004",
        "gold_run_id": "run004",
        "feature_set": "tfidf_dcw",
        "input_path": "",
    },
):
    assemble_inference_features = DockerOperator(
        task_id="assemble_inference_features",
        command=(
            "python include/model_pipeline/assemble_inference_features.py "
            "--batch-id {{ ds_nodash }} "
            "--feature-run-id {{ params.feature_run_id }} "
            "--gold-run-id {{ params.gold_run_id }} "
            "--feature-set {{ params.feature_set }}"
        ),
        execution_timeout=datetime.timedelta(hours=4),
        **COMMON,
    )

    run_batch_inference = DockerOperator(
        task_id="run_batch_inference",
        command=(
            "python include/inference/batch_inference.py "
            "--batch-id {{ ds_nodash }} "
            "--canary-percentage {{ params.canary_percentage }} "
            "{% if params.input_path %}"
            "--input-path '{{ params.input_path }}'"
            "{% else %}"
            "--input-path 's3a://cs611-project/gold/runs/{{ params.gold_run_id }}/X_unlabelled'"
            "{% endif %}"
        ),
        execution_timeout=datetime.timedelta(hours=4),
        **COMMON,
    )

    # Monitoring runs on the same batch (same ds_nodash batch-id) once predictions
    # are published, computing performance / PSI / CSI and writing the dashboards to
    # monitoring/{batch_id}/ on R2.
    run_monitoring = DockerOperator(
        task_id="run_monitoring",
        command="python include/monitoring/monitoring.py --batch-id {{ ds_nodash }}",
        execution_timeout=datetime.timedelta(hours=1),
        **COMMON,
    )

    assemble_inference_features >> run_batch_inference >> run_monitoring
