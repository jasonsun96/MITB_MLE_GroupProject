import datetime
import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG

IMAGE_NAME = "document_topic_tagger"
BATCH_ID_TEMPLATE = "{{ logical_date.in_timezone('Asia/Singapore').format('YYYYMMDD') }}"

R2_ENV = {
    "R2_ACCOUNT_ID": os.environ.get("R2_ACCOUNT_ID", ""),
    "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
    "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    "PYTHONPATH": "/app:/app/include:/app/include/gold",
}

COMMON = dict(
    image=IMAGE_NAME,
    environment=R2_ENV,
    auto_remove="force",
    docker_url="unix://var/run/docker.sock",
    mount_tmp_dir=False,
)

with DAG(
    dag_id="batch_inference_pipeline",
    start_date=datetime.datetime(2027, 1, 1),
    schedule="@monthly",
    catchup=False,
    max_active_runs=1,
    params={
        "feature_config": "config/batch_inference.yaml",
        "input_path": "",
    },
):
    assemble_inference_features = DockerOperator(
        task_id="assemble_inference_features",
        command=(
            "{% if params.input_path %}"
            "python -c \"print('Skipping assemble_inference_features: input_path provided')\""
            "{% else %}"
            "python include/model_pipeline/assemble_inference_features.py "
            f"--batch-id {BATCH_ID_TEMPLATE} "
            "--config {{ params.feature_config }}"
            "{% endif %}"
        ),
        execution_timeout=datetime.timedelta(hours=4),
        **COMMON,
    )

    run_batch_inference = DockerOperator(
        task_id="run_batch_inference",
        command=(
            "python include/inference/batch_inference.py "
            f"--batch-id {BATCH_ID_TEMPLATE} "
            "--feature-config {{ params.feature_config }} "
            "{% if params.input_path %}"
            "--input-path '{{ params.input_path }}'"
            "{% endif %}"
        ),
        execution_timeout=datetime.timedelta(hours=4),
        **COMMON,
    )

    # Monitoring runs on the same batch once production predictions are published.
    # If shadow predictions were staged, monitoring adds shadow performance to the
    # dashboard without publishing shadow outputs.
    run_monitoring = DockerOperator(
        task_id="run_monitoring",
        command=(
            "python include/monitoring/monitoring.py "
            f"--batch-id {BATCH_ID_TEMPLATE} "
            "--feature-config {{ params.feature_config }} "
            "{% if params.input_path %}"
            "--input-path '{{ params.input_path }}'"
            "{% endif %}"
        ),
        execution_timeout=datetime.timedelta(hours=1),
        **COMMON,
    )

    assemble_inference_features >> run_batch_inference >> run_monitoring
