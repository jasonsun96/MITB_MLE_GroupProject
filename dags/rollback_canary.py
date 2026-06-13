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

with DAG(
    dag_id="rollback_canary_pipeline",
    start_date=datetime.datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    params={"batch_id": ""},
):
    rollback_canary = DockerOperator(
        task_id="rollback_canary",
        image=IMAGE_NAME,
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        command=(
            "python include/inference/rollback_canary.py "
            "--batch-id '{{ params.batch_id }}'"
        ),
        execution_timeout=datetime.timedelta(hours=2),
    )
