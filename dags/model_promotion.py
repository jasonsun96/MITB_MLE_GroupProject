import datetime
import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG

IMAGE_NAME = "document_topic_tagger"

R2_ENV = {
    "R2_ACCOUNT_ID": os.environ.get("R2_ACCOUNT_ID", ""),
    "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
    "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    "PYTHONPATH": "/app:/app/include:/app/include/gold",
    "SPARK_MASTER": "local[2]",
    "SPARK_DRIVER_MEMORY": "8g",
}

COMMON = dict(
    image=IMAGE_NAME,
    environment=R2_ENV,
    auto_remove="force",
    docker_url="unix://var/run/docker.sock",
    mount_tmp_dir=False,
)

with DAG(
    dag_id="model_promotion",
    start_date=datetime.datetime(2005, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "alias": "shadow",
        "exp_id": "",
        "model_type": "logistic_regression",
        "model_date": "",
        "prediction_threshold": "",
        "actor": "airflow",
        "reason": "manual promotion",
        "sample_size": 3,
    },
):
    validate_candidate = DockerOperator(
        task_id="validate_candidate",
        command=(
            "python include/inference/model_registry.py validate "
            "--exp-id '{{ params.exp_id }}' "
            "--model-type '{{ params.model_type }}' "
            "{% if params.model_date %}--model-date '{{ params.model_date }}' {% endif %}"
            "{% if params.prediction_threshold|string %}--prediction-threshold '{{ params.prediction_threshold }}' {% endif %}"
        ),
        execution_timeout=datetime.timedelta(hours=1),
        **COMMON,
    )

    wait_for_manual_approval = DockerOperator(
        task_id="wait_for_manual_approval",
        command=("python -c \"raise SystemExit('Manual approval required: " "review validate_candidate, then mark this task successful to continue')\""),
        execution_timeout=datetime.timedelta(minutes=5),
        **COMMON,
    )

    update_alias = DockerOperator(
        task_id="update_alias",
        command=(
            "python include/inference/model_registry.py set "
            "--alias '{{ params.alias }}' "
            "--exp-id '{{ params.exp_id }}' "
            "--model-type '{{ params.model_type }}' "
            "{% if params.model_date %}--model-date '{{ params.model_date }}' {% endif %}"
            "{% if params.prediction_threshold|string %}--prediction-threshold '{{ params.prediction_threshold }}' {% endif %}"
            "--actor '{{ params.actor }}' "
            "--reason '{{ params.reason }}'"
        ),
        execution_timeout=datetime.timedelta(hours=1),
        **COMMON,
    )

    run_smoke_test = DockerOperator(
        task_id="run_smoke_test",
        command=("python include/inference/promotion_smoke_test.py " "--alias '{{ params.alias }}' " "--sample-size '{{ params.sample_size }}'"),
        execution_timeout=datetime.timedelta(hours=1),
        **COMMON,
    )

    validate_candidate >> wait_for_manual_approval >> update_alias >> run_smoke_test
