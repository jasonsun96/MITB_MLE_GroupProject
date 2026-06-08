import datetime
import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG

IMAGE_NAME = "document_topic_tagger"

R2_ENV = {
    "R2_ACCOUNT_ID": os.environ.get("R2_ACCOUNT_ID", ""),
    "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
    "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    "PYTHONPATH": "/app",
}

with DAG(
    dag_id="medallion_pipeline",
    start_date=datetime.datetime(2024, 1, 1),
    schedule="@monthly",
):
    bronze_ingest = DockerOperator(
        task_id="ingest_bronze",
        image=IMAGE_NAME,
        command="python include/bronze/ingest_bronze.py --start-date {{ ds }} --end-date {{ ds }}",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    silver_process_legal_docs = DockerOperator(
        task_id="process_silver_legal_docs",
        image=IMAGE_NAME,
        command="python include/silver/process_legal_docs.py",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    silver_process_wiki_docs = DockerOperator(
        task_id="process_silver_wiki_docs",
        image=IMAGE_NAME,
        command="python include/silver/process_wiki_docs.py",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    process_gold = DockerOperator(
        task_id="process_gold",
        image=IMAGE_NAME,
        command="python include/gold/sample_gold.py",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    bronze_ingest >> [silver_process_legal_docs, silver_process_wiki_docs] >> process_gold
