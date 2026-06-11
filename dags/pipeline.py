import datetime
import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG

IMAGE_NAME = "document_topic_tagger"

R2_ENV = {
    "R2_ACCOUNT_ID": os.environ.get("R2_ACCOUNT_ID", ""),
    "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
    "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
}

with DAG(
    dag_id="medallion_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@monthly",
):
    ingest_bronze = DockerOperator(
        task_id="ingest_bronze",
        image=IMAGE_NAME,
        command="python include/bronze/ingest_bronze.py --start-date {{ ds }} --end-date {{ ds }}",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    process_silver = DockerOperator(
        task_id="process_silver",
        image=IMAGE_NAME,
        command="python include/silver/sample_silver.py",
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

    extract_pos_counts = DockerOperator(
        task_id="extract_pos_counts",
        image=IMAGE_NAME,
        command="python include/gold/pos_counts.py",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    extract_pos_counts_wiki = DockerOperator(
        task_id="extract_pos_counts_wiki",
        image=IMAGE_NAME,
        command="python include/gold/wiki_pos_counts.py",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    extract_legal_embeddings = DockerOperator(
        task_id="extract_legal_embeddings",
        image=IMAGE_NAME,
        command="python include/gold/legal_embeddings.py",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    extract_wiki_embeddings = DockerOperator(
        task_id="extract_wiki_embeddings",
        image=IMAGE_NAME,
        command="python include/gold/wiki_embeddings.py",
        environment=R2_ENV,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
    )

    ingest_bronze >> process_silver >> process_gold
    ingest_bronze >> extract_pos_counts
    ingest_bronze >> extract_pos_counts_wiki
    ingest_bronze >> extract_legal_embeddings
    ingest_bronze >> extract_wiki_embeddings
