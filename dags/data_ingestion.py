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

COMMON = dict(
    image=IMAGE_NAME,
    environment=R2_ENV,
    auto_remove="force",
    docker_url="unix://var/run/docker.sock",
)

with DAG(
    dag_id="medallion_pipeline",
    start_date=datetime.datetime(2005, 1, 1),
    end_date=datetime.datetime(2005, 12, 31),
    schedule="@daily",
    catchup=False,
):
    # ---- bronze ----
    bronze_ingest = DockerOperator(
        task_id="ingest_bronze",
        command=(
            "python include/bronze/ingest_bronze.py "
            "--start-date {{ ds }} "
            "--end-date {{ ds }} "
            "--batch-id {{ ds_nodash }}"
        ),
        **COMMON,
    )

    # ---- silver ----
    silver_legal = DockerOperator(
        task_id="process_silver_legal_docs",
        command="python include/silver/process_legal_docs.py --snapshot-date {{ ds }}",
        **COMMON,
    )

    silver_wiki = DockerOperator(
        task_id="process_silver_wiki_docs",
        command="python include/silver/process_wiki_docs.py",
        **COMMON,
    )

    # ---- gold: label store (train/val/test/oot split assignment) ----
    build_label_store = DockerOperator(
        task_id="build_label_store",
        command="python include/gold/label_store.py --snapshot-date {{ ds }}",
        **COMMON,
    )

    # ---- gold: per-document features (all read from silver) ----
    gold_ngram_counts = DockerOperator(
        task_id="extract_ngram_counts",
        command="python include/gold/ngram_processing.py --snapshot-date {{ ds }}",
        **COMMON,
    )

    gold_pos_counts = DockerOperator(
        task_id="extract_pos_counts",
        command="python include/gold/pos_counts.py --snapshot-date {{ ds }}",
        **COMMON,
    )

    gold_legal_embeddings = DockerOperator(
        task_id="extract_legal_embeddings",
        command="python include/gold/legal_embeddings.py --snapshot-date {{ ds }}",
        **COMMON,
    )

    gold_pos_counts_wiki = DockerOperator(
        task_id="extract_pos_counts_wiki",
        command="python include/gold/wiki_pos_counts.py",
        **COMMON,
    )

    gold_wiki_embeddings = DockerOperator(
        task_id="extract_wiki_embeddings",
        command="python include/gold/wiki_embeddings.py",
        **COMMON,
    )

    # ---- dependencies ----
    # gold jobs read SILVER tables, so they must run after their silver task
    bronze_ingest >> [silver_legal, silver_wiki]
    silver_legal >> [build_label_store, gold_ngram_counts, gold_pos_counts, gold_legal_embeddings]
    silver_wiki >> [gold_pos_counts_wiki, gold_wiki_embeddings]

    # ---- not yet wired (pending label_store/labels table + column alignment) ----
    # tfidf_processing.py        << [gold_ngram_counts, build_label_store]
    # domain_concept_weight.py   << [gold_pos_counts, gold_pos_counts_wiki, build_label_store]
    # model_training.py          << [tfidf, dcw, gold_legal_embeddings, build_label_store]
