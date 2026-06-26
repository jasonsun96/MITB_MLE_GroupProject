# Legal Document Topic Tagger

This project builds a Spark + Airflow machine-learning pipeline for legal document topic classification. It ingests raw legal and wiki documents, cleans and enriches them through bronze/silver/gold Delta layers, trains multi-label classifiers, runs batch inference, and monitors production predictions.

Airflow orchestrates the pipeline. DAG tasks use Airflow's `DockerOperator` to run Python scripts from this repository inside the `document_topic_tagger` Docker image.

## Prerequisites

- Docker and Docker Compose
- A local `.env` file with the required credentials

## Configuration

Obtain the `.env` file from the Group Project Code and Report submission on eLearn and place into the root of the repository.

The `.env` file is excluded from the Docker image by `.dockerignore`. Docker Compose passes it into the Airflow services at runtime.

## Docker And Airflow Quickstart

Build the project runtime image:

```bash
docker compose build document-topic-tagger
```

Initialize the Airflow database:

```bash
docker compose up airflow-init
```

Open:

```text
http://localhost:8080
```

Login to Airflow with the following credentials:

```text
Username: airflow
Password: airflow
```

## Airflow Configuration

There are three main DAGs within Airflow.

`medallion_pipeline` runs the data ingestion and cleaning/transformation through
the medallion architecture.

`model_promotion` controls the deployment of production and shadow model aliases
and keeps track of model deployment history.

`batch_inference_pipeline` assembles features necessary for inference and generates
legal topic labels using production model. It also computes model performance and data
drift metrics.



## Jupyter Notebook

A JupyterLab server runs alongside Airflow inside the same `document_topic_tagger` image, so notebooks have full access to PySpark, Delta Lake, and R2 credentials.

Start the stack as usual (`docker compose up`) and open:

```text
http://localhost:8888
```

Inside a notebook you can use the shared Spark session helper exactly as the pipeline jobs do:

```python
from utils.spark_session import create_spark_session

spark = create_spark_session("notebook-exploration")
df = spark.read.format("delta").load("s3a://cs611-project/bronze/legal_docs_raw")
df.printSchema()
df.show(5, truncate=120)
```

To rebuild the image after adding dependencies to `requirements.txt`:

```bash
docker compose build document-topic-tagger
docker compose up -d jupyter
```
