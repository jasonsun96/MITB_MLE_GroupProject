# MLE Project

Airflow orchestrates a medallion pipeline that reads landing data from Cloudflare R2, writes bronze Delta tables, then runs silver and gold processing jobs.

The DAG is defined in `dags/pipeline.py`. Each task uses Airflow's `DockerOperator` to run a Python script from this repository inside the `document_topic_tagger` Docker image.

## Prerequisites

- Docker and Docker Compose
- Cloudflare R2 bucket credentials
- A local `.env` file with the required credentials

## Configuration

Create `.env` in the project root:

```bash
R2_ACCOUNT_ID=<your-account-id>
R2_ACCESS_KEY_ID=<your-access-key-id>
R2_SECRET_ACCESS_KEY=<your-secret-access-key>
RS_BUCKET=<your-bucket-name>
```

The `.env` file is excluded from the Docker image by `.dockerignore`. Docker Compose passes it into the Airflow services at runtime.

## Start Airflow

Build the project image and initialize Airflow:

```bash
docker compose up airflow-init
```

Start the Airflow services:

```bash
docker compose up
```

Open Airflow at:

```text
http://localhost:8080
```

Default local login:

```text
Username: airflow
Password: airflow
```

The main DAG is `medallion_pipeline`.

## Project Layout

```text
dags/pipeline.py              Airflow DAG
include/bronze/               Bronze ingestion jobs
include/silver/               Silver processing jobs
include/gold/                 Gold processing jobs
notebooks/                    Jupyter notebooks for exploration and prototyping
utils/spark_session.py        Spark + Delta + R2 session helper
schema.yaml                   Table paths and layer configuration
Dockerfile                    Runtime image used by DAG tasks
docker-compose.yml            Local Airflow stack
```

## Jupyter Notebook

A JupyterLab server runs alongside Airflow inside the same `document_topic_tagger` image, so notebooks have full access to PySpark, Delta Lake, and R2 credentials.

Start the stack as usual (`docker compose up`) and open:

```text
http://localhost:8888
```

By default no token is required (the port is bound to `127.0.0.1` only). To require a token, set `JUPYTER_TOKEN` in your `.env` file. The notebook working directory is the project root, so any file under `notebooks/` is editable from both the browser and your local filesystem. New notebooks should be saved under `notebooks/`.

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

## Add A Silver Job

Create a new Python file under `include/silver/`.

Example:

```python
import logging
from pathlib import Path

import yaml

from utils.spark_session import create_spark_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

spark = create_spark_session("my-silver-job")

bronze_path = f"{schema['bronze']['path']}/{schema['bronze']['tables']['legal_docs_raw']['path']}"
silver_path = f"{schema['silver']['path']}/{schema['silver']['tables']['legal_docs_processed']['path']}"

df = spark.read.format("delta").load(bronze_path)

# Add your transformations here.
processed = df

processed.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(silver_path)

logger.info("Silver job complete")
```

If the job writes a new table, add that table to `schema.yaml`:

```yaml
silver:
  path: "s3a://cs611-project/silver"
  tables:
    my_new_silver_table:
      path: my_new_silver_table
      partition_col: snapshot_date
```

Then add the job to `dags/pipeline.py`:

```python
my_silver_job = DockerOperator(
    task_id="my_silver_job",
    image=IMAGE_NAME,
    command="python include/silver/my_silver_job.py",
    environment=R2_ENV,
    auto_remove="force",
    docker_url="unix://var/run/docker.sock",
)
```

Attach it to the pipeline dependency chain:

```python
bronze_ingest >> my_silver_job >> process_gold
```

## Add A Gold Job

Create a new Python file under `include/gold/`.

Example:

```python
import logging
from pathlib import Path

import yaml

from utils.spark_session import create_spark_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

spark = create_spark_session("my-gold-job")

silver_path = f"{schema['silver']['path']}/{schema['silver']['tables']['legal_docs_processed']['path']}"
gold_path = f"{schema['gold']['path']}/{schema['gold']['tables']['my_gold_table']['path']}"

df = spark.read.format("delta").load(silver_path)

# Add your aggregations or final transformations here.
gold = df

gold.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(gold_path)

logger.info("Gold job complete")
```

If the job writes a new table, add that table to `schema.yaml`:

```yaml
gold:
  path: "s3a://cs611-project/gold"
  tables:
    my_gold_table:
      path: my_gold_table
```

Then add the job to `dags/pipeline.py`:

```python
my_gold_job = DockerOperator(
    task_id="my_gold_job",
    image=IMAGE_NAME,
    command="python include/gold/my_gold_job.py",
    environment=R2_ENV,
    auto_remove="force",
    docker_url="unix://var/run/docker.sock",
)
```

Attach it after the silver job it depends on:

```python
silver_process_legal_docs >> my_gold_job
```

## DAG Notes

- `IMAGE_NAME` in `dags/pipeline.py` must match the image built by Docker Compose: `document_topic_tagger`.
- The scheduler mounts `/var/run/docker.sock` so `DockerOperator` can start task containers.
- Airflow passes R2 credentials through `R2_ENV`.
- The DAG currently runs monthly and uses `{{ ds }}` as the execution date for bronze ingestion.
- After adding or changing Python jobs, rebuild the project image:

```bash
docker compose build document-topic-tagger
docker compose up
```

## Validate Changes

Check the Compose file:

```bash
docker compose config
```

Check that the DAG imports cleanly inside the Airflow image:

```bash
docker compose run --rm airflow-scheduler airflow dags list
```

Run a task from the Airflow UI or trigger the `medallion_pipeline` DAG manually.
