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
notebooks/eda.ipynb           Bronze EDA (quality checks + token analysis)
utils/spark_session.py        Spark + Delta + R2 session helper
docs/ngram_tfidf.md           Silver → gold pipeline and preprocessing reference
schema.yaml                   Table paths and layer configuration
Dockerfile                    Runtime image used by DAG tasks
docker-compose.yml            Local Airflow stack
```

## Model Deployment Aliases

Trained model artifacts are immutable under `model_bank/runs/{run_id}` in R2.
Deployment aliases assign the `production` and `shadow` roles without moving
or copying those artifacts. Each alias update also creates an immutable event
under the configured `model_bank.deployment_history` path.

Set an alias from the project runtime image:

```bash
python include/inference/model_registry.py set \
  --alias shadow \
  --run-id 20260613T120000Z \
  --actor airflow \
  --reason "June shadow evaluation"
```

Read the current alias:

```bash
python include/inference/model_registry.py get --alias shadow
```

The referenced `model_bank/runs/{run_id}` must already exist. Batch jobs should
resolve aliases once at startup and persist the resolved run IDs in their batch
manifest.

## Batch Shadow Inference

The `batch_inference_pipeline` DAG performs batch inference with optional shadow
scoring.
It expects an inference-ready Delta table at `gold/inference_features` with one
row per `document_id` and a Spark ML vector column named `features`. A different
input path can be supplied when manually triggering the DAG.

The DAG first assembles deployment-specific feature tables. Production features
are written to `gold/batch_inference/{batch_id}/features/production`; when a
distinct shadow model is enabled, shadow features are written to
`gold/batch_inference/{batch_id}/features/shadow`. Each table is built from that
model's own manifest, so frozen TF-IDF/IDF and DCW statistics come from the
correct `feature_run_id`.

On its first scoring attempt, the DAG reads the `production` and optional
`shadow` model configs and freezes their feature input paths and Delta versions
in a manifest. Retries reuse that manifest, then score the full batch with
production and, when configured, score the same full batch with shadow.

The manifest records both the inference input path and its Delta table version.
Inference retries read that exact version with `versionAsOf`, so later changes
to the input table do not change the batch.

Before scoring, the job validates every referenced model run. It checks metadata
and label mappings, model paths, run IDs, thresholds, label counts, loadability,
and that each deployment's feature table was assembled with the feature run its
model expects. Loaded models are reused for scoring.

Trigger it with an optional input path:

```json
{
  "input_path": "s3a://cs611-project/gold/inference_features"
}
```

The `production` alias is required. The `shadow` alias is optional. When no
shadow is set, or when it points to the same run as production, only production
is scored.

Outputs are written under:

```text
gold/batch_inference/{batch_id}/manifest.json
gold/batch_inference/{batch_id}/predictions
gold/batch_inference/{batch_id}/validation.json
gold/published_predictions
```

Prediction rows include `batch_id`, `document_id`, `model_run_id`,
`deployment_group`, `predicted_labels`, and the prediction timestamp. Staged
batch predictions include production rows and, when configured, shadow rows.
The canonical `gold/published_predictions` table receives production rows only.
This DAG does not promote the shadow alias; promotion remains a separate
deployment decision.

Before predictions are written, the job rejects empty inputs, null or duplicate
document IDs, missing or duplicate predictions, incorrect model assignments,
invalid labels, and incorrect production/shadow counts. A successful
`validation.json` records the input, prediction, production, and shadow row
counts while leaving the frozen manifest unchanged.

After validation, the job publishes into the canonical
`gold/published_predictions` Delta table. The first run creates the table;
later runs use a Delta `MERGE` keyed by `(batch_id, document_id)`, so retrying
the same Airflow batch updates its rows instead of creating duplicates.

Monitoring reads staged shadow rows from `gold/batch_inference/{batch_id}/predictions`
and plots shadow performance alongside production when reviewed labels are
available.

Inference Airflow tasks retry twice with exponential backoff: the first retry
waits five minutes and delays are capped at 30 minutes. Batch inference has a
four-hour execution timeout.

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
