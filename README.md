# MLE Project

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- A Cloudflare R2 account with a bucket and API credentials

## Configuration

Copy `.env.example` to `.env` (or create `.env`) and fill in your R2 credentials:

```
R2_ACCOUNT_ID=<your-account-id>
R2_ACCESS_KEY_ID=<your-access-key-id>
R2_SECRET_ACCESS_KEY=<your-secret-access-key>
RS_BUCKET=<your-bucket-name>
```

> The `.env` file is excluded from the Docker image via `.dockerignore` and must be passed at runtime (see below).

## Build

```bash
docker build -t mle-bronze .
```

## Run

Start an interactive shell inside the container, mounting your `.env` at runtime:

```bash
docker run --rm -it --env-file .env mle-bronze
```

Once inside the container, run the pipeline:

```bash
python main.py
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--log-level` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `--start-date` | from `schema.yaml` | Backfill start date (`YYYY-MM-DD`) |
| `--end-date` | from `schema.yaml` | Backfill end date (`YYYY-MM-DD`) |

Example — backfill a specific date range with debug logging:

```bash
python main.py --start-date 2024-01-01 --end-date 2024-03-31 --log-level DEBUG
```
