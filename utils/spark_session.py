import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def create_spark_session(app_name, log_level="ERROR"):
    ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
    ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
    SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
    R2_ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", R2_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", SECRET_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    )
    spark = configure_spark_with_delta_pip(
        builder,
        extra_packages=[
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        ],
    ).getOrCreate()
    spark.sparkContext.setLogLevel(log_level)

    return spark
