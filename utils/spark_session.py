import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

_JVM_OPENS = " ".join(
    [
        "-XX:+IgnoreUnrecognizedVMOptions",
        "-Dio.netty.tryReflectionSetAccessible=true",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.base/sun.misc=ALL-UNNAMED",
    ]
)

_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "4g")

os.environ["PYSPARK_SUBMIT_ARGS"] = f"--driver-memory {_DRIVER_MEMORY} " f"--driver-java-options='{_JVM_OPENS}' " f"--conf spark.executor.extraJavaOptions='{_JVM_OPENS}' " "pyspark-shell"


def create_spark_session(app_name, log_level="ERROR"):
    ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
    ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
    SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
    R2_ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

    master = os.environ.get("SPARK_MASTER", "local[*]")

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.extraJavaOptions", _JVM_OPENS)
        .config("spark.executor.extraJavaOptions", _JVM_OPENS)
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

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    return spark
