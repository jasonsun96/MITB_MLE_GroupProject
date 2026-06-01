import logging
import os

from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def process_bronze_table(table_name, table_config, source_path, bronze_path, spark, snapshot_date_str=None, file_prefix=None, filename=None):
    try:
        if file_prefix and snapshot_date_str:
            date_nodash = snapshot_date_str.replace("-", "")
            resolved_filename = f"{file_prefix}{date_nodash}.csv"
        elif filename:
            resolved_filename = filename
        else:
            resolved_filename = f"{table_name}.csv"
        source_file_path = os.path.join(source_path, resolved_filename)
        df = spark.read.csv(source_file_path, header=True, inferSchema=False, multiLine=True, maxCharsPerColumn=10000000, maxColumns=100000)

        if snapshot_date_str is not None:
            df = df.withColumn("snapshot_date", F.lit(snapshot_date_str))

        if snapshot_date_str is not None and not file_prefix:
            df = df.filter(F.col("snapshot_date") == snapshot_date_str)

        partition_col = table_config["partition_col"]
        output_path = os.path.join(bronze_path, table_config["table_dir"])

        writer = df.write.format("delta").option("mergeSchema", "true")
        if snapshot_date_str is not None:
            writer.partitionBy(partition_col).mode("overwrite").option("replaceWhere", f"snapshot_date = '{snapshot_date_str}'").save(output_path)
        else:
            writer.mode("overwrite").save(output_path)

        logger.info(f"Processed {table_name} for {snapshot_date_str}. Bronze table written to {output_path}")

    except Exception as e:
        logger.error(e)
