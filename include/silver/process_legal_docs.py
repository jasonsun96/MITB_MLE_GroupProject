import argparse
import html
import logging
import re
from pathlib import Path

import yaml
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Silver layer legal document processing pipeline")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "schema.yaml") as f:
    schema = yaml.safe_load(f)

BRONZE = schema["bronze"]
BRONZE_PATH = BRONZE["path"]
BRONZE_TABLES = BRONZE["tables"]
SILVER = schema["silver"]
SILVER_PATH = SILVER["path"]
SILVER_TABLES = SILVER["tables"]

spark = create_spark_session("sample-silver")

TEXT_COL = "act_raw_text"
CLEAN_TEXT_COL = "act_clean_text"
MERGED_CSV_RECORD_PATTERN = r"[\r\n]+\s*\"?[0-9][0-9A-Z()]{7,20},"
MOJIBAKE_MARKER_PATTERN = r"(Ã.|Â.|â€.|ðŸ|ï»¿|[\u0080-\u009f])"
CONTROL_CHAR_PATTERN = r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]"
HTML_ENTITY_PATTERN = r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);"
HTML_RESIDUE_PATTERN = r"(?i)<\/?(?:html|body|table|tbody|thead|tr|td|th|p|br|div|span|blk\d+)[^>]*>"
EXTRACTION_MARKUP_PATTERN = r'(?i)</?\(?BLK\d+\)?[A-Z0-9]*>|class=""page""'
PLACEHOLDER_PATTERN = r"(?i)^\s*(nan|null|none|na|n/a|character\(0\))\s*$"


def repair_mojibake_value(value):
    if value is None:
        return None

    repaired = value
    for _ in range(2):
        marker_count = len(re.findall(MOJIBAKE_MARKER_PATTERN, repaired))
        if marker_count == 0:
            break

        try:
            candidate = repaired.encode("latin1").decode("utf-8")
        except UnicodeError:
            break

        candidate_marker_count = len(re.findall(MOJIBAKE_MARKER_PATTERN, candidate))
        if candidate_marker_count >= marker_count:
            break

        repaired = candidate

    return repaired


def decode_html_entities_value(value):
    if value is None:
        return None
    return html.unescape(value)


repair_mojibake = F.udf(repair_mojibake_value, T.StringType())
decode_html_entities = F.udf(decode_html_entities_value, T.StringType())


def initialize_clean_text(df):
    return df.withColumn(CLEAN_TEXT_COL, F.col(TEXT_COL))


def add_missing_and_placeholder_flags(df):
    return (
        df.withColumn("is_text_null", F.col(TEXT_COL).isNull())
        .withColumn("is_text_empty", F.col(TEXT_COL).isNotNull() & (F.length(F.trim(F.col(TEXT_COL))) == 0))
        .withColumn("is_text_placeholder", F.col(TEXT_COL).rlike(PLACEHOLDER_PATTERN))
    )


def add_merged_csv_record_flag(df):
    return df.withColumn(
        "has_merged_csv_record",
        F.col(TEXT_COL).isNotNull() & F.col(TEXT_COL).rlike(MERGED_CSV_RECORD_PATTERN),
    )


def add_mojibake_flags(df):
    return df.withColumn(
        "has_mojibake",
        F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(MOJIBAKE_MARKER_PATTERN),
    )


def repair_text_encoding(df):
    return df.withColumn(
        CLEAN_TEXT_COL,
        F.when(F.col(CLEAN_TEXT_COL).isNull(), F.lit(None)).otherwise(repair_mojibake(F.col(CLEAN_TEXT_COL))),
    ).withColumn("encoding_repaired", F.col(TEXT_COL).isNotNull() & (F.col(TEXT_COL) != F.col(CLEAN_TEXT_COL)))


def add_html_entity_flag(df):
    return df.withColumn(
        "has_html_entities",
        F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(HTML_ENTITY_PATTERN),
    )


def decode_html_entities_in_text(df):
    return df.withColumn(
        CLEAN_TEXT_COL,
        F.when(F.col(CLEAN_TEXT_COL).isNull(), F.lit(None)).otherwise(decode_html_entities(F.col(CLEAN_TEXT_COL))),
    )


def add_html_residue_flag(df):
    return df.withColumn(
        "has_html_residue",
        F.col(CLEAN_TEXT_COL).isNotNull() & (F.col(CLEAN_TEXT_COL).rlike(HTML_RESIDUE_PATTERN) | F.col(CLEAN_TEXT_COL).rlike(EXTRACTION_MARKUP_PATTERN)),
    )


def remove_html_residue(df):
    return df.withColumn(
        CLEAN_TEXT_COL,
        F.regexp_replace(F.regexp_replace(F.col(CLEAN_TEXT_COL), HTML_RESIDUE_PATTERN, " "), EXTRACTION_MARKUP_PATTERN, " "),
    )


def add_soft_hyphen_flag(df):
    return df.withColumn(
        "has_soft_hyphen",
        F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).contains("\u00ad"),
    )


def remove_soft_hyphens(df):
    return df.withColumn(CLEAN_TEXT_COL, F.regexp_replace(F.col(CLEAN_TEXT_COL), "\u00ad", ""))


def add_literal_escape_flags(df):
    return df.withColumn(
        "has_literal_backslash_newline",
        F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(r"\\[rn]"),
    )


def repair_known_literal_escape_sequences(df):
    replacements = [
        (r"\\rticle", "Article"),
        (r"\\nnex", "Annex"),
        (r"\\rea", "Area"),
    ]

    cleaned = F.col(CLEAN_TEXT_COL)
    for pattern, replacement in replacements:
        cleaned = F.regexp_replace(cleaned, pattern, replacement)

    return df.withColumn(CLEAN_TEXT_COL, cleaned)


def add_control_character_flag(df):
    return df.withColumn(
        "has_control_characters",
        F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(CONTROL_CHAR_PATTERN),
    )


def remove_control_characters(df):
    return df.withColumn(CLEAN_TEXT_COL, F.regexp_replace(F.col(CLEAN_TEXT_COL), CONTROL_CHAR_PATTERN, ""))


def trim_and_normalize_whitespace(df):
    cleaned = F.regexp_replace(F.col(CLEAN_TEXT_COL), r"[ \t]{2,}", " ")
    cleaned = F.regexp_replace(cleaned, r"^\s+|\s+$", "")
    return df.withColumn(CLEAN_TEXT_COL, cleaned)


def add_length_outlier_flags(df):
    return (
        df.withColumn("raw_text_length", F.length(F.col(TEXT_COL)))
        .withColumn("clean_text_length", F.length(F.col(CLEAN_TEXT_COL)))
        .withColumn("is_short_text", F.col("clean_text_length").isNotNull() & (F.col("clean_text_length") <= 100))
        .withColumn("is_long_text_outlier", F.col("clean_text_length") >= 53305)
    )


def add_duplicate_text_flags(df):
    normalized_text_hash = F.sha2(F.lower(F.trim(F.col(CLEAN_TEXT_COL))), 256)
    duplicate_window = Window.partitionBy("clean_text_hash")

    return (
        df.withColumn("clean_text_hash", normalized_text_hash)
        .withColumn("duplicate_clean_text_count", F.count("*").over(duplicate_window))
        .withColumn(
            "is_duplicate_clean_text",
            F.col("clean_text_hash").isNotNull() & (F.col("duplicate_clean_text_count") > 1),
        )
    )


def add_cleaning_summary_flags(df):
    issue_columns = [
        "is_text_null",
        "is_text_empty",
        "is_text_placeholder",
        "has_merged_csv_record",
        "has_mojibake",
        "has_html_entities",
        "has_html_residue",
        "has_soft_hyphen",
        "has_literal_backslash_newline",
        "has_control_characters",
        "is_short_text",
        "is_long_text_outlier",
        "is_duplicate_clean_text",
    ]

    has_quality_issue = None
    for column_name in issue_columns:
        column_expr = F.coalesce(F.col(column_name), F.lit(False))
        has_quality_issue = column_expr if has_quality_issue is None else has_quality_issue | column_expr

    return df.withColumn("has_text_quality_issue", has_quality_issue)


def process_legal_docs(bronze_table_path, silver_table_path, spark):
    df = spark.read.format("delta").load(bronze_table_path)

    processed = df
    processed = initialize_clean_text(processed)
    processed = add_missing_and_placeholder_flags(processed)
    processed = add_merged_csv_record_flag(processed)
    processed = add_soft_hyphen_flag(processed)
    processed = remove_soft_hyphens(processed)
    processed = add_mojibake_flags(processed)
    processed = repair_text_encoding(processed)
    processed = add_html_entity_flag(processed)
    processed = decode_html_entities_in_text(processed)
    processed = add_html_residue_flag(processed)
    processed = remove_html_residue(processed)
    processed = add_literal_escape_flags(processed)
    processed = repair_known_literal_escape_sequences(processed)
    processed = add_control_character_flag(processed)
    processed = remove_control_characters(processed)
    processed = trim_and_normalize_whitespace(processed)
    processed = add_length_outlier_flags(processed)
    processed = add_duplicate_text_flags(processed)
    processed = add_cleaning_summary_flags(processed)

    try:
        writer = processed.write.format("delta").option("mergeSchema", "true")
        writer.mode("overwrite").save(silver_table_path)

    except Exception:
        logger.exception("Failed to process silver table from %s", bronze_table_path)
        raise


bronze_table_path = f"{BRONZE_PATH}/{BRONZE_TABLES['legal_docs_raw']['path']}"
silver_table_path = f"{SILVER_PATH}/{SILVER_TABLES['legal_docs_processed']['path']}"
process_legal_docs(bronze_table_path, silver_table_path, spark)


logger.info("Silver processing complete")
