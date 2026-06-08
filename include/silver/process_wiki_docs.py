import argparse
import html
import logging
from pathlib import Path

import yaml
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from utils.spark_session import create_spark_session

parser = argparse.ArgumentParser(description="Silver layer wiki document processing pipeline")
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

spark = create_spark_session("process-wiki-docs")

TEXT_COL = "text"
CLEAN_TEXT_COL = "text_clean"
PLACEHOLDER_PATTERN = r"(?i)^\s*(nan|null|none|na|n/a)\s*$"
HTML_ENTITY_PATTERN = r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);"
HTML_TAG_PATTERN = r"(?i)<\/?[A-Za-z][^>]{0,200}>"
WIKI_TEMPLATE_PATTERN = r"\{\{[^{}\n]{0,500}\}\}"
WIKI_TABLE_LINE_PATTERN = r"(?m)^\s*[!|][^\n]*$"
WIKI_FILE_LINK_PATTERN = r"\[\[(?:File|Image):[^\]]+\]\]"
WIKI_CATEGORY_LINK_PATTERN = r"\[\[Category:[^\]]+\]\]"
WIKI_LINK_PATTERN = r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]"
EXTERNAL_LINK_PATTERN = r"\[(https?://[^\s\]]+)(?:\s+([^\]]+))?\]"
REFERENCE_TAG_PATTERN = r"(?is)<ref\b[^>]*>.*?<\/ref>|<ref\b[^\/>]*/>"
NBSP_PATTERN = "\u00a0"
SOFT_HYPHEN_PATTERN = "\u00ad"
ZERO_WIDTH_PATTERN = r"[\u200b\u200e\u200f\ufeff]"
JOINER_CHARACTER_PATTERN = r"[\u200c\u200d]"
CONTROL_CHAR_PATTERN = r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]"
MOJIBAKE_MARKER_PATTERN = "(\u00c3.|\u00c2.|\u00e2\u20ac.|\u00ef\u00bb\u00bf)"
LONG_TEXT_OUTLIER_LENGTH = 7307


def decode_html_entities_value(value):
    if value is None:
        return None
    return html.unescape(value)


decode_html_entities = F.udf(decode_html_entities_value, T.StringType())


def initialize_clean_text(df):
    return df.withColumn(CLEAN_TEXT_COL, F.col(TEXT_COL))


def add_structural_quality_flags(df):
    return (
        df.withColumn("is_id_missing", F.col("id").isNull())
        .withColumn("is_id_malformed", F.col("id").isNotNull() & ~F.col("id").rlike(r"^\d+$"))
        .withColumn("is_url_missing", F.col("url").isNull())
        .withColumn("is_url_malformed", F.col("url").isNotNull() & ~F.col("url").rlike(r"^https?://"))
        .withColumn("is_title_missing", F.col("title").isNull())
    )


def add_missing_and_placeholder_text_flags(df):
    return (
        df.withColumn("is_text_null", F.col(TEXT_COL).isNull())
        .withColumn("is_text_empty", F.col(TEXT_COL).isNotNull() & (F.length(F.trim(F.col(TEXT_COL))) == 0))
        .withColumn("is_text_placeholder", F.col(TEXT_COL).rlike(PLACEHOLDER_PATTERN))
    )


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


def add_wiki_markup_flags(df):
    return (
        df.withColumn(
            "has_html_or_ref_tags",
            F.col(CLEAN_TEXT_COL).isNotNull() & (F.col(CLEAN_TEXT_COL).rlike(HTML_TAG_PATTERN) | F.col(CLEAN_TEXT_COL).rlike(REFERENCE_TAG_PATTERN)),
        )
        .withColumn(
            "has_wiki_templates",
            F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(WIKI_TEMPLATE_PATTERN),
        )
        .withColumn(
            "has_wiki_table_markup",
            F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(WIKI_TABLE_LINE_PATTERN),
        )
        .withColumn(
            "has_wiki_links",
            F.col(CLEAN_TEXT_COL).isNotNull()
            & (
                F.col(CLEAN_TEXT_COL).rlike(WIKI_FILE_LINK_PATTERN)
                | F.col(CLEAN_TEXT_COL).rlike(WIKI_CATEGORY_LINK_PATTERN)
                | F.col(CLEAN_TEXT_COL).rlike(WIKI_LINK_PATTERN)
                | F.col(CLEAN_TEXT_COL).rlike(EXTERNAL_LINK_PATTERN)
            ),
        )
    )


def remove_html_and_reference_tags(df):
    cleaned = F.regexp_replace(F.col(CLEAN_TEXT_COL), REFERENCE_TAG_PATTERN, " ")
    cleaned = F.regexp_replace(cleaned, HTML_TAG_PATTERN, " ")
    return df.withColumn(CLEAN_TEXT_COL, cleaned)


def remove_wiki_templates(df):
    cleaned = F.col(CLEAN_TEXT_COL)
    for _ in range(3):
        cleaned = F.regexp_replace(cleaned, WIKI_TEMPLATE_PATTERN, " ")
    return df.withColumn(CLEAN_TEXT_COL, cleaned)


def remove_wiki_table_lines(df):
    return df.withColumn(CLEAN_TEXT_COL, F.regexp_replace(F.col(CLEAN_TEXT_COL), WIKI_TABLE_LINE_PATTERN, " "))


def clean_wiki_links(df):
    cleaned = F.regexp_replace(F.col(CLEAN_TEXT_COL), WIKI_FILE_LINK_PATTERN, " ")
    cleaned = F.regexp_replace(cleaned, WIKI_CATEGORY_LINK_PATTERN, " ")
    cleaned = F.regexp_replace(cleaned, WIKI_LINK_PATTERN, "$1")
    cleaned = F.regexp_replace(cleaned, EXTERNAL_LINK_PATTERN, "$2")
    return df.withColumn(CLEAN_TEXT_COL, cleaned)


def add_unicode_artifact_flags(df):
    return (
        df.withColumn("has_nbsp", F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).contains(NBSP_PATTERN))
        .withColumn("has_soft_hyphen", F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).contains(SOFT_HYPHEN_PATTERN))
        .withColumn(
            "has_zero_width_artifacts",
            F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(ZERO_WIDTH_PATTERN),
        )
        .withColumn(
            "has_joiner_characters",
            F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(JOINER_CHARACTER_PATTERN),
        )
        .withColumn(
            "has_control_characters",
            F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(CONTROL_CHAR_PATTERN),
        )
        .withColumn("has_mojibake_markers", F.col(CLEAN_TEXT_COL).isNotNull() & F.col(CLEAN_TEXT_COL).rlike(MOJIBAKE_MARKER_PATTERN))
    )


def normalize_unicode_artifacts(df):
    cleaned = F.regexp_replace(F.col(CLEAN_TEXT_COL), NBSP_PATTERN, " ")
    cleaned = F.regexp_replace(cleaned, SOFT_HYPHEN_PATTERN, "")
    # Keep joiner characters used by Indic scripts; remove spacing/direction artifacts only.
    cleaned = F.regexp_replace(cleaned, ZERO_WIDTH_PATTERN, "")
    cleaned = F.regexp_replace(cleaned, CONTROL_CHAR_PATTERN, "")
    return df.withColumn(CLEAN_TEXT_COL, cleaned)


def trim_section_heading_spaces(df):
    return df.withColumn(CLEAN_TEXT_COL, F.regexp_replace(F.col(CLEAN_TEXT_COL), r"[ \t]+(?=\n)", ""))


def normalize_whitespace(df):
    cleaned = F.regexp_replace(F.col(CLEAN_TEXT_COL), r"[ \t]{2,}", " ")
    cleaned = F.regexp_replace(cleaned, r"(?m)^[ \t]+|[ \t]+$", "")
    cleaned = F.regexp_replace(cleaned, r"\n{3,}", "\n\n")
    cleaned = F.regexp_replace(cleaned, r"^\s+|\s+$", "")
    return df.withColumn(CLEAN_TEXT_COL, cleaned)


def add_length_outlier_flags(df):
    return (
        df.withColumn("raw_text_length", F.length(F.col(TEXT_COL)))
        .withColumn("clean_text_length", F.length(F.col(CLEAN_TEXT_COL)))
        .withColumn("is_short_text", F.col("clean_text_length").isNotNull() & (F.col("clean_text_length") <= 100))
        .withColumn("is_long_text_outlier", F.col("clean_text_length") >= LONG_TEXT_OUTLIER_LENGTH)
    )


def add_duplicate_text_flags(df):
    clean_text_hash = F.when(
        F.col(CLEAN_TEXT_COL).isNull() | (F.length(F.trim(F.col(CLEAN_TEXT_COL))) == 0),
        F.lit(None),
    ).otherwise(F.sha2(F.lower(F.trim(F.col(CLEAN_TEXT_COL))), 256))
    duplicate_window = Window.partitionBy("clean_text_hash")

    return (
        df.withColumn("clean_text_hash", clean_text_hash)
        .withColumn(
            "duplicate_clean_text_count",
            F.when(F.col("clean_text_hash").isNull(), F.lit(0)).otherwise(F.count("*").over(duplicate_window)),
        )
        .withColumn("is_duplicate_clean_text", F.col("duplicate_clean_text_count") > 1)
    )


def add_cleaning_summary_flags(df):
    issue_columns = [
        "is_id_missing",
        "is_id_malformed",
        "is_url_missing",
        "is_url_malformed",
        "is_title_missing",
        "is_text_null",
        "is_text_empty",
        "is_text_placeholder",
        "has_html_entities",
        "has_html_or_ref_tags",
        "has_wiki_templates",
        "has_wiki_table_markup",
        "has_wiki_links",
        "has_nbsp",
        "has_soft_hyphen",
        "has_zero_width_artifacts",
        "has_joiner_characters",
        "has_control_characters",
        "has_mojibake_markers",
        "is_short_text",
        "is_long_text_outlier",
        "is_duplicate_clean_text",
    ]

    has_quality_issue = None
    for column_name in issue_columns:
        column_expr = F.coalesce(F.col(column_name), F.lit(False))
        has_quality_issue = column_expr if has_quality_issue is None else has_quality_issue | column_expr

    return df.withColumn("has_text_quality_issue", has_quality_issue)


def process_wiki_docs(bronze_table_path, silver_table_path, spark):
    df = spark.read.format("delta").load(bronze_table_path)

    processed = df
    processed = initialize_clean_text(df)
    processed = add_structural_quality_flags(processed)
    processed = add_missing_and_placeholder_text_flags(processed)
    processed = add_html_entity_flag(processed)
    processed = decode_html_entities_in_text(processed)
    processed = add_wiki_markup_flags(processed)
    processed = remove_html_and_reference_tags(processed)
    processed = remove_wiki_templates(processed)
    processed = remove_wiki_table_lines(processed)
    processed = clean_wiki_links(processed)
    processed = add_unicode_artifact_flags(processed)
    processed = normalize_unicode_artifacts(processed)
    processed = trim_section_heading_spaces(processed)
    processed = normalize_whitespace(processed)
    processed = add_length_outlier_flags(processed)
    processed = add_duplicate_text_flags(processed)
    processed = add_cleaning_summary_flags(processed)

    try:
        writer = processed.write.format("delta").option("mergeSchema", "true")
        writer.mode("overwrite").save(silver_table_path)

    except Exception:
        logger.exception("Failed to process wiki silver table from %s", bronze_table_path)
        raise


bronze_table_path = f"{BRONZE_PATH}/{BRONZE_TABLES['wiki_docs_raw']['path']}"
silver_table_path = f"{SILVER_PATH}/{SILVER_TABLES['wiki_docs_processed']['path']}"
process_wiki_docs(bronze_table_path, silver_table_path, spark)


logger.info("Wiki silver processing complete")
