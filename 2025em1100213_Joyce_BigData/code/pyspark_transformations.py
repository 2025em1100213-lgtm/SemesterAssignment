"""PySpark Transformations and Actions — Average Engine Temperature Pipeline.

Student: Joyce Dorothy S (2025em1100213)

Ingests the real vehicle telemetry extract (vehicle_telemetry_dataset.csv),
filters null engine_temp readings, and computes the average engine
temperature per vehicle model. Each transformation is annotated with its
dependency type (narrow vs wide) and whether it is lazy or eager.

Run:
    python pyspark_transformations.py
"""

import os
import warnings

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


SCHEMA = StructType([
    StructField("vehicle_id", StringType(), True),
    StructField("vehicle_model", StringType(), True),
    StructField("engine_temp", DoubleType(), True),
    StructField("speed", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("battery_efficiency", DoubleType(), True),
    StructField("miles", DoubleType(), True),
])


def create_spark_session() -> SparkSession:
    """Initialize a SparkSession configured for local execution."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("TelemetryTransformations")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_telemetry_data(spark: SparkSession, csv_path: str):
    """Read telemetry CSV into a Spark DataFrame with an explicit schema.

    TRANSFORMATION (lazy): reading a CSV defines a DataFrame but does not
    immediately load all data into memory. Spark records this as the first
    node in the logical plan. An explicit schema (rather than
    inferSchema=True) avoids a second pass over the file for type inference.
    """
    return spark.read.csv(csv_path, header=True, schema=SCHEMA)


def filter_null_temperatures(df):
    """Remove records where engine_temp is null.

    TRANSFORMATION (lazy): filter() builds a new logical plan node but does
    not execute until an action is called.
    NARROW DEPENDENCY: each output partition depends on exactly one input
    partition. No shuffle is required — the filter is applied independently
    to each partition.
    """
    return df.filter(F.col("engine_temp").isNotNull())


def compute_avg_engine_temperature(df):
    """Group by vehicle model and compute mean engine temperature.

    TRANSFORMATION (lazy): groupBy() + agg() define a logical aggregation
    plan; no computation runs until an action triggers execution.
    WIDE DEPENDENCY: groupBy() requires a shuffle — records for the same
    vehicle_model may reside on different partitions and must be moved
    (shuffled) to the same partition for aggregation. This creates a new
    stage boundary in the DAG.
    """
    return (
        df.groupBy("vehicle_model")
        .agg(
            F.avg("engine_temp").alias("avg_engine_temp"),
            F.count("*").alias("n_records"),
        )
    )


def demonstrate_pipeline(spark: SparkSession, csv_path: str) -> None:
    """Execute the full pipeline with intermediate outputs for demonstration."""
    print("=" * 70)
    print("  PySpark Telemetry Pipeline: Average Engine Temperature")
    print("=" * 70)

    print("\n[Step 1] Loading telemetry data from CSV...")
    raw_df = load_telemetry_data(spark, csv_path)

    print("\n--- Raw DataFrame Schema ---")
    raw_df.printSchema()

    # ACTION (eager): count() triggers full DAG execution to count all rows.
    raw_count = raw_df.count()
    print(f"Total records loaded: {raw_count:,}")

    # ACTION (eager): show() triggers computation and displays rows.
    print("\n--- Sample Raw Data (first 5 rows) ---")
    raw_df.show(5, truncate=False)

    print("\n[Step 2] Filtering out records with null engine_temp...")
    filtered_df = filter_null_temperatures(raw_df)
    filtered_count = filtered_df.count()
    print(f"Records after filtering: {filtered_count:,}")
    print(f"Null records removed:    {raw_count - filtered_count:,}")

    print("\n[Step 3] Computing average engine temperature per vehicle model...")
    avg_temp_df = compute_avg_engine_temperature(filtered_df)

    print("\n--- Average Engine Temperature by Vehicle Model ---")
    avg_temp_df.orderBy("vehicle_model").show(truncate=False)

    print("\n--- Dependency Classification Summary ---")
    print(f"{'Operation':32}{'Dependency':14}{'Evaluation'}")
    print("-" * 62)
    print(f"{'spark.read.csv(...)':32}{'source':14}{'lazy (transformation)'}")
    print(f"{'.filter(isNotNull)':32}{'NARROW':14}{'lazy (transformation)'}")
    print(f"{'.groupBy(...).agg(avg(...))':32}{'WIDE':14}{'lazy (transformation)'}")
    print(f"{'.count() / .show()':32}{'-':14}{'eager (ACTION)'}")

    print("\n" + "=" * 70)
    print(f"  Vehicle models found: {avg_temp_df.count()}")
    print("=" * 70)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "vehicle_telemetry_dataset.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Telemetry CSV not found at: {csv_path}\n"
            "Place vehicle_telemetry_dataset.csv in the same folder as this script."
        )

    spark = create_spark_session()
    try:
        demonstrate_pipeline(spark, csv_path)
    finally:
        spark.stop()
        print("\nSparkSession stopped. Pipeline complete.")
