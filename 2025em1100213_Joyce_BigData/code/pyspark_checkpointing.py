"""PySpark Checkpointing for Long Lineage — Iterative Processing Pipeline.

Student: Joyce Dorothy S (2025em1100213)

Demonstrates how iterative transformations create increasingly long lineage
chains, and how strategic checkpointing truncates the lineage to prevent
StackOverflow errors and bound recovery time. Applies 15 successive narrow
transformations to the real telemetry DataFrame, checkpointing every 5th
iteration, and prints the measured lineage depth before/after each
checkpoint.

Run:
    python pyspark_checkpointing.py
"""

import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession, DataFrame
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
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("TelemetryCheckpointing")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def configure_checkpoint_directory(spark: SparkSession) -> str:
    """Set the checkpoint directory to a local temp path.

    In production this would point to a reliable distributed filesystem
    (HDFS, S3, ADLS) so checkpoints survive driver and node failures. A local
    temp directory is used here for demonstration only.
    """
    checkpoint_dir = tempfile.mkdtemp(prefix="telemetry_checkpoint_")
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    print(f"[Config] Checkpoint directory: {checkpoint_dir}")
    print("[Config] Note: production systems use HDFS/S3 for reliability.\n")
    return checkpoint_dir


def load_telemetry_data(spark: SparkSession, csv_path: str) -> DataFrame:
    return spark.read.csv(csv_path, header=True, schema=SCHEMA).filter(
        F.col("engine_temp").isNotNull()
    )


def lineage_depth(df: DataFrame) -> int:
    """Proxy for lineage length: number of nodes in the logical query plan.

    We use the Catalyst *logical* plan (via the internal JVM query-execution
    handle) rather than df.rdd.toDebugString(), because Spark's whole-stage
    code generation fuses consecutive narrow physical RDD operations into a
    single MapPartitionsRDD -- which makes the RDD-level debug string an
    unreliable proxy for how many transformations are actually chained. The
    logical plan, in contrast, retains one node per transformation exactly as
    written, so its depth grows and resets exactly where expected.
    """
    plan_str = df._jdf.queryExecution().logical().toString()
    return len([line for line in plan_str.split("\n") if line.strip()])


def apply_transformation(df: DataFrame, iteration: int) -> DataFrame:
    """One step of a simulated iterative workload.

    Cycles through narrow-dependency transformations (filter, withColumn
    arithmetic, conditional withColumn) to demonstrate lineage growth without
    the confound of shuffles. All three are NARROW dependencies -- they
    operate partition-locally with no data movement across the network.
    """
    kind = iteration % 3
    if kind == 0:
        return df.withColumn("engine_temp", F.col("engine_temp") + F.lit(0.01))
    elif kind == 1:
        return df.withColumn(
            "engine_temp",
            F.when(F.col("engine_temp") > 100, F.col("engine_temp")).otherwise(
                F.col("engine_temp") * 1.001
            ),
        )
    else:
        return df.filter(F.col("engine_temp") > 0)


def run_iterative_processing(
    df: DataFrame, num_iterations: int = 15, checkpoint_interval: int = 5
) -> DataFrame:
    """Apply num_iterations transformations, checkpointing every N-th step."""
    print(f"{'iter':>4}  {'lineage_depth':>14}  action")
    print("-" * 40)
    print(f"{0:>4}  {lineage_depth(df):>14}  start")

    for i in range(1, num_iterations + 1):
        df = apply_transformation(df, i)
        if i % checkpoint_interval == 0:
            # ACTION-like (eager materialization): checkpoint() writes the
            # DataFrame to reliable storage NOW and truncates the recorded
            # lineage -- this DataFrame becomes a new DAG root.
            df = df.checkpoint()
            print(f"{i:>4}  {lineage_depth(df):>14}  ** checkpoint -- lineage truncated **")
        else:
            print(f"{i:>4}  {lineage_depth(df):>14}  transform (no checkpoint)")

    print(f"\nFinal record count after {num_iterations} iterations: {df.count():,}")
    return df


def print_comparison_summary() -> None:
    print("\n" + "=" * 70)
    print("  Checkpoint vs Cache/Persist — Strategy Comparison")
    print("=" * 70)
    print("""
    +-------------------+---------------------------+---------------------------+
    | Dimension         | checkpoint()              | cache() / persist()      |
    +-------------------+---------------------------+---------------------------+
    | Lineage           | TRUNCATED (new root)      | RETAINED (full history)  |
    | Storage           | Reliable (HDFS/S3)        | Memory / local disk      |
    | Fault Tolerance   | High (survives crashes)   | Low (needs recompute)    |
    | Performance       | Slower (disk I/O)         | Faster (RAM speed)       |
    | Best For          | Iterative ML loops        | Multi-action queries     |
    +-------------------+---------------------------+---------------------------+

    Risks of long, un-truncated lineage chains:
      - StackOverflow: JVM stack exceeded during recursive DAG traversal
      - Slow recovery: a lost partition requires replaying the entire chain
      - Driver memory bloat: DAG metadata grows linearly with chain length

    Recommendation for the telemetry platform:
      -> Checkpoint every 5-10 iterations during iterative predictive-
         maintenance model training (bounds recovery time, avoids
         StackOverflow)
      -> Cache intermediate DataFrames reused across multiple actions within
         a single job (avoids redundant recomputation)
      -> The two techniques solve different problems and are complementary,
         not interchangeable
    """)


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
        configure_checkpoint_directory(spark)

        print("[Data] Loading telemetry data...")
        df = load_telemetry_data(spark, csv_path)
        print(f"[Data] Loaded {df.count():,} telemetry records.\n")

        result_df = run_iterative_processing(df, num_iterations=15, checkpoint_interval=5)

        print("\n[Result] Final DataFrame sample after 15 iterations:")
        result_df.show(5, truncate=True)

        print_comparison_summary()
    finally:
        spark.stop()
        print("\nSparkSession stopped. Checkpointing demonstration complete.")
