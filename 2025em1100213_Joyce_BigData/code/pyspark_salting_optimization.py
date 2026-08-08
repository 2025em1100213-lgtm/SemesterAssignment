"""PySpark Salting Strategy & Partitioning — Data Skew Mitigation.

Student: Joyce Dorothy S (2025em1100213)

Demonstrates the two-phase salting technique for mitigating data skew in
distributed aggregation. Uses the real telemetry extract (moderate ~2.3x
natural skew across vehicle models) and additionally synthesizes a severely
amplified copy of the dataset (several-hundred-x skew) to validate the
technique under conditions matching the assignment's stated "1000x more
logs" scenario.

Two-phase aggregation pattern:
    Phase 1: Aggregate by salted keys (distributes load evenly)
    Phase 2: Remove salt and aggregate partial results into the final answer

Run:
    python pyspark_salting_optimization.py
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

SALT_RANGE = 8  # number of salt buckets; tuned to executor core count


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("TelemetrySaltingOptimization")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_telemetry_data(spark: SparkSession, csv_path: str):
    return spark.read.csv(csv_path, header=True, schema=SCHEMA).filter(
        F.col("engine_temp").isNotNull()
    )


def show_data_skew(df) -> dict:
    """Display record distribution per vehicle_model and return counts."""
    print("\n" + "=" * 70)
    print("  DATA SKEW ANALYSIS (natural skew in the real extract)")
    print("=" * 70)

    total = df.count()
    print(f"\n  Total records: {total:,}\n")

    skew_df = (
        df.groupBy("vehicle_model")
        .agg(F.count("*").alias("record_count"))
        .orderBy(F.col("record_count").desc())
    )
    skew_df.show(truncate=False)

    counts = {r["vehicle_model"]: r["record_count"] for r in skew_df.collect()}
    ratio = max(counts.values()) / min(counts.values())
    print(f"  Observed skew ratio (largest / smallest): {ratio:.2f}x\n")
    return counts


def add_salted_key(df, salt_range: int = SALT_RANGE):
    """Prepend a random integer prefix to the group key.

    TRANSFORMATION (lazy), NARROW DEPENDENCY: each row's salt is computed
    independently — no shuffle required to add this column.
    """
    return df.withColumn(
        "salted_key",
        F.concat(
            F.floor(F.rand() * F.lit(salt_range)).cast("string"),
            F.lit("_"),
            F.col("vehicle_model"),
        ),
    )


def phase1_aggregate(salted_df):
    """Partial aggregation (sum, count) grouped by salted key.

    TRANSFORMATION (lazy), WIDE DEPENDENCY: groupBy on salted_key still
    shuffles, but load is now spread across salt_range x more distinct keys,
    so a previously hot vehicle_model no longer floods one partition.

    We compute SUM and COUNT rather than AVG directly, because averaging
    partial averages is mathematically incorrect — the correct approach
    carries (sum, count) forward and divides once, in Phase 2.
    """
    return salted_df.groupBy("salted_key").agg(
        F.sum("engine_temp").alias("partial_sum"),
        F.count("engine_temp").alias("partial_count"),
    )


def remove_salt(partial_agg_df):
    """Extract the original vehicle_model by stripping the salt prefix.

    TRANSFORMATION (lazy), NARROW DEPENDENCY: string manipulation is
    row-local, no shuffle needed.
    """
    return partial_agg_df.withColumn(
        "vehicle_model", F.element_at(F.split(F.col("salted_key"), "_"), 2)
    )


def phase2_aggregate(desalted_df):
    """Final aggregation: merge partials across all salt buckets per model.

    TRANSFORMATION (lazy), WIDE DEPENDENCY: a much smaller second shuffle —
    only salt_range x n_models rows are being grouped, not the full dataset.
    """
    return (
        desalted_df.groupBy("vehicle_model")
        .agg(
            (F.sum("partial_sum") / F.sum("partial_count")).alias("avg_engine_temp"),
            F.sum("partial_count").alias("n_records"),
        )
    )


def compute_direct_average(df):
    """Direct (un-salted) groupBy average — used only to verify correctness."""
    return df.groupBy("vehicle_model").agg(
        F.avg("engine_temp").alias("avg_engine_temp")
    )


def partition_sizes(df):
    """Row count per physical partition — used to visualize skew before/after."""
    return df.rdd.glom().map(len).collect()


def demonstrate_salting_pipeline(spark: SparkSession, csv_path: str) -> None:
    print("=" * 70)
    print("  PySpark Salting Optimization: Two-Phase Aggregation")
    print("=" * 70)

    df = load_telemetry_data(spark, csv_path)
    show_data_skew(df)

    # --- Phase 1: salted key + partial aggregation --------------------------
    print("=" * 70)
    print("  PHASE 1: Salting — Adding Random Prefix to Keys")
    print("=" * 70)
    salted_df = add_salted_key(df)
    salted_df.select("vehicle_model", "salted_key", "engine_temp").show(10, truncate=False)

    partial_agg_df = phase1_aggregate(salted_df)
    partial_agg_df.orderBy("salted_key").show(20, truncate=False)

    # --- Salt removal ---------------------------------------------------------
    print("=" * 70)
    print("  SALT REMOVAL & PHASE 2: Final Average by Original Key")
    print("=" * 70)
    desalted_df = remove_salt(partial_agg_df)
    final_df = phase2_aggregate(desalted_df).orderBy("vehicle_model")
    final_df.show(truncate=False)

    # --- Verification against direct aggregation -----------------------------
    print("=" * 70)
    print("  VERIFICATION: Salted Result vs Direct Aggregation")
    print("=" * 70)
    direct_df = compute_direct_average(df).orderBy("vehicle_model")
    direct_df.show(truncate=False)

    salted_results = {r["vehicle_model"]: round(r["avg_engine_temp"], 4) for r in final_df.collect()}
    direct_results = {r["vehicle_model"]: round(r["avg_engine_temp"], 4) for r in direct_df.collect()}
    all_match = all(
        abs(salted_results[m] - direct_results[m]) < 0.01 for m in direct_results
    )
    print(f"  All results match direct aggregation: {all_match}\n")

    # --- Severe-skew validation (matching the assignment's "1000x" scenario) -
    print("=" * 70)
    print("  SEVERE-SKEW VALIDATION (synthetic amplification)")
    print("=" * 70)
    counts = {
        r["vehicle_model"]: r["record_count"]
        for r in df.groupBy("vehicle_model").agg(F.count("*").alias("record_count")).collect()
    }
    hot_model = max(counts, key=counts.get)
    hot_rows_df = df.filter(F.col("vehicle_model") == hot_model)

    REPLICATION_FACTOR = 180
    skewed_df = df
    for _ in range(REPLICATION_FACTOR):
        skewed_df = skewed_df.unionByName(hot_rows_df)
    skewed_df = skewed_df.repartition(8)

    skew_counts = {
        r["vehicle_model"]: r["n"]
        for r in skewed_df.groupBy("vehicle_model").agg(F.count("*").alias("n")).collect()
    }
    print(f"  Synthetic skew ratio: {max(skew_counts.values()) / min(skew_counts.values()):.0f}x")

    before_key_df = skewed_df.groupBy("vehicle_model").agg(F.count("*").alias("n"))
    print(f"  Partition sizes AFTER groupBy on RAW key:    {partition_sizes(before_key_df)}")

    salted_skewed = add_salted_key(skewed_df)
    partial_skewed = phase1_aggregate(salted_skewed)
    print(f"  Partition sizes AFTER groupBy on SALTED key: {partition_sizes(partial_skewed)}")

    final_skewed = phase2_aggregate(remove_salt(partial_skewed)).orderBy("vehicle_model")
    final_skewed.show(truncate=False)

    print("=" * 70)
    print("  PARTITIONING STRATEGY: Hash Partitioning (chosen) vs Range Partitioning")
    print("=" * 70)
    print("""
  Hash partitioning distributes records by hash(key) % num_partitions,
  giving uniform partition sizes independent of key ordering -- ideal for
  groupBy/reduceByKey aggregation, which is exactly this workload. Combined
  with salting to widen the effective key space for the hot key, it directly
  neutralizes skew.

  Range partitioning distributes records by contiguous key ranges. It
  benefits ordered scans and range queries, but offers no defense against a
  single dominant key -- a popular vehicle model still forms one contiguous
  "range," so it does not solve this aggregation's skew problem. Range
  partitioning would be preferred elsewhere in this platform (e.g. the raw
  ingestion lake partitioned by timestamp for time-window queries), but not
  for this groupBy(vehicle_model) aggregation.
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
        demonstrate_salting_pipeline(spark, csv_path)
    finally:
        spark.stop()
        print("\nSparkSession stopped. Salting optimization demo complete.")
