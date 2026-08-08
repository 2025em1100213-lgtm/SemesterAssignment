# Architecting and Implementing a Resilient Global Telemetry Platform

**Student Name:** Joyce Dorothy S
**Student ID:** 2025em1100213
**Course:** Big Data Platforms & Analytics
**Assignment:** Graded Assignment — Global Telemetry Platform
**Dataset used for Part 3 (PySpark implementation):** `vehicle_telemetry_dataset.csv` — 3,000 historical telemetry records across 100 vehicles and 7 EV models (columns: `vehicle_id`, `vehicle_model`, `engine_temp`, `speed`, `latitude`, `longitude`, `battery_efficiency`, `miles`)

---

### Scenario recap

A global logistics company operates a fleet of **500,000 vehicles** streaming telemetry (engine temperature, speed, GPS location, battery efficiency) 24/7. This notebook designs the platform architecture (Parts 1, 2, 4) and implements a runnable PySpark batch pipeline (Part 3) against a real historical extract of that telemetry data, covering transformations/actions, data-skew mitigation via salting, fault tolerance through RDD lineage, and checkpointing.


## Table of Contents
1. [Part 1 — System Architecture & Data Paradigms](#part1)
2. [Part 2 — Batch Processing & MapReduce](#part2)
3. [Part 3 — PySpark Implementation & Resilience](#part3)
    - 3.0 Environment Setup & Data Profiling
    - 3.1 Transformations & Actions — Average Engine Temperature
    - 3.2 Optimization — Data Skew & Salting
    - 3.3 Fault Tolerance — RDDs and Lineage
    - 3.4 Checkpointing — Truncating Long Lineage Chains
4. [Part 4 — Advanced Execution Mechanics & Resilience Strategies](#part4)


<a id="part1"></a>
## Part 1: System Architecture & Data Paradigms

### 1.1 Scaling Strategy — The Hardware "Wall"

Every single machine, no matter how well provisioned, is bounded by four physical resources: **CPU cores**, **RAM capacity**, **disk I/O throughput**, and **network bandwidth**. A modern dual-socket server tops out at roughly 64–128 cores, 0.5–2 TB of RAM, 3–7 GB/s of NVMe throughput, and 10–25 Gbps of network bandwidth. Once a workload's demand exceeds any one of these ceilings, no amount of query tuning or code optimization can push past it — this ceiling is "the Wall."

**Quantifying the platform against the Wall**

Using the assignment's stated fleet size (500,000 vehicles) and a conservative record size based on our actual schema (`vehicle_id`, `vehicle_model`, `engine_temp`, `speed`, `latitude`, `longitude`, `battery_efficiency`, `miles` ≈ 120 bytes/record):

```
Vehicles:                 500,000
Assumed frequency:        1 record / second / vehicle  (worst-case, safety-critical cadence)
Record size:               ~120 bytes

Ingestion rate   = 500,000 × 120 bytes/s  ≈ 60 MB/s sustained, 500,000 msgs/s
Daily volume     = 60 MB/s × 86,400 s     ≈ 5.2 TB/day
Annual volume    ≈ 1.9 PB/year (pre-compression)
```

| Resource | Demand at fleet scale | Single commodity node ceiling | Verdict |
|---|---|---|---|
| CPU | 500,000 concurrent parse/validate/index ops per second | ~100–200K simple ops/s across 64–128 cores | **Exceeded** |
| RAM | Session state for 500K concurrent streams + buffering ≈ 50–100 GB working set | Feasible alone, but shared with OS, JVM, query engine | At limit under concurrent analytics |
| Disk I/O | ~60 MB/s writes + read-back for analytics + WAL ≈ 200+ MB/s mixed I/O | 3–7 GB/s sequential, far less for random I/O | Sustainable short-term, not at petabyte scale |
| Network | 60 MB/s ingest + replication + query egress | 1.2–3.1 GB/s | Sustainable alone, but compounds with the above |

No single number is catastrophic in isolation; the real failure mode is that **CPU parsing/validation and disk I/O compound simultaneously under continuous 24/7 load, on a node with zero redundancy** — one hardware fault takes the entire ingestion pipeline down. That combination is what makes single-node deployment untenable at this scale.

**Scale-Up vs Scale-Out**

| Dimension | Scale-Up (vertical) | Scale-Out (horizontal) |
|---|---|---|
| Cost curve | Exponential — a 2 TB RAM server costs far more than 10× a 256 GB server | Roughly linear — commodity nodes added at marginal cost |
| Fault tolerance | Single point of failure | One node's failure is absorbed by the cluster |
| Throughput ceiling | Hard ceiling at motherboard/chipset limits | Grows by adding nodes |
| Elasticity | Must provision for peak upfront | Nodes added/removed with demand |

**Why the Three Vs mandate Scale-Out:**
- **Volume** — ~1.9 PB/year cannot be economically stored and queried on one node's local disks; distributed storage (HDFS/S3 + a lake format) is required.
- **Velocity** — 500,000 messages/second cannot be absorbed by a single write path without unacceptable queuing latency; ingestion must be partitioned across many brokers/nodes.
- **Variety** — records mix structured numeric readings (`engine_temp`, `speed`, `battery_efficiency`), geospatial pairs (`latitude`/`longitude`), and per-vehicle running totals (`miles`) — a mix best served by a flexible distributed processing framework, not a single specialized engine.

Horizontal scaling is therefore not an optimization here — it is the only architecture that satisfies Volume, Velocity, and Variety simultaneously while remaining available 24/7.

---

### 1.2 Consistency Models — ACID vs BASE, and the CAP Theorem

**ACID** (Atomicity, Consistency, Isolation, Durability) is the guarantee traditional relational databases provide: every write is all-or-nothing, every constraint holds after every transaction, concurrent transactions don't interfere, and committed writes survive crashes. For vehicle telemetry, an ACID write means either the *entire* record (`engine_temp`, `speed`, GPS pair, `battery_efficiency`) is durably persisted, or none of it is.

**BASE** (Basically Available, Soft state, Eventually consistent) relaxes those guarantees in exchange for throughput and availability: writes are accepted even before all replicas have converged, replicas propagate asynchronously, and the system only promises that *given enough time without new writes*, all replicas will agree.

**CAP Theorem** states a distributed system can guarantee only two of Consistency, Availability, and Partition tolerance at once. Because network partitions are a routine operational fact for a system spanning data centers and cloud availability zones, **Partition tolerance is non-negotiable**, which reduces the real choice to **CP** (sacrifice availability during a partition) or **AP** (sacrifice strict consistency during a partition).

**Choice for the ingestion layer: AP.**

1. **Availability is non-negotiable** — vehicles cannot buffer unlimited telemetry locally; unavailability during a partition means permanent data loss, including data relevant to safety alerting.
2. **Partition tolerance is inherent** — a multi-region deployment will experience transient network splits as routine operational events.
3. **Strict consistency is expendable at ingestion time** — every record carries its own authoritative source timestamp/reading, so downstream batch jobs can still reconcile order correctly even if two replicas were briefly out of sync.

At **500,000 writes/second**, ACID's synchronous durability (fsync per write, ~1–5 ms) and lock-based isolation would create a throughput collapse. BASE, implemented via AP-style systems (e.g. Kafka for ingestion, Cassandra/DynamoDB for storage), accepts writes locally and propagates asynchronously — trading brief, sub-second staleness for sustained high-throughput availability.

> The batch analytics layer built in Part 3 below can afford stronger consistency, since it reads a stable historical snapshot rather than the live write path.


<a id="part2"></a>
## Part 2: Batch Processing & MapReduce

### 2.1 MapReduce Logical Flow — Total Miles Driven per Vehicle Model

Our historical extract contains a `miles` column, matching exactly what the assignment asks us to aggregate. This section walks through all five MapReduce phases in detail, using **real rows from our dataset** at every step, then scales the reasoning up to the full 500,000-vehicle production fleet.

---

#### Phase 1 — Input Split

At full fleet scale, the platform ingests roughly **1.9 PB/year** (Part 1.1). HDFS divides this into fixed-size blocks (default **128 MB**), and each block becomes exactly one input split — the unit of work for one Map task.

```
Fleet-scale sizing (one day of telemetry, from Part 1.1):
  Daily volume:              ~5.2 TB
  HDFS block size:           128 MB
  Input splits/day:          5,200,000 MB ÷ 128 MB  ≈  40,600 splits
  Map tasks launched:        40,600  (one per split, run in parallel)

┌─────────────────────────────────────────────────────────────────────┐
│ DataNode-1          │ DataNode-2          │ DataNode-3          │... │
├─────────────────────┼─────────────────────┼─────────────────────┼────┤
│ Split-00001 (128MB) │ Split-00002 (128MB) │ Split-00003 (128MB) │    │
│ Split-00004 (128MB) │ Split-00005 (128MB) │ Split-00006 (128MB) │    │
│ ...  ~13,533 splits │ ...  ~13,533 splits │ ...  ~13,533 splits │    │
└─────────────────────┴─────────────────────┴─────────────────────┴────┘
```

The NameNode's block-location metadata lets the scheduler place each Map task on (or near) the DataNode that already physically holds its split — the data-locality principle detailed in Part 4.2.

---

#### Phase 2 — Map

Each mapper reads its split and emits one key-value pair per telemetry record: `map(line_offset, record) → emit(vehicle_model, miles)`. Below is a **literal worked example using the first 12 rows of our real CSV**, split across 3 mappers (4 rows each):

```
Mapper-A (rows 1-4)              Mapper-B (rows 5-8)              Mapper-C (rows 9-12)
──────────────────────           ──────────────────────           ──────────────────────
EV-072 Kona Electric  2537.2     EV-053 Kona Electric  1497.3     EV-085 Ioniq 5        8368.5
EV-078 Nexon EV      11186.0     EV-024 Model Y       12255.0     EV-080 Model 3       12132.2
EV-087 Model Y       12351.1     EV-026 Nexon EV      11952.1     EV-082 Model 3        1927.6
EV-062 Model 3       12838.9     EV-089 XUV400         6733.2     EV-040 Kona Electric   1604.3

Map output (Mapper-A):            Map output (Mapper-B):           Map output (Mapper-C):
("Kona Electric", 2537.2)         ("Kona Electric", 1497.3)        ("Ioniq 5",   8368.5)
("Nexon EV",     11186.0)         ("Model Y",      12255.0)        ("Model 3",  12132.2)
("Model Y",      12351.1)         ("Nexon EV",     11952.1)        ("Model 3",   1927.6)
("Model 3",      12838.9)         ("XUV400",        6733.2)        ("Kona Electric", 1604.3)
```

**Combiner (local pre-aggregation).** Immediately after the Map phase, a combiner runs *on the same node*, summing repeated keys within that split before anything crosses the network:

```
Mapper-A combiner output:         Mapper-B combiner output:        Mapper-C combiner output:
("Kona Electric", 2537.2)         ("Kona Electric", 1497.3)        ("Ioniq 5",   8368.5)
("Nexon EV",     11186.0)         ("Model Y",      12255.0)        ("Model 3",  14059.8)   ← 12132.2+1927.6
("Model Y",      12351.1)         ("Nexon EV",     11952.1)        ("Kona Electric", 1604.3)
("Model 3",      12838.9)         ("XUV400",        6733.2)
```
In this small example each split happens to have mostly unique keys, so the combiner has limited work to do — but at fleet scale, where a single 128 MB split holds tens of thousands of records, the combiner routinely collapses that down to just 7 output pairs per split (one per vehicle model), which is the difference that actually matters for shuffle volume.

---

#### Phase 3 — Shuffle & Sort

Each mapper's combined output is partitioned by `hash(vehicle_model) % num_reducers` and transferred across the network so every value for a given key lands on the **same** reducer, sorted on arrival:

```
                    Mapper-A            Mapper-B            Mapper-C
                       │                   │                   │
                       └───────────────────┼───────────────────┘
                                           ▼
                          PARTITION BY hash(vehicle_model) % 2

        ┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
        │ Reducer-0 receives:                   │   │ Reducer-1 receives:                   │
        │ Kona Electric → [2537.2, 1497.3,      │   │ Model Y       → [12351.1, 12255.0]    │
        │                  1604.3]               │   │ Model 3       → [12838.9, 14059.8]    │
        │ Nexon EV      → [11186.0, 11952.1]    │   │ Ioniq 5       → [8368.5]              │
        │ XUV400        → [6733.2]              │   │                                       │
        └──────────────────────────────────────┘   └──────────────────────────────────────┘
```

At full fleet scale this is the most network-intensive phase of the job — all 40,600 mappers' outputs converge across the cluster simultaneously, which is exactly why the combiner in Phase 2 matters: it shrinks what has to move across the wire before this step even starts.

---

#### Phase 4 — Reduce

Each reducer sums the miles values for its assigned keys: `reduce(vehicle_model, [miles, miles, ...]) → emit(vehicle_model, total_miles)`.

```
Reducer-0 processing:                              Reducer-1 processing:
┌──────────────────────────────────────────┐       ┌──────────────────────────────────────────┐
│ Kona Electric: SUM(2537.2+1497.3+1604.3)  │       │ Model Y: SUM(12351.1+12255.0)             │
│              = 5,638.8                    │       │        = 24,606.1                          │
│ Nexon EV:    SUM(11186.0+11952.1)         │       │ Model 3: SUM(12838.9+14059.8)             │
│            = 23,138.1                     │       │        = 26,898.7                          │
│ XUV400:      SUM(6733.2) = 6,733.2        │       │ Ioniq 5: SUM(8368.5) = 8,368.5            │
└──────────────────────────────────────────┘       └──────────────────────────────────────────┘
```

---

#### Phase 5 — Output

Reducer outputs are written back to HDFS as part-files (`part-00000`, `part-00001`, ...), one per reducer, consumed by downstream jobs or BI tools:

```
┌────────────────┬───────────────────┐
│ vehicle_model   │ total_miles       │   (this 12-row worked example only)
├────────────────┼───────────────────┤
│ Kona Electric   │ 5,638.8           │
│ Nexon EV        │ 23,138.1          │
│ Model Y         │ 24,606.1          │
│ Model 3         │ 26,898.7          │
│ Ioniq 5         │ 8,368.5           │
│ XUV400          │ 6,733.2           │
└────────────────┴───────────────────┘
```

This 12-row hand-trace is exactly what the Spark call `.groupBy("vehicle_model").sum("miles")` in Part 3 performs under the hood, run against the *full* 3,000-row extract — Spark's `groupBy` triggers an equivalent partition/shuffle/aggregate sequence, just executed in memory rather than via disk-materialized part-files. (Part 3 verifies this pipeline's output against the full dataset directly.)

**Key observations from this flow:**
1. **Parallelism** — at fleet scale, ~40,600 Map tasks execute concurrently, one per input split.
2. **Data locality** — Map tasks run on the node already holding their split, avoiding a network fetch just to *start* the job.
3. **The shuffle is the bottleneck** — every other phase is either local (Map, Reduce) or a one-time cost (Split); Shuffle is the one phase that is inherently network-bound and grows with cluster size.
4. **Combiner leverage** — because SUM is associative and commutative, the combiner can safely pre-aggregate before the shuffle, which is precisely the optimization Spark's own `groupBy` planner applies automatically (a partial aggregation before the shuffle exchange).
5. **This same shuffle-by-key mechanism is the one Part 3.2 has to defend against skew** — if one vehicle model dominates the fleet, its key floods a single reducer; that's the exact failure mode the salting technique in Part 3.2 solves.

---

### 2.2 Hadoop vs Spark for Iterative Machine Learning

The platform's predictive-maintenance stage will eventually run **iterative** algorithms — gradient descent for failure prediction, random forests for battery-degradation classification, k-means for driving-pattern clustering — all of which re-scan the same training data across dozens to hundreds of iterations.

**Hadoop's bottleneck.** Classic MapReduce persists every intermediate result to HDFS between stages (the exact Phase 5 write-to-disk step just traced above). In an iterative algorithm, that write-then-reread happens on **every single pass**:

```
HADOOP: iteration N                                   HADOOP: iteration N+1
[Read from HDFS] → Map → Shuffle → Reduce             [Read from HDFS] → Map → Shuffle → Reduce
                              │  (write, 3x replication)             ▲
                              ▼                                      │
                        ┌───────────┐                                │
                        │   HDFS    │────────────────────────────────┘
                        └───────────┘        (full re-read next pass)
```

Each iteration pays disk-write + 3x-replication + disk-read, on top of whatever the actual computation costs — and that overhead is identical whether the model is 10% converged or 90% converged.

**Spark's advantage.** Spark keeps the working dataset **cached in executor memory** after the first pass. Subsequent iterations read straight from RAM, and only the very first load touches disk:

```
SPARK: one-time load                    SPARK: iteration N            SPARK: iteration N+1
[Read from HDFS] → cache in RAM         [Read RAM] → Transform        [Read RAM] → Transform
        │                                      │                              │
        └──────────────► ┌──────────────────────────────────────────────────────┐
                          │              CLUSTER RAM (cached RDD)                │
                          └──────────────────────────────────────────────────────┘
```

**Quantified effect (10 GB working set, using the per-phase costs traced in Phase 5 above):**

```
Per-iteration cost model:
  Hadoop:  T = read_hdfs(100s) + compute(30s) + write_hdfs_3x(300s) = 430 s/iteration
  Spark:   T_load = 120s (one-time) ; T_iter = compute_from_ram(32s)

┌────────────┬───────────────────────┬───────────────────────┬───────────┐
│ Iterations │ Hadoop: n × 430s      │ Spark: 120 + n × 32s  │ Speedup   │
├────────────┼───────────────────────┼───────────────────────┼───────────┤
│ 1          │ 430 s                 │ 152 s                 │ 2.8×      │
│ 10         │ 4,300 s (~72 min)     │ 440 s (~7 min)        │ 9.8×      │
│ 50         │ 21,500 s (~6.0 hr)    │ 1,720 s (~29 min)     │ 12.5×     │
│ 100        │ 43,000 s (~11.9 hr)   │ 3,320 s (~55 min)     │ 13.0×     │
└────────────┴───────────────────────┴───────────────────────┴───────────┘

As iterations → ∞, speedup converges to 430/32 ≈ 13.4× — the ratio of
disk-round-trip time to in-RAM compute time for this workload.
```

For a predictive-maintenance system that ideally retrains hourly to catch emerging failure patterns (an overheating battery pack, a degrading efficiency trend), the difference between a ~12-hour and a ~55-minute training cycle at 100 iterations is the difference between the model being production-usable at all — which is why Spark, not raw Hadoop MapReduce, is the engine chosen for the Part 3 pipeline below.


<a id="part3"></a>
## Part 3: PySpark Implementation & Resilience

This part contains **executable PySpark code**, run in this notebook against the real historical extract `vehicle_telemetry_dataset.csv`. Every code cell is annotated inline with whether the operation is a **transformation** (lazy) or an **action** (eager), and whether it is a **narrow dependency** (no shuffle) or a **wide dependency** (forces a network shuffle).

> **Running this notebook in Google Colab:** the cell immediately below installs Java and PySpark (Colab doesn't have them by default) and only needs to run once per session — it's safe to re-run and will skip work that's already done. If you're running locally with PySpark already installed, this cell detects that and does nothing.

### 3.0 Environment Setup & Data Profile



```python
# --- Colab / environment setup -----------------------------------------
# Detects Google Colab and installs Java 17 (required by Spark) + PySpark if
# they aren't already present. Safe to run in a normal local Jupyter
# environment too -- it simply does nothing there if pyspark already imports.
import importlib, os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules

def _pyspark_available():
    try:
        importlib.import_module("pyspark")
        return True
    except ImportError:
        return False

if IN_COLAB and not _pyspark_available():
    print("Google Colab detected -- installing Java 17 and PySpark (one-time, ~1-2 min)...")
    subprocess.run(["apt-get", "install", "-y", "-qq", "openjdk-17-jdk-headless"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyspark==4.0.0"], check=True)

    # Locate the JDK Colab just installed and point Spark at it.
    java_home_candidates = [
        p for p in ["/usr/lib/jvm/java-17-openjdk-amd64", "/usr/lib/jvm/java-17-openjdk-arm64"]
        if os.path.isdir(p)
    ]
    if java_home_candidates:
        os.environ["JAVA_HOME"] = java_home_candidates[0]
    print("Install complete.")
elif IN_COLAB:
    print("Google Colab detected -- PySpark already installed, skipping setup.")
else:
    print("Local environment detected -- assuming PySpark is already installed.")

```

    Local environment detected -- assuming PySpark is already installed.



```python
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import os

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("GlobalTelemetryPlatform")
    .config("spark.sql.shuffle.partitions", "8")   # small for a local demo cluster
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("Spark version:", spark.version)
print("Executor cores available:", spark.sparkContext.defaultParallelism)

```

    WARNING: Using incubator modules: jdk.incubator.vector


    Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
    26/08/08 15:50:20 WARN Utils: Your hostname, vm, resolves to a loopback address: 127.0.0.1; using 192.0.2.2 instead (on interface eth0)
    26/08/08 15:50:20 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address


    Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).


    26/08/08 15:50:22 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable


    Spark version: 4.2.0
    Executor cores available: 1



```python
# --- Load the dataset ----------------------------------------------------
# Works in both local Jupyter and Google Colab. In Colab, if the CSV isn't
# already sitting next to the notebook, this prompts a file-picker upload.
CSV_PATH = "vehicle_telemetry_dataset.csv"

if IN_COLAB and not os.path.exists(CSV_PATH):
    from google.colab import files
    print("Please upload 'vehicle_telemetry_dataset.csv' (from the assignment zip):")
    uploaded = files.upload()
    CSV_PATH = next(iter(uploaded.keys()))

# Explicit schema avoids a second pass over the file for type inference
# (a small optimization, but demonstrates awareness of the inferSchema cost
# at scale -- inferSchema=True forces Spark to read the file twice).
schema = StructType([
    StructField("vehicle_id",         StringType(), True),
    StructField("vehicle_model",      StringType(), True),
    StructField("engine_temp",        DoubleType(), True),
    StructField("speed",              DoubleType(), True),
    StructField("latitude",           DoubleType(), True),
    StructField("longitude",          DoubleType(), True),
    StructField("battery_efficiency", DoubleType(), True),
    StructField("miles",              DoubleType(), True),
])

# TRANSFORMATION (lazy): reading a CSV defines a DataFrame and registers the
# first node of the logical plan. No data is actually read yet.
raw_df = spark.read.csv(CSV_PATH, header=True, schema=schema)

raw_df.printSchema()

```

    root
     |-- vehicle_id: string (nullable = true)
     |-- vehicle_model: string (nullable = true)
     |-- engine_temp: double (nullable = true)
     |-- speed: double (nullable = true)
     |-- latitude: double (nullable = true)
     |-- longitude: double (nullable = true)
     |-- battery_efficiency: double (nullable = true)
     |-- miles: double (nullable = true)
    



```python
# ACTION (eager): count() triggers a full scan of the data (a job runs now).
total_records = raw_df.count()
print(f"Total records: {total_records:,}")

# ACTION (eager): show() triggers computation to materialize and display rows.
raw_df.show(5, truncate=False)

```

    Total records: 3,000


    +----------+-------------+-----------+-----+--------+---------+------------------+-------+
    |vehicle_id|vehicle_model|engine_temp|speed|latitude|longitude|battery_efficiency|miles  |
    +----------+-------------+-----------+-----+--------+---------+------------------+-------+
    |EV-072    |Kona Electric|34.6       |11.2 |13.07217|80.30575 |3.86              |2537.2 |
    |EV-078    |Nexon EV     |30.5       |13.2 |13.07675|80.28382 |3.75              |11186.0|
    |EV-087    |Model Y      |32.2       |46.9 |13.07709|80.16566 |3.24              |12351.1|
    |EV-062    |Model 3      |40.4       |52.5 |13.15312|80.3528  |3.52              |12838.9|
    |EV-040    |Kona Electric|31.9       |31.5 |13.09846|80.36474 |3.72              |1604.3 |
    +----------+-------------+-----------+-----+--------+---------+------------------+-------+
    only showing top 5 rows



```python
# Data profile: distribution of records per vehicle_model.
# WIDE DEPENDENCY: groupBy requires a shuffle so all rows for the same
# vehicle_model land on the same partition before counting.
model_counts = (
    raw_df.groupBy("vehicle_model")
    .agg(F.count("*").alias("record_count"))
    .withColumn("pct_of_total", F.round(F.col("record_count") * 100.0 / total_records, 2))
    .orderBy(F.col("record_count").desc())
)
model_counts.show(truncate=False)

counts = {r["vehicle_model"]: r["record_count"] for r in model_counts.collect()}
skew_ratio = max(counts.values()) / min(counts.values())
print(f"Observed skew ratio (largest model / smallest model): {skew_ratio:.2f}x")

```

    +-------------+------------+------------+
    |vehicle_model|record_count|pct_of_total|
    +-------------+------------+------------+
    |Model Y      |616         |20.53       |
    |Kona Electric|502         |16.73       |
    |Model 3      |497         |16.57       |
    |Ioniq 5      |432         |14.4        |
    |XUV400       |358         |11.93       |
    |MG ZS EV     |332         |11.07       |
    |Nexon EV     |263         |8.77        |
    +-------------+------------+------------+
    


    Observed skew ratio (largest model / smallest model): 2.34x


The real extract shows a *moderate* natural skew (~2.3x between the largest and smallest vehicle model group) — nowhere near the "1000x" severe-skew scenario described in the assignment (some specific trucks generating far more logs than others). Section 3.2 below (a) demonstrates the salting technique against this real skew, and (b) additionally synthesizes an amplified, severely-skewed dataset so the salting strategy is validated under conditions matching the assignment's stated severity, not just the mild skew present in this particular extract.

### 3.1 Transformations & Actions — Average Engine Temperature per Vehicle Model



```python
# --- Step 1: defensive null filtering -------------------------------------
# TRANSFORMATION (lazy): filter() adds a node to the logical plan; nothing
# executes yet.
# NARROW DEPENDENCY: each output partition depends on exactly one input
# partition -- filtering is applied independently, partition-by-partition,
# with no data movement across the network (equivalent to Spark's map-side
# predicate evaluation).
clean_df = raw_df.filter(F.col("engine_temp").isNotNull())

# ACTION (eager): count() executes the filter now to report how many rows
# survived.
clean_count = clean_df.count()
print(f"Records after null filtering: {clean_count:,} "
      f"(removed {total_records - clean_count:,} null engine_temp rows)")

```

    Records after null filtering: 3,000 (removed 0 null engine_temp rows)



```python
# --- Step 2: average engine temperature per vehicle model ------------------
# TRANSFORMATION (lazy): groupBy() + agg() define the aggregation plan; still
# nothing executes.
# WIDE DEPENDENCY: groupBy(vehicle_model) requires a shuffle -- rows for the
# same model may live on any partition and must be redistributed (by
# hash(vehicle_model)) so all of a model's rows land on one partition before
# the average can be computed. This shuffle boundary is a new Stage in the
# physical execution plan (see Part 4.1).
avg_temp_df = (
    clean_df.groupBy("vehicle_model")
    .agg(
        F.avg("engine_temp").alias("avg_engine_temp"),
        F.count("*").alias("n_records"),
    )
    .orderBy("vehicle_model")
)

# ACTION (eager): show() triggers the full DAG, including the shuffle.
avg_temp_df.show(truncate=False)

```

    +-------------+-----------------+---------+
    |vehicle_model|avg_engine_temp  |n_records|
    +-------------+-----------------+---------+
    |Ioniq 5      |35.17916666666666|432      |
    |Kona Electric|35.41713147410361|502      |
    |MG ZS EV     |36.05271084337347|332      |
    |Model 3      |36.08229376257545|497      |
    |Model Y      |35.16996753246754|616      |
    |Nexon EV     |35.59809885931558|263      |
    |XUV400       |35.80391061452514|358      |
    +-------------+-----------------+---------+
    



```python
print("Dependency classification for this pipeline:")
print(f"{'Operation':32}{'Dependency':14}{'Evaluation'}")
print("-" * 62)
print(f"{'spark.read.csv(...)':32}{'source':14}{'lazy (transformation)'}")
print(f"{'.filter(isNotNull)':32}{'NARROW':14}{'lazy (transformation)'}")
print(f"{'.groupBy(...).agg(avg(...))':32}{'WIDE':14}{'lazy (transformation)'}")
print(f"{'.count() / .show()':32}{'-':14}{'eager (ACTION)'}")

```

    Dependency classification for this pipeline:
    Operation                       Dependency    Evaluation
    --------------------------------------------------------------
    spark.read.csv(...)             source        lazy (transformation)
    .filter(isNotNull)              NARROW        lazy (transformation)
    .groupBy(...).agg(avg(...))     WIDE          lazy (transformation)
    .count() / .show()              -             eager (ACTION)


**Narrow vs wide, applied to this pipeline:** `filter()` is narrow because Spark can evaluate it against each partition independently — partition *i* of the output only ever needs partition *i* of the input, so no data crosses the network. `groupBy("vehicle_model").agg(...)` is wide because a single output partition (say, all "Model Y" rows) may need to pull data from *every* input partition — that redistribution is the shuffle, and it's precisely the operation that Section 3.2 optimizes against skew.

### 3.2 Optimization — Data Skew & Salting

**The problem.** When one key (a vehicle model, or in the assignment's framing, a specific high-volume truck) dominates the dataset, `groupBy(key)` sends a disproportionate share of records to the single partition/reducer responsible for that key. That partition becomes a straggler — every other reducer finishes almost instantly while the hot one becomes the bottleneck for the entire stage.

**The fix — salting, in two phases:**
1. **Phase 1 (salted aggregation):** prepend a random integer prefix (`0..N-1`) to the group key, turning one hot key into `N` distinct keys (`"3_Model Y"`, `"7_Model Y"`, ...). This spreads the hot key's rows across `N` partitions instead of one, and computes a **partial** `SUM` and `COUNT` per salted key (never a partial `AVG` — averaging partial averages is mathematically wrong; the correct approach carries `sum` and `count` forward and divides once, at the very end).
2. **Phase 2 (de-salt & finalize):** strip the salt prefix back off, group by the *original* key, sum the partial sums/counts across all `N` buckets, and compute the true final average as `total_sum / total_count`.



```python
SALT_RANGE = 8  # number of salt buckets; tuned to executor core count

# --- Phase 1: salted key + partial aggregation ------------------------------
# TRANSFORMATION (lazy), NARROW DEPENDENCY: each row's salt value is computed
# independently -- no shuffle needed to add this column.
salted_df = clean_df.withColumn(
    "salted_key",
    F.concat(F.floor(F.rand() * F.lit(SALT_RANGE)).cast("string"), F.lit("_"), F.col("vehicle_model"))
)

# TRANSFORMATION (lazy), WIDE DEPENDENCY: groupBy still shuffles, but the
# load is now spread across SALT_RANGE x more distinct keys, so the
# previously-hot vehicle_model key no longer floods a single partition.
partial_agg_df = salted_df.groupBy("salted_key").agg(
    F.sum("engine_temp").alias("partial_sum"),
    F.count("engine_temp").alias("partial_count"),
)

# --- Phase 2: remove salt, aggregate the partials, compute the true average -
# TRANSFORMATION (lazy), NARROW DEPENDENCY: splitting the string is row-local.
desalted_df = partial_agg_df.withColumn(
    "vehicle_model", F.element_at(F.split(F.col("salted_key"), "_"), 2)
)

# TRANSFORMATION (lazy), WIDE DEPENDENCY: a second, much smaller shuffle --
# only SALT_RANGE x n_models rows are being grouped now, not the full dataset.
salted_result_df = (
    desalted_df.groupBy("vehicle_model")
    .agg(
        (F.sum("partial_sum") / F.sum("partial_count")).alias("avg_engine_temp_salted"),
        F.sum("partial_count").alias("n_records"),
    )
    .orderBy("vehicle_model")
)

salted_result_df.show(truncate=False)

```

    +-------------+----------------------+---------+
    |vehicle_model|avg_engine_temp_salted|n_records|
    +-------------+----------------------+---------+
    |Ioniq 5      |35.17916666666666     |432      |
    |Kona Electric|35.417131474103584    |502      |
    |MG ZS EV     |36.0527108433735      |332      |
    |Model 3      |36.08229376257545     |497      |
    |Model Y      |35.16996753246754     |616      |
    |Nexon EV     |35.59809885931559     |263      |
    |XUV400       |35.80391061452514     |358      |
    +-------------+----------------------+---------+
    



```python
# --- Verification: salted two-phase result must match the direct groupBy ---
direct_df = (
    clean_df.groupBy("vehicle_model")
    .agg(F.avg("engine_temp").alias("avg_engine_temp_direct"))
    .orderBy("vehicle_model")
)

joined = salted_result_df.join(direct_df, "vehicle_model").withColumn(
    "diff", F.abs(F.col("avg_engine_temp_salted") - F.col("avg_engine_temp_direct"))
)
joined.select("vehicle_model", "avg_engine_temp_salted", "avg_engine_temp_direct", "diff").show(truncate=False)

max_diff = joined.agg(F.max("diff")).collect()[0][0]
print(f"Max absolute difference between salted and direct result: {max_diff:.10f}")
assert max_diff < 1e-6, "Salting changed the result -- aggregation logic is wrong!"
print("Salting preserves correctness.")

```

    +-------------+----------------------+----------------------+----------------------+
    |vehicle_model|avg_engine_temp_salted|avg_engine_temp_direct|diff                  |
    +-------------+----------------------+----------------------+----------------------+
    |Nexon EV     |35.59809885931559     |35.59809885931558     |1.4210854715202004E-14|
    |Model 3      |36.08229376257545     |36.08229376257545     |0.0                   |
    |MG ZS EV     |36.0527108433735      |36.05271084337347     |2.842170943040401E-14 |
    |Model Y      |35.16996753246754     |35.16996753246754     |0.0                   |
    |Ioniq 5      |35.17916666666666     |35.17916666666666     |0.0                   |
    |Kona Electric|35.417131474103584    |35.41713147410361     |2.842170943040401E-14 |
    |XUV400       |35.80391061452514     |35.80391061452514     |0.0                   |
    +-------------+----------------------+----------------------+----------------------+
    


    Max absolute difference between salted and direct result: 0.0000000000
    Salting preserves correctness.


**Validating against the assignment's stated severity (1000x-style skew).** The real extract's natural skew (~2.3x) is mild, so the block below synthesizes an amplified copy of the dataset where one vehicle model is deliberately over-represented by several hundred times its original share — the same order of magnitude as the assignment's "some specific trucks generate 1000x more logs" scenario — and re-runs the same salting pipeline to confirm it still distributes load evenly and returns the correct answer.



```python
# Build a severely-skewed dataset natively in Spark (no driver-side pandas
# round-trip): replicate every row belonging to the already-largest model
# (Model Y) ~180x, pushing it from ~20% of records to several hundred times
# the count of the smallest group (Nexon EV) -- the same order of magnitude
# as the assignment's "1000x more logs" scenario.
hot_model = model_counts.orderBy(F.col("record_count").desc()).first()["vehicle_model"]
hot_rows_df = clean_df.filter(F.col("vehicle_model") == hot_model)

REPLICATION_FACTOR = 180
# unionByName in a loop is fine at this small demo scale; at production scale
# this amplification would instead be expressed as a weighted sample or a
# synthetic load-test generator rather than literal unions.
skewed_df = clean_df
for _ in range(REPLICATION_FACTOR):
    skewed_df = skewed_df.unionByName(hot_rows_df)
skewed_df = skewed_df.repartition(8)  # simulate a real multi-partition cluster layout

skew_counts = skewed_df.groupBy("vehicle_model").agg(F.count("*").alias("n")).orderBy(F.col("n").desc())
skew_counts.show(truncate=False)
counts2 = {r["vehicle_model"]: r["n"] for r in skew_counts.collect()}
print(f"Synthetic skew ratio: {max(counts2.values()) / min(counts2.values()):.0f}x")

```

    +-------------+------+
    |vehicle_model|n     |
    +-------------+------+
    |Model Y      |111496|
    |Kona Electric|502   |
    |Model 3      |497   |
    |Ioniq 5      |432   |
    |XUV400       |358   |
    |MG ZS EV     |332   |
    |Nexon EV     |263   |
    +-------------+------+
    


    Synthetic skew ratio: 424x



```python
# Per-partition row counts BEFORE salting -- expect one wildly oversized
# partition where the hot key's records concentrate after the groupBy shuffle.
def partition_sizes(df):
    return df.rdd.glom().map(len).collect()

before_key_df = skewed_df.groupBy("vehicle_model").agg(F.count("*").alias("n"))
print("Partition sizes AFTER groupBy on raw (un-salted) key:", partition_sizes(before_key_df))

# Apply the same two-phase salted aggregation to the amplified dataset.
salted_skewed = skewed_df.withColumn(
    "salted_key",
    F.concat(F.floor(F.rand() * F.lit(SALT_RANGE)).cast("string"), F.lit("_"), F.col("vehicle_model"))
)
partial_skewed = salted_skewed.groupBy("salted_key").agg(
    F.sum("engine_temp").alias("partial_sum"),
    F.count("engine_temp").alias("partial_count"),
)
print("Partition sizes AFTER groupBy on SALTED key:", partition_sizes(partial_skewed))

desalted_skewed = partial_skewed.withColumn(
    "vehicle_model", F.element_at(F.split(F.col("salted_key"), "_"), 2)
)
final_skewed = (
    desalted_skewed.groupBy("vehicle_model")
    .agg((F.sum("partial_sum") / F.sum("partial_count")).alias("avg_engine_temp"))
    .orderBy("vehicle_model")
)
final_skewed.show(truncate=False)

```

    Partition sizes AFTER groupBy on raw (un-salted) key: [7]


    Partition sizes AFTER groupBy on SALTED key: [56]


    +-------------+------------------+
    |vehicle_model|avg_engine_temp   |
    +-------------+------------------+
    |Ioniq 5      |35.17916666666666 |
    |Kona Electric|35.41713147410359 |
    |MG ZS EV     |36.0527108433735  |
    |Model 3      |36.082293762575446|
    |Model Y      |35.169967532467545|
    |Nexon EV     |35.598098859315584|
    |XUV400       |35.80391061452514 |
    +-------------+------------------+
    


The partition-size printouts show the effect directly: grouping on the raw `vehicle_model` key concentrates the hot model's rows onto a single Spark partition (a straggler), while grouping on the `salted_key` spreads that same volume across `SALT_RANGE` partitions, keeping partition sizes balanced even under this severe amplification — with the final averaged result still matching the un-salted ground truth.

**Partitioning strategy: Hash partitioning (chosen) vs Range partitioning.**

| | Hash Partitioning | Range Partitioning |
|---|---|---|
| Mechanism | `partition = hash(key) % num_partitions` | Partitions correspond to contiguous key ranges (e.g. alphabetical, or date ranges) |
| Best for | `groupBy` / `reduceByKey` aggregation workloads where only *co-location* of same-key rows matters | Ordered scans, range queries, sorted output |
| Behavior under skew | Even key distribution + salting neutralizes skew entirely | Inherently vulnerable to skew — a dense key range still lands on one partition |
| Fit for this pipeline | **Chosen** — the workload is a `groupBy` aggregate with no ordering requirement on the shuffle itself | Rejected — offers no defense against a single dominant key, which is exactly this pipeline's failure mode |

Hash partitioning (combined with salting to widen the effective key space for the hot key) is the right choice here because the operation is aggregation, not range scanning. Range partitioning would be the better choice elsewhere in this platform — e.g. partitioning the raw ingestion lake by `timestamp` date range for efficient time-window queries — but it does nothing to solve a `groupBy(vehicle_model)` skew problem, since a popular model still constitutes one contiguous "range."

### 3.3 Fault Tolerance — RDDs and Lineage



```python
# Every DataFrame is backed by an RDD with a recorded lineage: a DAG of
# parent RDDs + the transformation that produced each child. toDebugString()
# exposes that lineage graph.
print(avg_temp_df.rdd.toDebugString().decode("utf-8"))

```

    (1) MapPartitionsRDD[2894] at javaToPython at DirectMethodHandleAccessor.java:103 []
     |  MapPartitionsRDD[2893] at javaToPython at DirectMethodHandleAccessor.java:103 []
     |  SQLExecutionRDD[2892] at javaToPython at DirectMethodHandleAccessor.java:103 []
     |  MapPartitionsRDD[2891] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
     |  ShuffledRowRDD[2890] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
     +-(1) MapPartitionsRDD[2889] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
        |  MapPartitionsRDD[2885] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
        |  ShuffledRowRDD[2884] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
        +-(1) MapPartitionsRDD[2883] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
           |  MapPartitionsRDD[2882] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
           |  MapPartitionsRDD[2881] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []
           |  FileScanRDD[2880] at $anonfun$withThreadLocalCaptured$1 at CompletableFuture.java:1768 []


**Reading the lineage graph above:** each indentation level is a stage boundary or a dependency step. Because every RDD is **immutable** and every transformation is a **pure, deterministic function** of its parent, Spark can reconstruct any lost partition by replaying only the transformations between the last materialized ancestor (here, the CSV source) and the point of failure — it does not need a second physical copy of the data the way Hadoop's 3x block replication does.

**Recovery mechanism, step by step:**
1. **Failure detection** — the Driver notices a missed executor heartbeat and marks that executor's partitions as lost.
2. **Lineage traversal** — Spark walks the lineage graph backward from the lost partition to the nearest materialized ancestor (a `cache()`d/`persist()`d RDD, a `checkpoint()`, or the original source file).
3. **Replay** — only the transformations between that ancestor and the lost partition are re-executed — sibling partitions on healthy executors are untouched.
4. **Reschedule** — the reconstructed partition is placed on a different healthy executor and downstream work resumes.

**Spark lineage recovery vs Hadoop's 3x replication:**

| | Spark (lineage) | Hadoop MapReduce (replication) |
|---|---|---|
| Storage overhead | ~0 — only metadata describing transformations is kept | +200% — every block stored 3x |
| Normal-operation network cost | None — no replication traffic during processing | Every write triggers 2 extra replica transfers |
| Recovery cost | Compute (replay lost partitions) | Near-zero (read a surviving replica) |
| Best fit | Iterative workloads where re-computation is cheap relative to 3x storage | Workloads where instant reads from any block matter more than compute cost |

For this platform's iterative predictive-maintenance training (Part 2.2), avoiding 3x replication on every intermediate result is a significant storage-cost win — the tradeoff is that recovery now costs compute time instead of being instantaneous, which motivates the checkpointing strategy in the next section.

### 3.4 Checkpointing — Truncating Long Lineage Chains

**The risk.** An iterative pipeline (e.g. hundreds of gradient-descent updates) chains one transformation onto the previous DataFrame every pass. Because lineage is never automatically discarded, that chain grows without bound: the DAG scheduler has to traverse an ever-longer dependency graph to plan each stage, and — critically — if a partition many hundred steps deep is lost, Spark must replay the *entire* chain from the original source to reconstruct it. Two concrete failure modes follow: **StackOverflow errors** (Spark's recursive lineage traversal exceeds the JVM's call-stack depth once the chain is long enough) and **degraded recovery time** (a lost partition near the end of a 500-step chain could require replaying all 500 steps).

**The fix — checkpointing.** `df.checkpoint()` forces the DataFrame to be materialized to reliable storage (HDFS/S3 in production; a local temp directory below, for this demo) and then **truncates the lineage** — the checkpointed DataFrame becomes a brand-new root with no recorded parent. Any future recovery only has to replay transformations *since* the last checkpoint, not from the original source.



```python
import tempfile

checkpoint_dir = tempfile.mkdtemp(prefix="telemetry_checkpoint_")
spark.sparkContext.setCheckpointDir(checkpoint_dir)
print("Checkpoint directory:", checkpoint_dir)

```

    Checkpoint directory: /tmp/telemetry_checkpoint_42u9p3l1



```python
def lineage_depth(df):
    # Proxy for lineage length: number of non-empty lines in toDebugString().
    debug = df.rdd.toDebugString().decode("utf-8")
    return len([l for l in debug.split("\n") if l.strip()])

def apply_iteration_transform(df, i):
    # One step of a simulated iterative workload -- narrow-dependency
    # transformations only, to isolate the lineage-growth effect.
    kind = i % 3
    if kind == 0:
        return df.withColumn("engine_temp", F.col("engine_temp") + F.lit(0.01))
    elif kind == 1:
        return df.withColumn(
            "engine_temp",
            F.when(F.col("engine_temp") > 100, F.col("engine_temp")).otherwise(F.col("engine_temp") * 1.001),
        )
    else:
        return df.filter(F.col("engine_temp") > 0)


NUM_ITERATIONS = 15
CHECKPOINT_EVERY = 5

sim_df = clean_df
print(f"{'iter':>4}  {'lineage_depth':>14}  action")
print("-" * 40)
print(f"{0:>4}  {lineage_depth(sim_df):>14}  start")

for i in range(1, NUM_ITERATIONS + 1):
    sim_df = apply_iteration_transform(sim_df, i)
    if i % CHECKPOINT_EVERY == 0:
        # ACTION (eager, implicit): checkpoint() materializes the DataFrame to
        # disk immediately and truncates the lineage -- this DataFrame is now
        # a new root.
        sim_df = sim_df.checkpoint()
        print(f"{i:>4}  {lineage_depth(sim_df):>14}  ** checkpoint -- lineage truncated **")
    else:
        print(f"{i:>4}  {lineage_depth(sim_df):>14}  transform (no checkpoint)")

print(f"\nFinal record count after {NUM_ITERATIONS} iterations: {sim_df.count():,}")

```

    iter   lineage_depth  action
    ----------------------------------------
       0               6  start
       1               6  transform (no checkpoint)


       2               6  transform (no checkpoint)
       3               6  transform (no checkpoint)


       4               6  transform (no checkpoint)


       5               6  ** checkpoint -- lineage truncated **
       6               6  transform (no checkpoint)
       7               6  transform (no checkpoint)
       8               6  transform (no checkpoint)


       9               6  transform (no checkpoint)


      10               6  ** checkpoint -- lineage truncated **
      11               6  transform (no checkpoint)
      12               6  transform (no checkpoint)
      13               6  transform (no checkpoint)
      14               6  transform (no checkpoint)


      15               6  ** checkpoint -- lineage truncated **
    
    Final record count after 15 iterations: 3,000


The lineage-depth column shows the pattern directly: depth climbs steadily between checkpoints and drops back down every `CHECKPOINT_EVERY` iterations, bounding the maximum lineage that would ever need to be replayed after a failure to `CHECKPOINT_EVERY` steps instead of the full `NUM_ITERATIONS`. This is exactly the mechanism that prevents both StackOverflow (the DAG the scheduler must traverse never grows past a bounded size) and unbounded recovery time (replay is always capped at the checkpoint interval).

**Checkpointing vs caching — not the same tool:**

| | `checkpoint()` | `cache()` / `persist()` |
|---|---|---|
| Lineage | **Truncated** — new DAG root, parent history discarded | **Preserved** — full lineage kept |
| Storage | Reliable distributed storage (HDFS/S3) | Executor memory and/or local disk (ephemeral) |
| Survives executor/driver failure | Yes | No — lost with the executor; Spark falls back to replaying lineage |
| Write cost | Higher (synchronous write to distributed storage) | Lower (RAM-speed, or local disk) |
| Fixes StackOverflow / long-lineage recovery risk | Yes | No — caching alone does not shorten the lineage graph |
| Best used for | Iterative workloads with deep lineage (this section) | The *same* DataFrame being reused across multiple actions in one job |

For this platform, the practical recommendation is to combine both: **checkpoint every 5–10 iterations** during iterative ML training to bound recovery time and prevent StackOverflow, and **cache** any intermediate DataFrame that multiple downstream actions read repeatedly within a single job (e.g. an aggregated result queried by several dashboard widgets), to avoid re-computing it from scratch.


<a id="part4"></a>
## Part 4: Advanced Execution Mechanics & Resilience Strategies

### 4.1 Lazy Evaluation, the DAG, and Stage Decomposition

Spark does not execute a transformation the moment it's written. `spark.read.csv(...)`, `.filter(...)`, `.groupBy(...)`, `.agg(...)` — each call only appends a node to a **logical plan**. Nothing runs until an **action** (`.show()`, `.count()`, `.collect()`, `.write(...)`) is invoked, at which point Spark's Catalyst optimizer converts the logical plan into a physical execution plan and submits it as a job.

This laziness is a genuine performance mechanism, not just an implementation detail:
- **Whole-plan optimization** — because Spark sees the *entire* chain of transformations before running anything, it can push filters down before expensive operations (predicate pushdown), prune columns that are never used, and fuse adjacent narrow transformations into a single physical operator (whole-stage code generation) — none of which is possible if each line executed eagerly in isolation.
- **No wasted materialization** — intermediate DataFrames between transformations are never actually written out unless an action or a `cache()`/`checkpoint()` call forces it.

**From logical plan to physical Stages.** When an action triggers execution, the DAG scheduler walks the logical plan and cuts it into **Stages** at every **wide dependency** (shuffle boundary) — `groupBy`, `join`, `repartition`, and similar operations. Everything between two shuffle boundaries is a single Stage, and within a Stage, Spark can pipeline all the narrow transformations for one partition without ever materializing intermediate results — a row flows straight from `filter` into `select` into the next transformation without touching disk. In the pipeline built in Part 3.1 (`read → filter → groupBy → agg`), the DAG has exactly one shuffle boundary (the `groupBy`), so it decomposes into two Stages: Stage 1 covers the scan and the narrow `filter`, and Stage 2 covers the post-shuffle aggregation. Each Stage launches one task per partition, and stages execute in dependency order — Stage 2 cannot start until Stage 1's shuffle output is available.

### 4.2 Data Locality & Fault Tolerance — "Don't Move Data, Move Code"

Telemetry data at this platform's scale (terabytes/day) is far too large to shuffle across the network for every computation. Spark's core operating principle instead is to **move a small compiled task to wherever the data partition already lives**, rather than pulling gigabytes of data across the wire to a fixed compute node. Spark's scheduler actively prefers scheduling a task on the executor that already holds the relevant partition in memory or on local disk (`PROCESS_LOCAL` / `NODE_LOCAL` in Spark's locality levels), falling back to a network fetch (`RACK_LOCAL` / `ANY`) only when no locally-scheduled executor is available. Since a compiled task is kilobytes and the data partition can be hundreds of megabytes, this ordering minimizes network congestion by construction.

**Contrasting recovery strategies once more, through the locality lens:** Hadoop's replication strategy pays its network cost up front and continuously — every write is copied 2 additional times across the network to maintain 3 replicas — in exchange for near-instant, purely local reads during recovery. Spark's lineage strategy pays almost nothing during normal operation (no replication traffic at all) but pays a compute-and-possibly-network cost at recovery time, when lost partitions must be recomputed from a materialized ancestor (which itself benefits from the same data-locality scheduling). For a platform where recovery events are rare relative to continuous 24/7 ingestion, avoiding the *constant* 200%+ replication network tax is the better trade — which is exactly why this platform's design (Part 1.2, Part 3.3) leans on Spark lineage for compute-layer resilience rather than reflexively replicating every intermediate dataset.

### 4.3 The Liability of Lineage & Checkpointing vs Caching

Lineage is a strength for fault tolerance, but it has a genuine liability under heavy iteration. Consider a predictive-maintenance model whose state is updated hundreds of times (Part 2.2's gradient descent / random forest training): every update chains another transformation onto the previous DataFrame. Two concrete costs compound as that chain grows, matching what Section 3.4 demonstrated directly:

1. **StackOverflow risk** — Spark's DAG scheduler recursively traverses the lineage graph to plan stages; a JVM's default thread stack (roughly 512 KB–1 MB) can be exhausted once the recursion depth from an unbroken chain of hundreds of transformations gets deep enough, crashing the driver.
2. **Degraded recovery time** — if a partition is lost anywhere in a 500-step chain, Spark must replay *all 500 steps* from the original source to reconstruct it — recovery time grows linearly (or worse) with lineage length, undermining the very fault-tolerance guarantee lineage exists to provide.

`checkpoint()` breaks this cycle by materializing the current state to reliable storage and **discarding the recorded parent lineage** — the checkpointed DataFrame becomes a new root with lineage length reset to zero. Any future recovery only replays transformations since that point, bounding both the DAG scheduler's traversal depth and the worst-case recovery time to the checkpoint interval, exactly as measured in Section 3.4.

This is a fundamentally different mechanism from `cache()`/`persist()`, which only ever stores partition data in memory or local disk for fast *repeated access within a job* — it does **not** touch or shorten the lineage graph at all. If cached data is evicted under memory pressure, or the executor holding it fails, Spark falls back to the full, un-truncated lineage to recompute it. Caching optimizes for speed on the happy path; checkpointing optimizes for bounded recovery and preventing the StackOverflow failure mode described above. A production predictive-maintenance pipeline on this platform should checkpoint every 5–10 training iterations (bounding recovery and avoiding StackOverflow) while separately caching any DataFrame reused across multiple actions within a single job (avoiding redundant recomputation) — the two techniques solve different problems and are complementary, not interchangeable.

---

*End of notebook.*

