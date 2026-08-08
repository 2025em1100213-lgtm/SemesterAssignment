# Part 3: PySpark Implementation & Resilience — Theory Companion

**Student Name:** Joyce Dorothy S
**Student ID:** 2025em1100213

---

> The executable PySpark code for this part lives in `code/pyspark_transformations.py`, `code/pyspark_salting_optimization.py`, and `code/pyspark_checkpointing.py`, and is also embedded with live outputs in `Global_Telemetry_Platform_Assignment.ipynb`. This file collects the accompanying theory and explanation.

## Part 3: PySpark Implementation & Resilience

This part contains **executable PySpark code**, run in this notebook against the real historical extract `vehicle_telemetry_dataset.csv`. Every code cell is annotated inline with whether the operation is a **transformation** (lazy) or an **action** (eager), and whether it is a **narrow dependency** (no shuffle) or a **wide dependency** (forces a network shuffle).

### 3.0 Environment Setup & Data Profile


## 3.1 Transformations & Actions — Narrow vs Wide Dependencies

The real extract shows a *moderate* natural skew (~2.3x between the largest and smallest vehicle model group) — nowhere near the "1000x" severe-skew scenario described in the assignment (some specific trucks generating far more logs than others). Section 3.2 below (a) demonstrates the salting technique against this real skew, and (b) additionally synthesizes an amplified, severely-skewed dataset so the salting strategy is validated under conditions matching the assignment's stated severity, not just the mild skew present in this particular extract.

### 3.1 Transformations & Actions — Average Engine Temperature per Vehicle Model


**Narrow vs wide, applied to this pipeline:** `filter()` is narrow because Spark can evaluate it against each partition independently — partition *i* of the output only ever needs partition *i* of the input, so no data crosses the network. `groupBy("vehicle_model").agg(...)` is wide because a single output partition (say, all "Model Y" rows) may need to pull data from *every* input partition — that redistribution is the shuffle, and it's precisely the operation that Section 3.2 optimizes against skew.

### 3.2 Optimization — Data Skew & Salting

**The problem.** When one key (a vehicle model, or in the assignment's framing, a specific high-volume truck) dominates the dataset, `groupBy(key)` sends a disproportionate share of records to the single partition/reducer responsible for that key. That partition becomes a straggler — every other reducer finishes almost instantly while the hot one becomes the bottleneck for the entire stage.

**The fix — salting, in two phases:**
1. **Phase 1 (salted aggregation):** prepend a random integer prefix (`0..N-1`) to the group key, turning one hot key into `N` distinct keys (`"3_Model Y"`, `"7_Model Y"`, ...). This spreads the hot key's rows across `N` partitions instead of one, and computes a **partial** `SUM` and `COUNT` per salted key (never a partial `AVG` — averaging partial averages is mathematically wrong; the correct approach carries `sum` and `count` forward and divides once, at the very end).
2. **Phase 2 (de-salt & finalize):** strip the salt prefix back off, group by the *original* key, sum the partial sums/counts across all `N` buckets, and compute the true final average as `total_sum / total_count`.


## 3.2 Optimization — Data Skew & Salting

**The problem.** When one key (a vehicle model, or in the assignment's framing, a specific high-volume truck) dominates the dataset, `groupBy(key)` sends a disproportionate share of records to the single partition/reducer responsible for that key. That partition becomes a straggler -- every other reducer finishes almost instantly while the hot one becomes the bottleneck for the entire stage.

**The fix -- salting, in two phases:**
1. **Phase 1 (salted aggregation):** prepend a random integer prefix (`0..N-1`) to the group key, turning one hot key into `N` distinct keys. This spreads the hot key's rows across `N` partitions instead of one, and computes a **partial** `SUM` and `COUNT` per salted key (never a partial `AVG` -- averaging partial averages is mathematically wrong).
2. **Phase 2 (de-salt & finalize):** strip the salt prefix, group by the *original* key, sum the partial sums/counts across all `N` buckets, and compute the true final average as `total_sum / total_count`.

See `code/pyspark_salting_optimization.py` for the full runnable implementation, including a validation against a severely-amplified (several-hundred-x) synthetic skew matching the assignment's stated '1000x more logs' scenario, and a correctness check against the direct (un-salted) aggregation.

**Partitioning strategy: Hash partitioning (chosen) vs Range partitioning.**

| | Hash Partitioning | Range Partitioning |
|---|---|---|
| Mechanism | `partition = hash(key) % num_partitions` | Partitions correspond to contiguous key ranges |
| Best for | `groupBy` / `reduceByKey` aggregation workloads | Ordered scans, range queries, sorted output |
| Behavior under skew | Even key distribution + salting neutralizes skew entirely | Vulnerable to skew -- a dense key range still lands on one partition |
| Fit for this pipeline | **Chosen** -- the workload is a `groupBy` aggregate with no ordering requirement | Rejected -- no defense against a single dominant key |

Hash partitioning, combined with salting to widen the effective key space for the hot key, is the right choice here because the operation is aggregation, not range scanning. Range partitioning would suit other parts of this platform better (e.g. partitioning the raw ingestion lake by `timestamp` for time-window queries), but it does nothing to solve a `groupBy(vehicle_model)` skew problem.

## 3.3 Fault Tolerance — RDDs and Lineage

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


## 3.4 Checkpointing — Truncating Long Lineage Chains

**The risk.** An iterative pipeline (e.g. hundreds of gradient-descent updates) chains one transformation onto the previous DataFrame every pass. Because lineage is never automatically discarded, that chain grows without bound: the DAG scheduler has to traverse an ever-longer dependency graph to plan each stage, and -- critically -- if a partition many hundred steps deep is lost, Spark must replay the *entire* chain from the original source to reconstruct it. Two concrete failure modes follow: **StackOverflow errors** (Spark's recursive lineage traversal exceeds the JVM's call-stack depth once the chain is long enough) and **degraded recovery time** (a lost partition near the end of a 500-step chain could require replaying all 500 steps).

**The fix -- checkpointing.** `df.checkpoint()` forces the DataFrame to be materialized to reliable storage (HDFS/S3 in production) and then **truncates the lineage** -- the checkpointed DataFrame becomes a brand-new root with no recorded parent. Any future recovery only has to replay transformations *since* the last checkpoint, not from the original source. See `code/pyspark_checkpointing.py` for the full runnable implementation, which applies 15 iterative transformations to the telemetry data, checkpointing every 5th iteration, and prints the measured lineage depth before and after each checkpoint to show the truncation directly.

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




---

*End of Part 3: PySpark Implementation & Resilience*
