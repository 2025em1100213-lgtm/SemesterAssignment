# Part 2: Batch Processing & MapReduce

**Student Name:** Joyce Dorothy S
**Student ID:** 2025em1100213

---


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

