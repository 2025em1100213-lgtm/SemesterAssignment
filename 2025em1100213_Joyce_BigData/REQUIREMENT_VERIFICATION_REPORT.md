# Requirement Verification Report

**Student Name:** Joyce Dorothy S
**Student ID:** 2025em1100213
**Assignment:** Architecting and Implementing a Resilient Global Telemetry Platform
**Dataset:** `vehicle_telemetry_dataset.csv` — 3,000 real telemetry records, 100 vehicles, 7 EV models (`vehicle_id`, `vehicle_model`, `engine_temp`, `speed`, `latitude`, `longitude`, `battery_efficiency`, `miles`)

This report maps every requirement in the assignment brief to where it is satisfied in the deliverables, and confirms execution status.

---

## Part 1: System Architecture & Data Paradigms

| Requirement | Deliverable | Status |
|---|---|---|
| Explain the physical hardware "Wall" | `part1_system_architecture.md` §1.1 / notebook Part 1 | ✅ Done |
| Justify horizontal over vertical scaling | `part1_system_architecture.md` §1.1 (Scale-Up vs Scale-Out table) | ✅ Done |
| Address the Three Vs (Volume, Velocity, Variety) | `part1_system_architecture.md` §1.1 | ✅ Done |
| Contrast ACID vs BASE | `part1_system_architecture.md` §1.2 | ✅ Done |
| Apply CAP Theorem to choose a model for the ingestion workload | `part1_system_architecture.md` §1.2 (AP selection + justification) | ✅ Done |

## Part 2: Batch Processing & MapReduce

| Requirement | Deliverable | Status |
|---|---|---|
| MapReduce logical flow: Split → Map → Shuffle → Sort → Reduce for total miles per vehicle model | `part2_batch_processing.md` §2.1, using real `miles` values from the dataset | ✅ Done |
| Explain why Spark's in-memory model beats Hadoop's disk-bound I/O for iterative ML | `part2_batch_processing.md` §2.2 | ✅ Done |

## Part 3: PySpark Implementation & Resilience

| Requirement | Deliverable | Status |
|---|---|---|
| Ingest data, compute average engine temperature per vehicle model | `code/pyspark_transformations.py`, notebook §3.1 | ✅ Executed against real data — 0 errors |
| Identify narrow vs wide dependencies in the code | Inline comments in `pyspark_transformations.py`; summary table printed at runtime | ✅ Done |
| Implement a salting strategy to mitigate data skew | `code/pyspark_salting_optimization.py`, notebook §3.2 — two-phase (salt → partial agg → de-salt → final agg) | ✅ Executed — result matches direct aggregation exactly |
| Validate salting at severe (assignment-stated "1000x") skew, not just the dataset's natural ~2.3x skew | Same script — synthetic amplification to ~424x, partition-size before/after comparison | ✅ Executed — skew neutralized, correctness preserved |
| Define partitioning strategy (Hash vs Range) | `pyspark_salting_optimization.py` closing section; notebook §3.2 | ✅ Done — Hash chosen, justified |
| Explain RDD + lineage-based fault tolerance without heavy replication | `part3_theory_fault_tolerance.md`; notebook §3.3, `rdd.toDebugString()` output | ✅ Done |
| Simulate long lineage growth and implement strategic checkpointing | `code/pyspark_checkpointing.py`, notebook §3.4 — 15 iterations, checkpoint every 5th, lineage depth logged before/after each checkpoint | ✅ Executed — lineage depth visibly truncates at each checkpoint |
| Explain how checkpointing prevents StackOverflow and stabilizes recovery time | `part3_theory_fault_tolerance.md`; inline comments in `pyspark_checkpointing.py` | ✅ Done |

## Part 4: Advanced Execution Mechanics & Resilience Strategies

| Requirement | Deliverable | Status |
|---|---|---|
| Explain Lazy Evaluation | `part4_advanced_execution.md` §4.1 | ✅ Done |
| Explain DAG construction and Stage decomposition at Wide Dependencies | `part4_advanced_execution.md` §4.1, mapped to the actual `read → filter → groupBy → agg` pipeline from Part 3 | ✅ Done |
| Explain "Don't move data, move code" / Data Locality | `part4_advanced_execution.md` §4.2 | ✅ Done |
| Contrast Spark lineage recovery vs Hadoop replication | `part4_advanced_execution.md` §4.2 (also covered in §3.3) | ✅ Done |
| Explain the "Liability of Lineage": StackOverflow risk + degraded recovery | `part4_advanced_execution.md` §4.3, tied back to the measured lineage-depth pattern in notebook §3.4 | ✅ Done |
| Differentiate checkpointing from caching | `part4_advanced_execution.md` §4.3 (comparison table) | ✅ Done |

---

## Execution Verification

All PySpark code in this submission was executed end-to-end against the real `vehicle_telemetry_dataset.csv` (not synthetic placeholder data) in a local Spark session (`local[*]`, PySpark 4.x), with zero runtime errors:

- `Global_Telemetry_Platform_Assignment.ipynb` — 26 cells (14 code cells), executed top to bottom, 0 errors.
- `code/pyspark_transformations.py` — runs standalone, produces the average-temperature table shown above.
- `code/pyspark_salting_optimization.py` — runs standalone, verifies salted result == direct aggregation result (max diff ≈ 0.0), and validates skew mitigation at ~424x synthetic amplification.
- `code/pyspark_checkpointing.py` — runs standalone, demonstrates lineage-depth growth and truncation across 15 iterations with checkpoints every 5th iteration.

## Notes on Data Fidelity

The assignment scenario describes a hypothetical 500,000-vehicle production fleet; the actual dataset provided is a 3,000-row historical batch extract from 100 vehicles. Part 1's quantitative "Wall" analysis uses the assignment's stated 500,000-vehicle fleet size (as instructed), while Part 3's PySpark pipeline processes the real historical extract directly, as instructed ("process a historical batch dataset of this telemetry data"). Where the real data's natural skew (~2.3x across vehicle models) was milder than the assignment's stated severity ("1000x more logs"), a synthetic amplification was added on top of the real data specifically to validate the salting strategy under that stated severity — this is clearly labeled as synthetic in both the notebook and the script.
