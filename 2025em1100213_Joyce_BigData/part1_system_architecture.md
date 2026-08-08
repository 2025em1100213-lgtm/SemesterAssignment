# Part 1: System Architecture & Data Paradigms

**Student Name:** Joyce Dorothy S
**Student ID:** 2025em1100213

---

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


---

*End of Part 1: System Architecture & Data Paradigms*
