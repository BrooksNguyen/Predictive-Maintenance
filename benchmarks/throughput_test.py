"""
Benchmarking Suite — IoT Predictive Maintenance Pipeline
────────────────────────────────────────────────────────
Measures key performance indicators for the streaming pipeline:

  1. Simulator Throughput   — records received per second over TCP
  2. ScyllaDB Write Latency — single-row insert P50/P95/P99
  3. ScyllaDB Read Latency  — single-partition query P50/P95/P99
  4. End-to-End Latency     — estimate from row counts over time

Usage:
  pip install cassandra-driver
  python3 benchmarks/throughput_test.py

  (Requires ScyllaDB + Simulator to be running via docker compose)
"""

import os
import sys
import time
import socket
import logging
import statistics
from datetime import datetime, timezone

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [BENCH]  %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────
SIMULATOR_HOST = os.getenv("SIMULATOR_HOST", "localhost")
SIMULATOR_PORT = int(os.getenv("SIMULATOR_PORT", 9999))
SCYLLA_HOST = os.getenv("SCYLLA_HOST", "localhost")
SCYLLA_PORT = int(os.getenv("SCYLLA_PORT", 9042))
SCYLLA_KEYSPACE = os.getenv("SCYLLA_KEYSPACE", "iot_maintenance")


def format_latency_stats(latencies_ms: list[float], label: str) -> str:
    """Format latency statistics for display."""
    if not latencies_ms:
        return f"{label}: no data"
    arr = sorted(latencies_ms)
    return (
        f"{label}:\n"
        f"  Count : {len(arr)}\n"
        f"  Mean  : {statistics.mean(arr):.2f} ms\n"
        f"  P50   : {arr[len(arr) // 2]:.2f} ms\n"
        f"  P95   : {arr[int(len(arr) * 0.95)]:.2f} ms\n"
        f"  P99   : {arr[int(len(arr) * 0.99)]:.2f} ms\n"
        f"  Min   : {arr[0]:.2f} ms\n"
        f"  Max   : {arr[-1]:.2f} ms"
    )


# ── Test 1: Simulator Throughput ────────────────────────────

def bench_simulator_throughput(duration_sec: int = 10) -> float:
    """Connect to the simulator TCP socket and count records/sec."""
    log.info("═" * 60)
    log.info("TEST 1: Simulator Throughput (duration=%ds)", duration_sec)
    log.info("═" * 60)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((SIMULATOR_HOST, SIMULATOR_PORT))
    except (ConnectionRefusedError, socket.timeout):
        log.error("Cannot connect to simulator at %s:%d", SIMULATOR_HOST, SIMULATOR_PORT)
        return 0.0

    buffer = ""
    count = 0
    start = time.monotonic()
    deadline = start + duration_sec

    try:
        while time.monotonic() < deadline:
            data = sock.recv(4096).decode("utf-8", errors="replace")
            if not data:
                break
            buffer += data
            lines = buffer.split("\n")
            # Last element may be incomplete
            buffer = lines[-1]
            count += len(lines) - 1
    except socket.timeout:
        pass
    finally:
        sock.close()

    elapsed = time.monotonic() - start
    throughput = count / elapsed if elapsed > 0 else 0
    log.info("  Records received : %d", count)
    log.info("  Elapsed          : %.2f s", elapsed)
    log.info("  Throughput       : %.1f records/sec", throughput)
    return throughput


# ── Test 2: ScyllaDB Write Latency ─────────────────────────

def bench_scylladb_write_latency(num_writes: int = 500) -> list[float]:
    """Measure single-row INSERT latency into sensor_readings."""
    log.info("═" * 60)
    log.info("TEST 2: ScyllaDB Write Latency (n=%d)", num_writes)
    log.info("═" * 60)

    try:
        from cassandra.cluster import Cluster
    except ImportError:
        log.error("cassandra-driver not installed. pip install cassandra-driver")
        return []

    cluster = Cluster([SCYLLA_HOST], port=SCYLLA_PORT)
    session = cluster.connect(SCYLLA_KEYSPACE)

    insert_cql = session.prepare(
        """INSERT INTO sensor_readings
           (machine_id, cycle, op_setting_1, op_setting_2, op_setting_3,
            sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6,
            sensor_7, sensor_8, sensor_9, sensor_10, sensor_11, sensor_12,
            sensor_13, sensor_14, sensor_15, sensor_16, sensor_17,
            sensor_18, sensor_19, sensor_20, sensor_21, sensor_avg)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    )

    latencies = []
    for i in range(num_writes):
        values = (
            f"bench_engine_{i % 10:03d}",  # machine_id
            100000 + i,                     # cycle (offset to avoid collisions)
            0.001, 0.0002, 100.0,           # op settings
            *[float(np.random.normal(500, 10)) for _ in range(21)],  # sensors
            500.0,                          # sensor_avg
        )
        t0 = time.monotonic()
        session.execute(insert_cql, values)
        latencies.append((time.monotonic() - t0) * 1000)

    # Cleanup benchmark data
    for i in range(num_writes):
        session.execute(
            "DELETE FROM sensor_readings WHERE machine_id = %s AND cycle = %s",
            (f"bench_engine_{i % 10:03d}", 100000 + i),
        )

    cluster.shutdown()
    log.info(format_latency_stats(latencies, "  Write latency"))
    return latencies


# ── Test 3: ScyllaDB Read Latency ──────────────────────────

def bench_scylladb_read_latency(num_reads: int = 200) -> list[float]:
    """Measure single-partition SELECT latency from sensor_readings."""
    log.info("═" * 60)
    log.info("TEST 3: ScyllaDB Read Latency (n=%d)", num_reads)
    log.info("═" * 60)

    try:
        from cassandra.cluster import Cluster
    except ImportError:
        log.error("cassandra-driver not installed. pip install cassandra-driver")
        return []

    cluster = Cluster([SCYLLA_HOST], port=SCYLLA_PORT)
    session = cluster.connect(SCYLLA_KEYSPACE)

    # Get available machine IDs
    rows = list(session.execute("SELECT DISTINCT machine_id FROM sensor_readings"))
    if not rows:
        log.warning("No data in sensor_readings — skipping read benchmark")
        cluster.shutdown()
        return []

    machine_ids = [r.machine_id for r in rows]

    read_cql = session.prepare(
        "SELECT * FROM sensor_readings WHERE machine_id = ? LIMIT 50"
    )

    latencies = []
    for i in range(num_reads):
        mid = machine_ids[i % len(machine_ids)]
        t0 = time.monotonic()
        list(session.execute(read_cql, (mid,)))
        latencies.append((time.monotonic() - t0) * 1000)

    cluster.shutdown()
    log.info(format_latency_stats(latencies, "  Read latency"))
    return latencies


# ── Test 4: Pipeline Row Growth Rate ───────────────────────

def bench_pipeline_throughput(measure_sec: int = 30) -> float:
    """Estimate end-to-end throughput by counting row growth in ScyllaDB."""
    log.info("═" * 60)
    log.info("TEST 4: Pipeline Throughput (duration=%ds)", measure_sec)
    log.info("═" * 60)

    try:
        from cassandra.cluster import Cluster
    except ImportError:
        log.error("cassandra-driver not installed.")
        return 0.0

    cluster = Cluster([SCYLLA_HOST], port=SCYLLA_PORT)
    session = cluster.connect(SCYLLA_KEYSPACE)

    def count_rows():
        # Approximate count using token range (faster than COUNT(*))
        rows = list(session.execute("SELECT COUNT(*) FROM sensor_readings"))
        return rows[0].count if rows else 0

    count_start = count_rows()
    log.info("  Starting row count: %d", count_start)
    time.sleep(measure_sec)
    count_end = count_rows()
    log.info("  Ending row count  : %d", count_end)

    new_rows = count_end - count_start
    throughput = new_rows / measure_sec if measure_sec > 0 else 0
    log.info("  New rows          : %d", new_rows)
    log.info("  Pipeline throughput: %.1f records/sec", throughput)

    cluster.shutdown()
    return throughput


# ── Main ────────────────────────────────────────────────────

def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  IoT Predictive Maintenance — Benchmark Suite          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("")

    results = {}

    # Run all benchmarks
    results["simulator_throughput_rps"] = bench_simulator_throughput(10)
    log.info("")
    results["write_latencies_ms"] = bench_scylladb_write_latency(500)
    log.info("")
    results["read_latencies_ms"] = bench_scylladb_read_latency(200)
    log.info("")
    results["pipeline_throughput_rps"] = bench_pipeline_throughput(30)

    # Summary
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  SUMMARY                                               ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Simulator throughput : %8.1f records/sec             ║",
             results["simulator_throughput_rps"])
    if results["write_latencies_ms"]:
        wl = sorted(results["write_latencies_ms"])
        log.info("║  Write latency P50   : %8.2f ms                    ║", wl[len(wl)//2])
        log.info("║  Write latency P99   : %8.2f ms                    ║", wl[int(len(wl)*0.99)])
    if results["read_latencies_ms"]:
        rl = sorted(results["read_latencies_ms"])
        log.info("║  Read latency  P50   : %8.2f ms                    ║", rl[len(rl)//2])
        log.info("║  Read latency  P99   : %8.2f ms                    ║", rl[int(len(rl)*0.99)])
    log.info("║  Pipeline throughput  : %8.1f records/sec             ║",
             results["pipeline_throughput_rps"])
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
