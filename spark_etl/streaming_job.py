"""
PySpark Structured Streaming ETL Job
─────────────────────────────────────
NASA C-MAPSS Turbofan Engine Degradation Dataset

Pipeline:
  1. Reads newline-delimited CSV rows from the IoT simulator TCP socket.
  2. Parses the 26-column schema (unit, cycle, 3 op_settings, 21 sensors).
  3. Computes sensor_avg — mean of all 21 sensor measurements.
  4. Writes raw data to ScyllaDB `sensor_readings` table.
  5. Computes rolling features (mean & std over 5-cycle window) per machine,
     writes to `sensor_features` table via cassandra-driver.
"""

import os
import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split
from pyspark.sql.types import DoubleType, IntegerType
from cassandra.cluster import Cluster as CassandraCluster
from cassandra.query import SimpleStatement

# ── Configuration (injected via Docker env vars) ───────────
SPARK_MASTER = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
SIMULATOR_HOST = os.getenv("SIMULATOR_HOST", "iot-simulator")
SIMULATOR_PORT = int(os.getenv("SIMULATOR_PORT", 9999))
SCYLLA_HOST = os.getenv("SCYLLA_HOST", "scylladb")
SCYLLA_PORT = os.getenv("SCYLLA_PORT", "9042")
SCYLLA_KEYSPACE = os.getenv("SCYLLA_KEYSPACE", "iot_maintenance")
SCYLLA_TABLE = os.getenv("SCYLLA_TABLE", "sensor_readings")

WINDOW_SIZE = 5  # Rolling window for feature engineering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [SPARK-ETL]  %(message)s",
)
log = logging.getLogger(__name__)

# ── Column definitions ─────────────────────────────────────
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Active sensors for rolling features (constant sensors excluded)
ACTIVE_SENSORS = [
    f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
]

# ── Lazy Cassandra connection for foreachBatch ─────────────
_cass_session = None


def get_cassandra_session():
    """Get or create a direct cassandra-driver session (for rolling features)."""
    global _cass_session
    if _cass_session is None:
        cluster = CassandraCluster([SCYLLA_HOST], port=int(SCYLLA_PORT))
        _cass_session = cluster.connect(SCYLLA_KEYSPACE)
        log.info("Cassandra direct session created for feature computation")
    return _cass_session


def create_spark_session() -> SparkSession:
    """Build a SparkSession wired to ScyllaDB via the Cassandra connector."""
    return (
        SparkSession.builder
        .appName("IoT-Predictive-Maintenance-ETL")
        .master(SPARK_MASTER)
        .config(
            "spark.jars.packages",
            "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
        )
        .config("spark.cassandra.connection.host", SCYLLA_HOST)
        .config("spark.cassandra.connection.port", SCYLLA_PORT)
        .config("spark.cassandra.output.consistency.level", "ONE")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .getOrCreate()
    )


# ── Rolling Feature Computation ────────────────────────────

def compute_and_write_features(machine_id: str) -> None:
    """
    Query the last WINDOW_SIZE readings for a machine from ScyllaDB,
    compute rolling mean & std for each active sensor,
    and write the features to the sensor_features table.
    """
    session = get_cassandra_session()

    rows = session.execute(
        SimpleStatement(
            "SELECT * FROM sensor_readings WHERE machine_id = %s LIMIT %s"
        ),
        (machine_id, WINDOW_SIZE),
    )
    df = pd.DataFrame(list(rows))
    if df.empty:
        return

    df = df.sort_values("cycle", ascending=True)
    latest_cycle = int(df.iloc[-1]["cycle"])

    # Build feature row
    values = {"machine_id": machine_id, "cycle": latest_cycle}
    for sensor in ACTIVE_SENSORS:
        values[f"{sensor}_rm5"] = float(df[sensor].mean())
        values[f"{sensor}_rs5"] = float(df[sensor].std()) if len(df) > 1 else 0.0

    cols = ", ".join(values.keys())
    placeholders = ", ".join(["%s"] * len(values))
    session.execute(
        SimpleStatement(f"INSERT INTO sensor_features ({cols}) VALUES ({placeholders})"),
        list(values.values()),
    )


# ── Micro-Batch Sink ──────────────────────────────────────

def write_to_scylladb(batch_df, batch_id):
    """
    foreachBatch sink:
      1. Write raw sensor data to sensor_readings (via Spark connector)
      2. Compute rolling features for each machine in the batch
    """
    if batch_df.rdd.isEmpty():
        return

    row_count = batch_df.count()
    log.info("Batch %d: %d rows — writing raw data …", batch_id, row_count)

    # ── Step 1: Write raw data via spark-cassandra-connector ──
    (
        batch_df.write
        .format("org.apache.spark.sql.cassandra")
        .mode("append")
        .options(table=SCYLLA_TABLE, keyspace=SCYLLA_KEYSPACE)
        .save()
    )

    # ── Step 2: Compute rolling features per machine ─────────
    machines = batch_df.select("machine_id").distinct().collect()
    for row in machines:
        try:
            compute_and_write_features(row.machine_id)
        except Exception as e:
            log.warning("Failed to compute features for %s: %s", row.machine_id, e)

    log.info(
        "Batch %d complete: %d rows written, features computed for %d machines",
        batch_id, row_count, len(machines),
    )


# ── Main ────────────────────────────────────────────────────

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    log.info("SparkSession created — connected to %s", SPARK_MASTER)

    # ── 1. Read raw TCP stream ──────────────────────────────
    raw_stream = (
        spark.readStream
        .format("socket")
        .option("host", SIMULATOR_HOST)
        .option("port", SIMULATOR_PORT)
        .load()
    )

    # ── 2. Parse CSV rows ───────────────────────────────────
    # Expected: machine_id,cycle,op1,op2,op3,s1,s2,...,s21  (26 fields)
    fields = split(col("value"), ",")

    parsed = raw_stream.select(
        fields.getItem(0).alias("machine_id"),
        fields.getItem(1).cast(IntegerType()).alias("cycle"),
        fields.getItem(2).cast(DoubleType()).alias("op_setting_1"),
        fields.getItem(3).cast(DoubleType()).alias("op_setting_2"),
        fields.getItem(4).cast(DoubleType()).alias("op_setting_3"),
        *[
            fields.getItem(5 + i).cast(DoubleType()).alias(f"sensor_{i + 1}")
            for i in range(21)
        ],
    )

    # ── 3. Compute sensor_avg (mean of all 21 sensors) ─────
    sensor_sum = sum(col(s) for s in SENSOR_COLS)
    enriched = parsed.withColumn("sensor_avg", sensor_sum / len(SENSOR_COLS))

    # ── 4. Write to ScyllaDB (raw + features) ──────────────
    query = (
        enriched.writeStream
        .outputMode("append")
        .foreachBatch(write_to_scylladb)
        .option("checkpointLocation", "/tmp/spark-checkpoints/iot-etl")
        .trigger(processingTime="5 seconds")
        .start()
    )

    log.info("Streaming query started — awaiting termination …")
    query.awaitTermination()


if __name__ == "__main__":
    main()
