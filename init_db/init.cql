-- ============================================================
--  ScyllaDB Schema Initialization
--  Keyspace: iot_maintenance
--
--  Tables:
--    1. sensor_readings  — raw ingested sensor data (from Spark ETL)
--    2. sensor_features  — rolling statistics (from ML service)
--    3. rul_predictions  — RUL forecasts + health status (from ML service)
--
--  NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)
--  26 raw columns: unit, cycle, 3 op_settings, 21 sensors
-- ============================================================

CREATE KEYSPACE IF NOT EXISTS iot_maintenance
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

USE iot_maintenance;

-- ── Table 1: Raw Sensor Readings ───────────────────────────
-- Partitioned by machine_id, clustered by cycle DESC.
-- Efficient for: "latest N readings for engine X"
CREATE TABLE IF NOT EXISTS sensor_readings (
    machine_id      TEXT,
    cycle           INT,
    op_setting_1    DOUBLE,
    op_setting_2    DOUBLE,
    op_setting_3    DOUBLE,
    sensor_1        DOUBLE,
    sensor_2        DOUBLE,
    sensor_3        DOUBLE,
    sensor_4        DOUBLE,
    sensor_5        DOUBLE,
    sensor_6        DOUBLE,
    sensor_7        DOUBLE,
    sensor_8        DOUBLE,
    sensor_9        DOUBLE,
    sensor_10       DOUBLE,
    sensor_11       DOUBLE,
    sensor_12       DOUBLE,
    sensor_13       DOUBLE,
    sensor_14       DOUBLE,
    sensor_15       DOUBLE,
    sensor_16       DOUBLE,
    sensor_17       DOUBLE,
    sensor_18       DOUBLE,
    sensor_19       DOUBLE,
    sensor_20       DOUBLE,
    sensor_21       DOUBLE,
    sensor_avg      DOUBLE,
    PRIMARY KEY (machine_id, cycle)
) WITH CLUSTERING ORDER BY (cycle DESC);


-- ── Table 2: Engineered Rolling Features ───────────────────
-- Rolling mean (_rm5) and rolling std (_rs5) over 5-cycle window
-- for the 14 active sensors (constant sensors excluded).
-- Written by the ML inference service after computing features.
CREATE TABLE IF NOT EXISTS sensor_features (
    machine_id      TEXT,
    cycle           INT,
    sensor_2_rm5    DOUBLE,
    sensor_2_rs5    DOUBLE,
    sensor_3_rm5    DOUBLE,
    sensor_3_rs5    DOUBLE,
    sensor_4_rm5    DOUBLE,
    sensor_4_rs5    DOUBLE,
    sensor_7_rm5    DOUBLE,
    sensor_7_rs5    DOUBLE,
    sensor_8_rm5    DOUBLE,
    sensor_8_rs5    DOUBLE,
    sensor_9_rm5    DOUBLE,
    sensor_9_rs5    DOUBLE,
    sensor_11_rm5   DOUBLE,
    sensor_11_rs5   DOUBLE,
    sensor_12_rm5   DOUBLE,
    sensor_12_rs5   DOUBLE,
    sensor_13_rm5   DOUBLE,
    sensor_13_rs5   DOUBLE,
    sensor_14_rm5   DOUBLE,
    sensor_14_rs5   DOUBLE,
    sensor_15_rm5   DOUBLE,
    sensor_15_rs5   DOUBLE,
    sensor_17_rm5   DOUBLE,
    sensor_17_rs5   DOUBLE,
    sensor_20_rm5   DOUBLE,
    sensor_20_rs5   DOUBLE,
    sensor_21_rm5   DOUBLE,
    sensor_21_rs5   DOUBLE,
    PRIMARY KEY (machine_id, cycle)
) WITH CLUSTERING ORDER BY (cycle DESC);


-- ── Table 3: RUL Predictions ──────────────────────────────
-- Predicted Remaining Useful Life + health classification.
-- Written by the ML inference service.
-- health_status: HEALTHY (RUL > 50), WARNING (15 < RUL ≤ 50), CRITICAL (RUL ≤ 15)
CREATE TABLE IF NOT EXISTS rul_predictions (
    machine_id      TEXT,
    cycle           INT,
    predicted_rul   DOUBLE,
    health_status   TEXT,
    predicted_at    TIMESTAMP,
    PRIMARY KEY (machine_id, cycle)
) WITH CLUSTERING ORDER BY (cycle DESC);
