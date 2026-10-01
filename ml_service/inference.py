"""
ML Inference Service — Continuous RUL Prediction
─────────────────────────────────────────────────
Polls ScyllaDB for the latest sensor readings, computes rolling features,
runs the trained XGBoost model, and writes RUL predictions + health status
back to ScyllaDB.

Loop:
  1. Get all active machine IDs from sensor_readings
  2. For each machine, fetch the last N cycles
  3. Compute the same rolling features used during training
  4. Run inference → predicted_rul
  5. Classify health: HEALTHY / WARNING / CRITICAL
  6. Write to rul_predictions table
"""

import os
import time
import logging
from datetime import datetime, timezone

try:
    import joblib
    import numpy as np
    import pandas as pd
    from cassandra.cluster import Cluster
except ImportError:
    pass
from cassandra.query import SimpleStatement

# ── Configuration ───────────────────────────────────────────
SCYLLA_HOST = os.getenv("SCYLLA_HOST", "scylladb")
SCYLLA_PORT = int(os.getenv("SCYLLA_PORT", 9042))
SCYLLA_KEYSPACE = os.getenv("SCYLLA_KEYSPACE", "iot_maintenance")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 5))
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

WINDOW_SIZE = 5
RUL_CAP = 125

# Health thresholds
HEALTHY_THRESHOLD = 50     # RUL > 50 → green
WARNING_THRESHOLD = 15     # 15 < RUL ≤ 50 → yellow
                           # RUL ≤ 15 → red

# Same sensors dropped during training (near-constant in FD001)
DROP_SENSORS = {
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
}
ACTIVE_SENSORS = [
    f"sensor_{i}" for i in range(1, 22)
    if f"sensor_{i}" not in DROP_SENSORS
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [ML-SERVICE]  %(message)s",
)
log = logging.getLogger(__name__)

# Track last-predicted cycle per machine to avoid redundant work
_last_predicted: dict[str, int] = {}


# ── Model Loading ──────────────────────────────────────────

def load_artifacts(model_dir: str):
    """Load the trained model, scaler, and feature column list."""
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
    log.info("Loaded model (%d features), scaler, feature_cols", len(feature_cols))
    return model, scaler, feature_cols


# ── Feature Engineering ────────────────────────────────────

def build_feature_vector(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame | None:
    """
    Given the last N rows for one machine (sorted by cycle ASC),
    build a single-row feature vector matching the training schema.
    """
    if df.empty:
        return None

    latest = df.iloc[-1].copy()

    # Compute rolling features for active sensors
    for sensor in ACTIVE_SENSORS:
        latest[f"{sensor}_rm{WINDOW_SIZE}"] = df[sensor].mean()
        if len(df) > 1:
            latest[f"{sensor}_rs{WINDOW_SIZE}"] = df[sensor].std()
        else:
            latest[f"{sensor}_rs{WINDOW_SIZE}"] = 0.0

    # Build DataFrame with the exact column order used during training
    try:
        X = pd.DataFrame([latest])[feature_cols].astype(float)
    except KeyError as e:
        log.warning("Missing feature columns: %s", e)
        return None

    return X


def get_health_status(rul: float) -> str:
    """Map predicted RUL to a health status label."""
    if rul > HEALTHY_THRESHOLD:
        return "HEALTHY"
    elif rul > WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "CRITICAL"


# ── Prediction Loop ────────────────────────────────────────

def predict_for_machine(
    session,
    model,
    scaler,
    feature_cols: list[str],
    machine_id: str,
) -> tuple[float, str] | None:
    """Fetch latest readings, predict RUL, write to rul_predictions."""

    # Query last WINDOW_SIZE readings (table clustered DESC so LIMIT works)
    rows = session.execute(
        SimpleStatement(
            "SELECT * FROM sensor_readings WHERE machine_id = %s LIMIT %s"
        ),
        (machine_id, WINDOW_SIZE),
    )
    df = pd.DataFrame(list(rows))
    if df.empty:
        return None

    df = df.sort_values("cycle", ascending=True).reset_index(drop=True)
    latest_cycle = int(df.iloc[-1]["cycle"])

    # Skip if we already predicted for this cycle
    if _last_predicted.get(machine_id) == latest_cycle:
        return None

    # Drop columns not needed for features
    df = df.drop(columns=["op_setting_3"] + list(DROP_SENSORS), errors="ignore")

    # Build feature vector
    X = build_feature_vector(df, feature_cols)
    if X is None:
        return None

    # Scale and predict
    X_scaled = scaler.transform(X)
    predicted_rul = float(model.predict(X_scaled)[0])
    predicted_rul = max(0.0, min(predicted_rul, float(RUL_CAP)))
    health_status = get_health_status(predicted_rul)

    # Write prediction to ScyllaDB
    session.execute(
        SimpleStatement(
            """INSERT INTO rul_predictions
               (machine_id, cycle, predicted_rul, health_status, predicted_at)
               VALUES (%s, %s, %s, %s, %s)"""
        ),
        (machine_id, latest_cycle, predicted_rul, health_status, datetime.now(timezone.utc)),
    )

    # Also write rolling features to sensor_features
    feature_values = {"machine_id": machine_id, "cycle": latest_cycle}
    for sensor in ACTIVE_SENSORS:
        feature_values[f"{sensor}_rm5"] = float(df[sensor].mean())
        feature_values[f"{sensor}_rs5"] = float(df[sensor].std()) if len(df) > 1 else 0.0

    cols = ", ".join(feature_values.keys())
    placeholders = ", ".join(["%s"] * len(feature_values))
    session.execute(
        SimpleStatement(f"INSERT INTO sensor_features ({cols}) VALUES ({placeholders})"),
        list(feature_values.values()),
    )

    _last_predicted[machine_id] = latest_cycle
    return predicted_rul, health_status


def main():
    # Load model artifacts
    model, scaler, feature_cols = load_artifacts(MODEL_DIR)

    # Connect to ScyllaDB (retry on startup)
    session = None
    for attempt in range(30):
        try:
            cluster = Cluster([SCYLLA_HOST], port=SCYLLA_PORT)
            session = cluster.connect(SCYLLA_KEYSPACE)
            log.info("Connected to ScyllaDB at %s:%d", SCYLLA_HOST, SCYLLA_PORT)
            break
        except Exception as e:
            log.warning("ScyllaDB not ready (attempt %d/30): %s", attempt + 1, e)
            time.sleep(5)

    if session is None:
        log.error("Could not connect to ScyllaDB — exiting.")
        return

    # ── Main prediction loop ────────────────────────────────
    log.info("Starting prediction loop (interval=%ds) …", POLL_INTERVAL)

    while True:
        try:
            # Get all active machines
            rows = session.execute("SELECT DISTINCT machine_id FROM sensor_readings")
            machines = [row.machine_id for row in rows]

            predictions_made = 0
            for machine_id in machines:
                result = predict_for_machine(
                    session, model, scaler, feature_cols, machine_id
                )
                if result:
                    rul, status = result
                    predictions_made += 1
                    emoji = {"HEALTHY": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}[status]
                    log.info(
                        "%s  %s  RUL=%.1f  %s",
                        emoji, machine_id, rul, status,
                    )

            if predictions_made > 0:
                log.info("Batch complete: %d predictions for %d machines",
                         predictions_made, len(machines))

        except Exception as e:
            log.error("Prediction loop error: %s", e, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

# ── Vercel Serverless Stub ──────────────────────────────────
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b"ML Service (Vercel Stub)"]
