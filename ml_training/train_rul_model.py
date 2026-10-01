"""
Offline RUL Model Training — NASA C-MAPSS FD001
────────────────────────────────────────────────
Trains an XGBoost Regressor to predict the Remaining Useful Life (RUL)
of turbofan engines based on sensor readings.

Pipeline:
  1. Load preprocessed sensor_data.csv (FD001)
  2. Compute RUL labels (capped at 125 cycles)
  3. Drop near-constant sensors (zero information gain)
  4. Engineer rolling features (mean & std over 5-cycle window)
  5. Train/test split by engine (no data leakage)
  6. Scale features → train XGBoost → evaluate
  7. Save model.pkl, scaler.pkl, feature_cols.pkl → models/

Run:
  cd Predictive-Maintenance
  pip install -r ml_training/requirements.txt
  python3 ml_training/train_rul_model.py
"""

import os
import sys
import json
import logging

try:
    import joblib
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    class Mock:
        def __getattr__(self, name): return Mock()
        def __call__(self, *args, **kwargs): return Mock()
    joblib = np = pd = xgb = train_test_split = StandardScaler = mean_squared_error = mean_absolute_error = r2_score = Mock()

# ── Configuration ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "iot_simulator", "data", "sensor_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

RUL_CAP = 125          # Cap RUL so model focuses on degradation phase
WINDOW_SIZE = 5         # Rolling window for feature engineering
TEST_SIZE = 0.2         # Fraction of engines held out for evaluation
RANDOM_STATE = 42

# Sensors with near-zero variance in FD001 (no predictive value)
DROP_SENSORS = [
    "sensor_1",   # Fan inlet temp — constant
    "sensor_5",   # Fan inlet pressure — constant
    "sensor_6",   # Bypass duct pressure — constant
    "sensor_10",  # Engine pressure ratio — constant
    "sensor_16",  # Burner fuel-air ratio — constant
    "sensor_18",  # Demanded fan speed — constant
    "sensor_19",  # Demanded corrected fan speed — constant
]

# Operational setting 3 is constant (100.0) in FD001
DROP_SETTINGS = ["op_setting_3"]

# Active sensor columns after filtering
ACTIVE_SENSORS = [
    f"sensor_{i}" for i in range(1, 22)
    if f"sensor_{i}" not in DROP_SENSORS
]
# → sensor_2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21  (14 sensors)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [TRAINING]  %(message)s",
)
log = logging.getLogger(__name__)


# ── Feature Engineering ─────────────────────────────────────

def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RUL = max_cycle_per_engine − current_cycle, capped at RUL_CAP."""
    max_cycles = df.groupby("machine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["machine_id", "max_cycle"]
    df = df.merge(max_cycles, on="machine_id")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CAP)
    df.drop("max_cycle", axis=1, inplace=True)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str],
    window: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Add rolling mean (_rm5) and rolling std (_rs5) per engine per sensor."""
    for sensor in sensors:
        grouped = df.groupby("machine_id")[sensor]
        df[f"{sensor}_rm{window}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{sensor}_rs{window}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
    return df


# ── Training Pipeline ───────────────────────────────────────

def main():
    # ── 1. Load data ────────────────────────────────────────
    log.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    n_engines = df["machine_id"].nunique()
    log.info("Loaded %d rows × %d columns (%d engines)", len(df), len(df.columns), n_engines)

    # ── 2. Compute RUL labels ───────────────────────────────
    df = compute_rul(df)
    log.info("RUL computed (capped at %d). Distribution:", RUL_CAP)
    log.info("  mean=%.1f  std=%.1f  min=%d  max=%d",
             df["rul"].mean(), df["rul"].std(), df["rul"].min(), df["rul"].max())

    # ── 3. Drop low-variance columns ───────────────────────
    df = df.drop(columns=DROP_SENSORS + DROP_SETTINGS, errors="ignore")
    log.info("Dropped %d constant columns → %d columns remaining",
             len(DROP_SENSORS) + len(DROP_SETTINGS), len(df.columns))

    # ── 4. Engineer rolling features ───────────────────────
    log.info("Computing rolling features (window=%d) for %d sensors …",
             WINDOW_SIZE, len(ACTIVE_SENSORS))
    df = add_rolling_features(df, ACTIVE_SENSORS, WINDOW_SIZE)
    log.info("Feature matrix: %d rows × %d columns", len(df), len(df.columns))

    # ── 5. Prepare feature matrix ──────────────────────────
    exclude = {"machine_id", "rul"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].astype(float)
    y = df["rul"].astype(float)

    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    # ── 6. Train/test split by engine (no leakage) ─────────
    engines = list(df["machine_id"].unique())
    train_engines, test_engines = train_test_split(
        engines, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_mask = df["machine_id"].isin(train_engines)

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    log.info("Split: %d train engines (%d rows) / %d test engines (%d rows)",
             len(train_engines), len(X_train), len(test_engines), len(X_test))

    # ── 7. Scale features ──────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 8. Train XGBoost ───────────────────────────────────
    log.info("Training XGBoost Regressor …")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50,
    )

    # ── 9. Evaluate ────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    log.info("=" * 50)
    log.info("  EVALUATION RESULTS")
    log.info("  RMSE : %.2f cycles", rmse)
    log.info("  MAE  : %.2f cycles", mae)
    log.info("  R²   : %.4f", r2)
    log.info("=" * 50)

    # ── 10. Feature importance (top 10) ────────────────────
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    log.info("Top 10 features by importance:")
    for feat, imp in importance.head(10).items():
        log.info("  %-25s  %.4f", feat, imp)

    # ── 11. Save artifacts ─────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    # Save metadata for reproducibility
    metadata = {
        "dataset": "NASA C-MAPSS FD001",
        "n_engines": int(n_engines),
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "active_sensors": ACTIVE_SENSORS,
        "dropped_sensors": DROP_SENSORS,
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "test_size": TEST_SIZE,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "model_params": model.get_params(),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("✅  Artifacts saved to %s", MODEL_DIR)
    log.info("    model.pkl, scaler.pkl, feature_cols.pkl, metadata.json")


if __name__ == "__main__":
    main()

# ── Vercel Serverless Stub ──────────────────────────────────
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b"ML Training (Vercel Stub)"]
