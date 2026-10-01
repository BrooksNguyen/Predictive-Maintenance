"""
Streamlit Real-Time Monitoring Dashboard
─────────────────────────────────────────
NASA C-MAPSS Turbofan Engine — Predictive Maintenance

Features:
  • Fleet overview with health status badges (🟢 🟡 🔴)
  • Per-engine RUL gauge and degradation curve
  • Multi-sensor time-series charts
  • Rolling feature visualization
  • Operational settings inspector
  • Raw data explorer
  • Auto-refresh (5s)
"""

import os
import time

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

# ── Configuration ───────────────────────────────────────────
SCYLLA_HOST = os.getenv("SCYLLA_HOST", "scylladb")
SCYLLA_PORT = int(os.getenv("SCYLLA_PORT", 9042))
SCYLLA_KEYSPACE = os.getenv("SCYLLA_KEYSPACE", "iot_maintenance")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 5))

RUL_CAP = 125
HEALTHY_THRESHOLD = 50
WARNING_THRESHOLD = 15

# Sensor labels (NASA C-MAPSS documentation)
SENSOR_LABELS = {
    "sensor_1":  "Fan inlet temp (°R)",
    "sensor_2":  "LPC outlet temp (°R)",
    "sensor_3":  "HPC outlet temp (°R)",
    "sensor_4":  "LPT outlet temp (°R)",
    "sensor_5":  "Fan inlet pressure (psia)",
    "sensor_6":  "Bypass-duct pressure (psia)",
    "sensor_7":  "HPC outlet pressure (psia)",
    "sensor_8":  "Physical fan speed (rpm)",
    "sensor_9":  "Physical core speed (rpm)",
    "sensor_10": "Engine pressure ratio",
    "sensor_11": "HPC outlet static pressure (psia)",
    "sensor_12": "Fuel flow ratio (pps/psia)",
    "sensor_13": "Corrected fan speed (rpm)",
    "sensor_14": "Corrected core speed (rpm)",
    "sensor_15": "Bypass ratio",
    "sensor_16": "Burner fuel-air ratio",
    "sensor_17": "Bleed enthalpy",
    "sensor_18": "Demanded fan speed (rpm)",
    "sensor_19": "Demanded corrected fan speed (rpm)",
    "sensor_20": "HPT coolant bleed (lbm/s)",
    "sensor_21": "LPT coolant bleed (lbm/s)",
}

ALL_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ACTIVE_SENSORS = [f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
DEFAULT_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]

# ── Page Config ─────────────────────────────────────────────
# Moved to main() to allow safe Vercel WSGI imports


# ── Database Connection ────────────────────────────────────

@st.cache_resource
def get_session():
    """Create and cache a ScyllaDB session with retries."""
    for attempt in range(15):
        try:
            cluster = Cluster(contact_points=[SCYLLA_HOST], port=SCYLLA_PORT)
            return cluster.connect(SCYLLA_KEYSPACE)
        except Exception as e:
            time.sleep(2)
    st.error("Failed to connect to ScyllaDB. Please check if the container is running.")
    st.stop()


# ── Data Fetching ──────────────────────────────────────────

def fetch_machine_ids(session) -> list[str]:
    rows = session.execute("SELECT DISTINCT machine_id FROM sensor_readings")
    return sorted([r.machine_id for r in rows])


def fetch_readings(session, machine_id: str, limit: int = 100) -> pd.DataFrame:
    rows = session.execute(
        SimpleStatement("SELECT * FROM sensor_readings WHERE machine_id = %s LIMIT %s"),
        (machine_id, limit),
    )
    df = pd.DataFrame(list(rows))
    if not df.empty:
        df = df.sort_values("cycle", ascending=True).reset_index(drop=True)
    return df


def fetch_predictions(session, machine_id: str, limit: int = 100) -> pd.DataFrame:
    rows = session.execute(
        SimpleStatement("SELECT * FROM rul_predictions WHERE machine_id = %s LIMIT %s"),
        (machine_id, limit),
    )
    df = pd.DataFrame(list(rows))
    if not df.empty:
        df = df.sort_values("cycle", ascending=True).reset_index(drop=True)
    return df


def fetch_latest_predictions(session, machine_ids: list[str]) -> pd.DataFrame:
    """Fetch the latest prediction for each machine (for fleet overview)."""
    records = []
    for mid in machine_ids:
        rows = list(session.execute(
            SimpleStatement("SELECT * FROM rul_predictions WHERE machine_id = %s LIMIT 1"),
            (mid,),
        ))
        if rows:
            records.append(rows[0]._asdict())
    return pd.DataFrame(records) if records else pd.DataFrame()


def fetch_features(session, machine_id: str, limit: int = 100) -> pd.DataFrame:
    rows = session.execute(
        SimpleStatement("SELECT * FROM sensor_features WHERE machine_id = %s LIMIT %s"),
        (machine_id, limit),
    )
    df = pd.DataFrame(list(rows))
    if not df.empty:
        df = df.sort_values("cycle", ascending=True).reset_index(drop=True)
    return df


# ── UI Components ──────────────────────────────────────────

def health_badge(status: str) -> str:
    """Return a colored emoji badge for health status."""
    return {"HEALTHY": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}.get(status, "⚪")


def health_color(status: str) -> str:
    return {"HEALTHY": "#22c55e", "WARNING": "#eab308", "CRITICAL": "#ef4444"}.get(status, "#6b7280")


def render_rul_gauge(predicted_rul: float, health_status: str):
    """Render a Plotly gauge for RUL prediction."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_rul,
        number={"suffix": " cycles", "font": {"size": 36}},
        title={"text": "Predicted Remaining Useful Life", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, RUL_CAP], "tickwidth": 1},
            "bar": {"color": health_color(health_status)},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, WARNING_THRESHOLD], "color": "rgba(239,68,68,0.15)"},
                {"range": [WARNING_THRESHOLD, HEALTHY_THRESHOLD], "color": "rgba(234,179,8,0.15)"},
                {"range": [HEALTHY_THRESHOLD, RUL_CAP], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": predicted_rul,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def render_fleet_overview(session, machines: list[str]):
    """Render a fleet-wide health status grid."""
    preds_df = fetch_latest_predictions(session, machines)

    if preds_df.empty:
        st.info("⏳ Waiting for ML service to generate predictions …")
        return

    # Summary stats
    status_counts = preds_df["health_status"].value_counts()
    total = len(preds_df)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Fleet Size", f"{total} engines")
    summary_cols[1].metric("🟢 Healthy", status_counts.get("HEALTHY", 0))
    summary_cols[2].metric("🟡 Warning", status_counts.get("WARNING", 0))
    summary_cols[3].metric("🔴 Critical", status_counts.get("CRITICAL", 0))

    st.divider()

    # Engine grid
    cols_per_row = 5
    preds_sorted = preds_df.sort_values("predicted_rul", ascending=True)

    for i in range(0, len(preds_sorted), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col_container in enumerate(cols):
            idx = i + j
            if idx >= len(preds_sorted):
                break
            row = preds_sorted.iloc[idx]
            badge = health_badge(row["health_status"])
            color = health_color(row["health_status"])
            col_container.markdown(
                f"""
                <div style="
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 12px;
                    text-align: center;
                    background: rgba(0,0,0,0.2);
                ">
                    <div style="font-size: 24px;">{badge}</div>
                    <div style="font-weight: bold; margin-top: 4px;">{row['machine_id']}</div>
                    <div style="font-size: 22px; color: {color}; font-weight: bold;">
                        RUL: {row['predicted_rul']:.0f}
                    </div>
                    <div style="font-size: 12px; color: #9ca3af;">cycle {row['cycle']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_engine_detail(session, machine_id: str, num_rows: int, selected_sensors: list[str]):
    """Render detailed view for a single engine."""
    df = fetch_readings(session, machine_id, limit=num_rows)
    preds_df = fetch_predictions(session, machine_id, limit=num_rows)
    features_df = fetch_features(session, machine_id, limit=num_rows)

    if df.empty:
        st.info(f"No readings found for **{machine_id}**.")
        return

    # ── RUL Prediction Section ──────────────────────────────
    if not preds_df.empty:
        latest_pred = preds_df.iloc[-1]
        rul = latest_pred["predicted_rul"]
        status = latest_pred["health_status"]

        rul_col, gauge_col = st.columns([1, 2])
        with rul_col:
            st.markdown(
                f"### {health_badge(status)} Health: **{status}**"
            )
            st.metric("Predicted RUL", f"{rul:.0f} cycles")
            st.metric("Current Cycle", f"{int(latest_pred['cycle'])}")
            if "predicted_at" in latest_pred.index and latest_pred["predicted_at"]:
                st.caption(f"Last prediction: {latest_pred['predicted_at']}")
        with gauge_col:
            st.plotly_chart(render_rul_gauge(rul, status), use_container_width=True)

        st.divider()

        # ── Degradation Curve (RUL over cycles) ────────────
        st.subheader("📉 Degradation Curve — Predicted RUL Over Time")
        fig_deg = go.Figure()
        fig_deg.add_trace(go.Scatter(
            x=preds_df["cycle"], y=preds_df["predicted_rul"],
            mode="lines+markers", name="Predicted RUL",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=4),
        ))
        # Threshold lines
        fig_deg.add_hline(y=WARNING_THRESHOLD, line_dash="dash",
                          line_color="#ef4444", annotation_text="CRITICAL threshold")
        fig_deg.add_hline(y=HEALTHY_THRESHOLD, line_dash="dash",
                          line_color="#eab308", annotation_text="WARNING threshold")
        fig_deg.update_layout(
            height=340, template="plotly_dark",
            xaxis_title="Cycle", yaxis_title="Predicted RUL (cycles)",
            yaxis=dict(range=[0, RUL_CAP + 10]),
        )
        st.plotly_chart(fig_deg, use_container_width=True)
    else:
        st.info("⏳ Waiting for ML predictions …")

    st.divider()

    # ── KPI Cards ──────────────────────────────────────────
    latest = df.iloc[-1]
    st.subheader(f"📊 Latest Sensor Readings — Cycle {int(latest['cycle'])}")
    kpi_cols = st.columns(6)
    kpi_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_avg"]
    kpi_labels = ["LPC Outlet", "HPC Outlet", "LPT Outlet", "HPC Press", "HPC Static", "Avg"]
    for col_c, sensor, label in zip(kpi_cols, kpi_sensors, kpi_labels):
        val = latest.get(sensor, 0)
        if len(df) >= 2:
            prev = df.iloc[-2].get(sensor, val)
            col_c.metric(label, f"{val:.2f}", f"{prev - val:+.2f}")
        else:
            col_c.metric(label, f"{val:.2f}")

    st.divider()

    # ── Sensor Trend Charts ────────────────────────────────
    if selected_sensors:
        st.subheader(f"📈 Sensor Trends — {machine_id}")
        melted = df.melt(
            id_vars=["cycle"],
            value_vars=selected_sensors,
            var_name="sensor",
            value_name="value",
        )
        fig = px.line(
            melted, x="cycle", y="value", color="sensor",
            title="Sensor Readings Over Engine Cycles",
            labels={"cycle": "Cycle", "value": "Reading", "sensor": "Sensor"},
        )
        fig.update_layout(
            height=460, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Rolling Features ───────────────────────────────────
    if not features_df.empty:
        with st.expander("📐 Rolling Features (5-cycle window)", expanded=False):
            rm_cols = [c for c in features_df.columns if c.endswith("_rm5")]
            if rm_cols:
                melted_rm = features_df.melt(
                    id_vars=["cycle"], value_vars=rm_cols,
                    var_name="feature", value_name="value",
                )
                fig_rm = px.line(
                    melted_rm, x="cycle", y="value", color="feature",
                    title="Rolling Mean (5-cycle) for Active Sensors",
                )
                fig_rm.update_layout(height=360, template="plotly_dark")
                st.plotly_chart(fig_rm, use_container_width=True)

    # ── Sensor Average ─────────────────────────────────────
    st.subheader("📊 Sensor Average Trend")
    fig_avg = px.area(
        df, x="cycle", y="sensor_avg",
        title=f"Mean Across All 21 Sensors — {machine_id}",
        labels={"cycle": "Cycle", "sensor_avg": "Average"},
    )
    fig_avg.update_layout(height=320, template="plotly_dark")
    st.plotly_chart(fig_avg, use_container_width=True)

    # ── Operational Settings ───────────────────────────────
    with st.expander("🛠 Operational Settings", expanded=False):
        op_melted = df.melt(
            id_vars=["cycle"],
            value_vars=["op_setting_1", "op_setting_2", "op_setting_3"],
            var_name="setting", value_name="value",
        )
        fig_op = px.line(op_melted, x="cycle", y="value", color="setting",
                         title="Operational Settings Over Cycles")
        fig_op.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig_op, use_container_width=True)

    # ── Raw Data Table ─────────────────────────────────────
    with st.expander("🗂 Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=400)


# ── Main ────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="IoT Predictive Maintenance — NASA Turbofan",
        page_icon="✈️",
        layout="wide",
    )
    session = get_session()

    # Header
    st.title("✈️ NASA Turbofan Engine — Predictive Maintenance Dashboard")
    st.caption(
        "Real-time RUL prediction & sensor monitoring · "
        "PySpark + ScyllaDB + XGBoost"
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        machines = fetch_machine_ids(session)
        if not machines:
            st.warning("No data yet — waiting for Spark ETL …")
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
            return

        view_mode = st.radio("View", ["🏭 Fleet Overview", "🔎 Engine Detail"], index=0)

        selected_machine = None
        num_rows = 100
        selected_sensors = DEFAULT_SENSORS

        if view_mode == "🔎 Engine Detail":
            selected_machine = st.selectbox("Select Engine", machines)
            num_rows = st.slider("Number of cycles", 20, 500, 100)
            selected_sensors = st.multiselect(
                "Sensors to display",
                options=ALL_SENSOR_COLS,
                default=DEFAULT_SENSORS,
                format_func=lambda s: f"{s}  ({SENSOR_LABELS.get(s, '')})",
            )

        auto_refresh = st.toggle("Auto-refresh", value=True)

        st.divider()
        st.caption(f"Connected to ScyllaDB at {SCYLLA_HOST}:{SCYLLA_PORT}")
        st.caption(f"Tracking {len(machines)} engines")

    # Main content
    if view_mode == "🏭 Fleet Overview":
        st.subheader("🏭 Fleet Health Overview")
        render_fleet_overview(session, machines)
    else:
        render_engine_detail(session, selected_machine, num_rows, selected_sensors)

    # Auto-refresh
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

# ── Vercel Serverless Stub ──────────────────────────────────
def app(environ, start_response):
    """Dummy WSGI app to satisfy Vercel's Python runtime requirements."""
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b"Dashboard Service (Vercel Stub)"]


if __name__ == "__main__":
    main()
