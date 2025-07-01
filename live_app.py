import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import os
import matplotlib.pyplot as plt
import shap
import warnings
import pytz

warnings.filterwarnings("ignore")

st.set_page_config(page_title="🚨 Xempla Fault Detection", layout="wide")
st.title("🏢 Smart Asset Intelligence Dashboard")

st.info(
    "📌 **Dashboard Overview**\n\n"
    "This smart asset monitoring dashboard simulates real-time sensor data from key building systems: HVAC (Temperature), Chiller (Power), and Solar (Efficiency). "
    "The charts display live sensor trends over time, with automatic detection of anomalies based on machine learning (XGBoost). Faults are visually marked with severity levels (Warning / Critical), and corresponding fault tickets are generated below.\n\n"
    "**Note:** All data currently shown is *synthetically generated* with random anomaly injections every 2 minutes for testing purposes only. "
    "Once connected to actual IoT sensors, this dashboard will operate in real-time using live building data for actionable fault detection."
)

st_autorefresh(interval=30000, key="auto-refresh")

data_path = "multi_asset_data.csv"
meta_path = "asset_metadata.csv"
log_path = "fault_tickets.csv"

tz = pytz.timezone("Asia/Kolkata")

df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata", ambiguous='NaT', nonexistent='NaT')

meta = pd.read_csv(meta_path)
df = df.merge(meta, on="asset_id", how="left")
df["hour"] = df["timestamp"].dt.hour

model = joblib.load("xgboost_model.pkl")
le_asset = joblib.load("le_asset.pkl")
le_metric = joblib.load("le_metric.pkl")

df["asset_id_enc"] = le_asset.transform(df["asset_id"])
df["metric_enc"] = le_metric.transform(df["metric"])

X = df[["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]]
df["fault_severity"] = model.predict(X)

severity_map = {0: "Normal", 1: "Warning", 2: "Critical"}
color_map = {0: "green", 1: "orange", 2: "red"}

if not os.path.exists(log_path):
    pd.DataFrame(columns=[
        "fault_code", "timestamp", "asset_id", "metric", "value", "severity",
        "location", "age_months", "suggested_action"
    ]).to_csv(log_path, index=False)

selected_assets = df["asset_id"].unique().tolist()
selected_metrics = df["metric"].unique().tolist()

st.subheader("⏱️ Select Time Range or Live View Mode")

view_mode = st.radio("View Mode", ["Live View (Last 2 Hours)", "Historical View"], horizontal=True)

if view_mode == "Live View (Last 2 Hours)":
    cutoff = datetime.now(tz) - timedelta(hours=2)
    st.info("Showing only the most recent 2 hours of data.")
else:
    time_range = st.selectbox("Select Historical Window", ["1 Day", "3 Days", "7 Days"], index=0)
    days_map = {"1 Day": 1, "3 Days": 3, "7 Days": 7}
    cutoff = datetime.now(tz) - timedelta(days=days_map[time_range])
    st.info(f"Showing data from the last {time_range.lower()}.")

filtered_df = df[
    (df['asset_id'].isin(selected_assets)) &
    (df['metric'].isin(selected_metrics)) &
    (df['timestamp'] >= cutoff)
]

st.subheader("📈 Sensor Graphs with Fault Annotations")
for asset in selected_assets:
    for metric in selected_metrics:
        sub_df = filtered_df[(filtered_df["asset_id"] == asset) & (filtered_df["metric"] == metric)]
        if sub_df.empty:
            continue

        rolling_df = sub_df.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_df["timestamp"], y=rolling_df["value"], mode='lines+markers',
            name="Reading", line=dict(color="blue")
        ))

        for sev, color in color_map.items():
            fault_pts = rolling_df[rolling_df["fault_severity"] == sev]
            if sev > 0 and not fault_pts.empty:
                fig.add_trace(go.Scatter(
                    x=fault_pts["timestamp"], y=fault_pts["value"],
                    mode="markers", name=f"{severity_map[sev]}",
                    marker=dict(color=color, size=10, symbol="x")
                ))

        fig.update_layout(title=f"{asset} - {metric}", height=300)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Fault Tickets (Generated in real-time)")

latest_faults = df[df["fault_severity"] > 0].sort_values("timestamp", ascending=False).drop_duplicates(
    subset=["asset_id", "metric"]
)

cols = st.columns(3)

for idx, (_, row) in enumerate(latest_faults.iterrows()):
    time_code = row.timestamp.strftime("%H%M")
    fault_code = f"{row.asset_id[:2].upper()}-{row.metric[:2].upper()}-{time_code}"
    severity = severity_map[row.fault_severity]
    action = "🔧 Immediate Check" if row.fault_severity == 2 else "🕵️ Monitor Closely"

    with cols[idx % 3]:
        st.markdown(f"**🆔 Fault Code:** `{fault_code}`")
        st.markdown(f"**Asset:** {row.asset_id} ({row.location})")
        st.markdown(f"**Sensor:** {row.metric} | Value: `{row.value} {row.unit}`")
        st.markdown(f"**Severity:** `{severity}`")
        st.markdown(f"**Timestamp:** {row.timestamp}")
        st.markdown(f"**Asset Age:** {row.asset_age_months} months")
        st.markdown(f"**Suggested Action:** {action}")
        st.markdown("---")

    log_df = pd.read_csv(log_path)
    if fault_code not in log_df["fault_code"].values:
        new_ticket = {
            "fault_code": fault_code,
            "timestamp": row.timestamp,
            "asset_id": row.asset_id,
            "metric": row.metric,
            "value": row.value,
            "severity": severity,
            "location": row.location,
            "age_months": row.asset_age_months,
            "suggested_action": action
        }
        pd.concat([log_df, pd.DataFrame([new_ticket])], ignore_index=True).to_csv(log_path, index=False)

st.subheader("🔍 SHAP Explainability (Last Fault per Asset)")

feature_cols = ["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]
explainer = shap.Explainer(model)

last_faults = (
    df[df["fault_severity"] > 0]
    .sort_values("timestamp", ascending=False)
    .drop_duplicates(subset="asset_id", keep="first")
)

for _, row in last_faults.iterrows():
    input_row = row[feature_cols].values.reshape(1, -1)
    shap_vals = explainer(input_row)
    predicted_class = model.predict(input_row)[0]

    shap_for_class = shap.Explanation(
        values=shap_vals.values[0][:, predicted_class],
        base_values=shap_vals.base_values[0][predicted_class],
        data=input_row[0],
        feature_names=feature_cols
    )

    timestamp_str = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    title = f"{row.asset_id} - {row.metric} | {severity_map[row.fault_severity]} | {timestamp_str}"

    fig, ax = plt.subplots()
    shap.plots.bar(shap_for_class, show=False)
    plt.title(title, fontsize=10, color="black", pad=20)
    st.pyplot(fig)

if "data_cycle" not in st.session_state:
    st.session_state.data_cycle = 0

def simulate_live_data():
    ts = datetime.now(tz)
    new_data = []
    inject_anomaly = (st.session_state.data_cycle % 4 == 0)

    hvac_temp = 22 + 3 * np.sin(2 * np.pi * (ts.hour / 24)) + np.random.normal(0, 0.5)
    if inject_anomaly and np.random.rand() < 0.5:
        hvac_temp += np.random.uniform(6, 9)
    new_data.append([ts, "HVAC", "Temperature", round(hvac_temp, 2), "°C"])

    chiller_power = 150 + 10 * np.cos(2 * np.pi * (ts.hour / 24)) + np.random.normal(0, 2)
    if inject_anomaly and np.random.rand() < 0.5:
        chiller_power += np.random.uniform(20, 30)
    new_data.append([ts, "Chiller", "Power", round(chiller_power, 2), "kW"])

    is_day = 6 <= ts.hour <= 18
    base_eff = 90 if is_day else 0
    solar_eff = base_eff + np.random.normal(0, 2)
    if inject_anomaly and is_day and np.random.rand() < 0.5:
        solar_eff -= np.random.uniform(25, 35)
    solar_eff = max(0, solar_eff)
    new_data.append([ts, "Solar", "Efficiency", round(solar_eff, 2), "%"])

    df_new = pd.DataFrame(new_data, columns=["timestamp", "asset_id", "metric", "value", "unit"])
    df_new["timestamp"] = df_new["timestamp"].dt.tz_localize(None)
    df_new.to_csv(data_path, mode='a', header=not os.path.exists(data_path), index=False)

    st.session_state.data_cycle += 1

simulate_live_data()
