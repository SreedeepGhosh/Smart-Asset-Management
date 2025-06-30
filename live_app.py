# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# from sklearn.ensemble import IsolationForest

# from streamlit_autorefresh import st_autorefresh

# # Page config
# st.set_page_config(page_title="🏢 Smart Asset Dashboard", layout="wide")
# st.title("🏢 Smart Asset Intelligence Dashboard")
# st.markdown("""
# Built to reflect Xempla's principles: predictive monitoring, sustainability insights, and operational visibility across assets.
# """)

# # Load dataset
# data_path = "multi_asset_data.csv"
# # Auto-refresh every 60 seconds (60000 ms)
# st_autorefresh(interval=60000, key="data_refresh")

# @st.cache_data
# def load_data():
#     df = pd.read_csv(data_path, parse_dates=['timestamp'])
#     return df

# df = load_data()

# # Sidebar filters
# assets = df['asset_id'].unique().tolist()
# metrics = df['metric'].unique().tolist()

# st.sidebar.header("🔍 Filters")
# selected_assets = st.sidebar.multiselect("Select Assets", assets, default=assets)
# selected_metrics = st.sidebar.multiselect("Select Metrics", metrics, default=metrics)
# date_range = st.sidebar.date_input("Select Date Range", [df['timestamp'].min().date(), df['timestamp'].max().date()])

# # Filter data
# filtered_df = df[
#     (df['asset_id'].isin(selected_assets)) &
#     (df['metric'].isin(selected_metrics)) &
#     (df['timestamp'].dt.date >= date_range[0]) &
#     (df['timestamp'].dt.date <= date_range[1])
# ]

# # Train Isolation Forest on each metric/asset pair
# st.markdown("## 🔍 AI-Powered Anomaly Detection")
# anomaly_results = []
# for asset in selected_assets:
#     for metric in selected_metrics:
#         subset = filtered_df[(filtered_df['asset_id'] == asset) & (filtered_df['metric'] == metric)].copy()
#         if len(subset) >= 20:
#             model = IsolationForest(contamination=0.01, random_state=42)
#             subset['anomaly'] = model.fit_predict(subset[['value']])
#             subset['is_anomaly'] = subset['anomaly'].apply(lambda x: 1 if x == -1 else 0)
#             anomaly_results.append(subset)

# # Combine and display
# if anomaly_results:
#     df_anomalies = pd.concat(anomaly_results)

#     for asset in selected_assets:
#         for metric in selected_metrics:
#             plot_df = df_anomalies[(df_anomalies['asset_id'] == asset) & (df_anomalies['metric'] == metric)]
#             if not plot_df.empty:
#                 st.markdown(f"### 📈 {asset} - {metric}")
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df['value'], mode='lines', name='Value'))
#                 anomalies = plot_df[plot_df['is_anomaly'] == 1]
#                 fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['value'], mode='markers', name='Anomaly', marker=dict(color='red', size=6)))
#                 fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
#                 st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("Not enough data to train models or no data matching filters.")


# 6th version without shap

# import streamlit as st
# st.set_page_config(page_title="⚠️ Fault Detection Dashboard", layout="wide")
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime
# from streamlit_autorefresh import st_autorefresh
# import plotly.graph_objects as go
# import os

# # Auto-refresh every 60 seconds
# st_autorefresh(interval=60000, key="refresh")

# st.title("⚠️ AI-Powered Fault Detection Dashboard")
# st.markdown("Built with XGBoost | Updated live every minute | Fault Code + Severity")

# # Load latest data and metadata
# df = pd.read_csv("multi_asset_data.csv", parse_dates=["timestamp"])
# meta = pd.read_csv("asset_metadata.csv")
# df = df.merge(meta, on="asset_id", how="left")
# df["hour"] = df["timestamp"].dt.hour

# # Load model and encoders
# model = joblib.load("xgboost_model.pkl")
# le_asset = joblib.load("le_asset.pkl")
# le_metric = joblib.load("le_metric.pkl")

# # Get latest values per asset/metric
# latest_df = df.sort_values("timestamp").groupby(["asset_id", "metric"]).tail(1)

# # Prepare data for prediction
# latest_df["asset_id_enc"] = le_asset.transform(latest_df["asset_id"])
# latest_df["metric_enc"] = le_metric.transform(latest_df["metric"])
# X_latest = latest_df[["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]]

# # Predict fault severity
# preds = model.predict(X_latest)
# latest_df["fault_severity"] = preds

# # Severity mapping
# severity_map = {0: "Normal", 1: "Warning", 2: "Critical"}
# color_map = {0: "green", 1: "orange", 2: "red"}

# # Create log if not present
# LOG_FILE = "fault_tickets.csv"
# if not os.path.exists(LOG_FILE):
#     pd.DataFrame(columns=["fault_code", "timestamp", "asset_id", "metric", "value", "severity", "location", "age_months", "suggested_action"]).to_csv(LOG_FILE, index=False)

# # Display fault tickets
# st.subheader("📋 Fault Tickets")
# log_df = pd.read_csv(LOG_FILE)

# for _, row in latest_df.iterrows():
#     severity = severity_map[row.fault_severity]
#     if row.fault_severity > 0:
#         fault_code = f"{row.asset_id}-{row.metric.upper()}-{severity.upper()}-{str(row.timestamp)[11:13]}{str(row.timestamp)[14:16]}"
#         suggested = "Immediate maintenance required" if row.fault_severity == 2 else "Monitor closely"

#         # Check if fault already logged
#         if not ((log_df["fault_code"] == fault_code) & (log_df["timestamp"] == str(row.timestamp))).any():
#             # Log fault
#             fault_entry = {
#                 "fault_code": fault_code,
#                 "timestamp": row.timestamp,
#                 "asset_id": row.asset_id,
#                 "metric": row.metric,
#                 "value": row.value,
#                 "severity": severity,
#                 "location": row.location,
#                 "age_months": row.asset_age_months,
#                 "suggested_action": suggested
#             }
#             log_df = pd.concat([log_df, pd.DataFrame([fault_entry])], ignore_index=True)
#             log_df.to_csv(LOG_FILE, index=False)

#         # Display ticket
#         with st.container():
#             st.markdown(f"### 🔧 Fault Code: `{fault_code}`")
#             st.markdown(f"**Asset:** {row.asset_id} ({row.location})")
#             st.markdown(f"**Sensor:** {row.metric} — Value: `{row.value} {row.unit}`")
#             st.markdown(f"**Severity:** `{severity}`")
#             st.markdown(f"**Timestamp:** {row.timestamp}")
#             st.markdown(f"**Asset Age:** {row.asset_age_months} months")
#             st.markdown(f"**Suggested Action:** {suggested}")
#             st.markdown("---")

# # Sensor trend graphs
# st.subheader("📈 Sensor Overview")
# for asset in df["asset_id"].unique():
#     asset_df = df[df["asset_id"] == asset]
#     for metric in asset_df["metric"].unique():
#         metric_df = asset_df[asset_df["metric"] == metric].sort_values("timestamp").tail(100)
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=metric_df["timestamp"], y=metric_df["value"], mode="lines+markers", name=metric))
#         fig.update_layout(title=f"{asset} - {metric}", height=300, margin=dict(t=30, b=30))
#         st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from xgboost import XGBClassifier
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="🚨 Xempla Fault Detection", layout="wide")
st.title("🏢 Smart Asset Intelligence Dashboard")


st_autorefresh(interval=30000, key="auto-refresh")  # 30 seconds

# Load data
data_path = "multi_asset_data.csv"
meta_path = "asset_metadata.csv"
log_path = "fault_tickets.csv"


# --- SIMULATED LIVE DATA LOGIC ---
if "data_cycle" not in st.session_state:
    st.session_state.data_cycle = 0

def simulate_live_data():
    ts = datetime.now()
    new_data = []
    inject_anomaly = (st.session_state.data_cycle % 4 == 0)  # Every 4th cycle (2 minutes)

    # HVAC Temperature
    hvac_temp = 22 + 3 * np.sin(2 * np.pi * (ts.hour / 24)) + np.random.normal(0, 0.5)
    if inject_anomaly and np.random.rand() < 0.5:
        hvac_temp += np.random.uniform(6, 9)
    new_data.append([ts, "HVAC", "Temperature", round(hvac_temp, 2), "°C"])

    # Chiller Power
    chiller_power = 150 + 10 * np.cos(2 * np.pi * (ts.hour / 24)) + np.random.normal(0, 2)
    if inject_anomaly and np.random.rand() < 0.5:
        chiller_power += np.random.uniform(20, 30)
    new_data.append([ts, "Chiller", "Power", round(chiller_power, 2), "kW"])

    # Solar Efficiency
    is_day = 6 <= ts.hour <= 18
    base_eff = 90 if is_day else 0
    solar_eff = base_eff + np.random.normal(0, 2)
    if inject_anomaly and is_day and np.random.rand() < 0.5:
        solar_eff -= np.random.uniform(25, 35)
    solar_eff = max(0, solar_eff)
    new_data.append([ts, "Solar", "Efficiency", round(solar_eff, 2), "%"])

    # Append to CSV
    df_new = pd.DataFrame(new_data, columns=["timestamp", "asset_id", "metric", "value", "unit"])
    df_new.to_csv(data_path, mode='a', header=not os.path.exists(data_path), index=False)

    st.session_state.data_cycle += 1
    # st.success(f"🟢 Simulated data at {ts.strftime('%H:%M:%S')} | Anomaly: {inject_anomaly}") --> HEADER FOR ANOMALY INSERTION

# Run the simulation on each refresh
simulate_live_data()

df = pd.read_csv(data_path, parse_dates=["timestamp"])
meta = pd.read_csv(meta_path)
df = df.merge(meta, on="asset_id", how="left")
df["hour"] = df["timestamp"].dt.hour

# Load model & encoders
model = joblib.load("xgboost_model.pkl")
le_asset = joblib.load("le_asset.pkl")
le_metric = joblib.load("le_metric.pkl")

# Encode features
df["asset_id_enc"] = le_asset.transform(df["asset_id"])
df["metric_enc"] = le_metric.transform(df["metric"])

# XGBoost input
X = df[["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]]
df["fault_severity"] = model.predict(X)

# Severity mapping
severity_map = {0: "Normal", 1: "Warning", 2: "Critical"}
color_map = {0: "green", 1: "orange", 2: "red"}

# Prepare ticket logging
if not os.path.exists(log_path):
    pd.DataFrame(columns=[
        "fault_code", "timestamp", "asset_id", "metric", "value", "severity",
        "location", "age_months", "suggested_action"
    ]).to_csv(log_path, index=False)

selected_assets = df["asset_id"].unique().tolist()
selected_metrics = df["metric"].unique().tolist()




# Sidebar: Time Range Selection
st.subheader("⏱️ Select Time Range or Live View Mode")

view_mode = st.radio("View Mode", ["Live View (Last 2 Hours)", "Historical View"], horizontal=True)

if view_mode == "Live View (Last 2 Hours)":
    cutoff = datetime.now() - timedelta(hours=2)
    st.info("Showing only the most recent 2 hours of data.")
else:
    time_range = st.selectbox("Select Historical Window", ["1 Day", "3 Days", "7 Days"], index=0)
    days_map = {"1 Day": 1, "3 Days": 3, "7 Days": 7}
    cutoff = datetime.now() - timedelta(days=days_map[time_range])
    st.info(f"Showing data from the last {time_range.lower()}.")


# Filter data
filtered_df = df[
    (df['asset_id'].isin(selected_assets)) &
    (df['metric'].isin(selected_metrics)) &
    (df['timestamp'] >= cutoff)
]


# 🔍 Plot sensor data + fault markers
st.subheader("📈 Sensor Graphs with Fault Annotations")
for asset in selected_assets:
    for metric in selected_metrics:
        sub_df = filtered_df[(filtered_df["asset_id"] == asset) & (filtered_df["metric"] == metric)]
        if sub_df.empty:
            continue

        # Use already-filtered sub_df based on selected view
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


st.info(
    "📌 **Dashboard Overview**\n\n"
    "This smart asset monitoring dashboard simulates real-time sensor data from key building systems: HVAC (Temperature), Chiller (Power), and Solar (Efficiency). "
    "The charts display live sensor trends over time, with automatic detection of anomalies based on machine learning (XGBoost). Faults are visually marked with severity levels (Warning / Critical), and corresponding fault tickets are generated below.\n\n"
    "**Note:** All data currently shown is *synthetically generated* with random anomaly injections every 2 minutes for testing purposes only. "
    "Once connected to actual IoT sensors, this dashboard will operate in real-time using live building data for actionable fault detection."
)

# 📋 Fault Tickets
st.subheader("📋 Fault Tickets (Generated in real-time)")

# Get the latest fault per asset-metric
latest_faults = df[df["fault_severity"] > 0].sort_values("timestamp", ascending=False).drop_duplicates(
    subset=["asset_id", "metric"]
)

# Define 3 columns for layout
cols = st.columns(3)

for idx, (_, row) in enumerate(latest_faults.iterrows()):
    # Ensure timestamp is datetime
    if isinstance(row.timestamp, str):
        row.timestamp = pd.to_datetime(row.timestamp)

    time_code = row.timestamp.strftime("%H%M")
    fault_code = f"{row.asset_id[:2].upper()}-{row.metric[:2].upper()}-{time_code}"
    severity = severity_map[row.fault_severity]
    action = "🔧 Immediate Check" if row.fault_severity == 2 else "🕵️ Monitor Closely"

    # Write to one of the 3 columns
    with cols[idx % 3]:
        st.markdown(f"**🆔 Fault Code:** `{fault_code}`")
        st.markdown(f"**Asset:** {row.asset_id} ({row.location})")
        st.markdown(f"**Sensor:** {row.metric} | Value: `{row.value} {row.unit}`")
        st.markdown(f"**Severity:** `{severity}`")
        st.markdown(f"**Timestamp:** {row.timestamp}")
        st.markdown(f"**Asset Age:** {row.asset_age_months} months")
        st.markdown(f"**Suggested Action:** {action}")
        st.markdown("---")

    # Log fault ticket if not already saved
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

# 📊 SHAP Explanations


# st.subheader("🔍 SHAP Explainability (Last Fault per Asset)")

# # Define input features
# feature_cols = ["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]

# # Prepare SHAP explainer
# explainer = shap.Explainer(model)

# # Get last fault per asset (most recent fault per asset)
# last_faults = (
#     df[df["fault_severity"] > 0]
#     .sort_values("timestamp", ascending=False)
#     .drop_duplicates(subset="asset_id", keep="first")
# )

# for i, (_, row) in enumerate(last_faults.iterrows()):
#     st.markdown(f"🔹 Fault {i+1}: {row.asset_id} - {row.metric} | Severity: `{severity_map[row.fault_severity]}`")
    
#     # Prepare input row
#     input_row = row[feature_cols].values.reshape(1, -1)
    
#     # Predict class to choose correct SHAP slice
#     predicted_class = model.predict(input_row)[0]
    
#     # Get SHAP values
#     shap_vals = explainer(input_row)

#     # Extract SHAP values for the predicted class
#     shap_for_class = shap.Explanation(
#         values=shap_vals.values[0][:, predicted_class],
#         base_values=shap_vals.base_values[0][predicted_class],
#         data=input_row[0],
#         feature_names=feature_cols
#     )

#     # Plot SHAP values for the predicted class
#     fig, ax = plt.subplots()
#     shap.plots.bar(shap_for_class, show=False)
#     st.pyplot(fig)
# -----------------------------------------------------------------------
# st.subheader("🔍 SHAP Explainability (Last Fault per Asset)")

# # Define input features
# feature_cols = ["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]

# # Prepare SHAP explainer
# explainer = shap.Explainer(model)

# # Get last fault per asset (most recent fault per asset)
# last_faults = (
#     df[df["fault_severity"] > 0]
#     .sort_values("timestamp", ascending=False)
#     .drop_duplicates(subset="asset_id", keep="first")
# )

# for i, (_, row) in enumerate(last_faults.iterrows()):
#     fault_time = row["timestamp"]
#     st.markdown(
#         f"🔹 Fault {i+1}: **{row.asset_id} - {row.metric}** | "
#         f"**Severity:** `{severity_map[row.fault_severity]}` | "
#         f"🕒 **Time:** `{fault_time}`"
#     )

#     # Prepare input row
#     input_row = row[feature_cols].values.reshape(1, -1)

#     # Predict class to choose correct SHAP slice
#     predicted_class = model.predict(input_row)[0]

#     # Get SHAP values
#     shap_vals = explainer(input_row)

#     # Extract SHAP values for the predicted class
#     shap_for_class = shap.Explanation(
#         values=shap_vals.values[0][:, predicted_class],
#         base_values=shap_vals.base_values[0][predicted_class],
#         data=input_row[0],
#         feature_names=feature_cols
#     )

#     # Plot SHAP values for the predicted class
#     fig, ax = plt.subplots()
#     shap.plots.bar(shap_for_class, show=False)
#     st.pyplot(fig)

# st.subheader("🔍 SHAP Explainability (Last Fault per Asset)")

# # Define input features
# feature_cols = ["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]

# # Prepare SHAP explainer
# explainer = shap.Explainer(model)

# # Get last fault per asset (most recent fault per asset)
# last_faults = (
#     df[df["fault_severity"] > 0]
#     .sort_values("timestamp", ascending=False)
#     .drop_duplicates(subset="asset_id", keep="first")
# )

# for i, (_, row) in enumerate(last_faults.iterrows()):
#     predicted_class = model.predict(row[feature_cols].values.reshape(1, -1))[0]
#     input_row = row[feature_cols].values.reshape(1, -1)
#     shap_vals = explainer(input_row)

#     shap_for_class = shap.Explanation(
#         values=shap_vals.values[0][:, predicted_class],
#         base_values=shap_vals.base_values[0][predicted_class],
#         data=input_row[0],
#         feature_names=feature_cols
#     )

#     # Style severity and timestamp with high contrast
#     timestamp_str = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
#     severity_text = severity_map[row.fault_severity]
#     severity_color = {
#         "Critical": "#d90429",   # Bold Red
#         "Warning": "#ff9f1c",    # Bold Orange
#         "Normal": "#2b9348"      # Bold Green
#     }[severity_text]

#     st.markdown(f"""
#         <div style="padding:12px; margin-bottom:8px; border:1px solid #ccc; border-radius:8px;
#                     background-color:#ffffff; color:#000000; font-size:15px;">
#             <b>🔹 Fault {i+1}:</b> <span style="color:#003049">{row.asset_id} - {row.metric}</span><br>
#             <b>Severity:</b> <span style="color:{severity_color}; font-weight:bold">{severity_text}</span><br>
#             <b>Timestamp:</b> <span style="color:#1d3557; font-weight:bold">{timestamp_str}</span>
#         </div>
#     """, unsafe_allow_html=True)

#     fig, ax = plt.subplots()
#     shap.plots.bar(shap_for_class, show=False)
#     st.pyplot(fig)


st.subheader("🔍 SHAP Explainability (Last Fault per Asset)")

# Define input features
feature_cols = ["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]

# Prepare SHAP explainer
explainer = shap.Explainer(model)

# Get last fault per asset (most recent fault per asset)
last_faults = (
    df[df["fault_severity"] > 0]
    .sort_values("timestamp", ascending=False)
    .drop_duplicates(subset="asset_id", keep="first")
)

for i, (_, row) in enumerate(last_faults.iterrows()):
    predicted_class = model.predict(row[feature_cols].values.reshape(1, -1))[0]
    input_row = row[feature_cols].values.reshape(1, -1)
    shap_vals = explainer(input_row)

    shap_for_class = shap.Explanation(
        values=shap_vals.values[0][:, predicted_class],
        base_values=shap_vals.base_values[0][predicted_class],
        data=input_row[0],
        feature_names=feature_cols
    )

    # Extract metadata
    timestamp_str = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    severity_text = severity_map[row.fault_severity]
    title = f"{row.asset_id} - {row.metric} | {severity_text} | {timestamp_str}"

    # Plot SHAP values and add title to plot
    fig, ax = plt.subplots()
    shap.plots.bar(shap_for_class, show=False)
    plt.title(title, fontsize=10, color="black", pad=20)
    st.pyplot(fig)
