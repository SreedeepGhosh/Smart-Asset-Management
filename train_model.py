import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Load datasets
df = pd.read_csv("multi_asset_data.csv", parse_dates=["timestamp"])
meta = pd.read_csv("asset_metadata.csv")
df = df.merge(meta, on="asset_id", how="left")

# Add hour feature
df["hour"] = df["timestamp"].dt.hour

# Label generation based on rule-based thresholds
def assign_fault(row):
    if row["asset_id"] == "HVAC":
        if row["value"] > 30:
            return 2  # Critical
        elif row["value"] > 27:
            return 1  # Warning
    elif row["asset_id"] == "Chiller":
        if row["value"] > 175:
            return 2
        elif row["value"] > 160:
            return 1
    elif row["asset_id"] == "Solar" and 6 <= row["hour"] <= 18:
        if row["value"] < 60:
            return 2
        elif row["value"] < 70:
            return 1
    return 0  # Normal

df["fault_severity"] = df.apply(assign_fault, axis=1)

# Encode categorical features
le_asset = LabelEncoder()
le_metric = LabelEncoder()
df["asset_id_enc"] = le_asset.fit_transform(df["asset_id"])
df["metric_enc"] = le_metric.fit_transform(df["metric"])

# Define features and target
X = df[["value", "asset_age_months", "hour", "asset_id_enc", "metric_enc"]]
y = df["fault_severity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Predict and report
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

report = classification_report(y_test, y_pred, target_names=["Normal", "Warning", "Critical"])
print("\n📊 Classification Report:\n")
print(report)

# Save model and encoders
joblib.dump(model, "xgboost_model.pkl")
joblib.dump(le_asset, "le_asset.pkl")
joblib.dump(le_metric, "le_metric.pkl")
print("\n✅ Model and encoders saved successfully.")