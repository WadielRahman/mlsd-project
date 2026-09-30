import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Read training parameters
params = yaml.safe_load(open("params.yaml"))["train"]

# CLI:
# python train.py <train_csv> <model_output>
train_csv = sys.argv[1]
model_out = Path(sys.argv[2])

model_out.parent.mkdir(parents=True, exist_ok=True)

# Load training data
train_df = pd.read_csv(train_csv)

# Separate features and target
X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

# Select numeric features
X = X.select_dtypes(include=["number"])

# Create Random Forest classifier
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"],
    n_jobs=-1,
)

# Train
model.fit(X, y)

# Save model
joblib.dump(model, model_out)

print(f"Saved model -> {model_out}")