import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_path = sys.argv[1]
test_csv = sys.argv[2]
metrics_path = Path(sys.argv[3])

metrics_path.parent.mkdir(parents=True, exist_ok=True)

model = joblib.load(model_path)

test_df = pd.read_csv(test_csv)

X = test_df.drop(columns=["Survived"])
y_true = test_df["Survived"]

X = X.select_dtypes(include=["number"])

y_pred = model.predict(X)

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0))
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))