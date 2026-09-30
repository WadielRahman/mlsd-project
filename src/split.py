import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Read split parameters
params = yaml.safe_load(open("params.yaml"))["split"]

# CLI:
# python split.py <input_csv> <output_dir>
input_path = sys.argv[1]
output_dir = Path(sys.argv[2])

output_dir.mkdir(parents=True, exist_ok=True)

# Load Titanic dataset
df = pd.read_csv(input_path)

# Remove PassengerId if present
if "PassengerId" in df.columns:
    df = df.drop(columns=["PassengerId"])

# Split data (stratified so train/test keep the same survival ratio)
train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["Survived"],
)

# Save outputs
train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")