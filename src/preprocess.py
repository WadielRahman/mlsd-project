import sys
from pathlib import Path

import pandas as pd

TARGET = "Survived"
EMBARKED_CATEGORIES = ["C", "Q", "S"]


def fit_stats(train: pd.DataFrame) -> dict:
    """Learn imputation values from the TRAIN split only (avoids leakage)."""
    return {
        "age_median": float(train["Age"].median()),
        "fare_median": float(train["Fare"].median()),
        "embarked_mode": train["Embarked"].mode()[0],
    }


def transform(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[TARGET] = df[TARGET]
    out["Pclass"] = df["Pclass"]
    out["Sex"] = (df["Sex"] == "female").astype(int)
    out["Age"] = df["Age"].fillna(stats["age_median"])
    out["SibSp"] = df["SibSp"]
    out["Parch"] = df["Parch"]
    out["Fare"] = df["Fare"].fillna(stats["fare_median"])

    # Engineered features
    family_size = df["SibSp"] + df["Parch"] + 1
    out["FamilySize"] = family_size
    out["IsAlone"] = (family_size == 1).astype(int)
    out["HasCabin"] = df["Cabin"].notna().astype(int)

    # One-hot Embarked with FIXED categories so train/test columns always match
    embarked = pd.Categorical(
        df["Embarked"].fillna(stats["embarked_mode"]),
        categories=EMBARKED_CATEGORIES,
    )
    dummies = pd.get_dummies(embarked, prefix="Embarked", dtype=int)
    dummies.index = df.index

    return pd.concat([out, dummies], axis=1)


if __name__ == "__main__":
    # python preprocess.py <input_dir> <output_dir>
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(in_dir / "train.csv")
    test_df = pd.read_csv(in_dir / "test.csv")

    stats = fit_stats(train_df)  # learned from train only
    transform(train_df, stats).to_csv(out_dir / "train.csv", index=False)
    transform(test_df, stats).to_csv(out_dir / "test.csv", index=False)

    print(f"Processed: train {len(train_df)} rows, test {len(test_df)} rows")