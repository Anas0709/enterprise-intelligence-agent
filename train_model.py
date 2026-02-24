#!/usr/bin/env python3
"""
Train churn prediction model.
Loads dataset, preprocesses features, trains LogisticRegression, saves model.pkl.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data(csv_path: str) -> pd.DataFrame:
    """Load customer data from CSV."""
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess features for churn prediction.
    Features: age, total_spend, region (encoded)
    Target: churn
    """
    region_map = {"north": 0, "south": 1, "east": 2, "west": 3}
    df = df.copy()
    df["region_encoded"] = df["region"].str.lower().map(region_map).fillna(0).astype(int)

    features = ["age", "total_spend", "region_encoded"]
    X = df[features]
    y = df["churn"]
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "sample_data.csv"),
        help="Path to sample_data.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "model.pkl"),
        help="Output path for model.pkl",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args.data)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training LogisticRegression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    joblib.dump(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
