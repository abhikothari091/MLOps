# src/train.py
import os
import argparse
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train(random_state: int = 42, test_size: float = 0.2):
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    return model, (X_test, y_test)


def main():
    parser = argparse.ArgumentParser(description="Train an iris RF model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("MODEL_DIR", "models"),
        help="where to save the trained model inside the container",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="iris_model.pkl",
        help="output file name",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="test split size",
    )
    args = parser.parse_args()

    model, _ = train(test_size=args.test_size)

    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / args.model_name
    joblib.dump(model, out_path)

    print(f"[train] model saved to {out_path}")


if __name__ == "__main__":
    main()