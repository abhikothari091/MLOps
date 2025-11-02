# src/evaluate.py
import os
import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved iris model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("MODEL_PATH", "models/iris_model.pkl"),
        help="path to the trained model",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)

    iris = load_iris()
    X, y = iris.data, iris.target

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    print(f"[evaluate] accuracy on full iris = {acc:.4f}")


if __name__ == "__main__":
    main()