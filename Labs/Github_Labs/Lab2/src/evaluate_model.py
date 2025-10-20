import argparse, os, json, pickle
from pathlib import Path
import numpy as np
from joblib import load
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--model", default="rf", choices=["rf", "logreg"])
    args = parser.parse_args()

    ts = args.timestamp
    data_dir   = Path("data")
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    ensure_dir(metrics_dir)

    model_path = models_dir / f"model_{ts}_{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(data_dir / "data.pickle", "rb") as f: X = pickle.load(f)
    with open(data_dir / "target.pickle", "rb") as f: y = pickle.load(f)

    clf = load(model_path)
    yhat = clf.predict(X)
    metrics = {
        "timestamp": ts,
        "model": args.model,
        "accuracy": float(accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
    }
    #  proba metrics
    if hasattr(clf, "predict_proba"):
        p1 = clf.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, p1)
        pr_auc = auc(recall, precision)
        try:
            roc_auc = roc_auc_score(y, p1)
        except ValueError:
            roc_auc = float("nan")
        metrics.update({"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)})

    metrics_path = metrics_dir / f"{ts}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[evaluate] saved: {metrics_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()