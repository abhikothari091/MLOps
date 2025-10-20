import argparse, os, json, pickle, random, datetime
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_data(seed: int, n_samples: int = 1000):
    # robust: never 0 rows; deterministic but uses timestamp-derived seed
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=max(200, n_samples),  # avoid tiny/zero samples
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=rng,
        shuffle=True,
    )
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True, help="Timestamp from workflow (YYYYmmddHHMMSS)")
    parser.add_argument("--model", default="rf", choices=["rf", "logreg"])
    args = parser.parse_args()

    ts = args.timestamp
    seed = int(ts[-6:])  

    # Paths
    data_dir = Path("data")
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    for d in (data_dir, models_dir, metrics_dir):
        ensure_dir(d)

    # Data
    X, y = make_data(seed=seed)
    with open(data_dir / "data.pickle", "wb") as f: pickle.dump(X, f)
    with open(data_dir / "target.pickle", "wb") as f: pickle.dump(y, f)

    # Model
    if args.model == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, n_jobs=None, random_state=seed)

    clf.fit(X, y)
    yhat = clf.predict(X)
    acc = accuracy_score(y, yhat)
    f1  = f1_score(y, yhat)

    # Save model (timestamped)
    model_stem = f"model_{ts}_{args.model}"
    model_path = models_dir / f"{model_stem}.joblib"
    dump(clf, model_path)

    # Save quick metrics (train-only snapshot; full eval happens in evaluate step)
    snap = {"timestamp": ts, "model": args.model, "train_acc": acc, "train_f1": f1}
    with open(metrics_dir / f"{ts}_train_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)

    print(f"[train] saved: {model_path}")
    print(f"[train] snapshot metrics: acc={acc:.4f}, f1={f1:.4f}")

if __name__ == "__main__":
    main()