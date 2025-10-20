import argparse, json, pickle
from pathlib import Path
from joblib import load, dump
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--model", default="rf", choices=["rf", "logreg"])
    parser.add_argument("--method", default="isotonic", choices=["isotonic", "sigmoid"])
    args = parser.parse_args()

    ts = args.timestamp
    models_dir = Path("models")
    data_dir   = Path("data")
    metrics_dir = Path("metrics")
    ensure_dir(models_dir); ensure_dir(metrics_dir)

    model_path = models_dir / f"model_{ts}_{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing base model: {model_path}")

    with open(data_dir / "data.pickle", "rb") as f: X = pickle.load(f)
    with open(data_dir / "target.pickle", "rb") as f: y = pickle.load(f)

    base = load(model_path)
    # use CV=3 for quick consistency
    calib = CalibratedClassifierCV(estimator=base, method=args.method, cv=3)
    calib.fit(X, y)

    # Evaluate calibration via Brier score
    if hasattr(calib, "predict_proba"):
        p1 = calib.predict_proba(X)[:, 1]
        brier = brier_score_loss(y, p1)
    else:
        brier = float("nan")

    out_path = models_dir / f"model_{ts}_{args.model}_cal_{args.method}.joblib"
    dump(calib, out_path)

    cal_metrics = {"timestamp": ts, "model": args.model, "method": args.method, "brier": float(brier)}
    with open(metrics_dir / f"{ts}_calibration_{args.method}.json", "w") as f:
        json.dump(cal_metrics, f, indent=2)

    print(f"[calibrate] saved calibrated model: {out_path}")
    print(json.dumps(cal_metrics, indent=2))

if __name__ == "__main__":
    main()