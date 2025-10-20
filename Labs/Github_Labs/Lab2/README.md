Lab 2 — GitHub Actions for Model Training, Evaluation & Calibration

This lab automates a simple ML lifecycle using GitHub Actions: train a model, evaluate it, calibrate probabilities, and version artifacts. You can run everything locally or let Actions run it on every push to main.

⸻

Changes made (summary)
	•	Cleaned & modularized scripts
	•	train_model.py: deterministic data generation, safer I/O, timestamped model names (model_<TS>_<model>.joblib), quick train snapshot metrics.
	•	evaluate_model.py: accuracy, F1, PR AUC, ROC AUC; saves JSON metrics.
	•	calibrate_model.py: isotonic or sigmoid calibration using CalibratedClassifierCV; logs Brier score; saves calibrated models as model_<TS>_<model>_cal_<method>.joblib.
	•	Improved CI
	•	Modern Actions versions (checkout@v4, setup-python@v5) with pip cache.
	•	Training/eval workflow runs on every push to main and uploads artifacts.
	•	Calibration workflow finds latest model and produces calibrated variants.
	•	Commits models/ and metrics/ back to the repo (and uploads artifacts).
	•	Compatibility note
	•	CalibratedClassifierCV now uses estimator= (scikit-learn ≥ 1.4). If needed, add a small try/except to support older versions.

⸻

Run locally (quick test)

Recommended before pushing—verifies your environment and scripts.

# 1) Create venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train
TS=$(date +%Y%m%d%H%M%S)
python src/train_model.py --timestamp "$TS" --model rf

# 4) Evaluate
python src/evaluate_model.py --timestamp "$TS" --model rf

# 5) (Optional) Calibrate
python src/calibrate_model.py --timestamp "$TS" --model rf --method isotonic

# 6) Check outputs
ls -1 models
ls -1 metrics

Expected outputs:
	•	models/model_<TS>_rf.joblib
	•	metrics/<TS>_metrics.json
	•	(if calibrated) models/model_<TS>_rf_cal_isotonic.joblib, metrics/<TS>_calibration_isotonic.json

⸻

Run in GitHub Actions
	1.	Push to main (or open a PR and merge):

git add .
git commit -m "lab2: ci pipelines + model scripts"
git push origin main

	2.	Go to GitHub → Actions:

	•	Model Retraining on Push runs automatically:
	•	Trains and evaluates model.
	•	Saves artifacts to models/ and metrics/.
	•	Uploads zipped artifacts to the workflow run.
	•	Commits artifacts to the repo.
	•	Model Calibration on Push runs automatically after push:
	•	Detects latest model timestamp.
	•	Produces isotonic and sigmoid calibrated models.
	•	Saves calibration metrics (Brier score).
	•	Uploads artifacts and commits to the repo.

	3.	Download artifacts from the run or browse the committed files in your repo.

⸻

Authentication / permissions
	•	You do not need a personal token (PAT). The built-in GITHUB_TOKEN is used.
	•	Ensure repo setting: Settings → Actions → General → Workflow permissions → Read and write permissions (enable).

⸻

Parameters & versioning
	•	All scripts accept a --timestamp (format YYYYmmddHHMMSS). CI supplies this automatically.
	•	Models are saved as models/model_<timestamp>_<model>.joblib.
	•	Calibrated models add _cal_<method> suffix.
	•	Metrics JSONs are saved in metrics/ with matching timestamps.

⸻

Notes for grading (what changed & why it matters)
	•	Added calibration with Brier score → improves probability quality (useful for decision thresholds).
	•	Expanded metrics (PR AUC, ROC AUC) for better performance visibility.
	•	Robust CI that version-controls models & metrics and publishes artifacts automatically.
	•	Deterministic-ish runs (seed from timestamp) and no “0 samples” edge cases.

⸻