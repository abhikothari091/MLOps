# Airflow Lab 3 — README

## 🚀 Overview

An Apache Airflow pipeline that:

1. Loads the **Advertising** dataset
2. Preprocesses & trains a **Logistic Regression** model
3. Evaluates metrics and stores artifacts
4. Optionally sends a success email notification

Artifacts are saved to:

```
Labs/Airflow_Labs/Lab_3/model/
```

---

## 📁 Project Layout (relevant paths)

```
Labs/
└── Airflow_Labs/
    └── Lab_3/
        ├── airflow.cfg              # Airflow config in $AIRFLOW_HOME
        ├── airflow.db               # SQLite metadata (auto-created)
        ├── dags/
        │   ├── my_dag.py            # your DAG (aka sample_dag)
        │   ├── data/advertising.csv # dataset (relative path)
        │   ├── src/
        │   │   ├── model_development.py
        │   │   └── success_email.py
        │   └── templates/           # (if used for emails)
        └── model/
            ├── model.sav            # trained model
            └── metrics.json         # evaluation metrics
```

---

## 🔧 Notable Changes Made

* **macOS stability & discovery**

  * Executor set to **`DebugExecutor`** (works with SQLite, avoids fork issues).
  * Ensured DAG is created **at import time** (no heavy imports at module top).
* **UI log fix**

  * Webserver bound to `127.0.0.1` and base URL set:
    `web_server_host = 127.0.0.1`, `base_url = http://127.0.0.1:8080`
  * Use local file logs in UI: `serve_logs = False`.
* **Path hardening**

  * Dataset loaded **relative to DAG**: `dags/data/advertising.csv`.
  * Artifacts saved under `Labs/Airflow_Labs/Lab_3/model/`.
* **Email notification**

  * `success_email.py` uses Airflow connection **`email_smtp`** (Gmail SMTP shown; safe fallback prints/logs if unset).

---

## ✅ Requirements

* Python **3.9**
* Apache Airflow **2.9.x** (SQLite metadata for local dev)
* `pandas`, `scikit-learn`, `jinja2`

> Tip: Airflow recommends installing with pinned constraints for your Python version.

---

## 🛠️ Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv airflow_lab3_venv
source airflow_lab3_venv/bin/activate
```

### 2) Install dependencies

```bash
pip install "apache-airflow==2.9.2" pandas scikit-learn jinja2
```

### 3) Set Airflow home (this repo’s Lab 3)

```bash
export AIRFLOW_HOME=/Users/DELL/MLOPS/MLOPS-1/Labs/Airflow_Labs/Lab_3
```

### 4) Initialize the Airflow DB

```bash
airflow db init
```

### 5) Create an admin user (first run only)

```bash
airflow users create \
  --username abhi \
  --firstname Abhishek \
  --lastname K \
  --role Admin \
  --email you@example.com \
  --password "YourStrongPassword"
```

### 6) Configure Airflow (key snippets in `airflow.cfg`)

**[core]**

```ini
executor = DebugExecutor
sql_alchemy_conn = sqlite:////Users/DELL/MLOPS/MLOPS-1/Labs/Airflow_Labs/Lab_3/airflow.db
load_examples = False
```

**[webserver]**

```ini
web_server_host = 127.0.0.1
web_server_port = 8080
base_url = http://127.0.0.1:8080
serve_logs = False
enable_proxy_fix = True
expose_hostname = True
```

> These settings fix the "missing http(s) protocol" UI log error and ensure local logs are shown reliably.

---

## ▶️ Run

Open **two terminals** (both with the venv activated and `AIRFLOW_HOME` exported):

**Terminal A — Webserver**

```bash
airflow webserver --port 8080 --debug
```

**Terminal B — Scheduler**

```bash
airflow scheduler
```

Open the UI: [http://127.0.0.1:8080](http://127.0.0.1:8080)

* Ensure your DAG (e.g., `sample_dag` / `my_dag`) is **unpaused**.
* Click **Trigger DAG**.
* Inspect **Graph** → select a task → **Log** (should load local file logs).

---

## 🧪 What to Expect

* **prepare** → **train** → **evaluate** → **notify** (linear flow).
* On success, you’ll find:

  * `Labs/Airflow_Labs/Lab_3/model/model.sav`
  * `Labs/Airflow_Labs/Lab_3/model/metrics.json`

---

## 🧰 Troubleshooting

* **DAG not visible**: confirm `dags_folder` points to `.../Lab_3/dags/` and restart webserver + scheduler.
* **Import errors**: `airflow dags list-import-errors`.
* **Logs still not showing in UI**: ensure `[webserver] serve_logs = False`.
  You can always read raw files under:
  `$AIRFLOW_HOME/logs/<dag_id>/<task_id>/<run_id>/1.log`.
* **macOS fork issues**: stick to `DebugExecutor` (no Postgres/MySQL needed).

---

## 📦 Submission Notes

Include in your PR/branch:

* Updated DAG (`dags/my_dag.py`) with relative paths and clean task separation.
* `dags/src/model_development.py`, `dags/src/success_email.py`.
* `airflow.cfg` changes: `DebugExecutor`, `web_server_host=127.0.0.1`, `base_url`, `serve_logs=False`.
* A short run log snippet or screenshot of a **green** run.

---

## 📝 Change Log (Summary)

* Switched executor to **DebugExecutor** for macOS/SQLite compatibility.
* Bound webserver to **127.0.0.1** and set **base_url**; forced **serve_logs=False** to fix UI log fetching.
* Made dataset/model paths **relative** and saved artifacts to `model/`.
* Added optional **email** notification via `email_smtp` connection.
* Moved heavy imports **inside tasks**; ensured DAG object is created at import time.
