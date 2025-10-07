import os
import json
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# paths 
HERE = Path(__file__).resolve().parent
DATA_CSV = HERE.parent / "data" / "advertising.csv"          # Labs/Airflow_Labs/Lab_3/dags/data/advertising.csv
MODEL_DIR = HERE.parent.parent / "model"                      # Labs/Airflow_Labs/Lab_3/model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"                          # saved pipeline

def load_data() -> pd.DataFrame:
    """Load the CSV from the repo-relative path."""
    df = pd.read_csv(DATA_CSV)
    return df

def split_preprocess(df: pd.DataFrame):
    """Prepare X/y, split, and build the preprocessing pipeline."""
    X = df.drop(['Timestamp', 'Clicked on Ad', 'Ad Topic Line', 'Country', 'City'], axis=1)
    y = df['Clicked on Ad'].astype(int)

    num_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop", 
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    return pipe, X_train, X_test, y_train, y_test

def train_and_save(pipe: Pipeline, X_train, y_train):
    pipe.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return str(MODEL_PATH)

def evaluate(X_test, y_test) -> dict:
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    # also write a tiny artifact for the email
    artifact = MODEL_DIR / "metrics.json"
    artifact.write_text(json.dumps({"accuracy": acc, "report": report}, indent=2))
    return {"accuracy": acc, "artifact": str(artifact)}