# FastAPI_Labs/src/train.py
import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Save model here to keep compatibility with the Streamlit lab path
MODEL_PATH = Path(__file__).resolve().parent / "iris_model.pkl"

def main():
    data = load_iris(as_frame=True)
    X = data.data  # columns: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
    y = data.target

    # Column order we expect from the UI/JSON
    feature_order = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = X.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    X = X[feature_order]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest is robust; we keep a ColumnTransformer for future proofing (e.g., scaling or mixing types)
    pre = ColumnTransformer(transformers=[], remainder="passthrough")

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=data.target_names))

    joblib.dump({"pipeline": pipe, "target_names": list(data.target_names)}, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")

if __name__ == "__main__":
    main()