# Docker Lab – Containerized Model Training (Iris)

This lab demonstrates how to containerize a simple machine learning training workflow using Docker. The project trains a Random Forest classifier on the Iris dataset inside a Docker container, evaluates it, and saves the trained model to a configurable directory. The goal is to help you understand how to build, run, and persist ML models in isolated, reproducible environments.

---

## 1. Overview of Improvements

We enhanced the starter code to make it more modular and closer to a real-world setup:

### Training pipeline modularization
- Moved all logic to `src/train.py`
- Separated evaluation into `src/evaluate.py`

### Configurable model saving
- Trained model is written to a configurable folder using:
  - env var: `MODEL_DIR`
  - or CLI arg: `--model-dir`

### Dependency management
- Docker installs Python dependencies from `requirements.txt`

### Local model persistence
- You can mount a local folder (e.g. `./models`) to persist the trained model even after the container exits

### Docker best practices
- Added a `.dockerignore` to avoid copying unnecessary files
- Structured the Dockerfile to use caching effectively

---

## 2. Project Structure

```
.
├── src
│   ├── train.py         # Trains the Random Forest classifier
│   ├── evaluate.py      # Evaluates the saved model
│   └── utils.py         # (optional) helper utilities
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── .dockerignore        # Files to exclude during image build
└── README.md            # Documentation
```

---

## 3. How It Works

### Training workflow

1. The container starts by running `train.py` (default CMD in Dockerfile).
2. `train.py`:
   - Loads the Iris dataset from `sklearn.datasets`
   - Splits data into train/test
   - Trains a `RandomForestClassifier`
   - Saves the trained model (default: `iris_model.pkl`) to `/app/models` or to the path specified by `MODEL_DIR`

### Evaluation workflow

- The `src/evaluate.py` script can be run:
  - Manually, or
  - As a separate Docker command
- It:
  - Loads the saved model
  - Loads the Iris dataset again
  - Reports metrics (accuracy / F1)

---

## 4. Dockerfile (final version)

Key points:
- `--no-cache-dir` keeps the image smaller
- We create `/app/models` so the script has a place to write
- Code + config are separated — you can change output dir at runtime

---

## 5. Building and Running the Container

### 5.1 Build the image

```bash
docker build -t iris-train:latest .
```

### 5.2 Run training (ephemeral)

```bash
docker run --rm iris-train:latest
```

You should see something like:
```
[train] model saved to /app/models/iris_model.pkl
```

This means the container ran, trained the model, and wrote it inside the container.

---

### 5.3 Run and persist model output

By default, anything written inside the container is lost when it stops.

To keep the model locally:

```bash
mkdir -p models
docker run --rm -v $(pwd)/models:/app/models iris-train:latest
```

Now the model will be available on your Mac at:
```
./models/iris_model.pkl
```

---

### 5.4 Run evaluation

Once the model is saved (either inside the image or via a mounted volume), you can run:

```bash
docker run --rm -v $(pwd)/models:/app/models iris-train:latest python src/evaluate.py
```

This:
- Mounts your local `./models` → `/app/models` in the container
- Runs the evaluation script
- Loads `/app/models/iris_model.pkl`
- Prints evaluation metrics

---
