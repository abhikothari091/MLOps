Docker Lab – Containerized Model Training (Iris)

This lab demonstrates how to containerize a simple machine learning training workflow using Docker. The project trains a Random Forest classifier on the Iris dataset inside a Docker container, evaluates it, and saves the trained model to a configurable directory. The goal is to help you understand how to build, run, and persist ML models in isolated, reproducible environments.

⸻

1. Overview of Improvements

We enhanced the starter code to make it more production-like and modular:
	•	Training pipeline modularization:
	•	Moved all logic to src/train.py and separated evaluation into src/evaluate.py.
	•	Configurable model saving:
	•	The trained model is written to a configurable folder using MODEL_DIR or --model-dir.
	•	Dependency management:
	•	Docker installs Python dependencies from requirements.txt.
	•	Local model persistence:
	•	You can mount a local folder (./models) to persist the trained model even after the container exits.
	•	Docker best practices applied:
	•	Added a .dockerignore to avoid copying unnecessary files.
	•	Structured the Dockerfile to use caching effectively.

⸻

2. Project Structure

.
├── src
│   ├── train.py         # Trains the Random Forest classifier
│   ├── evaluate.py      # Evaluates the saved model
│   └── utils.py         # Optional helper utilities
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── .dockerignore        # Files to exclude during image build
└── README.md            # Documentation


⸻

3. How It Works

Training Workflow
	1.	The container starts by running train.py.
	2.	train.py:
	•	Loads the Iris dataset from sklearn.datasets.
	•	Splits data into training and test sets.
	•	Trains a Random Forest Classifier.
	•	Saves the trained model (iris_model.pkl) to /app/models or the path specified in the environment variable MODEL_DIR.

Evaluation Workflow
	1.	The evaluation script (evaluate.py) can be run manually or as a separate container command.
	2.	It loads the saved model and computes accuracy/F1-score on the Iris dataset.

⸻

4. Dockerfile Breakdown

# Use an official Python runtime
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY src/ ./src
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model output directory
RUN mkdir -p /app/models

# Run training script by default
CMD ["python", "src/train.py"]

Key Improvements:
	•	Uses --no-cache-dir for smaller image size.
	•	Follows the 12-factor app principle by separating code and config.
	•	Creates the model directory inside the image.

⸻

5. Building and Running the Container

Build the Image

docker build -t iris-train:latest .

Run Training (Ephemeral)

docker run --rm iris-train:latest

➡️ This will train the model and print:

[train] model saved to /app/models/iris_model.pkl

Run and Persist Model Output

To keep the trained model locally:

mkdir -p models
docker run --rm -v $(pwd)/models:/app/models iris-train:latest

➡️ The model file will be available at:

./models/iris_model.pkl

Run Evaluation

Once the model is saved:

docker run --rm -v $(pwd)/models:/app/models iris-train:latest python src/evaluate.py


⸻