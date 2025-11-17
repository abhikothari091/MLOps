# Experiment Tracking with Weights & Biases (W&B)

This lab demonstrates how to use Weights & Biases (W&B) to track experiments, visualize metrics, log hyperparameters, and compare multiple training runs. The original starter code trained an XGBoost model on the UCI Dermatology dataset with minimal logging. We significantly improved and modernized the lab by converting it into a clean, modular, reproducible workflow using Random Forest, richer W&B visualizations, and improved dataset handling.

---

## 🔧 What We Changed (Summary of Improvements)

### 1. Replaced XGBoost with Random Forest

The original lab used XGBoost with outdated parameters. We replaced it with a `RandomForestClassifier` to simplify training and reduce dependency complexity.

### 2. Added full W&B experiment structure

We introduced:
- `wandb.init(project=..., name=...)`
- `wandb.config` for hyperparameter tracking
- Logging of accuracy, F1 scores (per class), macro F1, ROC-AUC
- Confusion matrix visualizations
- Feature importance plots via `wandb.Table`

### 3. Cleaned and fixed dataset parsing

The original dataset parsing used `np.loadtxt` with weird converters. We:
- Loaded the dataset using pandas
- Properly handled missing values represented by `?`
- Shifted class labels to 0–5 (required for classification)

### 4. Added a reusable training function

We wrapped the workflow in:

```python
train_random_forest(config)
```

so the student can run multiple W&B sweeps later.

### 5. Improved experiment logging

We added:
- Classification report logging
- Per-class F1
- ROC-AUC-OvR
- Feature importance visualization
- All artifacts logged cleanly in W&B

---

## 📁 Final Folder Structure

```
.
├── Lab1.ipynb            # Your updated notebook using RF + W&B
├── README.md             # This file
└── dermatology.data      # UCI dataset used in the lab
```

---

## 🚀 Updated Lab 1 Workflow

Below is the high-level flow of the updated experiment.

### 1. Load libraries and login to W&B

We authenticate once using:

```python
import wandb
wandb.login()
```

### 2. Download & prep dataset

The dataset is fetched from UCI, loaded with pandas, and cleaned.

### 3. Configure and initialize W&B run

We track hyperparameters:

```python
run = wandb.init(project="Lab1-visualize-models", name="rf-baseline")
wandb.config.update(config)
```

### 4. Train the model

A Random Forest classifier is trained with:

```python
model = RandomForestClassifier(**config)
model.fit(X_train, y_train)
```

### 5. Log all evaluation metrics

We compute:
- Validation accuracy
- Macro F1 score
- ROC AUC (OvR)
- Per-class F1 scores

All metrics are logged to W&B.

### 6. Confusion matrix visualization

W&B's confusion matrix tool is used with integer labels to avoid `KeyError` issues.

### 7. Feature Importance Visualization

We log feature importances using a W&B table.

### 8. Finish the run

```python
run.finish()
```

---

## 🎯 What You Learned in This Lab

By completing the enhanced Lab 1, you now understand how to:

### ✔ Use W&B for experiment tracking:
- `wandb.init()`
- `wandb.config`
- `wandb.log()`
- `wandb.plot.confusion_matrix()`

### ✔ Track ML workflows end-to-end:
- Dataset parsing
- Model training
- Evaluation
- Visualization

### ✔ Add meaningful visual diagnostics:
- Confusion matrix
- Per-class metrics
- Feature importances
- ROC curves

### ✔ Structure a reproducible ML experiment

These improvements make the lab more realistic and closer to actual industry workflows.

---

## 📌 Final Notes

1. This version of the lab is significantly richer and fully aligned with experiment-tracking best practices. The code is modular, visual, and ready for extension (e.g., sweeps, hyperparameter tuning, comparing models, logging artifacts).
2. I have gitignored the reports and the model weights and just kept the lab1 and readme, you can trace the results in the ipynb outputs.
