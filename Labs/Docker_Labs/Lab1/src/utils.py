# src/utils.py
import os

def get_model_dir(default="models"):
    return os.environ.get("MODEL_DIR", default)