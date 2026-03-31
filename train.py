import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"


def main():
    # ✅ Create artifacts folder
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ✅ Load sample data (replace with your dataset later)
    data = load_iris()
    X, y = data.data, data.target

    # ✅ Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # ✅ Save model
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    main()