#!/usr/bin/env python3
"""
Usage:
    python run_model.py --input "[5.1, 3.5, 1.4, 0.2]"
"""

import argparse
import json
from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path("artifacts/model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def parse_input(input_str):
    try:
        features = json.loads(input_str)
        if not isinstance(features, list):
            raise ValueError("Input must be a list")
        return np.array(features).reshape(1, -1)
    except json.JSONDecodeError:
        raise ValueError(
            '❌ Invalid input. Use JSON list, e.g. --input "[5.1,3.5,1.4,0.2]"'
        )


def main():
    parser = argparse.ArgumentParser()

    # ✅ Make input optional (important for CI)
    parser.add_argument(
        "--input",
        default="[5.1, 3.5, 1.4, 0.2]",
        help="Feature list as JSON string"
    )

    args = parser.parse_args()

    # Parse input safely
    X = parse_input(args.input)

    # Load model
    model = load_model()

    # Predict
    pred = model.predict(X)

    print(json.dumps({"prediction": pred.tolist()}))


if __name__ == "__main__":
    main()