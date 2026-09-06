"""
train_predictor.py
-------------------
Trains a small from-scratch neural network to predict a crop's production
(1000 tons) for a given state, crop, year, and cultivated area.

Inputs (features):
    - year (normalized)
    - area (1000 ha, normalized)
    - state (one-hot encoded)
    - crop (one-hot encoded)
Output:
    - predicted production (1000 tons)

Run:
    python train_predictor.py
Produces:
    ../weights/predictor.npz          (trained network weights)
    ../weights/predictor_meta.json    (feature encoding + normalization info)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from neural_net import NeuralNetwork

DATA_PATH = Path(__file__).parent.parent / "data" / "crops_long_state.csv"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


def build_features(df, states, crops):
    state_idx = {s: i for i, s in enumerate(states)}
    crop_idx = {c: i for i, c in enumerate(crops)}

    n = len(df)
    X = np.zeros((n, 2 + len(states) + len(crops)))

    year_mean, year_std = df["year"].mean(), df["year"].std()
    area_mean, area_std = df["area"].mean(), df["area"].std()

    X[:, 0] = (df["year"].values - year_mean) / year_std
    X[:, 1] = (df["area"].values - area_mean) / area_std
    for i, row in enumerate(df.itertuples()):
        X[i, 2 + state_idx[row.state]] = 1.0
        X[i, 2 + len(states) + crop_idx[row.crop]] = 1.0

    norm = {"year_mean": year_mean, "year_std": year_std,
            "area_mean": area_mean, "area_std": area_std}
    return X, norm


def main():
    df = pd.read_csv(DATA_PATH)
    states = sorted(df["state"].unique())
    crops = sorted(df["crop"].unique())

    X, norm = build_features(df, states, crops)

    y = df["production"].values.reshape(-1, 1)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std  # normalize target for stable training

    # simple train/test split (80/20), shuffled
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]

    net = NeuralNetwork(
        n_input=X.shape[1], n_hidden1=32, n_hidden2=16, n_output=1,
        mode="regression",
    )

    print(f"Training predictor on {len(train_idx)} examples "
          f"({len(states)} states x {len(crops)} crops x years)...")
    net.train(X[train_idx], y_norm[train_idx], epochs=2000, lr=0.05, verbose_every=200)

    # Evaluate on held-out test set (de-normalized, in real units)
    pred_norm = net.predict(X[test_idx])
    pred = pred_norm * y_std + y_mean
    actual = y[test_idx]
    mae = np.mean(np.abs(pred - actual))
    print(f"\nTest MAE: {mae:.1f} (1000 tons)  |  mean production: {actual.mean():.1f}")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    net.save(WEIGHTS_DIR / "predictor.npz")

    meta = {
        "states": states,
        "crops": crops,
        "norm": norm,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
    }
    with open(WEIGHTS_DIR / "predictor_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved weights to {WEIGHTS_DIR / 'predictor.npz'}")


if __name__ == "__main__":
    main()
