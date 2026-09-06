"""
train_shortage.py
------------------
Trains a small from-scratch neural network to classify whether a given
crop-year was a "shortage" year at the national level.

Label definition (a heuristic, not ground truth): a crop-year is labeled
a shortage if national production fell more than 15% below the trailing
3-year average production for that crop. This mirrors how "shortage" is
often discussed informally (a noticeable drop versus recent trend), and
gives the classifier a learnable signal to generalize from -- covering
things like the real 2014-2015 pulse production dip visible in this data.

Inputs (features):
    - year (normalized)
    - crop (one-hot encoded)
    - area (1000 ha, normalized)
    - production (1000 tons, normalized)
    - trailing 3-year average production (normalized)
Output:
    - probability of "shortage" (sigmoid, binary classification)

Run:
    python train_shortage.py
Produces:
    ../weights/shortage.npz
    ../weights/shortage_meta.json
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

SHORTAGE_THRESHOLD = 0.85  # production below 85% of trailing 3yr avg = shortage


def build_national_series(df):
    nat = df.groupby(["year", "crop"], as_index=False)[["area", "production"]].sum()
    nat = nat.sort_values(["crop", "year"]).reset_index(drop=True)

    # trailing 3-year average (using only prior years, so first 3 years per
    # crop don't have a full window and are dropped)
    nat["trail_avg"] = (
        nat.groupby("crop")["production"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    nat = nat.dropna(subset=["trail_avg"]).reset_index(drop=True)
    nat["shortage"] = (nat["production"] < SHORTAGE_THRESHOLD * nat["trail_avg"]).astype(int)
    return nat


def build_features(nat, crops):
    crop_idx = {c: i for i, c in enumerate(crops)}
    n = len(nat)
    X = np.zeros((n, 3 + len(crops)))

    year_mean, year_std = nat["year"].mean(), nat["year"].std()
    area_mean, area_std = nat["area"].mean(), nat["area"].std()
    prod_mean, prod_std = nat["production"].mean(), nat["production"].std()
    trail_mean, trail_std = nat["trail_avg"].mean(), nat["trail_avg"].std()

    X[:, 0] = (nat["year"].values - year_mean) / year_std
    X[:, 1] = (nat["production"].values - prod_mean) / prod_std
    X[:, 2] = (nat["trail_avg"].values - trail_mean) / trail_std
    for i, row in enumerate(nat.itertuples()):
        X[i, 3 + crop_idx[row.crop]] = 1.0

    norm = {
        "year_mean": year_mean, "year_std": year_std,
        "prod_mean": prod_mean, "prod_std": prod_std,
        "trail_mean": trail_mean, "trail_std": trail_std,
        "area_mean": area_mean, "area_std": area_std,
    }
    return X, norm


def main():
    df = pd.read_csv(DATA_PATH)
    crops = sorted(df["crop"].unique())

    nat = build_national_series(df)
    print(f"Built {len(nat)} crop-year examples "
          f"({nat['shortage'].sum()} labeled as shortage years)")

    X, norm = build_features(nat, crops)
    y = nat["shortage"].values.reshape(-1, 1).astype(float)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    split = max(1, int(0.8 * len(X)))
    train_idx, test_idx = idx[:split], idx[split:]

    net = NeuralNetwork(
        n_input=X.shape[1], n_hidden1=16, n_hidden2=8, n_output=1,
        mode="binary",
    )

    print("Training shortage classifier...")
    net.train(X[train_idx], y[train_idx], epochs=1500, lr=0.1, verbose_every=200)

    if len(test_idx) > 0:
        pred = net.predict(X[test_idx])
        pred_label = (pred > 0.5).astype(int)
        acc = np.mean(pred_label == y[test_idx])
        print(f"\nTest accuracy: {acc:.2f} (on {len(test_idx)} held-out examples)")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    net.save(WEIGHTS_DIR / "shortage.npz")

    meta = {"crops": crops, "norm": norm, "threshold": SHORTAGE_THRESHOLD}
    with open(WEIGHTS_DIR / "shortage_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Also save the national series itself -- the conversational layer needs
    # it to look up actual production/trend numbers, not just the label.
    nat.to_csv(WEIGHTS_DIR / "national_crop_series.csv", index=False)

    print(f"Saved weights to {WEIGHTS_DIR / 'shortage.npz'}")


if __name__ == "__main__":
    main()
