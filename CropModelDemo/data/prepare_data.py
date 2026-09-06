"""
prepare_data.py
----------------
Reshapes the raw Indian crop dataset (wide format: one row per district-year,
one column per crop x metric) into a clean long format:

    year, state, crop, area (1000 ha), production (1000 tons), yield (kg/ha)

aggregated to state level (summed across districts).

Source data: district-level crop statistics for India, 2010-2017, covering
20 states, 311 districts, and 22 crops (cereals, pulses, oilseeds,
sugarcane, cotton). Originally compiled from government agriculture
statistics and republished on GitHub.

Run:
    python prepare_data.py
Produces:
    crops_long_state.csv  (in this same folder)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).parent / "crops_data_raw.csv"
OUT_PATH = Path(__file__).parent / "crops_long_state.csv"

# Crops that have AREA / PRODUCTION / YIELD triples in the raw file
CROPS = [
    "RICE", "WHEAT", "KHARIF SORGHUM", "RABI SORGHUM", "SORGHUM",
    "PEARL MILLET", "MAIZE", "FINGER MILLET", "BARLEY", "CHICKPEA",
    "PIGEONPEA", "MINOR PULSES", "GROUNDNUT", "SESAMUM",
    "RAPESEED AND MUSTARD", "SAFFLOWER", "CASTOR", "LINSEED",
    "SUNFLOWER", "SOYABEAN", "OILSEEDS", "SUGARCANE", "COTTON",
]


def prepare(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    rows = []
    for crop in CROPS:
        area_col = f"{crop} AREA (1000 ha)"
        prod_col = f"{crop} PRODUCTION (1000 tons)"
        if area_col not in df.columns or prod_col not in df.columns:
            continue
        sub = df[["Year", "State Name", area_col, prod_col]].copy()
        sub.columns = ["year", "state", "area", "production"]
        sub["crop"] = crop.title()
        rows.append(sub)

    long_df = pd.concat(rows, ignore_index=True)

    # Aggregate district-level rows up to state level
    state_df = long_df.groupby(["year", "state", "crop"], as_index=False)[
        ["area", "production"]
    ].sum()

    # Drop rows with negligible/zero area (yield undefined) or negative production
    state_df = state_df[(state_df["area"] > 0.5) & (state_df["production"] >= 0)].copy()
    state_df["yield"] = (state_df["production"] / state_df["area"] * 1000).round(2)
    state_df = state_df.replace([np.inf, -np.inf], np.nan).dropna()
    state_df = state_df.sort_values(["crop", "state", "year"]).reset_index(drop=True)

    state_df.to_csv(out_path, index=False)
    return state_df


if __name__ == "__main__":
    result = prepare()
    print(f"Wrote {len(result)} rows to {OUT_PATH}")
    print(f"Years: {sorted(result['year'].unique())}")
    print(f"States: {result['state'].nunique()}")
    print(f"Crops: {sorted(result['crop'].unique())}")
