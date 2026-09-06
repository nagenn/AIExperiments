"""
nlp_interface.py
-----------------
The conversational front-end. Uses spaCy (a real, independently-trained
open-source NLP library) to tokenize and tag incoming questions, matches
extracted terms against the known vocabulary of states/crops/years from
the dataset itself, detects a rough intent from keywords, and routes the
question to one of:

    - direct data lookup    (fastest, most reliable -- used whenever the
                              exact fact already exists in the dataset)
    - the trained predictor network   (for years outside the dataset)
    - the trained shortage classifier (for "was there a shortage" questions)

This file does NOT do any machine learning itself -- all the learning
happened in models/train_predictor.py and models/train_shortage.py. This
is just the language layer that decides which trained model to ask.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

sys.path.append(str(Path(__file__).parent / "models"))
from neural_net import NeuralNetwork

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "crops_long_state.csv"
WEIGHTS_DIR = ROOT / "weights"

# A few common short forms / aliases for states
STATE_ALIASES = {
    "up": "Uttar Pradesh",
    "mp": "Madhya Pradesh",
    "tn": "Tamil Nadu",
    "ap": "Andhra Pradesh",
    "wb": "West Bengal",
    "hp": "Himachal Pradesh",
}

# Generic terms that should expand to a group of crops
CROP_GROUPS = {
    "pulses": ["Chickpea", "Pigeonpea", "Minor Pulses"],
    "pulse": ["Chickpea", "Pigeonpea", "Minor Pulses"],
    "oilseed": ["Oilseeds"],
    "oilseeds": ["Oilseeds"],
}

SHORTAGE_WORDS = ["shortage", "deficit", "scarcity", "shortfall", "short of", "shortfall"]
PREDICT_WORDS = ["predict", "forecast", "estimate", "expect", "will be", "projection"]
COMPARE_WORDS = ["normal", "usual", "typical", "average", "higher than", "lower than",
                  "closer to", "than usual", "compared to", "trend"]

# same 15% band used to train the shortage classifier, kept consistent here
LOWER_BAND = 0.85
UPPER_BAND = 1.15


class CropAssistant:
    def __init__(self):
        print("Loading spaCy language model...")
        self.nlp = spacy.load("en_core_web_sm")

        self.df = pd.read_csv(DATA_PATH)
        self.states = sorted(self.df["state"].unique())
        self.crops = sorted(self.df["crop"].unique())
        self.states_lower = {s.lower(): s for s in self.states}
        self.crops_lower = {c.lower(): c for c in self.crops}

        print("Loading trained models...")
        self.predictor = NeuralNetwork.load(WEIGHTS_DIR / "predictor.npz")
        with open(WEIGHTS_DIR / "predictor_meta.json") as f:
            self.pred_meta = json.load(f)

        self.shortage_net = NeuralNetwork.load(WEIGHTS_DIR / "shortage.npz")
        with open(WEIGHTS_DIR / "shortage_meta.json") as f:
            self.shortage_meta = json.load(f)
        self.national_series = pd.read_csv(WEIGHTS_DIR / "national_crop_series.csv")

        self.year_min = self.pred_meta["year_min"]
        self.year_max = self.pred_meta["year_max"]
        print("Ready.\n")

    # ---------- parsing ----------

    def parse(self, text):
        doc = self.nlp(text)
        lower = text.lower()

        # year: any 4-digit token 19xx/20xx
        year_matches = re.findall(r"\b((?:19|20)\d{2})\b", text)
        year = int(year_matches[0]) if year_matches else None

        # crop: match multi-word crop names first, then single tokens/groups
        crop_list = []
        for group_term, expansion in CROP_GROUPS.items():
            if group_term in lower:
                crop_list = expansion
                break
        if not crop_list:
            for crop_l, crop_orig in self.crops_lower.items():
                if crop_l in lower:
                    crop_list = [crop_orig]
                    break

        # state: match aliases first, then full names
        state = None
        for alias, full in STATE_ALIASES.items():
            if re.search(rf"\b{alias}\b", lower):
                state = full
                break
        if state is None:
            for state_l, state_orig in self.states_lower.items():
                if state_l in lower:
                    state = state_orig
                    break

        # intent (checked in order of specificity)
        if any(w in lower for w in SHORTAGE_WORDS):
            intent = "shortage"
        elif any(w in lower for w in COMPARE_WORDS):
            intent = "compare"
        elif any(w in lower for w in PREDICT_WORDS):
            intent = "predict"
        else:
            intent = "lookup"

        return {
            "doc": doc,
            "year": year,
            "crops": crop_list,
            "state": state,
            "intent": intent,
        }

    # ---------- routing ----------

    def answer(self, text):
        parsed = self.parse(text)
        intent, year, crops, state = (
            parsed["intent"], parsed["year"], parsed["crops"], parsed["state"]
        )

        if intent == "shortage":
            return self._answer_shortage(crops, year)
        elif intent == "compare":
            return self._answer_compare(crops, year)
        elif intent == "predict":
            return self._answer_predict(crops, state, year)
        else:
            return self._answer_lookup(crops, state, year)

    def _answer_shortage(self, crops, year):
        if not crops:
            return ("I can check for a shortage, but I need to know which crop "
                    "(or 'pulses') you mean.")
        if not year:
            return "Which year would you like me to check?"

        results = []
        for crop in crops:
            row = self.national_series[
                (self.national_series["crop"] == crop) & (self.national_series["year"] == year)
            ]
            if row.empty:
                results.append(
                    f"{crop}: no trend data available for {year} "
                    f"(need at least 3 prior years in range {self.year_min}-{self.year_max})."
                )
                continue
            row = row.iloc[0]
            X = self._shortage_features(row)
            prob = float(self.shortage_net.predict(X)[0, 0])
            is_shortage = prob > 0.5
            pct_of_trend = 100 * row["production"] / row["trail_avg"]
            verdict = "YES, a shortage" if is_shortage else "no shortage"
            results.append(
                f"{crop} in {year}: {verdict} (model confidence {prob:.0%}). "
                f"Actual production was {row['production']:.0f} thousand tons, "
                f"which is {pct_of_trend:.0f}% of the trailing 3-year average "
                f"({row['trail_avg']:.0f})."
            )
        return "\n".join(results)

    def _answer_compare(self, crops, year):
        """Answers 'was production higher/lower/normal' questions for ANY
        crop (not just pulses), using the same trailing-3-year-average logic
        the shortage classifier was trained on -- but reporting all three
        outcomes (higher / normal / lower), not just a shortage yes/no."""
        if not crops:
            return "Which crop would you like me to compare?"
        if not year:
            return "Which year would you like me to check?"

        results = []
        for crop in crops:
            row = self.national_series[
                (self.national_series["crop"] == crop) & (self.national_series["year"] == year)
            ]
            if row.empty:
                # not enough trailing history for this crop-year -- fall back
                # to just reporting the raw figure, with no trend comparison
                raw = self.df[(self.df["crop"] == crop) & (self.df["year"] == year)]
                if raw.empty:
                    results.append(f"{crop}: no data for {year}.")
                else:
                    total = raw["production"].sum()
                    results.append(
                        f"{crop} in {year}: production was {total:.0f} thousand tons "
                        f"(all-India), but I don't have enough prior-year history "
                        f"in this dataset to say whether that's higher or lower than normal."
                    )
                continue

            row = row.iloc[0]
            ratio = row["production"] / row["trail_avg"]
            pct = ratio * 100
            if ratio >= UPPER_BAND:
                verdict = "HIGHER than normal"
            elif ratio <= LOWER_BAND:
                verdict = "LOWER than normal"
            else:
                verdict = "close to normal"

            results.append(
                f"{crop} in {year}: {verdict}. Actual production was "
                f"{row['production']:.0f} thousand tons, {pct:.0f}% of the "
                f"trailing 3-year average ({row['trail_avg']:.0f})."
            )
        return "\n".join(results)

    def _shortage_features(self, row):
        norm = self.shortage_meta["norm"]
        crops = self.shortage_meta["crops"]
        X = np.zeros((1, 3 + len(crops)))
        X[0, 0] = (row["year"] - norm["year_mean"]) / norm["year_std"]
        X[0, 1] = (row["production"] - norm["prod_mean"]) / norm["prod_std"]
        X[0, 2] = (row["trail_avg"] - norm["trail_mean"]) / norm["trail_std"]
        X[0, 3 + crops.index(row["crop"])] = 1.0
        return X

    def _answer_lookup(self, crops, state, year):
        if not crops:
            return "I can look that up, but I need to know which crop you mean."
        subset = self.df[self.df["crop"].isin(crops)]
        if state:
            subset = subset[subset["state"] == state]
        if year:
            subset = subset[subset["year"] == year]

        if subset.empty:
            return (f"I don't have data matching that "
                    f"(dataset covers {self.year_min}-{self.year_max}).")

        if state and year:
            row = subset.iloc[0]
            return (f"{row['crop']} in {state}, {year}: "
                    f"production {row['production']:.0f} thousand tons, "
                    f"area {row['area']:.0f} thousand ha, "
                    f"yield {row['yield']:.0f} kg/ha.")
        else:
            agg = subset.groupby("crop")[["production", "area"]].sum()
            lines = [f"{c}: {r['production']:.0f} thousand tons across "
                     f"{r['area']:.0f} thousand ha" for c, r in agg.iterrows()]
            scope = f" in {state}" if state else " (all states)"
            when = f" for {year}" if year else " (summed across all years in data)"
            return f"Data{scope}{when}:\n" + "\n".join(lines)

    def _answer_predict(self, crops, state, year):
        if not crops or not state or not year:
            missing = [n for n, v in [("crop", crops), ("state", state), ("year", year)] if not v]
            return f"For a prediction I need crop, state, and year. Missing: {', '.join(missing)}."

        crop = crops[0]
        if crop not in self.pred_meta["crops"] or state not in self.pred_meta["states"]:
            return "I don't have that crop/state in the trained model's vocabulary."

        # Use the average area for this crop+state as a stand-in if we don't
        # have an exact area for a future year
        hist = self.df[(self.df["crop"] == crop) & (self.df["state"] == state)]
        area = hist["area"].mean() if not hist.empty else self.df["area"].mean()

        X = self._predict_features(crop, state, year, area)
        pred_norm = self.predictor.predict(X)[0, 0]
        pred = pred_norm * self.pred_meta["y_std"] + self.pred_meta["y_mean"]

        caveat = ""
        if year > self.year_max or year < self.year_min:
            caveat = (f" (note: this is extrapolated -- the model was only trained on "
                      f"{self.year_min}-{self.year_max}, so treat this as a rough estimate)")

        return (f"Predicted {crop} production in {state} for {year}: "
                f"~{pred:.0f} thousand tons{caveat}")

    def _predict_features(self, crop, state, year, area):
        norm = self.pred_meta["norm"]
        states = self.pred_meta["states"]
        crops = self.pred_meta["crops"]
        X = np.zeros((1, 2 + len(states) + len(crops)))
        X[0, 0] = (year - norm["year_mean"]) / norm["year_std"]
        X[0, 1] = (area - norm["area_mean"]) / norm["area_std"]
        X[0, 2 + states.index(state)] = 1.0
        X[0, 2 + len(states) + crops.index(crop)] = 1.0
        return X
