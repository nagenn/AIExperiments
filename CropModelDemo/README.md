# Indian Crop Data Assistant — a small AI model built from scratch

A minimal, fully working example of the whole pipeline behind an "AI model":
real data → a neural network trained from scratch (no PyTorch/TensorFlow) →
a conversational interface on top of it. Built as a teaching demo, not a
production tool — the goal is to make every step visible and readable.

## What this actually is

Two small neural networks, written by hand in NumPy (see
`models/neural_net.py` — full forward pass, backpropagation, and gradient
descent, no autograd library doing the work for you):

1. **A production predictor** — given a crop, state, year, and cultivated
   area, predicts expected production (in 1000 tons).
2. **A shortage classifier** — given a crop and year, predicts whether
   national production was significantly below the recent trend (a proxy
   for "shortage").

On top of that sits a **conversational layer** (`nlp_interface.py`) that
uses [spaCy](https://spacy.io) — a real, independently-trained open-source
NLP library — to parse a plain-English question, pull out the crop/state/year,
and route it to the right place: a direct lookup in the data, the predictor
network, or the shortage classifier.

**Important honesty note:** the two neural nets are the "from scratch AI
model" part of this project. The conversational layer is *not* a language
model that talks freely — it's rule-based routing on top of spaCy's parsing.
That's a deliberate, realistic design: most "chat with your data" tools work
exactly this way, with a thin language layer routing to real trained models
underneath, rather than one giant model doing everything.

## The data

Real Indian district-level agriculture statistics, aggregated to state
level: 20 states, 22 crops, **years 2010–2017**. Originally compiled from
government agriculture statistics and republished on GitHub. See
`data/crops_data_raw.csv` for the raw source and `data/prepare_data.py`
for how it's cleaned.

**Coverage caveat:** this is the most complete dataset reachable from a
sandboxed environment at build time — it does *not* extend to the present.
Questions about years outside 2010–2017 get an explicit extrapolation
warning rather than a confident-sounding wrong answer.

## Setup

Requires Python 3.9+.

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy's small English model (needed for the NLP layer)
python -m spacy download en_core_web_sm
```

## Run it

The trained model weights are already included in `weights/`, so you can
go straight to chatting:

```bash
python app.py
```

Example questions to try:
```
What was rice production in Punjab in 2015?
Did we have a pulse shortage in 2015?
Was there a chickpea shortage in 2014?
Predict wheat production in Haryana for 2018.
How much cotton did Gujarat produce?
```

## Retrain the models yourself

Want to see the actual training happen, not just use the pre-trained
weights? Run:

```bash
cd data
python prepare_data.py          # rebuilds data/crops_long_state.csv

cd ../models
python train_predictor.py       # trains + saves weights/predictor.npz
python train_shortage.py        # trains + saves weights/shortage.npz
```

You'll see the loss print every 200 epochs, going down — that's the
network learning. Test accuracy/error is printed at the end of each script.

## Project structure

```
crop-ai-demo/
├── README.md
├── requirements.txt
├── app.py                    # CLI entry point — chat with the assistant
├── nlp_interface.py           # spaCy parsing + routing logic
├── data/
│   ├── crops_data_raw.csv     # raw source data
│   ├── crops_long_state.csv   # cleaned, reshaped data (generated)
│   └── prepare_data.py        # reshape script
├── models/
│   ├── neural_net.py          # from-scratch NumPy neural network
│   ├── train_predictor.py     # trains the production predictor
│   └── train_shortage.py      # trains the shortage classifier
└── weights/                   # trained weights + metadata (generated)
    ├── predictor.npz
    ├── predictor_meta.json
    ├── shortage.npz
    ├── shortage_meta.json
    └── national_crop_series.csv
```

## Ideas for extending this (good exercises for learners)

- Swap the CLI in `app.py` for a small Flask/FastAPI web app.
- Add a rainfall or soil dataset and see if predictions improve.
- Extend `neural_net.py` with a third hidden layer, or try a different
  activation function, and compare training loss curves.
- Change the shortage threshold in `train_shortage.py` (currently 15%
  below trailing average) and see how the label distribution shifts.
- Add more crop aliases / state aliases to `nlp_interface.py` to make the
  parser more robust to phrasing.
