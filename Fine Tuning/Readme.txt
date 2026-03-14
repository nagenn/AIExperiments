Here's everything you need to run the demo:


**First time only — install dependencies**
```
pip install transformers torch scikit-learn pandas datasets tornado
```

**First time only — fine-tune the model**
```
python train.py
```
Takes 3–5 minutes. Watch the loss drop each epoch. Model saves to `./ticket-classifier/` when done.

**Every time — run the web app**
```
python app.py
```
Then open `http://localhost:8686` in your browser.

///////////////\\\\\\\\\\\\\\\\\\\\\\\

**Key things to remember**

- `train.py` only needs to run once unless you change the data or training parameters
- The first run of `train.py` will also download DistilBERT (~65MB) — this only happens once, it caches locally
- `app.py` loads the saved model from `./ticket-classifier/` — so train.py must have been run at least once before you start the app
- All four files (`train.py`, `app.py`, `index.html`, `support_tickets.csv`) must be in the same folder

**The four ticket categories the model learns**
- 💳 Billing — charges, invoices, refunds
- 🔧 Technical — bugs, errors, app issues
- 📦 Delivery — shipping, tracking, couriers
- ↩️ Returns — returns, exchanges, replacements
