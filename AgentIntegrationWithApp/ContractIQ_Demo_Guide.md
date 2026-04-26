# ContractIQ — Setup & Demo Guide

---

## Folder Structure

```
files/
├── app.py
├── analyze_contract.py              ← module used by the app (do not rename)
├── analyze_contract_standalone.py   ← run this from terminal for the demo
├── compliance_rules.json
├── seed_contracts.py
├── .env
├── contracts/
│   ├── sample_contract.pdf
│   ├── nexus_consulting_sow.pdf
│   ├── cloudbridge_saas_agreement.pdf
│   └── vertex_data_services.pdf
└── static/
    ├── index.html          ← active file (swap this to switch modes)
    ├── index_before.html   ← manual review only
    └── index_after.html    ← full agent version
```

---

## One-Time Setup

### 1. Install dependencies
```
pip install fastapi uvicorn pdfplumber openai requests python-dotenv reportlab
```

### 2. Create your .env file
Create a file called .env in the same folder as app.py:
```
OPENAI_API_KEY=sk-...
```

### 3. Create the contracts folder and add PDFs
```
mkdir contracts
```
Drop all your PDF contracts into this folder.

### 4. Seed the database
```
python3.9 seed_contracts.py
```
This creates two pre-reviewed history records so the queue looks
lived-in before the demo starts. Run once only.

### 5. Verify app.py serves index.html
In app.py, confirm this line reads:
```
return FileResponse("static/index.html")
```

---

## Switching Between Before and After Modes

### Before mode (manual review only — Act 1)
Rename index_before.html to index.html in the static/ folder.
The app serves whatever file is named index.html — no code change needed.

### After mode (agent review — Act 2 and 3)
Rename index.html back to index_before.html.
Rename index_after.html to index.html.

After renaming, hard refresh in Safari: Cmd + Option + R

---

## Running the Demo

### Start the app
```
uvicorn app:app --reload --port 8282
```
Open: http://127.0.0.1:8282

### Load your contracts
Click the "Check for New" button in the top right of the queue.
The app scans the contracts/ folder and registers any new PDFs.
Reviewed contracts are never affected — only new files are added.
You can drop new PDFs into contracts/ at any time and click again.

---

## Demo Flow

---

### PART 1 — The Raw Agent (2 minutes)

Open analyze_contract_standalone.py in your editor.
Walk through the code briefly — point out:
- extract_contract_text()  — the PDF reader (perception)
- compliance_rules.json    — the knowledge store (memory)
- the OpenAI call          — the brain
- the JSON output          — structured judgment

Then run it from the terminal:

```
python3.9 analyze_contract_standalone.py sample_contract.pdf
```

Watch the result print. Then say:

  "This is the agent working. It read the contract, applied the rules,
   made a judgment. But the output is JSON in a terminal. Nobody in
   your legal team is going to use this. This is where integration begins."

---

### PART 2 — The Manual Way (index_before.html active)

Make sure index_before.html is renamed to index.html.
Open http://127.0.0.1:8282 in Safari.

1. Click "Check for New" — contracts appear as Pending Review
2. Click sample_contract.pdf
3. Work through the checklist slowly
4. Deliberately miss the payment terms checkbox
5. Enter your name, select Medium risk, add a note
6. Click Submit Review — status updates in the queue
7. Say:
   "That is one review. We have more in the queue.
    Each one takes 20-30 minutes. A reviewer's entire morning.
    And I still missed something."

---

### PART 3 — The Agent in the Workflow (index_after.html active)

1. Click "Reset for Demo" on sample_contract.pdf
2. Rename files to activate index_after.html
3. Hard refresh: Cmd + Option + R (Safari)
4. Click sample_contract.pdf
5. Click the "Agent Review" tab
6. Click "Run Agent Review"
7. Watch the trace panel fill in real time
8. Result tab opens automatically when complete
9. Say:
   "Same agent as the terminal. Same logic. Same rules.
    But now it's integrated — it updated the system automatically,
    it caught the payment terms issue the manual reviewer missed,
    and the human reviews the finding, not the contract.
    That is the shift from doer to reviewer."

---

### Showing other contracts

Run these from the terminal to show the raw agent output first,
then click "Run Agent Review" in the app to show the integrated version:

```
python3.9 analyze_contract_standalone.py nexus_consulting_sow.pdf
python3.9 analyze_contract_standalone.py cloudbridge_saas_agreement.pdf
python3.9 analyze_contract_standalone.py vertex_data_services.pdf
```

---

## Contract Summary

| Contract                        | Expected Result         | Key Issues                                               |
|---------------------------------|-------------------------|----------------------------------------------------------|
| sample_contract.pdf             | HIGH — Escalated        | 45-day payment, foreign jurisdiction, 3 missing clauses  |
| vertex_data_services.pdf        | HIGH — Escalated        | Unlimited liability, auto-renewal, foreign jurisdiction  |
| nexus_consulting_sow.pdf        | LOW — Agent Cleared     | Fully compliant, Net 28 days, all clauses present        |
| cloudbridge_saas_agreement.pdf  | LOW — Agent Cleared     | Fully compliant, Net 30 days, all clauses present        |

---

## Resetting for the Next Run

Click "Reset for Demo" on any reviewed contract in the result panel.

Or reset everything cleanly:
```
rm contracts.db
python3.9 seed_contracts.py
uvicorn app:app --reload --port 8282
```
Then click "Check for New" in the browser.
Your PDF files in contracts/ are never affected by a database reset.

---

## Troubleshooting

### Browser showing old version after file rename
Hard refresh in Safari:  Cmd + Option + R
Hard refresh in Chrome:  Cmd + Shift + R

### compliance_rules.json not found
Make sure compliance_rules.json is in the same folder as analyze_contract.py.

### PDF not found error when running agent
Make sure the PDF is inside the contracts/ folder, not the root folder.
Filenames are case-sensitive.

### Agent returns unknown risk
Check the uvicorn terminal for a Python traceback. Most likely cause
is an OpenAI API key issue — verify your .env file has the correct key.

### Port already in use
```
lsof -i :8282
kill -9 <PID>
```
