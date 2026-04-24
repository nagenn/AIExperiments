# ContractIQ — Setup & Demo Guide

---

## Folder Structure

```
files/
├── app.py
├── analyze_contract.py
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
Create a file called `.env` in the same folder as `app.py`:
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
(Change folder contains the other)

After renaming, do a hard refresh in Safari: Cmd + Option + R

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
Reviewed contracts are never affected — only new unregistered files are added.
You can drop new PDFs into contracts/ at any time and click the button again.

---

## Demo Flow

### ACT 1 — The Manual Way (index_before.html active)

1. Open the app — show the queue with history records already present
2. Click "Check for New" — your PDF contracts appear as Pending Review
3. Click sample_contract.pdf
4. Work through the checklist slowly — deliberately miss the payment terms checkbox
5. Enter your name, select Medium risk, add a note
6. Click Submit Review — status updates in the queue immediately
7. Say: "That is one review. We have more in the queue.
        Each one takes 20-30 minutes. A reviewer's entire morning."

### ACT 2 — Reset
1. In the result panel click "Reset for Demo"
2. Rename files to activate index_after.html (see Switching section above)
3. Hard refresh the browser: Cmd + Option + R (Safari)
4. Say: "Same contract. Same system. Now watch what changes."

### ACT 3 — The Agent (index_after.html active)
1. Click sample_contract.pdf in the queue
2. Click the "Agent Review" tab
3. Click "Run Agent Review"
4. Watch the trace panel fill in real time — clause by clause
5. Result tab opens automatically when complete
6. Queue status updates to Escalated for Legal Review

The line to say:
"The agent caught the payment terms violation the manual reviewer missed.
 It reviewed the contract in seconds, not minutes. And it updated the
 system automatically — no one touched a keyboard after clicking that button.
 The human is still in the loop — but they are reviewing the agent's findings,
 not doing the work."

### Reviewing the other contracts
- nexus_consulting_sow.pdf   — should return LOW risk, Agent Cleared
- cloudbridge_saas_agreement.pdf — should return LOW risk, Agent Cleared
- vertex_data_services.pdf   — should return HIGH risk, Escalated for Legal Review

Use these to show the agent is not just flagging everything —
it correctly clears compliant contracts and escalates the problematic ones.

---

## Resetting for the Next Run

Click "Reset for Demo" on any reviewed contract in the result panel.

Or reset all contracts via the terminal:
```
rm contracts.db
python3.9 seed_contracts.py
```
Then click "Check for New" in the browser to reload the PDFs.

Note: deleting contracts.db does not affect your PDF files in contracts/.

---

## Troubleshooting

### Browser showing old version after file rename
Hard refresh in Safari: Cmd + Option + R
Hard refresh in Chrome: Cmd + Shift + R

### compliance_rules.json not found
Make sure compliance_rules.json is in the same folder as analyze_contract.py.

### PDF not found error when running agent
Make sure the PDF file is inside the contracts/ folder, not the root folder.
Filenames are case-sensitive.

### Agent returns unknown risk
Check the uvicorn terminal for a Python error. Most likely cause is
an OpenAI API key issue — verify your .env file has the correct key.

### Port already in use
```
lsof -i :8282
kill -9 <PID>
```

---

## Contract Summary for the Demo

| Contract                       | Expected Result              | Key Issues                                              |
|-------------------------------|------------------------------|---------------------------------------------------------|
| sample_contract.pdf           | HIGH — Escalated             | 45-day payment, foreign jurisdiction, 3 missing clauses |
| vertex_data_services.pdf      | HIGH — Escalated             | Unlimited liability, auto-renewal, foreign jurisdiction  |
| nexus_consulting_sow.pdf      | LOW — Agent Cleared          | Fully compliant, Net 28 days, all clauses present        |
| cloudbridge_saas_agreement.pdf| LOW — Agent Cleared          | Fully compliant, Net 30 days, all clauses present        |
