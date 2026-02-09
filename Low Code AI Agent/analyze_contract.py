
import json
import pdfplumber
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
OPENAI_MODEL = "gpt-4o-mini"  # cost-effective + strong reasoning
client = OpenAI(api_key="<your-api-key>")
#You will need to get this from the AI service you chose
# For example: https://platform.openai.com/api-keys for GPT. This is usually a paid service.


# ----------------------------
# STEP 1: Extract contract text
# ----------------------------
def extract_contract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:  
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# ----------------------------
# STEP 2: Load compliance rules
# ----------------------------
def load_compliance_rules(path="compliance_rules.json"):
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------
# STEP 3: Build reasoning prompt
# ----------------------------
def build_prompt(contract_text, rules):
    return f"""
You are a legal contract review assistant.

Your task is to review the contract strictly against the provided compliance rules.
Do NOT invent rules. Do NOT provide legal advice.

CONTRACT TEXT:
{contract_text}

COMPLIANCE RULES:
{json.dumps(rules, indent=2)}

Follow these steps:
1. Identify missing required clauses.
2. Identify prohibited or risky terms.
3. Assess overall risk (Low, Medium, High).
4. Extract key obligations or deadlines.
5. Provide recommendations if risk is Medium or High.
6. Estimate your confidence (0.0–1.0).

Return ONLY valid JSON in this format:
{{
  "risk_score": "Low | Medium | High",
  "missing_clauses": [],
  "problematic_terms": [],
  "key_obligations": [],
  "recommendations": "",
  "confidence": 0.0
}}
"""


# ----------------------------
# STEP 4: Call GPT
# ----------------------------
def analyze_contract(contract_text, compliance_rules):
    prompt = build_prompt(contract_text, compliance_rules)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    raw_output = response.choices[0].message.content

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "risk_score": "Unknown",
            "missing_clauses": [],
            "problematic_terms": [],
            "key_obligations": [],
            "recommendations": "AI response could not be parsed.",
            "confidence": 0.0
        }


# ----------------------------
# STEP 5: Run the agent
# ----------------------------
if __name__ == "__main__":
    contract_path = "sample_contract.pdf"

    contract_text = extract_contract_text(contract_path)
    rules = load_compliance_rules()

    result = analyze_contract(contract_text, rules)

    print("\n=== CONTRACT REVIEW RESULT ===")
    print(json.dumps(result, indent=2))

    if result["confidence"] < rules["escalation_rules"]["confidence_below"]:
        print("\n⚠️ Human review required (low confidence)")
    else:
        print("\n✅ Safe for first-pass review")
