import os
import fitz  # PyMuPDF
from transformers import pipeline
import torch
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
open_positions_dir = "Open Positions"   # folder containing JD PDFs
resume_dir = "resumes"           # folder containing resume PDFs

# Relative filtering: only keep candidates whose score is at least
# Z_SCORE_CUTOFF standard deviations above the mean for that JD.
# Raise this to be stricter (fewer candidates); lower it to be looser.
Z_SCORE_CUTOFF = 0.5

# ── Helpers ─────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def get_embedding(text, extractor):
    with torch.no_grad():
        output = extractor(text)
        embedding = np.mean(output[0], axis=0)
        return embedding / np.linalg.norm(embedding)


def cosine_similarity(a, b):
    return float(np.dot(a, b))


# ── Step 1: Load resumes (done once) ────────────────────────────────────────
print("📂 Step 1: Loading resumes from folder...")
resume_files = sorted(
    [os.path.join(resume_dir, f) for f in os.listdir(resume_dir) if f.endswith(".pdf")]
)
if not resume_files:
    raise FileNotFoundError(f"No PDF resumes found in '{resume_dir}'")

resumes_text = [extract_text_from_pdf(f) for f in resume_files]
resume_names = [os.path.splitext(os.path.basename(f))[0] for f in resume_files]
print(f"✅ {len(resume_files)} resumes loaded.\n")

# ── Step 2: Load model (done once) ──────────────────────────────────────────
print("🧠 Step 2: Loading Hugging Face embedding model...")
feature_extractor = pipeline(
    "feature-extraction",
    model="sentence-transformers/paraphrase-MiniLM-L6-v2",
)
print("✅ Model ready.\n")

# ── Step 3: Pre-compute resume embeddings (done once) ───────────────────────
print("⚙️  Step 3: Generating resume embeddings...")
resume_embeddings = [get_embedding(text, feature_extractor) for text in resumes_text]
print("✅ Resume embeddings ready.\n")

# ── Step 4: Load all job descriptions ───────────────────────────────────────
print(f"🔍 Step 4: Reading job descriptions from '{open_positions_dir}'...")
jd_files = sorted(
    [
        os.path.join(open_positions_dir, f)
        for f in os.listdir(open_positions_dir)
        if f.endswith(".pdf")
    ]
)
if not jd_files:
    raise FileNotFoundError(f"No PDF job descriptions found in '{open_positions_dir}'")

print(f"✅ {len(jd_files)} open position(s) found.\n")

# ── Step 5: Match resumes to each JD ────────────────────────────────────────
for jd_path in jd_files:
    jd_name = os.path.splitext(os.path.basename(jd_path))[0]
    print("=" * 60)
    print(f"📋 Position: {jd_name}")
    print("=" * 60)

    job_description = extract_text_from_pdf(jd_path)
    job_embedding = get_embedding(job_description, feature_extractor)

    all_scores = [
        (resume_names[i], cosine_similarity(job_embedding, emb))
        for i, emb in enumerate(resume_embeddings)
    ]
    all_scores.sort(key=lambda x: x[1], reverse=True)

    # Relative filtering: compute a dynamic cutoff from this JD's score distribution.
    # Candidates below (mean + Z_SCORE_CUTOFF * std) are filtered out as poor fits.
    raw_scores = np.array([s for _, s in all_scores])
    mean_score = np.mean(raw_scores)
    std_score  = np.std(raw_scores)
    dynamic_threshold = mean_score + Z_SCORE_CUTOFF * std_score

    print(f"  [Score stats — mean: {mean_score:.4f}, std: {std_score:.4f}, cutoff: {dynamic_threshold:.4f}]")

    results  = [(name, score) for name, score in all_scores if score >= dynamic_threshold]
    excluded = [(name, score) for name, score in all_scores if score < dynamic_threshold]


    if not results:
        print("\n❌ No relevant candidates found above the threshold for this position.")
        print()
        continue

    print(f"\n🎯 Ranked Resume Fit Scores ({len(results)} relevant candidate(s)):")
    for name, score in results:
        print(f"  {name}: {score:.4f}")

    # Business-style recommendation
    print("\n📌 Business Insight:")
    top_candidate, top_score = results[0]
    print(
        f"  {top_candidate} appears to be the strongest fit for the "
        f"'{jd_name}' role based on AI analysis."
    )

    if len(results) > 1:
        second_candidate, second_score = results[1]
        gap = top_score - second_score
        if gap >= 0.05:
            print(
                f"  You may consider reviewing {second_candidate}, but the gap "
                f"suggests a significantly better match with {top_candidate}."
            )
        else:
            print(
                f"  {second_candidate} also shows a relatively close fit. "
                f"Consider both for further evaluation."
            )

    print()

print("✅ All positions processed.")
