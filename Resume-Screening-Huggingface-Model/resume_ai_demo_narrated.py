
import os
import fitz  # PyMuPDF
from transformers import pipeline
import torch
import numpy as np

# Step 1: Setup
resume_dir = "sample_resumes"
job_description_pdf = "job_description_backend_engineer.pdf"

print("🔍 Step 1: Reading job description from PDF...")
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

job_description = extract_text_from_pdf(job_description_pdf)
print("✅ Job description successfully extracted.\n")

print("📂 Step 2: Loading resumes from folder...")
resume_files = sorted([os.path.join(resume_dir, f) for f in os.listdir(resume_dir) if f.endswith(".pdf")])
resumes_text = [extract_text_from_pdf(f) for f in resume_files]
resume_names = [os.path.splitext(os.path.basename(f))[0] for f in resume_files]
print(f"✅ {len(resume_files)} resumes loaded.\n")

print("🧠 Step 3: Loading Hugging Face model and generating embeddings...")
feature_extractor = pipeline("feature-extraction", model="sentence-transformers/paraphrase-MiniLM-L6-v2")

def get_embedding(text):
    with torch.no_grad():
        output = feature_extractor(text)
        embedding = np.mean(output[0], axis=0)
        return embedding / np.linalg.norm(embedding)

job_embedding = get_embedding(job_description)
resume_embeddings = [get_embedding(text) for text in resumes_text]
print("✅ Embeddings generated for job description and resumes.\n")

print("📊 Step 4: Calculating similarity scores...")
def cosine_similarity(a, b):
    return np.dot(a, b)

results = [(resume_names[i], cosine_similarity(job_embedding, emb)) for i, emb in enumerate(resume_embeddings)]
results.sort(key=lambda x: x[1], reverse=True)

print("\n🎯 Ranked Resume Fit Scores:")
for name, score in results:
    print(f"{name}: {score:.4f}")

# Step 5: Business-Style Recommendation
print("\n📌 Business Insight:")
top_candidate, top_score = results[0]
print(f"{top_candidate} appears to be the strongest fit for the Backend Engineer role based on AI analysis.")

if len(results) > 1:
    second_candidate, second_score = results[1]
    gap = top_score - second_score
    if gap >= 0.05:
        print(f"You may consider reviewing {second_candidate}, but the gap suggests a significantly better match with {top_candidate}.")
    else:
        print(f"{second_candidate} also shows a relatively close fit. Consider both for further evaluation.")
