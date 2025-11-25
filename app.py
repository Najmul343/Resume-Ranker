# ELITE RESUME SCREENER 2025 

!pip install -q pymupdf sentence-transformers pandas faiss-cpu groq

import os, zipfile, fitz, re, numpy as np, pandas as pd, faiss, base64
from sentence_transformers import SentenceTransformer
from groq import Groq
from google.colab import userdata
from IPython.display import HTML, display

# ==================== SETTINGS ====================
zip_path = '/content/INFORMATION-TECHNOLOGY.zip'
extract_folder = '/content/extracted_resumes/INFORMATION-TECHNOLOGY'
os.makedirs(extract_folder, exist_ok=True)

must_have_keywords = []          
job_description = "Senior Python Developer with FastAPI and AWS"

# ==================== HELPERS ====================
def has_all_keywords(text, kws):
    if not kws: return True
    t = text.lower()
    return all(any(f in t for f in [k.lower(), k.replace(" ", ""), k.replace("-", "")]) for k in kws)

def extract_text(path):
    try:
        with fitz.open(path) as doc:
            text = "".join(p.get_text() for p in doc)
            return text[-22_000:], text
    except: return "", ""

# ==================== EXTRACT & FILTER ====================
with zipfile.ZipFile(zip_path) as z: z.extractall(extract_folder)
pdfs = [os.path.join(r, f) for r, _, fs in os.walk(extract_folder) for f in fs if f.lower().endswith('.pdf')]

candidates = []
for p in pdfs:
    recent, full = extract_text(p)
    if len(recent) < 100: continue
    if not has_all_keywords(full, must_have_keywords): continue
    candidates.append({"file": os.path.basename(p), "text": recent, "path": p})

print(f"After filter: {len(candidates)} resumes")

if len(candidates) == 0:
    print("No resumes passed filtering.")
else:
    while True:
        try:
            top_n = int(input(f"\nHow many TOP resumes do you want to score? (1–{min(50, len(candidates))}) → "))
            if 1 <= top_n <= min(50, len(candidates)):
                break
        except: pass

    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    embs = model.encode([c["text"] for c in candidates], normalize_embeddings=True).astype('float32')
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    q = model.encode([job_description], normalize_embeddings=True).astype('float32')
    faiss.normalize_L2(q)
    D, I = index.search(q, top_n)

    client = Groq(api_key=userdata.get("GROQ_API_KEY"))
    accepted = []

    print(f"\nScoring TOP {top_n} with Groq (super fast)...")

    for rank, idx in enumerate(I[0], 1):
        c = candidates[idx]
        
        # === FIXED PROMPT: Real senior HR thinking ===
        prompt = f"""You are a senior technical recruiter with 15+ years experience.

Job Description:
{job_description}

Resume:
{c["text"][:20000]}

Think like a real HR: 
- Does this person actually have strong, recent experience in the core stack?
- Is their seniority and depth appropriate?
- Would you confidently shortlist them?

Be strict and honest.

Reply exactly:
SCORE: 0-100
DECISION: ACCEPT or REJECT
REASON: 1 short sentence"""

        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80
            ).choices[0].message.content.strip()

            score = int(re.search(r"SCORE:\s*(\d+)", resp).group(1))
            decision = "ACCEPT" if "ACCEPT" in resp.upper() else "REJECT"
            reason = re.search(r"REASON:\s*(.+)", resp, re.DOTALL)
            reason = reason.group(1).strip() if reason else "Strong match"

            if decision == "ACCEPT":
                accepted.append({
                    "File": c["file"], 
                    "Path": c["path"], 
                    "Score": score, 
                    "Why Selected": reason,
                    "Text": c["text"]
                })
                print(f"{len(accepted)}. {c['file']} → {score}/100")

        except Exception as e:
            print(f"Error: {e}")

    # ==================== FINAL TABLE — FIXED SORTING & DISPLAY ====================
    print("\n" + "═" * 120)
    print("FINAL ACCEPTED CANDIDATES — CLICK TO DOWNLOAD + WHY SELECTED")
    print("═" * 120)

    if not accepted:
        print("No candidate was accepted.")
    else:
        # Keep original filename for sorting
        df = pd.DataFrame(accepted)
        df["Original_File"] = df["File"]

        # Sort by score first
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)

        # Now make clickable
        def make_clickable(row):
            with open(row["Path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f'<a href="data:application/pdf;base64,{b64}" download="{row["Original_File"]}">{row["Original_File"]}</a>'

        df["File_Link"] = df.apply(make_clickable, axis=1)
        df = df[["Rank", "File_Link", "Score", "Why Selected"]]
        df.columns = ["Rank", "File", "Score", "Why Selected"]

        display(HTML(df.to_html(escape=False, index=False)))

        # ==================== TALK TO RESUME ====================
        print("\n" + "═" * 80)
        print("TALK TO ANY RESUME")
        print("Type filename to chat (or 'exit')")
        print("═" * 80)

        while True:
            user_input = input("\nEnter filename (or 'exit'): ").strip()
            if user_input.lower() == 'exit': break
            selected = next((a for a in accepted if a["File"] == user_input), None)
            if not selected:
                print("Not found.")
                continue

            print(f"\nChatting with: {selected['File']} | Score: {selected['Score']}")
            while True:
                q = input("You: ")
                if q.lower() in ["exit", "back"]: break
                chat_prompt = f"Answer ONLY from this resume:\n{selected['Text'][:25000]}\n\nQuestion: {q}\nAnswer briefly:"
                try:
                    ans = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": chat_prompt}],
                        temperature=0.3, max_tokens=200
                    ).choices[0].message.content.strip()
                    print(f"Resume: {ans}")
                except: print("Error")

    print(f"\nDone! {len(accepted)} accepted candidates.")
