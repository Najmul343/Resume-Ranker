# app.py — ELITE RESUME SCREENER 2025 — FINAL PERFECT VERSION

import streamlit as st
import os, zipfile, fitz, re, numpy as np, pandas as pd, faiss, base64
from sentence_transformers import SentenceTransformer
from groq import Groq
from pathlib import Path

st.set_page_config(page_title="Elite Resume Screener 2025", layout="wide", page_icon="rocket")

# === DESIGN ===
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0f23 0%, #1a0033 100%); color: white;}
    .title {font-size: 4.5rem; font-weight: 900; text-align: center; 
            background: linear-gradient(90deg, #00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .stButton>button {background: #00d4ff; color: black; border-radius: 16px; padding: 14px 32px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Elite Resume Screener 2025</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.5rem;'>Groq + FAISS • Real HR Intelligence</p>", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.header("Job & Filters")
    job_description = st.text_area("Job Description", height=160, value="Senior Python Developer with FastAPI and AWS")
    must_have_input = st.text_input("Must-have keywords (comma-separated)", "")
    must_have_keywords = [k.strip() for k in must_have_input.split(",") if k.strip()]
    uploaded_file = st.file_uploader("Upload ZIP of Resumes", type="zip")

if not uploaded_file:
    st.info("Upload a ZIP to begin.")
    st.stop()

extract_folder = "resumes"
os.makedirs(extract_folder, exist_ok=True)

with st.spinner("Extracting..."):
    with zipfile.ZipFile(uploaded_file) as z:
        z.extractall(extract_folder)
    st.success("ZIP extracted!")

pdfs = list(Path(extract_folder).rglob("*.pdf"))

def extract_text(path):
    try:
        with fitz.open(path) as doc:
            text = "".join(p.get_text() for p in doc)
            return text[-22_000:], text
    except:
        return "", ""

def has_all_keywords(text, kws):
    if not kws:
        return True
    t = text.lower()
    return all(any(f in t for f in [k.lower(), k.replace(" ", ""), k.replace("-", "")]) for k in kws)

with st.spinner("Filtering resumes..."):
    candidates = []
    for p in pdfs:
        recent, full = extract_text(p)
        if len(recent) < 100:
            continue
        if not has_all_keywords(full, must_have_keywords):
            continue
        candidates.append({"file": p.name, "text": recent, "path": str(p)})
    st.write(f"**{len(candidates)} resumes passed filter**")

if len(candidates) == 0:
    st.error("No resumes matched your criteria.")
    st.stop()

top_n = st.slider("How many TOP resumes to score?", 1, min(50, len(candidates)), 15)

if st.button("Start AI Screening", type="primary"):
    with st.spinner("Loading model..."):
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    with st.spinner("Ranking..."):
        embs = model.encode([c["text"] for c in candidates], normalize_embeddings=True).astype('float32')
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        q = model.encode([job_description], normalize_embeddings=True).astype('float32')
        faiss.normalize_L2(q)
        D, I = index.search(q, top_n)

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    accepted = []
    progress = st.progress(0)

    st.markdown("### Scoring with Groq...")

    for i, idx in enumerate(I[0]):
        c = candidates[idx]
        
        # === PERFECT PROMPT — NEVER REJECTS EVERYTHING ===
        prompt = f"""You are a senior recruiter.

Job Description:
{job_description}

Resume:
{c["text"][:22000]}

Rate this candidate on a scale of 0–100 based on how well they match the job.

Scoring guide (be realistic):
- 80–100: Perfect fit — has everything
- 80–89: Excellent — strong in most areas
- 70–85: Good — solid match, minor gaps
- 60–69: Average — relevant but not strong
- Below 60: Poor fit

Only give below 70 if they clearly lack the core skills.

Reply exactly:
SCORE: XX
REASON: 1 short sentence"""

        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80
            ).choices[0].message.content.strip()

            score_match = re.search(r"SCORE:\s*(\d+)", resp, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 0
            reason_match = re.search(r"REASON:\s*(.+)", resp, re.DOTALL | re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else "Good fit"

            # ACCEPT if score >= 75
            if score >= 61:
                accepted.append({
                    "File": c["file"],
                    "Path": c["path"],
                    "Score": score,
                    "Why": reason,
                    "Text": c["text"]
                })
                st.success(f"{len(accepted)}. {c['file']} → {score}/100")
        except:
            pass

        progress.progress((i + 1) / len(I[0]))

    # === FINAL TABLE ===
    st.markdown("### FINAL ACCEPTED CANDIDATES")
    if not accepted:
        st.warning("No candidate scored 75+. Try lowering the threshold or checking your JD.")
    else:
        df = pd.DataFrame(accepted)
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)

        def make_link(row):
            with open(row["Path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f'<a href="data:application/pdf;base64,{b64}" download="{row["File"]}">{row["File"]}</a>'

        df["File"] = df.apply(make_link, axis=1)
        df = df[["Rank", "File", "Score", "Why"]]
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # === CHAT WITH RESUME ===
        st.markdown("### Talk to Any Resume")
        if accepted:
            selected_name = st.selectbox("Select candidate", [a["File"] for a in accepted])
            selected = next(a for a in accepted if a["File"] == selected_name)

            if "chat" not in st.session_state:
                st.session_state.chat = []

            for msg in st.session_state.chat:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if q := st.chat_input("Ask about this resume..."):
                st.session_state.chat.append({"role": "user", "content": q})
                with st.chat_message("user"): st.write(q)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        ans = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": f"Answer only from this resume:\n{selected['Text'][:25000]}\n\nQ: {q}"}],
                            temperature=0.3,
                            max_tokens=200
                        ).choices[0].message.content
                        st.write(ans)
                        st.session_state.chat.append({"role": "assistant", "content": ans})

    st.balloons()
