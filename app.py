# app.py — ELITE RESUME SCREENER 2025

import streamlit as st
import os, zipfile, fitz, re, numpy as np, pandas as pd, faiss, base64
from sentence_transformers import SentenceTransformer
from groq import Groq
from pathlib import Path
import time

st.set_page_config(page_title="Elite Resume Screener 2025", layout="wide", page_icon="rocket")

# === CUSTOM CSS FOR ELEGANCE ===
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 100%); color: white;}
    .stButton>button {background: #00d4ff; color: black; border-radius: 12px; padding: 12px 30px; font-weight: bold;}
    .stTextInput>div>div>input {background: #2a2a3a; color: white; border-radius: 12px;}
    .stFileUploader>div>div {background: #3a3a5a; border-radius: 16px; padding: 20px;}
    .title {font-size: 4rem; font-weight: 900; text-align: center; background: linear-gradient(90deg, #00d4ff, #ff00c8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; font-size: 1.4rem; opacity: 0.9;}
    .card {background: rgba(255,255,255,0.1); padding: 20px; border-radius: 16px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);}
    .chat-bubble {background: #00d4ff; color: black; padding: 12px 20px; border-radius: 20px; max-width: 80%; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# === TITLE ===
st.markdown("<h1 class='title'>Elite Resume Screener 2025</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered • Lightning Fast </p>", unsafe_allow_html=True)

# === SIDEBAR INPUTS ===
with st.sidebar:
    st.header("Configuration")
    job_description = st.text_area("Job Description", height=150, value="Senior Python Developer with FastAPI and AWS")
    must_have_input = st.text_input("Must-have keywords (comma-separated, optional)", "")
    must_have_keywords = [k.strip() for k in must_have_input.split(",") if k.strip()]
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload ZIP of Resumes", type="zip")

if not uploaded_file:
    st.info("Please upload a ZIP file to begin.")
    st.stop()

# === MAIN PROCESSING ===
with st.spinner("Extracting resumes..."):
    extract_folder = "extracted_resumes"
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(uploaded_file) as z:
        z.extractall(extract_folder)
    st.success("ZIP extracted!")

pdfs = list(Path(extract_folder).rglob("*.pdf"))

def extract_text(path):
    try:
        with fitz.open(path) as doc:
            text = "".join(p.get_text() for p in doc)
            return text[-22_000:], text
    except: return "", ""

def has_all_keywords(text, kws):
    if not kws: return True
    t = text.lower()
    return all(any(f in t for f in [k.lower(), k.replace(" ", ""), k.replace("-", "")]) for k in kws)

st.info(f"Found {len(pdfs)} PDFs. Filtering...")

candidates = []
for p in pdfs:
    recent, full = extract_text(p)
    if len(recent) < 100: continue
    if not has_all_keywords(full, must_have_keywords): continue
    candidates.append({"file": p.name, "text": recent, "path": str(p)})

st.write(f"**{len(candidates)} resumes passed filter**")

if len(candidates) == 0:
    st.error("No resumes matched your criteria.")
    st.stop()

top_n = st.slider("How many TOP resumes to score with AI?", 1, min(50, len(candidates)), 10)

if st.button("Start AI Screening", type="primary"):
    with st.spinner("Loading AI model..."):
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    with st.spinner("Running semantic search..."):
        embs = model.encode([c["text"] for c in candidates], normalize_embeddings=True).astype('float32')
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        q = model.encode([job_description], normalize_embeddings=True).astype('float32')
        faiss.normalize_L2(q)
        D, I = index.search(q, top_n)

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    accepted = []
    progress = st.progress(0)

    st.write("### Scoring with Groq AI...")

    for i, idx in enumerate(I[0]):
        c = candidates[idx]
        prompt = f"""JD: {job_description}\n\nResume:\n{c["text"][:20000]}\n\nReply only:\nSCORE: 0-100\nDECISION: ACCEPT or REJECT\nREASON: 1-2 short sentences why this candidate is a good match"""

        try:
            resp = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=80).choices[0].message.content.strip()
            score = int(re.search(r"SCORE:\s*(\d+)", resp).group(1))
            decision = "ACCEPT" if "ACCEPT" in resp.upper() else "REJECT"
            reason = re.search(r"REASON:\s*(.+)", resp, re.DOTALL)
            reason = reason.group(1).strip() if reason else "Strong match"

            if decision == "ACCEPT":
                accepted.append({"File": c["file"], "Path": c["path"], "Score": score, "Why Selected": reason, "Text": c["text"]})
                st.success(f"{len(accepted)}. {c['file']} → {score}/100")
        except: pass

        progress.progress((i + 1) / len(I[0]))

    # === FINAL TABLE ===
    st.markdown("### FINAL ACCEPTED CANDIDATES")
    if not accepted:
        st.warning("No candidate was accepted.")
    else:
        def make_clickable(row):
            with open(row["Path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f'<a href="data:application/pdf;base64,{b64}" download="{row["File"]}">Download {row["File"]}</a>'

        df = pd.DataFrame(accepted)
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        df["File"] = df.apply(make_clickable, axis=1)
        df = df[["Rank", "File", "Score", "Why Selected"]]
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # === TALK TO RESUME ===
        st.markdown("### Talk to Any Resume")
        selected_file = st.selectbox("Choose a resume to chat with", [a["File"] for a in accepted])
        selected = next(a for a in accepted if a["File"] == selected_file)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ask anything about this resume..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            chat_prompt = f"""Answer ONLY from this resume:\n{selected["Text"][:25000]}\n\nQuestion: {prompt}\nAnswer briefly:"""
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": chat_prompt}], temperature=0.3, max_tokens=200).choices[0].message.content
                    st.write(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

    st.balloons()
