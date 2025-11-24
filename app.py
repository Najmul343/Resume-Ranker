
import streamlit as st
import PyPDF2, re, io, zipfile, pandas as pd, base64, numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import faiss
import time

# ========================= CONFIG =========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # This will read from secrets.toml
client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.1-8b-instant"

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@st.cache_resource
def load_embedder():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

embedder = load_embedder()

st.set_page_config(page_title="Elite Resume Ranker", layout="wide")
st.title("⚡ Elite Resume Ranker 2025")
st.markdown("**with the power of LLMs be Blazing Fast**")

# ========================= UI =========================
c1, c2 = st.columns([3,1])
with c1:
    job_desc = st.text_area("Job Description", height=180, placeholder="google,teacher,Senior Python Engineer, FastAPI, Docker, AWS...")
with c2:
    boost = st.text_input("This Boosts keywords (+20 pts)", placeholder="FastAPI, Docker, AWS, Redis")

uploaded_zip = st.file_uploader("Upload ZIP of PDF resumes", type="zip")

# ========================= ULTRA-FAST TEXT EXTRACTION =========================
def extract_text(pdf_bytes):
    try:
        text = "".join(p.extract_text() or "" for p in PyPDF2.PdfReader(io.BytesIO(pdf_bytes)).pages)
        if len(text) > 600:
            return re.sub(r'\s+', ' ', text)[:24000]
    except: pass
    try:
        img = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)[0]
        return pytesseract.image_to_string(img, config='--psm 6')[:24000]
    except:
        return text[:24000] if 'text' in locals() else ""

# ========================= MAIN LOGIC =========================
if st.button("Click to Start Ranking ", type="primary", use_container_width=True):
    if not job_desc.strip():
        st.error("Job Description required!")
        st.stop()

    start_time = time.time()
    with zipfile.ZipFile(uploaded_zip) as z:
        files = [(n, z.read(n)) for n in z.namelist() if n.lower().endswith(".pdf")]

    boost_words = set(boost.lower().split()) if boost else set()

    # PHASE 1: Fast text + keyword pre-filter
    with st.spinner("Phase 1: Extracting text + keyword filter..."):
        def process(p):
            name, data = p
            text = extract_text(data)
            score = sum(w in text.lower() for w in re.findall(r'\w+', job_desc.lower())) + 3*sum(w in text.lower() for w in boost_words)
            return name, data, text, score
        
        with ThreadPoolExecutor(32) as ex:
            items = list(ex.map(process, files))
        items.sort(key=lambda x: x[3], reverse=True)
        candidates = items[:220]  # Top 220 survive

    # PHASE 2: Semantic ranking (MiniLM + FAISS)
    with st.spinner("Phase 2: Semantic ranking ..."):
        texts = [c[2] for c in candidates]
        embeddings = embedder.encode(texts, batch_size=128, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        query = embedder.encode([job_desc + " " + boost], normalize_embeddings=True)
        _, I = index.search(query.astype('float32'), min(130, len(candidates)))
        final_resumes = [candidates[i] for i in I[0]]

    # PHASE 3: Final scoring with Llama-3.1-8B-instant
    def score_one(item):
        name, data, text, _ = item
        prompt = f"""Score 0–100 how well this resume matches the JD.

JD: {job_desc}
BOOST: {boost}

Resume: {text}

Reply exactly:
SCORE: XX
REASON: [1 short sentence]"""
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=120
            ).choices[0].message.content
            score = int(re.search(r"SCORE:\s*(\d+)", resp, re.I).group(1))
            reason = resp.split("REASON:")[-1].strip() if "REASON:" in resp else "Strong match"
        except:
            score, reason = 30, "Error"
        return {"File": name, "Score": score, "Why": reason, "PDF": data}

    with st.spinner("Phase 3: Final scoring with LLM..."):
        with ThreadPoolExecutor(28) as ex:
            results = list(ex.map(score_one, final_resumes[:130]))

    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df)+1)

    st.success(f"🎯 Done in {int(time.time()-start_time)} seconds — #1: {df.iloc[0]['File']} ({df.iloc[0]['Score']})")

    def link(n,d): 
        return f'<a href="data:application/pdf;base64,{base64.b64encode(d).decode()}" download="{n}" style="color:#0066cc;font-weight:600;">{n}</a>'
    df["Candidate"] = df.apply(lambda r: link(r["File"], r["PDF"]), axis=1)

    st.markdown(df[["Rank", "Candidate", "Score", "Why"]].to_html(escape=False, index=False), unsafe_allow_html=True)

    # Download top 20
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for _, r in df.head(20).iterrows():
            z.writestr(f"{r['Score']:03d}_{r['File']}", r["PDF"])
    buf.seek(0)
    st.download_button("📥 Download Top 20 Ranked Resumes", buf, "TOP20_FINAL.zip", "application/zip", use_container_width=True)

    st.balloons()
