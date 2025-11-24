#beutiful HTML

import streamlit as st
import PyPDF2, re, io, zipfile, pandas as pd, base64
from pdf2image import convert_from_bytes
import pytesseract
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import faiss
import time

# ========================= PREMIUM SAAS UI (INSTANT UPGRADE) =========================
st.set_page_config(page_title="Elite Resume Ranker 2025", layout="wide")
st.markdown("""
<style>
    .big-title {font-size: 4.5rem !important; font-weight: 900; text-align: center;
                background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom:0;}
    .subtitle {text-align:center; font-size:1.5rem; color:#64748b; font-weight:600; margin-top:10px;}
    .stButton>button {background: linear-gradient(135deg, #7c3aed, #ec4899); color:white; 
                      border:none; border-radius:16px; height:68px; font-size:1.4rem; 
                      font-weight:800; box-shadow:0 10px 30px rgba(139,92,246,0.4);}
    .stButton>button:hover {transform:translateY(-4px); box-shadow:0 20px 40px rgba(139,92,246,0.5);}
    .stTextArea>div>div>textarea, .stTextInput>div>div>input {border-radius:16px; border:2px solid #e2e8f0; box-shadow:0 4px 20px rgba(0,0,0,0.05);}
    .stSuccess {background:linear-gradient(90deg,#10b981,#34d399); color:white; border-radius:16px; padding:1.5rem; font-size:1.4rem; text-align:center;}
    .stDownloadButton>button {background:linear-gradient(135deg,#f59e0b,#fbbf24); color:black; border-radius:16px; height:68px; font-weight:800;}
    table {border-radius:16px; overflow:hidden; box-shadow:0 10px 40px rgba(0,0,0,0.1);}
    th {background:#1e1b4b !important; color:white !important;}
    td {background:#f8fafc;}
</style>
<div class="big-title">Elite Resume Ranker 2025</div>
<div class="subtitle">AI-Powered • Blazing Fast • Built for Recruiters</div>
""", unsafe_allow_html=True)

# ========================= CONFIG =========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@st.cache_resource
def load_embedder():
    from torch import cuda
    device = "cuda" if cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
embedder = load_embedder()

# ========================= UI =========================
c1, c2 = st.columns([3,1])
with c1:
    job_desc = st.text_area("Job Description", height=180, placeholder="e.g. Senior Python Engineer, FastAPI, Docker, AWS...")
with c2:
    boost = st.text_input("Boost Keywords (+20 pts)", placeholder="FastAPI, Docker, AWS, Redis")
uploaded_zip = st.file_uploader("Upload ZIP of PDF resumes", type="zip")

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

# ========================= MAIN LOGIC + FILTERS =========================
if st.button("Start AI Ranking", type="primary", use_container_width=True):
    if not job_desc.strip():
        st.error("Job Description required!")
        st.stop()

    start_time = time.time()
    with zipfile.ZipFile(uploaded_zip) as z:
        all_files = [(n, z.read(n)) for n in z.namelist() if n.lower().endswith(".pdf")]

    # NEW: Filter >4 pages or too short (<80 words)
    valid_files = []
    rejected = 0
    for name, data in all_files:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            pages = len(reader.pages)
            if pages > 4:
                rejected += 1
                continue
            text = extract_text(data)
            if len(text.split()) < 80:
                rejected += 1
                continue
            valid_files.append((name, data))
        except:
            rejected += 1
            continue

    total = len(all_files)
    st.info(f"Quality Filter: **{len(valid_files)}/{total}** resumes passed (>4 pages or too short removed)")

    if len(valid_files) == 0:
        st.error("No valid resumes after filtering!")
        st.stop()

    boost_words = set(boost.lower().split()) if boost else set()
    files = valid_files  # only clean ones go forward

    # PHASE 1: Keyword pre-filter
    with st.spinner("Phase 1: Keyword matching..."):
        def process(p):
            name, data = p
            text = extract_text(data)
            score = sum(w in text.lower() for w in re.findall(r'\w+', job_desc.lower())) + \
                    3*sum(w in text.lower() for w in boost_words)
            return name, data, text, score
        with ThreadPoolExecutor(32) as ex:
            items = list(ex.map(process, files))
        items.sort(key=lambda x: x[3], reverse=True)
        candidates = items[:220]

    # PHASE 2: Semantic ranking
    with st.spinner("Phase 2: Deep AI understanding..."):
        texts = [c[2] for c in candidates]
        embeddings = embedder.encode(texts, batch_size=128, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        query = embedder.encode([job_desc + " " + boost], normalize_embeddings=True)
        _, I = index.search(query.astype('float32'), min(130, len(candidates)))
        final_resumes = [candidates[i] for i in I[0]]

    # PHASE 3: LLM final scoring
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
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=120
            ).choices[0].message.content
            score = int(re.search(r"SCORE:\s*(\d+)", resp, re.I).group(1))
            reason = resp.split("REASON:")[-1].strip() if "REASON:" in resp else "Strong match"
        except:
            score, reason = 30, "Error"
        return {"File": name, "Score": score, "Why": reason, "PDF": data}

    with st.spinner("Phase 3: Final AI scoring..."):
        with ThreadPoolExecutor(28) as ex:
            results = list(ex.map(score_one, final_resumes[:130]))

    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df)+1)
    st.success(f"Done in {int(time.time()-start_time)}s — Top Match: {df.iloc[0]['File']} ({df.iloc[0]['Score']}/100)")

    def link(n,d): 
        return f'<a href="data:application/pdf;base64,{base64.b64encode(d).decode()}" download="{n}" style="color:#8b5cf6; font-weight:700;">{n}</a>'
    df["Candidate"] = df.apply(lambda r: link(r["File"], r["PDF"]), axis=1)
    st.markdown(df[["Rank", "Candidate", "Score", "Why"]].to_html(escape=False, index=False), unsafe_allow_html=True)

    # Download Top 20
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for _, r in df.head(20).iterrows():
            z.writestr(f"{r['Score']:03d}_{r['File']}", r["PDF"])
    buf.seek(0)
    st.download_button("Download Top 20 Ranked Resumes", buf, "TOP20_ELITE.zip", "application/zip", use_container_width=True)
    st.balloons()
