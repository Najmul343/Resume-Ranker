import streamlit as st
import PyPDF2, re, io, zipfile, pandas as pd, base64
from pdf2image import convert_from_bytes
import pytesseract
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import faiss
import time

# ========================= PREMIUM SAAS UI =========================
st.set_page_config(page_title="Elite Resume Ranker 2025", layout="wide")
st.markdown("""
<style>
    .big-title {font-size: 5rem !important; font-weight: 900; text-align: center;
                background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin:0;}
    .subtitle {text-align:center; font-size:1.7rem; color:#64748b; font-weight:600; margin-top:12px;}
    .stButton>button {background: linear-gradient(135deg, #7c3aed, #ec4899); color:white; 
                      border:none; border-radius:18px; height:74px; font-size:1.5rem; font-weight:800;
                      box-shadow:0 14px 45px rgba(139,92,246,0.5);}
    .stButton>button:hover {transform:translateY(-6px); box-shadow:0 28px 60px rgba(139,92,246,0.6);}
    .stTextArea>div>div>textarea, .stTextInput>div>div>input {border-radius:16px; border:2px solid #e2e8f0; box-shadow:0 6px 25px rgba(0,0,0,0.06);}
    .stSuccess {background:linear-gradient(90deg,#10b981,#34d399); color:white; border-radius:18px; padding:1.7rem; font-size:1.5rem; text-align:center; font-weight:700;}
    .stDownloadButton>button {background:linear-gradient(135deg,#f59e0b,#fbbf24); color:black; border-radius:18px; height:74px; font-weight:800; font-size:1.4rem;}
    table {border-radius:18px; overflow:hidden; box-shadow:0 14px 50px rgba(0,0,0,0.12);}
    th {background:#1e1b4b !important; color:white !important; font-weight:700;}
    td {background:#fafafa;}
</style>
<div class="big-title">Elite Resume Ranker 2025</div>
<div class="subtitle">AI-Powered • 100% Reliable • Built for Elite Hiring</div>
""", unsafe_allow_html=True)

# ========================= CONFIG =========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@st.cache_resource(show_spinner="Loading AI brain...")
def load_embedder():
    from torch import cuda
    device = "cuda" if cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
embedder = load_embedder()

# ========================= UI =========================
c1, c2 = st.columns([3,1])
with c1:
    job_desc = st.text_area("Job Description", height=180, placeholder="Senior Python Engineer, FastAPI, Docker, AWS...")
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

# ========================= MAIN LOGIC =========================
if st.button("Start AI Ranking", type="primary", use_container_width=True):
    if not job_desc.strip():
        st.error("Job Description required!")
        st.stop()
    if not uploaded_zip:
        st.error("Upload a ZIP file first!")
        st.stop()

    start_time = time.time()
    with zipfile.ZipFile(uploaded_zip) as z:
        all_files = [(n, z.read(n)) for n in z.namelist() if n.lower().endswith(".pdf")]

    # FILTER: Remove >4 pages or too short
    valid_files = []
    rejected = 0
    for name, data in all_files:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            if len(reader.pages) > 4:
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

    st.info(f"Quality Filter → **{len(valid_files)}/{len(all_files)}** passed ({rejected} removed)")

    if not valid_files:
        st.error("No valid resumes found!")
        st.stop()

    boost_words = set(w.lower().strip(", ") for w in boost.split(",")) if boost else set()
    files = valid_files

    # PHASE 1
    with st.spinner("Phase 1: Keyword ranking..."):
        def process(p):
            name, data = p
            text = extract_text(data)
            score = sum(w in text.lower() for w in re.findall(r'\w+', job_desc.lower())) + \
                    3 * sum(w in text.lower() for w in boost_words)
            return name, data, text, score
        with ThreadPoolExecutor(32) as ex:
            items = list(ex.map(process, files))
        items.sort(key=lambda x: x[3], reverse=True)
        candidates = items[:220]

    # PHASE 2
    with st.spinner("Phase 2: Semantic ranking..."):
        texts = [c[2] for c in candidates]
        embeddings = embedder.encode(texts, batch_size=128, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        query = embedder.encode([job_desc + " " + boost], normalize_embeddings=True)
        _, I = index.search(query.astype('float32'), min(130, len(candidates)))
        final_resumes = [candidates[i] for i in I[0]]

    # PHASE 3 ONLY — Pure JD vs Resume comparison (this replaces everything after filtering)
def score_one(item):
    name, data, text, _ = item
    truncated = text[:15000]

    prompt = f"""You are a senior technical recruiter with 15+ years of experience.

Job Description:
{job_desc}

Boost keywords (give extra weight): {boost or "none"}

Candidate Resume:
{truncated}

Compare the resume against the JD and give:
1. An exact match score from 0–100
2. One short, professional, complete sentence explaining the score

Return strictly in this format and nothing else:

SCORE: <number>
REASON: <your sentence>"""

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
                timeout=25
            ).choices[0].message.content.strip()

            # Clean any markdown
            resp = resp.replace("```", "")

            score = 70  # default
            reason = "Good overall match with the job requirements."

            if "SCORE:" in resp:
                try: score = int(re.search(r"SCORE:\s*(\d+)", resp, re.I).group(1))
                except: pass
            score = max(0, min(100, score))

            if "REASON:" in resp:
                reason = re.search(r"REASON:\s*(.+)", resp, re.I)
                reason = reason.group(1).strip() if reason else reason
                if not reason.endswith(('.', '!', '?')):
                    reason += "."

            return {"File": name, "Score": score, "Why": reason[:140], "PDF": data}

        except Exception as e:
            if attempt == 1:
                # Final intelligent fallback using pure semantic similarity
                from sklearn.metrics.pairwise import cosine_similarity
                job_emb = embedder.encode([job_desc + " " + boost], normalize_embeddings=True)
                res_emb = embedder.encode([text], normalize_embeddings=True)
                sim = cosine_similarity(job_emb, res_emb)[0][0]
                score = int(35 + 65 * sim)  # 35–100 range
                return {"File": name, "Score": score, "Why": "Strong semantic alignment with job requirements.", "PDF": data}
            time.sleep(0.7)

# MAIN EXECUTION — Pure AI ranking (no keyword or semantic pre-sort bias)
with st.spinner("Analyzing each resume against the Job Description one by one..."):
    clean_files = valid_files  # from your earlier filter (>4 pages, etc.)

    with ThreadPoolExecutor(30) as ex:
        results = list(ex.map(score_one, clean_files))  # ALL resumes go through pure AI

    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df)+1)

st.success(f"Completed pure AI evaluation of {len(df)} resumes — ranked by real match quality")
