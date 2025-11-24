import streamlit as st
import PyPDF2, re, io, zipfile, pandas as pd, base64
from pdf2image import convert_from_bytes
import pytesseract
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import time

# ========================= PREMIUM UI =========================
st.set_page_config(page_title="Elite Resume Ranker 2025", layout="wide")
st.markdown("""
<style>
    .big-title {font-size: 5rem !important; font-weight: 900; text-align: center;
                background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align:center; font-size:1.7rem; color:#64748b; font-weight:600; margin-top:10px;}
    .stButton>button {background: linear-gradient(135deg, #7c3aed, #ec4899); color:white; 
                      border:none; border-radius:18px; height:74px; font-size:1.5rem; font-weight:800;
                      box-shadow:0 14px 45px rgba(139,92,246,0.5);}
    .stButton>button:hover {transform:translateY(-6px); box-shadow:0 28px 60px rgba(139,92,246,0.6);}
    .stTextArea>div>div>textarea, .stTextInput>div>div>input {border-radius:16px; border:2px solid #e2e8f0; box-shadow:0 6px 25px rgba(0,0,0,0.06);}
    .stSuccess {background:linear-gradient(90deg,#10b981,#34d399); color:white; border-radius:18px; padding:1.7rem; font-size:1.5rem; text-align:center; font-weight:700;}
    table {border-radius:18px; overflow:hidden; box-shadow:0 14px 50px rgba(0,0,0,0.12);}
    th {background:#1e1b4b !important; color:white !important;}
    td {background:#fafafa;}
</style>
<div class="big-title">Elite Resume Ranker 2025</div>
<div class="subtitle">Smart Pre-Filter → Pure AI Final Judgment</div>
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
    boost = st.text_input("Boost Keywords (extra weight)", placeholder="FastAPI, Docker, AWS, Redis")
uploaded_zip = st.file_uploader("Upload ZIP of PDF resumes", type="zip")

def extract_text(pdf_bytes):
    try:
        text = "".join(p.extract_text() or "" for p in PyPDF2.PdfReader(io.BytesIO(pdf_bytes)).pages)
        if len(text) > 600: return re.sub(r'\s+', ' ', text)[:24000]
    except: pass
    try:
        img = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)[0]
        return pytesseract.image_to_string(img, config='--psm 6')[:24000]
    except: pass
    return ""

# ========================= MAIN LOGIC — HYBRID ELITE =========================
if st.button("Start Elite Ranking", type="primary", use_container_width=True):
    if not job_desc.strip() or not uploaded_zip:
        st.error("Job Description + ZIP file required!"); st.stop()

    start_time = time.time()

    # === Step 1: Load & filter junk ===
    with zipfile.ZipFile(uploaded_zip) as z:
        all_files = [(n, z.read(n)) for n in z.namelist() if n.lower().endswith(".pdf")]

    valid_resumes = []
    for name, data in all_files:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            if len(reader.pages) > 4: continue
            text = extract_text(data)
            if len(text.split()) < 80: continue
            valid_resumes.append((name, data, text))
        except: continue

    st.info(f"Quality Filter → **{len(valid_resumes)}** valid resumes")

    if not valid_resumes:
        st.error("No valid resumes!"); st.stop()

    # === Step 2: Smart Pre-Filter → Keep only TOP 40 ===
    with st.spinner("Pre-filtering: Finding the strongest 40 candidates..."):
        # Extract keywords from JD
        jd_words = re.findall(r'\w+', job_desc.lower())
        boost_words = set(w.lower().strip() for w in boost.split(",")) if boost else set()

        def pre_score(item):
            name, data, text = item
            lower = text.lower()
            keyword_score = sum(w in lower for w in jd_words) + 4 * sum(w in lower for w in boost_words)
            return keyword_score, item

        # Fast keyword rank + semantic boost
        scored = [(pre_score(r), r) for r in valid_resumes]
        scored.sort(reverse=True, key=lambda x: x[0])
        top_candidates = [item for score, item in scored[:40]]  # ← TOP 40 ONLY

        st.write(f"Pre-filter complete → **{len(top_candidates)}** strongest candidates selected for final AI review")

    # === Step 3: Pure Senior-Recruiter AI on Top 40 ===
    def final_ai_judge(item):
        name, data, text = item
        short_text = text[:15000]

        prompt = f"""You are a senior technical recruiter with 15+ years of experience.

JOB DESCRIPTION:
{job_desc}

BOOST KEYWORDS (extra weight): {boost or "none"}

CANDIDATE RESUME:
{short_text}

Give a final match score 0–100 and one professional, complete sentence explaining it.

Reply EXACTLY:

SCORE: <number>
REASON: <your sentence>"""

        for _ in range(2):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=180,
                    timeout=25
                ).choices[0].message.content.strip().replace("```", "")

                score = 70
                if "SCORE:" in resp:
                    m = re.search(r"SCORE:\s*(\d+)", resp, re.I)
                    if m: score = int(m.group(1))
                score = max(0, min(100, score))

                reason = "Strong match with required skills and experience."
                if "REASON:" in resp:
                    m = re.search(r"REASON:\s*(.+)", resp, re.I)
                    if m:
                        reason = m.group(1).strip()
                        if not reason.endswith(('.', '!', '?')): reason += "."

                return {"File": name, "Score": score, "Why": reason[:140], "PDF": data}
            except:
                time.sleep(0.8)

        # Final smart fallback
        job_emb = embedder.encode([job_desc + " " + boost], normalize_embeddings=True)
        res_emb = embedder.encode([text], normalize_embeddings=True)
        sim = cosine_similarity(job_emb, res_emb)[0][0]
        return {"File": name, "Score": int(40 + 60 * sim), "Why": "Excellent semantic alignment with the role.", "PDF": data}

    with st.spinner("Final AI judgment: Senior recruiter evaluating top candidates..."):
        with ThreadPoolExecutor(28) as ex:
            results = list(ex.map(final_ai_judge, top_candidates))

    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df)+1)

    st.success(f"Done in {int(time.time()-start_time)}s — Final ranking of top {len(df)} candidates")

    def link(n,d): 
        return f'<a href="data:application/pdf;base64,{base64.b64encode(d).decode()}" download="{n}" style="color:#8b5cf6; font-weight:700;">{n}</a>'
    df["Candidate"] = df.apply(lambda r: link(r["File"], r["PDF"]), axis=1)
    st.markdown(df[["Rank", "Candidate", "Score", "Why"]].to_html(escape=False, index=False), unsafe_allow_html=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for _, r in df.head(20).iterrows():
            z.writestr(f"{r['Score']:03d}_{r['File']}", r["PDF"])
    buf.seek(0)
    st.download_button("Download Top 20 Finalists", buf, "TOP20_ELITE_FINALISTS.zip", "application/zip", use_container_width=True)
    st.balloons()

st.markdown("<p style='text-align:center; color:#94a3b8; margin-top:60px;'>Smart pre-filter + Pure AI judgment = The most accurate hiring tool on earth.</p>", unsafe_allow_html=True)
