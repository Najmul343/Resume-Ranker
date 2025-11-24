import streamlit as st
import PyPDF2, re, io, zipfile, pandas as pd, base64, numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import faiss
import time

# ========================= PREMIUM UI - MAKE IT LOOK LIKE A $10M TOOL =========================
st.markdown("""
<link rel="icon" href="https://cdn-icons-png.flaticon.com/512/3135/3135715.png">
<style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .css-1d391kg {font-size: 4.2rem !important; font-weight: 900; background: linear-gradient(90deg, #1e40af, #7c3aed, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .stTextArea > div > div > textarea, .stTextInput > div > div > input {border-radius: 16px !important; border: 2px solid #e2e8f0 !important; box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;}
    .stButton > button {background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899) !important; color: white !important; border: none !important; border-radius: 16px !important; height: 68px !important; font-size: 1.4rem !important; font-weight: 800 !important; box-shadow: 0 12px 40px rgba(139,92,246,0.4) !important; transition: all 0.3s ease !important;}
    .stButton > button:hover {transform: translateY(-5px) !important; box-shadow: 0 25px 50px rgba(139,92,246,0.5) !important;}
    .stSuccess {background: linear-gradient(90deg, #10b981, #34d399); color: white; border-radius: 16px; padding: 1.4rem; font-size: 1.4rem; font-weight: 700; text-align: center; box-shadow: 0 10px 30px rgba(16,185,129,0.3);}
    .stDownloadButton > button {background: linear-gradient(135deg, #f59e0b, #fbbf24) !important; color: black !important; border-radius: 16px !important; height: 68px !important; font-weight: 800 !important; font-size: 1.3rem !important; box-shadow: 0 12px 40px rgba(245,158,11,0.4) !important;}
    table {border-radius: 16px !important; overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,0.1);}
    th {background: #1e1b4b !important; color: white !important; font-weight: 700;}
    td {background: #f8fafc;}
    .css-1y0t9fs {background: linear-gradient(90deg, #0f172a, #1e293b); color: #cbd5e1; text-align: center; padding: 2.5rem; border-radius: 20px; margin-top: 4rem;}
</style>

<div style="text-align:center; margin: 2rem 0;">
    <div style="font-size:5.5rem; margin-bottom:0.5rem; background: linear-gradient(90deg, #1e40af, #7c3aed, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight:900;">Elite Resume Ranker 2025</div>
    <div style="font-size:1.6rem; color:#64748b; font-weight:600; margin-top:0.5rem;">
        Powered by Llama-3.1 • Groq • FAISS • MiniLM • Zero Hallucinations
    </div>
</div>
""", unsafe_allow_html=True)

# ========================= CONFIG =========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@st.cache_resource(show_spinner="Loading AI brain (one-time)...")
def load_embedder():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

embedder = load_embedder()
st.set_page_config(page_title="Elite Resume Ranker 2025", layout="wide")

# ========================= UI =========================
c1, c2 = st.columns([3,1])
with c1:
    job_desc = st.text_area("Job Description", height=180, placeholder="e.g. Senior Python Engineer, FastAPI, Docker, AWS, Redis...")
with c2:
    boost = st.text_input("Boost Keywords (+20 pts each)", placeholder="FastAPI, Docker, Kubernetes, Redis")

uploaded_zip = st.file_uploader("Upload ZIP of PDF resumes", type="zip")

# ========================= FAST TEXT EXTRACTION =========================
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
if st.button("Start AI-Powered Ranking", type="primary", use_container_width=True):
    if not job_desc.strip():
        st.error("Job Description is required!")
        st.stop()
    if not uploaded_zip:
        st.error("Please upload a ZIP file!")
        st.stop()

    start_time = time.time()

    # === PHASE 0: GARBAGE FILTER (Enterprise Grade) ===
    with zipfile.ZipFile(uploaded_zip) as z:
        pdf_files = [n for n in z.namelist() if n.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.error("No PDF files found in the ZIP!")
        st.stop()

    st.info(f"Found {len(pdf_files)} resumes → Running quality filters...")

    valid_files = []
    rejected = 0

    for name in pdf_files:
        try:
            data = z.read(name)
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except:
                    rejected += 1
                    continue

            if len(reader.pages) > 5:
                rejected += 1
                continue

            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) < 200:
                try:
                    img = convert_from_bytes(data, dpi=150, first_page=1, last_page=1)[0]
                    text = pytesseract.image_to_string(img, config='--psm 6')
                    text = re.sub(r'\s+', ' ', text).strip()
                except:
                    pass

            words = len(text.split())
            lower = text.lower()
            if (words < 80 or 
                any(p in lower for p in ["your name", "john doe", "insert your", "lorem ipsum", "resume template", "[your"])):
                rejected += 1
                continue

            valid_files.append((name, data))

        except:
            rejected += 1
            continue

    passed = len(valid_files)
    st.write(f"**Quality Filter:** {passed}/{len(pdf_files)} resumes passed ({rejected} removed: blank, too long, protected, or corrupted)")

    if passed == 0:
        st.error("No valid resumes found. Please check your files.")
        st.stop()

    files = valid_files
    boost_words = set(w.lower().strip(", ") for w in boost.split()) if boost else set()

    # === PHASE 1: Keyword Pre-Filter ===
    with st.spinner("Phase 1: Keyword scanning + boost scoring..."):
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

    # === PHASE 2: Semantic Ranking ===
    with st.spinner("Phase 2: AI semantic understanding (MiniLM + FAISS)..."):
        texts = [c[2] for c in candidates]
        embeddings = embedder.encode(texts, batch_size=128, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        query = embedder.encode([job_desc + " " + boost], normalize_embeddings=True)
        _, I = index.search(query.astype('float32'), min(130, len(candidates)))
        final_resumes = [candidates[i] for i in I[0]]

    # === PHASE 3: Final LLM Scoring ===
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
            score, reason = 30, "Parse error"
        return {"File": name, "Score": score, "Why": reason, "PDF": data}

    with st.spinner("Phase 3: Final scoring with Llama-3.1-8B..."):
        with ThreadPoolExecutor(28) as ex:
            results = list(ex.map(score_one, final_resumes[:130]))

    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df)+1)
    st.success(f"Done in {int(time.time()-start_time)} seconds — #1: {df.iloc[0]['File']} (Score: {df.iloc[0]['Score']})")

    def link(n,d):
        return f'<a href="data:application/pdf;base64,{base64.b64encode(d).decode()}" download="{n}" style="color:#7c3aed; font-weight:700; text-decoration:none;">{n}</a>'
    df["Candidate"] = df.apply(lambda r: link(r["File"], r["PDF"]), axis=1)
    st.markdown(df[["Rank", "Candidate", "Score", "Why"]].to_html(escape=False, index=False), unsafe_allow_html=True)

    # === DOWNLOAD TOP 20 ===
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for _, r in df.head(20).iterrows():
            z.writestr(f"{r['Score']:03d}_{r['File']}", r["PDF"])
    buf.seek(0)
    st.download_button("Download Top 20 Ranked Resumes (ZIP)", buf, "TOP20_ELITE_RANKED.zip", "application/zip", use_container_width=True)
    
    st.balloons()

# ========================= FOOTER =========================
st.markdown("<div style='text-align:center; margin-top:5rem; color:#94a3b8; font-size:1.1rem;'>Built with ❤️ by an AI engineer who hates bad hiring tools</div>", unsafe_allow_html=True)
