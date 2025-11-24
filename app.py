"""
Production-ready Streamlit app: Elite Resume Ranker
Improvements vs original:
- Robust PDF/text extraction with clear fallbacks and logging
- Safe ZIP handling (file count / size limits)
- Regex word-boundary keyword matching
- Proper boost parsing (regex)
- FAISS usage with normalized float32 embeddings
- Optional weighting of boost terms in semantic search
- Resume summarization before LLM scoring to save tokens
- Robust LLM response parsing with retries
- Tesseract auto-detect fallback
- Caching for embedder
- Progress and user-friendly spinners/messages

Notes:
- Expects GROQ_API_KEY in Streamlit secrets.toml as before.
- Tune THREAD_COUNTS and limits for your deployment environment.
"""

import streamlit as st
import PyPDF2, re, io, zipfile, pandas as pd, base64, numpy as np, os, math
from pdf2image import convert_from_bytes
import pytesseract
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import faiss
import time
import textwrap
from typing import List, Tuple

# ========================= CONFIG =========================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY missing in secrets.toml — LLM scoring will be disabled.")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODEL = "llama-3.1-8b-instant"

# Tunables
MAX_ZIP_FILES = 800
MAX_ZIP_BYTES = 250 * 1024 * 1024  # 250 MB
EXTRACTION_THREADS = 8  # tune per CPU
LLM_THREADS = 8
MAX_CANDIDATES = 220
FINAL_LLM_TOP = 40
SUMMARY_CHAR_LIMIT = 2500  # summarise resume to ~2500 chars for LLM scoring

# Auto-detect tesseract
try:
    if not pytesseract.get_tesseract_version():
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
except Exception:
    # best-effort; let pytesseract raise later when used
    pass

st.set_page_config(page_title="Elite Resume Ranker — Production", layout="wide")
st.title("⚡ Elite Resume Ranker — Production-ready")
st.markdown("**Improved reliability, safety, and cost-efficiency**")

# ========================= HELPERS & CACHES =========================
@st.cache_resource
def load_embedder(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

embedder = load_embedder()

def safe_read_zip(file) -> List[Tuple[str, bytes]]:
    """Reads zip file contents but enforces file count and size limits to avoid ZIP bombs."""
    file.seek(0)
    data = file.read()
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"Uploaded zip exceeds maximum allowed size of {MAX_ZIP_BYTES//1024//1024} MB")
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.pdf')]
        if len(names) > MAX_ZIP_FILES:
            raise ValueError(f"Too many PDF files in ZIP ({len(names)}). Max allowed is {MAX_ZIP_FILES}.")
        items = [(n, z.read(n)) for n in names]
    return items


def extract_text(pdf_bytes: bytes, ocr_first_page_only: bool = True) -> str:
    """Try: 1) PyPDF2 text extraction 2) If short or empty -> OCR first page 3) return truncated text"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for p in reader.pages:
            # some pages may return None
            pg = p.extract_text()
            if pg:
                text += pg + "\n"
        text = re.sub(r'\s+', ' ', text).strip()
        # if sufficient text extracted, return
        if len(text) > 300:
            return text[:24000]
    except Exception as e:
        # log silently and fallback to OCR
        st.write(f"PyPDF2 failed: {e}")

    # Fallback: OCR first page
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)
        if imgs:
            text_ocr = pytesseract.image_to_string(imgs[0], config='--psm 6')
            text_ocr = re.sub(r'\s+', ' ', text_ocr).strip()
            if text_ocr:
                # combine with previous if any
                combined = (text + '\n' + text_ocr).strip()
                return combined[:24000]
    except Exception as e:
        st.write(f"OCR fallback failed: {e}")

    # Final fallback: return whatever we have or explicit marker
    if not text:
        return ""  # explicit empty so downstream can detect
    return text[:24000]


def parse_keywords(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def keyword_score(text: str, jd_tokens: List[str], boost_tokens: List[str]) -> int:
    """Score using word-boundary matching; boost tokens weighted higher."""
    text_low = text.lower()
    score = 0
    for w in set(jd_tokens):
        if re.search(rf"\b{re.escape(w)}\b", text_low):
            score += 1
    for w in set(boost_tokens):
        if re.search(rf"\b{re.escape(w)}\b", text_low):
            score += 3
    return score


def summarize_for_llm(text: str, char_limit: int = SUMMARY_CHAR_LIMIT) -> str:
    """A lightweight summarizer: returns the first useful paragraphs up to limit.
    (We intentionally keep it simple to avoid extra tokens and complexity.)
    """
    if not text:
        return ""
    # Prefer the top paragraph blocks separated by two newlines
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    out = ""
    for p in paragraphs:
        if len(out) + len(p) + 2 > char_limit:
            break
        out += p + "\n\n"
    if not out:
        out = text[:char_limit]
    return out.strip()


def robust_parse_score(resp_text: str) -> Tuple[int, str]:
    """Try to extract score robustly from LLM response."""
    if not resp_text:
        return 30, "Empty LLM response"
    s = resp_text
    # normalize
    s_norm = s.replace('\r', '\n')
    # search for digits near 'score'
    m = re.search(r'score[^0-9]{0,6}(\d{1,3})', s_norm, re.I)
    if not m:
        m = re.search(r'^(\d{1,3})\b', s_norm)
    if m:
        try:
            score = int(m.group(1))
            score = max(0, min(100, score))
        except:
            score = 30
    else:
        score = 30
    # reason extraction
    reason = ""
    m2 = re.search(r'reason[:\-\s]{1,20}(.+)', s_norm, re.I | re.S)
    if m2:
        reason = m2.group(1).strip().split('\n')[0][:200]
    else:
        # fallback: first sentence after score
        parts = re.split(r'\n', s_norm)
        if len(parts) > 1:
            reason = parts[1].strip()[:200]
        else:
            reason = s_norm.strip()[:200]
    return score, reason


def call_llm_score(client: Groq, prompt: str, model: str = MODEL, retries: int = 2) -> str:
    if client is None:
        return ''
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=180
            )
            return resp.choices[0].message.content
        except Exception as e:
            st.write(f"LLM call failed (attempt {attempt}): {e}")
            time.sleep(0.8 + attempt * 0.5)
    return ''

# ========================= UI and Main Flow =========================

c1, c2 = st.columns([3,1])
with c1:
    job_desc = st.text_area("Job Description", height=200, placeholder="e.g. Senior Python Engineer, FastAPI, Docker, AWS...")
with c2:
    boost = st.text_input("Boost keywords (comma separated)", placeholder="FastAPI, Docker, AWS, Redis")

uploaded_zip = st.file_uploader("Upload ZIP of PDF resumes", type="zip")

if st.button("Start Ranking", type="primary"):
    if not job_desc or not job_desc.strip():
        st.error("Job Description is required.")
        st.stop()
    if not uploaded_zip:
        st.error("Please upload a ZIP file containing PDF resumes.")
        st.stop()

    try:
        files = safe_read_zip(uploaded_zip)
    except Exception as e:
        st.error(f"Error reading ZIP: {e}")
        st.stop()

    st.info(f"Found {len(files)} PDF resumes in ZIP — starting processing.")

    jd_tokens = parse_keywords(job_desc)
    boost_tokens = parse_keywords(boost) if boost else []

    start_time = time.time()

    # Phase 1: parallel extraction + keyword pre-filter
    extractor_progress = st.empty()
    extractor_progress.text("Phase 1: extracting text and keyword filtering...")

    candidates = []
    with ThreadPoolExecutor(max_workers=min(EXTRACTION_THREADS, max(2, len(files)))) as ex:
        futures = {ex.submit(lambda t: (t[0], extract_text(t[1])), item): item[0] for item in files}
        results = []
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                name, text = fut.result()
            except Exception as e:
                st.write(f"Extraction failed for {name}: {e}")
                text = ""
            score = keyword_score(text, jd_tokens, boost_tokens) if text else 0
            results.append((name, text, score))

    # sort by keyword score and keep top N
    results.sort(key=lambda x: x[2], reverse=True)
    survivors = results[:MAX_CANDIDATES]
    extractor_progress.text(f"Phase 1 done — {len(survivors)} survivors. Time: {int(time.time()-start_time)}s")

    # Phase 2: semantic ranking using embeddings + FAISS
    sem_progress = st.empty()
    sem_progress.text("Phase 2: semantic ranking (embeddings + FAISS)...")

    texts = [r[1] if r[1] else "" for r in survivors]
    # Create embeddings in batches
    try:
        embeddings = embedder.encode([t if t else "" for t in texts], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
    except Exception as e:
        st.error(f"Embedding error: {e}")
        st.stop()

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Query embedding: consider weighting boost tokens by repeating them
    query_text = job_desc
    if boost_tokens:
        # append boost tokens repeated to give them extra weight
        query_text = job_desc + ' ' + ' '.join(boost_tokens * 3)
    q_emb = embedder.encode([query_text], normalize_embeddings=True).astype('float32')

    k = min(40, len(survivors))
    D, I = index.search(q_emb, k)
    ranked = [survivors[i] for i in I[0]]
    sem_progress.text(f"Phase 2 done — semantic top {len(ranked)} selected. Time: {int(time.time()-start_time)}s")

    # Phase 3: Final scoring with LLM (on summarized text to save tokens)
    llm_progress = st.empty()
    llm_progress.text("Phase 3: final scoring with LLM (summarized inputs)...")

    # Prepare PDF bytes lookup for download later
    # Re-open zip to fetch PDFs as bytes for top candidates
    uploaded_zip.seek(0)
    zip_buf = io.BytesIO(uploaded_zip.read())
    with zipfile.ZipFile(zip_buf) as z:
        name_to_bytes = {n: z.read(n) for n in z.namelist() if n.lower().endswith('.pdf')}

    final_items = ranked[:FINAL_LLM_TOP]
    results_for_df = []

    def score_item(name, text):
        summary = summarize_for_llm(text)
        prompt = f"""Score 0-100 how well this resume matches the JD. Reply EXACTLY in this format:\nSCORE: <integer 0-100>\nREASON: <one short sentence>\n\nJD: {job_desc}\nBOOST: {boost}\n\nResumeSummary: {summary}\n\nIf you cannot score, return SCORE: 30 and REASON: Unable to evaluate."""
        resp = call_llm_score(client, prompt)
        score, reason = robust_parse_score(resp)
        return score, reason

    # Use ThreadPoolExecutor but limit concurrency to control API usage
    with ThreadPoolExecutor(max_workers=min(LLM_THREADS, max(1, len(final_items)))) as ex:
        future_to_name = {ex.submit(score_item, itm[0], itm[1]): itm[0] for itm in final_items}
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                score, reason = fut.result()
            except Exception as e:
                st.write(f"LLM scoring failed for {name}: {e}")
                score, reason = 30, "LLM error"
            pdf_bytes = name_to_bytes.get(name, b"")
            results_for_df.append({"File": name, "Score": score, "Why": reason, "PDF": pdf_bytes})

    df = pd.DataFrame(results_for_df).sort_values("Score", ascending=False).reset_index(drop=True)
    if df.empty:
        st.error("No scored resumes — nothing to show.")
        st.stop()

    df["Rank"] = range(1, len(df) + 1)

    st.success(f"Done in {int(time.time()-start_time)} seconds — Top: {df.iloc[0]['File']} ({df.iloc[0]['Score']})")

    def link(n, d):
        return f'<a href="data:application/pdf;base64,{base64.b64encode(d).decode()}" download="{n}" style="color:#0066cc;font-weight:600;">{n}</a>'

    df["Candidate"] = df.apply(lambda r: link(r["File"], r["PDF"]), axis=1)

    st.markdown(df[["Rank", "Candidate", "Score", "Why"]].to_html(escape=False, index=False), unsafe_allow_html=True)

    # Download top N
    TOP_N = 20
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for _, r in df.head(TOP_N).iterrows():
            name = f"{int(r['Score']):03d}_{r['File']}"
            z.writestr(name, r['PDF'])
    buf.seek(0)
    st.download_button("📥 Download Top 20 Ranked Resumes", buf, "TOP20_FINAL.zip", "application/zip", use_container_width=True)

    st.balloons()

# End of app
