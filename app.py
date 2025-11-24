"""
Ultra-fast production-ready Streamlit app: Elite Resume Ranker — Optimized

Key optimizations applied:
- pypdf for fast text extraction (first N pages only)
- OCR used ONLY when PyPDF extraction returns empty
- Controlled thread pools for IO vs network
- Embedding caching with st.cache_resource for repeated runs
- Summarization before embedding (short snippets)
- FAISS HNSW index for fast approximate search
- Batch LLM scoring (5 resumes per prompt) to reduce RPCs
- Safe ZIP handling (file count/size limits)
- Minimal on-screen logging; use Python logging for debug

Notes:
- Requires GROQ_API_KEY in Streamlit secrets.toml for LLM scoring.
- Tweak THREAD counts and batch sizes according to your deployment (CPU vs GPU).

"""

import streamlit as st
import io, zipfile, re, time, os, math, base64
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Fast PDF reader
try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader  # fallback

# OCR (only used as last resort)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
except Exception:
    convert_from_bytes = None
    pytesseract = None

# Embeddings + FAISS
from sentence_transformers import SentenceTransformer
import faiss

# LLM client (Groq)
from groq import Groq

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ========================= CONFIG =========================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL = "llama-3.1-8b-instant"

# Tunables — adjust for your infra
MAX_ZIP_FILES = 800
MAX_ZIP_BYTES = 300 * 1024 * 1024  # 300 MB
EXTRACTION_THREADS = 12
LLM_THREADS = 6
MAX_CANDIDATES = 220
FINAL_LLM_TOP = 120
SUMMARY_CHAR_LIMIT = 1800  # small summaries for embedding + LLM
EMBED_BATCH = 64
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
HNSW_M = 32
BATCH_LLM_SIZE = 5  # resumes per LLM batch prompt
PAGES_TO_EXTRACT = 3  # only first N pages

st.set_page_config(page_title="Elite Resume Ranker — Ultra-fast", layout="wide")
st.title("⚡ Elite Resume Ranker — Ultra-fast")
st.caption("Optimized pipeline: fast extraction → cached embeddings → HNSW FAISS → batched LLM scoring")

# ========================= CACHES & HELPERS =========================
@st.cache_resource
def get_embedder(model_name: str = EMBED_MODEL):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedder on {device}")
    return SentenceTransformer(model_name, device=device)

embedder = get_embedder()

@st.cache_resource
def embed_texts(texts: Tuple[str]):
    # accept tuple to be cacheable
    emb = embedder.encode(list(texts), batch_size=EMBED_BATCH, show_progress_bar=False, normalize_embeddings=True)
    return np.array(emb, dtype='float32')


def safe_read_zip(uploaded_file) -> List[Tuple[str, bytes]]:
    uploaded_file.seek(0)
    data = uploaded_file.read()
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"ZIP size exceeds {MAX_ZIP_BYTES//1024//1024} MB")
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.pdf')]
        if len(names) > MAX_ZIP_FILES:
            raise ValueError(f"Too many PDFs in ZIP ({len(names)}). Limit {MAX_ZIP_FILES}.")
        items = [(n, z.read(n)) for n in names]
    return items


def fast_pdf_text(pdf_bytes: bytes, pages: int = PAGES_TO_EXTRACT) -> str:
    """Extract text from first N pages using pypdf/PyPDF2. If no text and OCR available, OCR first page only."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        # pypdf and PyPDF2 both expose .pages
        for p in reader.pages[:pages]:
            pg = p.extract_text()
            if pg:
                text += pg + "\n"
        text = re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.info(f"PDF parsing error: {e}")
        text = ""

    # Only OCR if extraction returned NOTHING
    if (not text or len(text.strip()) == 0) and convert_from_bytes and pytesseract:
        try:
            imgs = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)
            if imgs:
                text = pytesseract.image_to_string(imgs[0], config='--psm 6')
                text = re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logger.info(f"OCR failed: {e}")
            text = text or ""

    return text[:24000]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def keyword_score(text: str, jd_tokens: List[str], boost_tokens: List[str]) -> int:
    text_low = (text or "").lower()
    score = 0
    for w in set(jd_tokens):
        if re.search(rf"\b{re.escape(w)}\b", text_low):
            score += 1
    for w in set(boost_tokens):
        if re.search(rf"\b{re.escape(w)}\b", text_low):
            score += 3
    return score


def summarize(text: str, limit: int = SUMMARY_CHAR_LIMIT) -> str:
    if not text:
        return ""
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    out = ""
    for p in paras:
        if len(out) + len(p) + 2 > limit:
            break
        out += p + "\n\n"
    if not out:
        out = text[:limit]
    return out.strip()

# Robust LLM parsing for batched responses

def parse_batch_llm_response(resp: str, count: int) -> Dict[int, Tuple[int, str]]:
    """Expected formats in response (examples):
    1: SCORE=85, REASON=Good match
    or
    1) SCORE: 85 REASON: Good match
    Returns mapping idx->(score, reason)
    """
    out = {}
    if not resp:
        return {i: (30, 'Empty response') for i in range(1, count+1)}
    lines = [l.strip() for l in re.split(r'\n|\r', resp) if l.strip()]
    # combine lines that look like entries
    candidate_text = ' '.join(lines)
    # try to find patterns like '1: SCORE=85, REASON=...'
    matches = re.findall(r'(\d+)[\):\.]?\s*(?:score[:=]?\s*(\d{1,3}))?[,;\-\s]+reason[:=]?\s*([^\d]+?)(?=\d+\)|\d+:|$)', candidate_text, re.I)
    if matches:
        for m in matches:
            idx = int(m[0])
            sc = int(m[1]) if m[1] and m[1].isdigit() else 30
            reason = m[2].strip()[:250]
            out[idx] = (max(0, min(100, sc)), reason)
        # fill missing
        for i in range(1, count+1):
            if i not in out:
                out[i] = (30, 'No parse')
        return out

    # fallback: look for all numbers and reasons sequentially
    scores = re.findall(r'\b(\d{1,3})\b', candidate_text)
    reasons = re.split(r'\b\d{1,3}\b', candidate_text)
    for i in range(1, count+1):
        sc = int(scores[i-1]) if i-1 < len(scores) else 30
        reason = reasons[i][:200].strip() if i <= len(reasons)-1 else ''
        out[i] = (max(0, min(100, sc)), reason or 'Parsed')
    return out


def call_llm_batch(client: Groq, jd: str, boost: str, summaries: List[str]) -> Dict[int, Tuple[int, str]]:
    """Send up to BATCH_LLM_SIZE summaries in one prompt and parse response."""
    if client is None:
        # return neutral scores
        return {i+1: (50, 'LLM disabled') for i in range(len(summaries))}
    # Build prompt
    prompt_lines = ["You are an expert recruiter. Score each resume 0-100 and give one short reason."]
    prompt_lines.append(f"JD: {jd}")
    if boost:
        prompt_lines.append(f"BOOST: {boost}")
    prompt_lines.append("\nRate these resumes. Reply in a numbered list: '1: SCORE=XX, REASON=one short sentence'\n")
    for i, s in enumerate(summaries, start=1):
        prompt_lines.append(f"{i}) {s[:1500].replace('\n',' ')}")
    prompt = '\n'.join(prompt_lines)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,
            max_tokens=400
        )
        text = resp.choices[0].message.content
    except Exception as e:
        logger.info(f"LLM batch call failed: {e}")
        text = ''
    return parse_batch_llm_response(text, len(summaries))

# ========================= UI =========================
c1, c2 = st.columns([3,1])
with c1:
    job_desc = st.text_area("Job Description", height=200, placeholder="e.g. Senior Python Engineer, FastAPI, Docker, AWS...")
with c2:
    boost = st.text_input("Boost keywords (comma separated)", placeholder="FastAPI, Docker, AWS, Redis")

uploaded_zip = st.file_uploader("Upload ZIP of PDF resumes", type=["zip"])

if st.button("Start (Ultra-fast)"):
    if not job_desc or not job_desc.strip():
        st.error("Job description required")
        st.stop()
    if not uploaded_zip:
        st.error("Please upload a ZIP of PDFs")
        st.stop()

    try:
        files = safe_read_zip(uploaded_zip)
    except Exception as e:
        st.error(f"ZIP error: {e}")
        st.stop()

    st.info(f"Processing {len(files)} PDFs — running fast pipeline")
    start = time.time()

    jd_tokens = tokenize(job_desc)
    boost_tokens = tokenize(boost)

    # Phase 1: Fast parallel extraction + keyword prefilter
    extracted = []
    with ThreadPoolExecutor(max_workers=min(EXTRACTION_THREADS, max(2, len(files)))) as ex:
        futures = {ex.submit(fast_pdf_text, data): name for name, data in files}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                logger.info(f"Extraction failed {name}: {e}")
                text = ""
            score = keyword_score(text, jd_tokens, boost_tokens)
            extracted.append((name, text, score))

    extracted.sort(key=lambda x: x[2], reverse=True)
    survivors = extracted[:MAX_CANDIDATES]
    st.write(f"Phase1 done — {len(survivors)} survivors — {int(time.time()-start)}s")

    # Phase 2: Summarize + embed (cacheable)
    texts = [s[1] or "" for s in survivors]
    summaries = [summarize(t) for t in texts]

    # Use cached embedder: embed summaries
    embeddings = embed_texts(tuple(summaries))

    # Build HNSW index
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = 40
    index.add(embeddings)

    # Query embedding: emphasize boost by repeating
    query_text = job_desc + (' ' + boost) * 2 if boost else job_desc
    q_emb = embedder.encode([query_text], normalize_embeddings=True).astype('float32')
    k = min(130, len(summaries))
    D, I = index.search(q_emb, k)
    ranked = [survivors[i] for i in I[0]]
    st.write(f"Phase2 done — top {len(ranked)} semantic candidates — {int(time.time()-start)}s")

    # Prepare bytes lookup for download
    uploaded_zip.seek(0)
    buf = io.BytesIO(uploaded_zip.read())
    with zipfile.ZipFile(buf) as z:
        name_to_bytes = {n: z.read(n) for n in z.namelist() if n.lower().endswith('.pdf')}

    # Phase 3: Batched LLM scoring
    to_score = ranked[:FINAL_LLM_TOP]
    results = []
    # prepare batches
    batches = [to_score[i:i+BATCH_LLM_SIZE] for i in range(0, len(to_score), BATCH_LLM_SIZE)]

    for batch_idx, batch in enumerate(batches, start=1):
        batch_summaries = [summarize(item[1]) for item in batch]
        res_map = call_llm_batch(client, job_desc, boost, batch_summaries)
        # parse and attach
        for i, item in enumerate(batch, start=1):
            name, text, _ = item
            sc, reason = res_map.get(i, (30, 'No score'))
            pdfb = name_to_bytes.get(name, b"")
            results.append({"File": name, "Score": sc, "Why": reason, "PDF": pdfb})
        st.write(f"Batched LLM: processed batch {batch_idx}/{len(batches)}")

    df = pd.DataFrame(results).sort_values('Score', ascending=False).reset_index(drop=True)
    if df.empty:
        st.warning('No results')
        st.stop()
    df['Rank'] = range(1, len(df)+1)

    st.success(f"Done in {int(time.time()-start)}s — Top: {df.iloc[0]['File']} ({df.iloc[0]['Score']})")

    def link(n,d):
        return f'<a href="data:application/pdf;base64,{base64.b64encode(d).decode()}" download="{n}">{n}</a>'

    df['Candidate'] = df.apply(lambda r: link(r['File'], r['PDF']), axis=1)
    st.markdown(df[['Rank','Candidate','Score','Why']].to_html(escape=False, index=False), unsafe_allow_html=True)

    # Download top 20
    outbuf = io.BytesIO()
    with zipfile.ZipFile(outbuf, 'w') as z:
        for _, r in df.head(20).iterrows():
            z.writestr(f"{int(r['Score']):03d}_{r['File']}", r['PDF'])
    outbuf.seek(0)
    st.download_button('📥 Download Top 20', outbuf, 'top20.zip', 'application/zip')

    st.balloons()

# End
