import io
import os
import uuid
import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


st.set_page_config(page_title="Minimal PDF + Pinecone", layout="centered")
st.title("PooC RAG")
st.write(
    "Query Pinecone directly or upload a PDF to index. "
    "If your index already has data, you can skip the upload."
)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "test")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")

if not PINECONE_API_KEY:
    st.error("Missing PINECONE_API_KEY environment variable.")
    st.stop()
if not OPENROUTER_API_KEY:
    st.error("Missing OPENROUTER_API_KEY environment variable.")
    st.stop()
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return [c for c in chunks if c.strip()]


def extract_sectioned_chunks(text: str) -> list[dict]:
    import re

    lines = [ln.strip() for ln in text.splitlines()]
    chunks = []
    current_section = ""
    current_title = ""
    pending_section = ""

    buffer = []

    def flush_buffer():
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if not body:
            return
        header = ""
        if current_section or current_title:
            header = f"{current_section} {current_title}".strip()
        chunks.append(
            {
                "text": (f"{header}\n{body}" if header else body),
                "section": current_section,
                "title": current_title,
            }
        )

    for ln in lines:
        if not ln:
            continue
        # Case: "9.1 Effectuer un reset du réseau" on one line
        m_inline = re.match(r"^(\d+(\.\d+)*)(\s+)(.+)$", ln)
        if m_inline:
            flush_buffer()
            buffer = []
            current_section = m_inline.group(1)
            current_title = m_inline.group(4).strip()
            pending_section = ""
            continue
        # Matches section numbers like "9.1" or "10" on their own line
        if re.match(r"^\d+(\.\d+)*$", ln):
            # New section starts; flush previous buffer
            flush_buffer()
            buffer = []
            pending_section = ln
            current_section = ln
            current_title = ""
            continue

        # If we just saw a section number, take the next line as title
        if pending_section:
            current_title = ln
            pending_section = ""
            continue

        buffer.append(ln)

    flush_buffer()

    # If no sections found, fallback to plain chunking
    if not chunks:
        return [{"text": c, "section": "", "title": ""} for c in chunk_text(text)]

    # Now chunk within each section
    final_chunks = []
    for item in chunks:
        sub_chunks = chunk_text(item["text"])
        for sc in sub_chunks:
            final_chunks.append(
                {
                    "text": sc,
                    "section": item["section"],
                    "title": item["title"],
                }
            )
    return final_chunks


def embed_texts(texts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors], EMBEDDING_MODEL


def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    from sentence_transformers import CrossEncoder

    if not chunks:
        return []
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, c) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def answer_with_llm(question: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    system = (
        "You are a helpful assistant. Use only the provided context to answer. "
        "If the answer is not in the context, say you don't know."
    )
    user = f"Question:\n{question}\n\nContext:\n{context}"
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def get_index(dimension: int):
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX)


uploaded_file = st.file_uploader(
    "Upload a PDF (optional, only needed to add new data)", type=["pdf"]
)
if uploaded_file is not None:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        st.warning("PDF extraction needs `pypdf`. Run: `uv add pypdf`.")
        st.stop()

    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not raw_text.strip():
        st.info("No text could be extracted from this PDF.")
        st.stop()

    chunks = extract_sectioned_chunks(raw_text)
    st.caption(f"Chunks: {len(chunks)}")
    with st.expander("Debug: sample chunks", expanded=False):
        for i, item in enumerate(chunks[:8], start=1):
            st.markdown(f"**Chunk {i}**")
            st.text(item["text"][:1000])

    if st.button("Index in Pinecone"):
        with st.spinner("Embedding and upserting..."):
            vectors, model = embed_texts([c["text"] for c in chunks])
            index = get_index(dimension=len(vectors[0]))
            batch = []
            for item, vec in zip(chunks, vectors):
                batch.append(
                    (
                        str(uuid.uuid4()),
                        vec,
                        {
                            "text": item["text"],
                            "source": uploaded_file.name,
                            "model": model,
                            "section": item["section"],
                            "title": item["title"],
                        },
                    )
                )
            index.upsert(vectors=batch)
        st.success("Indexed successfully.")

query = st.text_input("Search query")
if query:
    with st.spinner("Searching..."):
        qvecs, model = embed_texts([query])
        index = get_index(dimension=len(qvecs[0]))
        results = index.query(vector=qvecs[0], top_k=12, include_metadata=True)
    st.subheader("Top Matches")
    
    context_chunks = []
    
    for match in results.get("matches", []):
        score = match.get("score", 0)
        text = match.get("metadata", {}).get("text", "")
        if text:
            context_chunks.append(text)
        #st.markdown(f"**Score:** {score:.4f}")
        #st.write(text)
    
    if context_chunks:
        context_chunks = rerank(query, context_chunks, top_k=5)
        st.subheader("Answer")
        with st.spinner("Generating answer..."):
            context = "\n\n---\n\n".join(context_chunks)
            answer = answer_with_llm(query, context)
        st.write(answer)
