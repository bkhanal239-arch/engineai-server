import os
import json
import time
import sqlite3
import re
import numpy as np
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

DB_DIR = os.environ.get(
    "VECTOR_DB_PATH",
    os.path.join(os.path.dirname(__file__), "vector_db")
)
CACHE_DB    = os.path.join(os.path.dirname(__file__), "answer_cache.db")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt_template.md")

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
CACHE_THRESHOLD  = float(os.environ.get("CACHE_THRESHOLD", "0.92"))


# ── Helpers ─────────────────────────────────────────────────────

def load_system_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0,
        max_tokens=4096,
        default_headers={
            "HTTP-Referer": "https://engineai.app",
            "X-Title": "EngineAI",
        },
    )


# ── Semantic Cache ───────────────────────────────────────────────

def init_cache():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS answer_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name    TEXT    NOT NULL,
            question    TEXT    NOT NULL,
            embedding   TEXT    NOT NULL,
            answer      TEXT,
            code_ref    TEXT,
            snippet     TEXT,
            chunks      TEXT,
            reformulated TEXT,
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def cache_lookup(question: str, pdf_key: str, embeddings) -> Optional[dict]:
    q_emb = embeddings.embed_query(question)
    conn  = sqlite3.connect(CACHE_DB)
    rows  = conn.execute(
        "SELECT question, embedding, answer, code_ref, snippet, chunks, reformulated "
        "FROM answer_cache WHERE pdf_name=?",
        (pdf_key,)
    ).fetchall()
    conn.close()

    best_sim, best = 0.0, None
    for row in rows:
        sim = _cosine(q_emb, json.loads(row[1]))
        if sim > best_sim:
            best_sim, best = sim, row

    if best_sim >= CACHE_THRESHOLD and best:
        return {
            "answer":       best[2],
            "code_ref":     best[3],
            "snippet":      best[4],
            "raw_chunks":   json.loads(best[5] or "[]"),
            "reformulated": best[6],
            "from_cache":   True,
            "cache_similarity": round(best_sim, 3),
        }
    return None


def cache_save(question: str, pdf_key: str, embeddings, result: dict):
    q_emb = embeddings.embed_query(question)
    conn  = sqlite3.connect(CACHE_DB)
    conn.execute("""
        INSERT INTO answer_cache
            (pdf_name, question, embedding, answer, code_ref, snippet, chunks, reformulated, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        pdf_key, question, json.dumps(q_emb),
        result.get("answer", ""),
        result.get("code_ref", ""),
        result.get("snippet", ""),
        json.dumps(result.get("raw_chunks", [])),
        result.get("reformulated"),
        datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


# ── Vector Search ────────────────────────────────────────────────

def vector_search(query: str, pdf_name: Optional[str], embeddings, k: int = 8):
    if pdf_name:
        safe    = os.path.splitext(pdf_name)[0].replace(" ", "_")
        db_path = os.path.join(DB_DIR, safe)
        if not os.path.exists(db_path):
            return [], f"No database for '{pdf_name}'"
        db   = Chroma(persist_directory=db_path, embedding_function=embeddings)
        docs = db.as_retriever(search_kwargs={"k": k}).invoke(query)
        return docs, pdf_name

    ingested = [n for n in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, n))]
    docs = []
    for name in ingested:
        db   = Chroma(persist_directory=os.path.join(DB_DIR, name), embedding_function=embeddings)
        docs.extend(db.as_retriever(search_kwargs={"k": 3}).invoke(query))
    return docs, "all PDFs"


def fmt_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        pg  = doc.metadata.get("page", 0) + 1
        parts.append(f"[Chunk {i} | Source: {src} | Page: {pg}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def parse_response(raw: str) -> dict:
    def extract(label):
        m = re.search(
            rf"\*\*{label}:\*\*\s*(.*?)(?=\n\n\*\*|\Z)", raw, re.DOTALL | re.IGNORECASE
        )
        return m.group(1).strip().strip('"').strip("'") if m else ""

    return {
        "answer":   extract("Answer") or raw.strip(),
        "code_ref": extract("Code Reference"),
        "snippet":  extract("Exact Snippet"),
    }


def _invoke_llm(llm, messages, retries=3):
    for attempt in range(retries):
        try:
            return llm.invoke(messages).content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# ── Main Agent Entry Point ────────────────────────────────────────

def hermes_agent(question: str, pdf_name: Optional[str]) -> dict:
    embeddings = get_embeddings()
    init_cache()
    pdf_key = pdf_name or "ALL"

    # 1. Semantic cache — instant return if similar Q answered before
    hit = cache_lookup(question, pdf_key, embeddings)
    if hit:
        return hit

    system_prompt = load_system_prompt()

    # 2. Search ChromaDB with multiple query variants (no LLM needed)
    #    This replaces the 3-call agentic loop with smart keyword search
    docs, label = vector_search(question, pdf_name, embeddings, k=8)

    # Also search with a cleaned-up version (remove filler words)
    filler = {"what", "is", "the", "are", "for", "in", "of", "a", "an", "how", "does", "do"}
    keywords = " ".join(w for w in question.lower().split() if w not in filler)
    if keywords and keywords != question.lower():
        alt_docs, _ = vector_search(keywords, pdf_name, embeddings, k=4)
        seen = {d.metadata.get("page") for d in docs}
        for d in alt_docs:
            if d.metadata.get("page") not in seen:
                docs.append(d)
                seen.add(d.metadata.get("page"))

    context = fmt_docs(docs)

    # 3. Single LLM call — system/user split so Llama follows the strict rules properly
    llm = get_llm()
    raw = _invoke_llm(llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            "RETRIEVED CONTEXT:\n" + context
            + "\n\n---\nUSER QUESTION:\n" + question
            + "\n\nRespond using the exact format: **Answer:** / **Code Reference:** / **Exact Snippet:**"
            + "\nAlways start the Answer with *Interpreted as: ...*  Never truncate values or lists."
        )),
    ])
    parsed = parse_response(raw)

    # Extract reformulated query from the answer's "Interpreted as:" line
    reformulated = None
    interp_match = re.search(r"\*Interpreted as:\s*(.+?)\.\*", parsed["answer"])
    if interp_match:
        reformulated = interp_match.group(1).strip()

    raw_chunks = [
        {
            "source": os.path.basename(d.metadata.get("source", "unknown")),
            "page":   d.metadata.get("page", 0) + 1,
            "text":   d.page_content.strip()[:400],
        }
        for d in docs
    ]

    result = {
        "answer":       parsed["answer"],
        "code_ref":     parsed["code_ref"],
        "snippet":      parsed["snippet"],
        "raw_chunks":   raw_chunks,
        "reformulated": reformulated,   # extracted from "Interpreted as:" in answer
        "from_cache":   False,
        "searched":     label,
    }

    # 6. Cache for instant recall next time
    if len(parsed["answer"]) > 50:
        try:
            cache_save(question, pdf_key, embeddings, result)
        except Exception:
            pass

    return result
