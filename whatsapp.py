import os
import sqlite3
from datetime import datetime
from typing import Optional

SESSIONS_DB = os.path.join(os.path.dirname(__file__), "whatsapp_sessions.db")
MAX_WA_LEN  = 4000   # WhatsApp message character limit


# ── Session storage (which PDF each user is searching) ──────────

def init_sessions():
    conn = sqlite3.connect(SESSIONS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone      TEXT PRIMARY KEY,
            pdf_name   TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_session_pdf(phone: str) -> Optional[str]:
    conn = sqlite3.connect(SESSIONS_DB)
    row  = conn.execute("SELECT pdf_name FROM sessions WHERE phone=?", (phone,)).fetchone()
    conn.close()
    return row[0] if row else None


def set_session_pdf(phone: str, pdf_name: Optional[str]):
    conn = sqlite3.connect(SESSIONS_DB)
    conn.execute("""
        INSERT INTO sessions (phone, pdf_name, updated_at) VALUES (?,?,?)
        ON CONFLICT(phone) DO UPDATE SET pdf_name=excluded.pdf_name, updated_at=excluded.updated_at
    """, (phone, pdf_name, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# ── Format agent result as WhatsApp message ──────────────────────

def format_response(result: dict, active_pdf: Optional[str]) -> str:
    parts = []

    if active_pdf:
        parts.append(f"📂 _{active_pdf.replace('_', ' ')}_\n")

    if result.get("from_cache"):
        sim = result.get("cache_similarity", 0)
        parts.append(f"⚡ _Instant recall ({round(sim * 100)}% match)_\n")

    if result.get("reformulated"):
        parts.append(f"🔍 *Searched as:* {result['reformulated']}\n")

    answer = (result.get("answer") or "").strip()
    if answer:
        parts.append(f"✅ *Answer*\n{answer}")

    code_ref = (result.get("code_ref") or "").strip()
    if code_ref:
        parts.append(f"\n📌 *Code Reference*\n{code_ref}")

    snippet = (result.get("snippet") or "").strip()
    if snippet:
        if len(snippet) > 350:
            snippet = snippet[:347] + "..."
        parts.append(f"\n📄 *Exact Snippet*\n_{snippet}_")

    chunks = result.get("raw_chunks") or []
    if chunks:
        pages    = list(dict.fromkeys(c["page"] for c in chunks))[:8]
        page_str = "  ".join(f"p.{p}" for p in pages)
        parts.append(f"\n📑 *Pages:* {page_str}")

    msg = "\n".join(parts)
    if len(msg) > MAX_WA_LEN:
        msg = msg[:MAX_WA_LEN - 20] + "\n\n_[truncated]_"
    return msg


# ── Main message handler ─────────────────────────────────────────

def handle_message(phone: str, message: str, available_pdfs: list) -> str:
    from agent import hermes_agent

    init_sessions()
    msg       = message.strip()
    msg_lower = msg.lower()

    # ── Help / greeting ──
    if msg_lower in ("help", "hi", "hello", "hey", "?", "start"):
        pdf_list = "\n".join(f"  • {p.replace('_', ' ')}" for p in available_pdfs[:10])
        current  = get_session_pdf(phone)
        cur_str  = f"\n📂 *Active doc:* {current.replace('_',' ')}" if current else ""
        return (
            "👋 *EngineAI — Structural Code Assistant*\n\n"
            "Ask any structural engineering question and I'll search the code for you.\n\n"
            "*Commands:*\n"
            "  `list` — available documents\n"
            "  `use ACI 318` — set active document\n"
            "  `clear` — search all documents\n"
            "  `help` — show this message\n"
            f"{cur_str}\n\n"
            f"*Loaded documents:*\n{pdf_list}"
        )

    # ── List PDFs ──
    if msg_lower in ("list", "pdfs", "documents", "docs"):
        if not available_pdfs:
            return "⚠️ No documents loaded on the server yet."
        items    = "\n".join(f"  {i+1}. {p.replace('_',' ')}" for i, p in enumerate(available_pdfs[:15]))
        current  = get_session_pdf(phone)
        cur_str  = f"\n\n📂 *Active:* {current.replace('_',' ')}" if current else "\n\n📂 Searching all documents"
        return f"📚 *Available Documents:*\n{items}{cur_str}"

    # ── Set active PDF ──
    if msg_lower.startswith("use "):
        query   = msg[4:].strip()
        norm    = query.lower().replace("_", " ").replace(".pdf", "")
        matched = next(
            (p for p in available_pdfs if norm in p.lower().replace("_", " ")),
            None
        )
        if matched:
            set_session_pdf(phone, matched)
            return f"✅ Now searching:\n*{matched.replace('_', ' ')}*\n\nAsk your question!"
        else:
            opts = "\n".join(f"  • {p.replace('_',' ')}" for p in available_pdfs[:10])
            return f"❌ No match for _{query}_\n\nAvailable:\n{opts}"

    # ── Clear / reset to all PDFs ──
    if msg_lower in ("clear", "all", "reset", "any"):
        set_session_pdf(phone, None)
        return "✅ Searching all documents.\n\nAsk your question!"

    # ── Engineering question → Hermes agent ──
    current_pdf = get_session_pdf(phone)
    try:
        result  = hermes_agent(msg, current_pdf)
        return format_response(result, current_pdf)
    except Exception as e:
        err = str(e)[:200]
        if "503" in err or "UNAVAILABLE" in err:
            return "⏳ Hermes is busy right now. Please try again in a few seconds."
        if "429" in err or "rate" in err.lower():
            return "⏳ Rate limit reached. Please wait a moment and try again."
        return f"⚠️ Error: {err}\n\nPlease try again."
