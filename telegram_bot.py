import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
from whatsapp import get_session_pdf, set_session_pdf, init_sessions
from agent import hermes_agent
from rag import get_ingested_pdfs

load_dotenv()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_LEN = 4000


# ── Formatting ───────────────────────────────────────────────────

def fmt(result: dict, active_pdf: str | None) -> str:
    parts = []

    if active_pdf:
        parts.append(f"📂 <i>{active_pdf.replace('_', ' ')}</i>\n")

    if result.get("from_cache"):
        sim = result.get("cache_similarity", 0)
        parts.append(f"⚡ <i>Instant recall ({round(sim * 100)}% match)</i>\n")

    if result.get("reformulated"):
        parts.append(f"🔍 <b>Searched as:</b> <i>{result['reformulated']}</i>\n")

    answer = (result.get("answer") or "").strip()
    if answer:
        parts.append(f"✅ <b>Answer</b>\n{answer}")

    code_ref = (result.get("code_ref") or "").strip()
    if code_ref:
        parts.append(f"\n📌 <b>Code Reference</b>\n<code>{code_ref}</code>")

    snippet = (result.get("snippet") or "").strip()
    if snippet:
        if len(snippet) > 400:
            snippet = snippet[:397] + "..."
        parts.append(f"\n📄 <b>Exact Snippet</b>\n<i>{snippet}</i>")

    chunks = result.get("raw_chunks") or []
    if chunks:
        pages    = list(dict.fromkeys(c["page"] for c in chunks))[:8]
        page_str = "  ".join(f"p.{p}" for p in pages)
        parts.append(f"\n📑 <b>Pages:</b> {page_str}")

    msg = "\n".join(parts)
    if len(msg) > MAX_LEN:
        msg = msg[:MAX_LEN - 20] + "\n\n<i>[truncated]</i>"
    return msg


def chat_key(chat_id: int) -> str:
    return f"tg_{chat_id}"


# ── Command handlers ─────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_sessions()
    pdfs    = get_ingested_pdfs()
    pdf_list = "\n".join(f"  • {p.replace('_', ' ')}" for p in pdfs[:10])
    await update.message.reply_html(
        "👋 <b>EngineAI — Structural Code Assistant</b>\n\n"
        "Ask any structural engineering question and I'll search the code for you.\n\n"
        "<b>Commands:</b>\n"
        "  /list — available documents\n"
        "  /use ACI 318 — set active document\n"
        "  /clear — search all documents\n"
        "  /help — show this message\n\n"
        f"<b>Loaded documents:</b>\n{pdf_list or '  (none yet)'}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pdfs = get_ingested_pdfs()
    if not pdfs:
        await update.message.reply_text("⚠️ No documents loaded on the server yet.")
        return
    items   = "\n".join(f"  {i+1}. {p.replace('_',' ')}" for i, p in enumerate(pdfs[:15]))
    key     = chat_key(update.effective_chat.id)
    current = get_session_pdf(key)
    cur_str = f"\n\n📂 <b>Active:</b> {current.replace('_',' ')}" if current else \
              "\n\n📂 Searching <b>all documents</b>"
    await update.message.reply_html(f"📚 <b>Available Documents:</b>\n{items}{cur_str}")


async def cmd_use(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_sessions()
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /use ACI 318  or  /use ASCE 7")
        return
    pdfs  = get_ingested_pdfs()
    norm  = query.lower().replace("_", " ").replace(".pdf", "")
    match = next((p for p in pdfs if norm in p.lower().replace("_", " ")), None)
    key   = chat_key(update.effective_chat.id)
    if match:
        set_session_pdf(key, match)
        await update.message.reply_html(
            f"✅ Now searching:\n<b>{match.replace('_', ' ')}</b>\n\nAsk your question!"
        )
    else:
        opts = "\n".join(f"  • {p.replace('_',' ')}" for p in pdfs[:10])
        await update.message.reply_html(
            f"❌ No match for <i>{query}</i>\n\nAvailable:\n{opts}"
        )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_sessions()
    key = chat_key(update.effective_chat.id)
    set_session_pdf(key, None)
    await update.message.reply_text("✅ Searching all documents.\n\nAsk your question!")


# ── Question handler ─────────────────────────────────────────────

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_sessions()
    question = (update.message.text or "").strip()
    if not question:
        return

    key         = chat_key(update.effective_chat.id)
    active_pdf  = get_session_pdf(key)

    thinking = await update.message.reply_html(
        f"🔎 <i>Searching{'  ' + active_pdf.replace('_',' ') if active_pdf else ' all documents'}…</i>"
    )

    try:
        result = hermes_agent(question, active_pdf)
        reply  = fmt(result, active_pdf)
        await thinking.delete()
        await update.message.reply_html(reply)
    except Exception as e:
        err = str(e)[:200]
        await thinking.delete()
        if "503" in err or "UNAVAILABLE" in err:
            await update.message.reply_text("⏳ Hermes is busy. Please try again in a few seconds.")
        elif "429" in err or "rate" in err.lower():
            await update.message.reply_text("⏳ Rate limit reached. Please wait a moment.")
        else:
            await update.message.reply_text(f"⚠️ Error: {err}")


# ── Entry point ──────────────────────────────────────────────────

def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("list",  cmd_list))
    app.add_handler(CommandHandler("use",   cmd_use))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    log.info("EngineAI Telegram bot starting (polling mode)…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
