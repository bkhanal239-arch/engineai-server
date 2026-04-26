import os
import re
import time
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import build_chain, get_ingested_pdfs, parse_response, expand_query

load_dotenv()

PDF_DIR          = os.environ.get("PDF_DIR", os.path.join(os.path.dirname(__file__), "my_pdfs"))
USE_HERMES_AGENT = os.environ.get("USE_HERMES_AGENT", "false").lower() == "true"

app = FastAPI(title="EngineAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    question: str
    pdf_name: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/pdfs")
def list_pdfs():
    pdfs = get_ingested_pdfs()
    return {"pdfs": [p.replace("_", " ") for p in pdfs]}


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Hermes agent path (OpenRouter cloud) ──────────────────────
    if USE_HERMES_AGENT:
        try:
            from agent import hermes_agent
            result = hermes_agent(request.question, request.pdf_name)
            return {
                "answer":           result.get("answer", ""),
                "code_ref":         result.get("code_ref", ""),
                "snippet":          result.get("snippet", ""),
                "searched":         result.get("searched", ""),
                "query":            request.question,
                "reformulated":     result.get("reformulated"),
                "raw_chunks":       result.get("raw_chunks", []),
                "from_cache":       result.get("from_cache", False),
                "cache_similarity": result.get("cache_similarity"),
                "engine":           "openrouter",
                "model":            os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct").split("/")[-1],
            }
        except Exception as e:
            msg = str(e)
            # On rate limit or unavailable → silently fall back to Gemini
            is_rate   = "429" in msg or "rate" in msg.lower() or "rate_limit" in msg.lower()
            is_unavail = "503" in msg or "502" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower()
            if not (is_rate or is_unavail):
                raise HTTPException(status_code=500, detail=msg)
            # Fall through to Gemini below

    # ── Gemini path (primary when Hermes off, fallback when rate-limited) ──
    expanded     = expand_query(request.question)
    reformulated = expanded if expanded.lower() != request.question.lower().strip() else None
    chain, retriever, label = build_chain(request.pdf_name)

    if chain is None:
        raise HTTPException(status_code=404, detail=label)

    try:
        raw = None
        for attempt in range(3):
            try:
                raw = chain.invoke(expanded)
                break
            except Exception as e:
                if attempt < 2 and ("503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)):
                    time.sleep(2 ** attempt)
                else:
                    raise
        parsed   = parse_response(raw)
        docs     = retriever.invoke(expanded)
        raw_snips = [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page":   doc.metadata.get("page", 0) + 1,
                "text":   doc.page_content.strip()[:400],
            }
            for doc in docs
        ]
        return {
            "answer":       parsed["answer"],
            "code_ref":     parsed["code_ref"],
            "snippet":      parsed["snippet"],
            "searched":     label,
            "query":        request.question,
            "reformulated": reformulated,
            "raw_chunks":   raw_snips,
            "from_cache":   False,
            "engine":       "gemini",
        }
    except Exception as e:
        msg = str(e)
        if "503" in msg or "UNAVAILABLE" in msg:
            raise HTTPException(status_code=503, detail="Gemini is temporarily overloaded. Please try again.")
        if "429" in msg or "quota" in msg.lower():
            raise HTTPException(status_code=429, detail="API rate limit reached. Please wait a moment.")
        raise HTTPException(status_code=500, detail=msg)


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        messages = [SystemMessage(content=(
            "You are a helpful structural engineering assistant. "
            "You can answer general engineering questions, explain concepts, "
            "help with calculations, and discuss codes and standards."
        ))]
        for m in request.history[-10:]:
            if m.role == "user":
                messages.append(HumanMessage(content=m.content))
            else:
                messages.append(AIMessage(content=m.content))
        messages.append(HumanMessage(content=request.message))
        response = llm.invoke(messages)
        return {"reply": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def find_pdf(name: str) -> str | None:
    if not os.path.isdir(PDF_DIR):
        return None
    norm = name.lower().replace("_", " ").replace(".pdf", "").strip()
    for f in os.listdir(PDF_DIR):
        if not f.lower().endswith(".pdf"):
            continue
        if norm in f.lower().replace("_", " ").replace(".pdf", "").strip():
            return os.path.join(PDF_DIR, f)
    return None


def _find_highlight_rects(pg, highlight: str):
    """Return list of fitz.Rect to highlight on the page. Empty list = nothing found."""
    import fitz
    clean = re.sub(r'\s+', ' ', highlight.strip())

    # Stage 1: exact phrase, progressively shortened
    for length in [90, 60, 35]:
        rects = pg.search_for(clean[:length])
        if rects:
            return rects

    # Stage 2: block-level match — find the text block (paragraph) with most matching words
    # Returns a single rect around the best matching paragraph, not scattered word rects
    search_words = {w.lower().rstrip('.,;:()') for w in re.split(r'\W+', clean) if len(w) >= 4}
    if not search_words:
        return []

    blocks = pg.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    best_score, best_rect = 0, None
    for block in blocks:
        if len(block) < 6 or block[6] != 0:  # skip images / non-text
            continue
        block_text = block[4].lower()
        score = sum(1 for w in search_words if w in block_text)
        if score > best_score:
            best_score = score
            best_rect = fitz.Rect(block[0], block[1], block[2], block[3])

    if best_score >= 2 and best_rect:
        return [best_rect]
    return []


@app.get("/page-count")
def page_count(pdf: str = Query(...)):
    import fitz
    found = find_pdf(pdf)
    if not found:
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf}")
    doc = fitz.open(found)
    n   = len(doc)
    doc.close()
    return {"count": n}


@app.get("/page-image")
def page_image(
    pdf:       str   = Query(...),
    page:      int   = Query(...),
    scale:     float = Query(1.5),
    highlight: str   = Query(default=""),
):
    import fitz
    scale = max(0.5, min(3.0, scale))
    found = find_pdf(pdf)
    if not found:
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf}")
    doc      = fitz.open(found)
    page_idx = max(0, min(page - 1, len(doc) - 1))
    pg       = doc[page_idx]

    # Draw yellow highlight on matching text if requested
    if highlight and len(highlight.strip()) > 6:
        rects = _find_highlight_rects(pg, highlight)
        if rects:
            shape = pg.new_shape()
            for rect in rects:
                shape.draw_rect(rect)
            shape.finish(color=None, fill=(1, 0.93, 0), fill_opacity=0.75)
            shape.commit()

    pix       = pg.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img_bytes = pix.tobytes("png")
    doc.close()
    # No cache when highlighted so normal page isn't cached as highlighted
    cache = "no-store" if highlight else "public, max-age=3600"
    return Response(content=img_bytes, media_type="image/png",
                    headers={"Cache-Control": cache})


@app.get("/snippet-image")
def snippet_image(
    pdf:   str   = Query(...),
    page:  int   = Query(...),
    text:  str   = Query(default=""),
    scale: float = Query(2.0),
):
    """Crop the PDF page to the region containing the snippet text and return as PNG."""
    import fitz
    scale = max(1.0, min(3.0, scale))
    found = find_pdf(pdf)
    if not found:
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf}")

    doc      = fitz.open(found)
    page_idx = max(0, min(page - 1, len(doc) - 1))
    pg       = doc[page_idx]
    pw, ph   = pg.rect.width, pg.rect.height

    # Search for the snippet text — try progressively shorter candidates
    found_rects = []
    if text and len(text.strip()) > 8:
        for length in [100, 60, 35]:
            candidate = text.strip()[:length]
            rects = pg.search_for(candidate)
            if rects:
                found_rects = rects
                break

    # Build clip rectangle around found text (with generous padding)
    if found_rects:
        x0 = min(r.x0 for r in found_rects)
        y0 = min(r.y0 for r in found_rects)
        x1 = max(r.x1 for r in found_rects)
        y1 = max(r.y1 for r in found_rects)
        clip_rect = fitz.Rect(
            max(0,  x0 - 35),
            max(0,  y0 - 70),
            min(pw, x1 + 35),
            min(ph, y1 + 70),
        )
    else:
        # Text not found — show upper-middle portion of the page
        clip_rect = fitz.Rect(0, ph * 0.2, pw, ph * 0.6)

    # Draw yellow highlight rectangle directly on a separate drawing layer
    # (avoids annotation API issues across PyMuPDF versions)
    if found_rects:
        shape = pg.new_shape()
        for rect in found_rects:
            shape.draw_rect(rect)
        shape.finish(color=None, fill=(1, 0.95, 0), fill_opacity=0.4)
        shape.commit()

    # Render cropped region as PNG (reliable across all PyMuPDF versions)
    pix       = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip_rect)
    img_bytes = pix.tobytes("png")
    doc.close()

    return Response(content=img_bytes, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """Twilio WhatsApp webhook — receives messages and replies via Hermes agent."""
    from twilio.twiml.messaging_response import MessagingResponse
    from twilio.request_validator import RequestValidator
    from whatsapp import handle_message

    form_data = dict(await request.form())
    body  = form_data.get("Body", "").strip()
    phone = form_data.get("From", "").replace("whatsapp:", "")

    # Validate Twilio signature (skip if no auth token set)
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    if auth_token:
        validator = RequestValidator(auth_token)
        signature = request.headers.get("X-Twilio-Signature", "")
        url       = str(request.url)
        if not validator.validate(url, form_data, signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    available_pdfs = get_ingested_pdfs()
    reply = handle_message(phone, body, available_pdfs)

    resp = MessagingResponse()
    resp.message(reply)
    return Response(content=str(resp), media_type="application/xml")


@app.get("/pdf-file")
def pdf_file(pdf: str = Query(...)):
    from fastapi.responses import FileResponse as FR
    found = find_pdf(pdf)
    if not found:
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf}")
    return FR(found, media_type="application/pdf", filename=os.path.basename(found))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
