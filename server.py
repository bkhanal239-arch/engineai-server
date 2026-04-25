import os
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
                "answer":       result.get("answer", ""),
                "code_ref":     result.get("code_ref", ""),
                "snippet":      result.get("snippet", ""),
                "searched":     result.get("searched", ""),
                "query":        request.question,
                "reformulated": result.get("reformulated"),
                "raw_chunks":   result.get("raw_chunks", []),
                "from_cache":   result.get("from_cache", False),
                "cache_similarity": result.get("cache_similarity"),
            }
        except Exception as e:
            msg = str(e)
            if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                raise HTTPException(status_code=503, detail="Hermes is temporarily unavailable. Please try again.")
            if "429" in msg or "rate" in msg.lower():
                raise HTTPException(status_code=429, detail="Rate limit reached. Please wait a moment.")
            raise HTTPException(status_code=500, detail=msg)

    # ── Gemini fallback path ──────────────────────────────────────
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
def page_image(pdf: str = Query(...), page: int = Query(...), scale: float = Query(1.5)):
    import fitz
    scale = max(0.5, min(3.0, scale))
    found = find_pdf(pdf)
    if not found:
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf}")
    doc      = fitz.open(found)
    page_idx = max(0, min(page - 1, len(doc) - 1))
    pix      = doc[page_idx].get_pixmap(matrix=fitz.Matrix(scale, scale))
    img_bytes = pix.tobytes("png")
    doc.close()
    return Response(content=img_bytes, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/snippet-image")
def snippet_image(
    pdf:   str   = Query(...),
    page:  int   = Query(...),
    text:  str   = Query(default=""),
    scale: float = Query(2.0),
):
    """Return a cropped PNG of the page region containing the given text, with yellow highlight."""
    import fitz
    scale = max(1.0, min(3.0, scale))
    found = find_pdf(pdf)
    if not found:
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf}")

    doc      = fitz.open(found)
    page_idx = max(0, min(page - 1, len(doc) - 1))
    pg       = doc[page_idx]
    pw, ph   = pg.rect.width, pg.rect.height

    clip_rect  = None
    found_rects = []

    # Search for the snippet text on the page
    if text and len(text.strip()) > 8:
        search_candidates = [
            text.strip()[:120],
            text.strip()[:60],
            text.strip()[:40],
        ]
        for candidate in search_candidates:
            rects = pg.search_for(candidate)
            if rects:
                found_rects = rects
                break

    if found_rects:
        x0 = min(r.x0 for r in found_rects)
        y0 = min(r.y0 for r in found_rects)
        x1 = max(r.x1 for r in found_rects)
        y1 = max(r.y1 for r in found_rects)
        pad_x, pad_y = 30, 55
        clip_rect = fitz.Rect(
            max(0,  x0 - pad_x),
            max(0,  y0 - pad_y),
            min(pw, x1 + pad_x),
            min(ph, y1 + pad_y),
        )
    else:
        # Fall back: show middle third of the page
        clip_rect = fitz.Rect(0, ph * 0.25, pw, ph * 0.65)

    # Add temporary yellow highlight annotations for found text
    added_annots = []
    for rect in found_rects:
        try:
            hl = pg.add_highlight_annot(rect)
            hl.set_colors(stroke=(1.0, 0.85, 0.0))
            hl.update()
            added_annots.append(hl)
        except Exception:
            pass

    pix       = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip_rect)
    img_bytes = pix.tobytes("png")

    # Remove annotations so the stored PDF is not modified
    for annot in added_annots:
        try:
            pg.delete_annot(annot)
        except Exception:
            pass

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
