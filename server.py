import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import build_chain, get_ingested_pdfs, parse_response

load_dotenv()

PDF_DIR = os.environ.get("PDF_DIR", os.path.join(os.path.dirname(__file__), "my_pdfs"))

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
    role: str   # "user" or "assistant"
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

    chain, retriever, label = build_chain(request.pdf_name)

    if chain is None:
        raise HTTPException(status_code=404, detail=label)

    try:
        raw = chain.invoke(request.question)
        parsed = parse_response(raw)

        docs = retriever.invoke(request.question)
        raw_snippets = [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", 0) + 1,
                "text": doc.page_content.strip()[:400],
            }
            for doc in docs
        ]

        return {
            "answer":     parsed["answer"],
            "code_ref":   parsed["code_ref"],
            "snippet":    parsed["snippet"],
            "searched":   label,
            "query":      request.question,
            "raw_chunks": raw_snippets,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/page-image")
def page_image(pdf: str = Query(...), page: int = Query(...)):
    import fitz
    found = find_pdf(pdf)
    if not found:
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf}")
    doc = fitz.open(found)
    page_idx = max(0, min(page - 1, len(doc) - 1))
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    img_bytes = pix.tobytes("png")
    doc.close()
    return Response(content=img_bytes, media_type="image/png")


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
