import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import build_chain, get_ingested_pdfs

load_dotenv()

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
        answer = chain.invoke(request.question)
        docs = retriever.invoke(request.question)
        snippets = [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", 0) + 1,
                "text": doc.page_content.strip()[:300]
            }
            for doc in docs
        ]
        return {"answer": answer, "searched": label, "snippets": snippets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
