#!/usr/bin/env python3
"""Ingest all PDFs in PDF_DIR into ChromaDB. Skips already-ingested PDFs."""
import os
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = os.environ.get("PDF_DIR",  os.path.join(os.path.dirname(__file__), "my_pdfs"))
DB_DIR  = os.environ.get("VECTOR_DB_PATH", os.path.join(os.path.dirname(__file__), "vector_db"))


def ingest_pdf(pdf_path: str):
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    pdf_name  = os.path.basename(pdf_path)
    safe_name = os.path.splitext(pdf_name)[0].replace(" ", "_")
    db_path   = os.path.join(DB_DIR, safe_name)

    if os.path.exists(db_path):
        print(f"  [skip] {pdf_name} — already ingested")
        return

    print(f"  [ingest] {pdf_name} ...")
    docs   = PyMuPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    emb    = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(chunks, emb, persist_directory=db_path)
    print(f"  [done]  {len(chunks)} chunks → {db_path}")


if __name__ == "__main__":
    os.makedirs(DB_DIR, exist_ok=True)

    if not os.path.isdir(PDF_DIR):
        print(f"PDF_DIR not found: {PDF_DIR}")
    else:
        pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        if not pdfs:
            print(f"No PDFs found in {PDF_DIR}")
        else:
            print(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")
            for pdf_file in sorted(pdfs):
                ingest_pdf(os.path.join(PDF_DIR, pdf_file))

    print("Ingestion complete.")
