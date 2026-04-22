import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

DB_DIR = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def db_path_for(pdf_name):
    base_name = os.path.splitext(pdf_name)[0]
    return os.path.join(DB_DIR, base_name.replace(" ", "_"))


def get_ingested_pdfs():
    if not os.path.exists(DB_DIR):
        return []
    return [name for name in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, name))]


def format_docs_with_snippets(docs):
    parts = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        parts.append(f"[{source} | Page {page + 1}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_retriever(pdf_name=None):
    embeddings = get_embeddings()

    if pdf_name:
        safe_name = os.path.splitext(pdf_name)[0].replace(" ", "_")
        db_path = os.path.join(DB_DIR, safe_name)
        if not os.path.exists(db_path):
            return None, f"No database found for '{pdf_name}'."
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 4}), pdf_name

    ingested = get_ingested_pdfs()
    if not ingested:
        return None, "No databases found on server."

    dbs = [Chroma(persist_directory=os.path.join(DB_DIR, name), embedding_function=embeddings) for name in ingested]
    retrievers = [db.as_retriever(search_kwargs={"k": 2}) for db in dbs]

    def multi_retrieve(query):
        docs = []
        for r in retrievers:
            docs.extend(r.invoke(query))
        return docs

    return RunnableLambda(multi_retrieve), "all PDFs"


def build_chain(pdf_name=None):
    retriever, label = build_retriever(pdf_name)
    if retriever is None:
        return None, None, label

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = PromptTemplate.from_template(
        "Use the following context from the documents to answer the question.\n"
        "If the answer is not in the context, say 'I don't have that information in the provided documents.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = (
        {"context": retriever | format_docs_with_snippets, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever, label
