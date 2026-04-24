import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

DB_DIR = os.environ.get(
    "VECTOR_DB_PATH",
    os.path.join(os.path.dirname(__file__), "vector_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "prompt_template.md")


def load_system_prompt():
    try:
        with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def db_path_for(pdf_name):
    base_name = os.path.splitext(pdf_name)[0]
    return os.path.join(DB_DIR, base_name.replace(" ", "_"))


def get_ingested_pdfs():
    if not os.path.exists(DB_DIR):
        return []
    return [name for name in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, name))]


def expand_query(question: str) -> str:
    """Reformulate a vague or incomplete question into a precise engineering code query.
    Falls back to the original question if Gemini is unavailable."""
    import time
    from langchain_core.messages import HumanMessage, SystemMessage
    system = (
        "You are a structural engineering expert. Your only job is to rewrite the user's question "
        "into a complete, precise structural engineering code question that can be used to search "
        "technical PDF documents like ACI 318, ACI 350, ASCE 7, IBC, etc.\n"
        "Rules:\n"
        "- If the question is vague (e.g. 'flood' or 'slab'), expand it to the most likely full engineering question.\n"
        "- If it is already specific and complete, return it unchanged.\n"
        "- Output ONLY the reformulated question — no explanation, no prefix, no quotes."
    )
    for attempt in range(3):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            result = llm.invoke([SystemMessage(content=system), HumanMessage(content=question)])
            expanded = result.content.strip().strip('"').strip("'")
            return expanded if expanded else question
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)   # 1 s, then 2 s
            else:
                return question            # fall back to original on repeated failure


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", 0)
        parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page + 1}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def parse_response(raw: str) -> dict:
    def extract(label):
        pattern = rf"\*\*{label}:\*\*\s*(.*?)(?=\n\n\*\*|\Z)"
        m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip().strip('"').strip("'") if m else ""

    answer    = extract("Answer")
    code_ref  = extract("Code Reference")
    snippet   = extract("Exact Snippet")

    if not answer:
        answer = raw.strip()

    return {
        "answer":    answer,
        "code_ref":  code_ref,
        "snippet":   snippet,
    }


def build_retriever(pdf_name=None):
    embeddings = get_embeddings()

    if pdf_name:
        safe_name = os.path.splitext(pdf_name)[0].replace(" ", "_")
        db_path = os.path.join(DB_DIR, safe_name)
        if not os.path.exists(db_path):
            return None, f"No database found for '{pdf_name}'."
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 8}), pdf_name

    ingested = get_ingested_pdfs()
    if not ingested:
        return None, "No databases found on server."

    dbs = [Chroma(persist_directory=os.path.join(DB_DIR, name), embedding_function=embeddings) for name in ingested]
    retrievers = [db.as_retriever(search_kwargs={"k": 3}) for db in dbs]

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

    system_prompt = load_system_prompt()

    prompt_text = (
        system_prompt
        + "\n\n---\nRETRIEVED CONTEXT:\n{context}"
        + "\n\n---\nUSER QUESTION:\n{question}"
        + "\n\nRespond using the exact format specified above."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        top_p=1,
        max_output_tokens=4096,
    )

    prompt = PromptTemplate.from_template(prompt_text)

    chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever, label
