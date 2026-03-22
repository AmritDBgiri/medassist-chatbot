import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# ─── Config ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAw896DrAxehIdHXUwerklqzfANBSCtQ0g")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
vectorstore = None
chat_chain = None
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=6
)
uploaded_files: List[str] = []

# ─── LLM & Embeddings ────────────────────────────────────────────────────────
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True,
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

# ─── RAG Chain Builder ───────────────────────────────────────────────────────
SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are MedAssist AI, a knowledgeable medical information assistant.
Your role is to provide clear, accurate, and helpful medical information based on the uploaded clinical documents, research papers, or medical literature.

Guidelines:
- Answer based on the provided medical context when available.
- Use proper medical terminology but also explain terms in plain language.
- Structure answers with headings, bullet points, and dosage/symptom tables when relevant.
- Always remind users that your information is for educational purposes only and does not replace professional medical advice.
- If information isn't in the context, use your general medical knowledge but clearly state it.
- Never diagnose or prescribe — always recommend consulting a licensed healthcare provider for personal medical decisions.

Context from medical documents:
{context}

Chat History:
{chat_history}

Medical Question: {question}

Answer (with disclaimer if needed):"""
)

def build_chain():
    global chat_chain
    if vectorstore is None:
        return
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT},
        output_key="answer",
    )

# ─── Document Loader ──────────────────────────────────────────────────────────
def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

# ─── Routes ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class URLRequest(BaseModel):
    url: str

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore
    allowed = {".pdf", ".txt", ".docx", ".doc"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type {ext} not supported. Use: {allowed}")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = load_document(str(dest))
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)

        uploaded_files.append(file.filename)
        build_chain()
        return {
            "success": True,
            "filename": file.filename,
            "chunks": len(chunks),
            "total_files": len(uploaded_files),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/add-url")
async def add_url(req: URLRequest):
    global vectorstore
    try:
        loader = WebBaseLoader(req.url)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)

        uploaded_files.append(req.url)
        build_chain()
        return {"success": True, "url": req.url, "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    # No documents — general chat with Gemini
    if chat_chain is None:
        llm = get_llm()
        response = llm.invoke(req.message)
        return {"answer": response.content, "sources": []}

    try:
        result = chat_chain.invoke({"question": req.message})
        sources = list({
            doc.metadata.get("source", "")
            for doc in result.get("source_documents", [])
            if doc.metadata.get("source")
        })
        return {"answer": result["answer"], "sources": sources}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/documents")
def list_documents():
    return {"files": uploaded_files}

@app.delete("/documents")
def clear_documents():
    global vectorstore, chat_chain, uploaded_files
    vectorstore = None
    chat_chain = None
    uploaded_files = []
    memory.clear()
    for f in UPLOAD_DIR.iterdir():
        f.unlink(missing_ok=True)
    return {"success": True, "message": "All documents cleared"}

@app.post("/clear-chat")
def clear_chat():
    memory.clear()
    return {"success": True}

# Mount static files last
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
