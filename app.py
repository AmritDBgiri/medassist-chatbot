import os
import shutil
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MedAssist RAG API")

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
upload_status = {}  # filename -> {status, msg, chunks}

# ─── LLM & Embeddings ────────────────────────────────────────────────────────
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True,
        max_retries=1,  # Don't retry on quota errors
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )

# ─── RAG Chain ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are MedAssist AI, a knowledgeable medical information assistant.
Your role is to provide clear, accurate, and helpful medical information based on uploaded clinical documents, research papers, or medical literature.

Guidelines:
- Answer based on the provided medical context when available.
- Use proper medical terminology but also explain terms in plain language.
- Structure answers with headings and bullet points when relevant.
- Always remind users this is for educational purposes only, not personal medical advice.
- If information is not in the context, use your general medical knowledge but clearly state it.
- Never diagnose or prescribe — always recommend consulting a licensed healthcare provider.

Context from medical documents:
{context}

Chat History:
{chat_history}

Medical Question: {question}

Answer:"""
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

# ─── Document Loader ─────────────────────────────────────────────────────────
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

# ─── Batched Embedding (avoids rate limits & timeouts) ───────────────────────
def embed_in_batches(chunks, embeddings, batch_size=8):
    global vectorstore
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        for attempt in range(2):  # Only 2 attempts max
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                print(f"  Embedded batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size}")
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "exhausted" in err.lower():
                    if attempt == 0:
                        print(f"  Rate limit hit, waiting 15s...")
                        time.sleep(15)
                    else:
                        raise Exception("API quota exceeded. Please use a new API key.")
                else:
                    raise e
        if i + batch_size < total:
            time.sleep(2)

# ─── Background Indexing ─────────────────────────────────────────────────────
def process_document_background(file_path: str, filename: str):
    global upload_status
    try:
        upload_status[filename] = {"status": "processing", "msg": "Loading document..."}
        docs = load_document(file_path)

        upload_status[filename]["msg"] = "Splitting text..."
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        total_chunks = len(chunks)
        upload_status[filename]["msg"] = f"Embedding {total_chunks} chunks (this may take a minute)..."

        embeddings = get_embeddings()
        embed_in_batches(chunks, embeddings, batch_size=8)

        uploaded_files.append(filename)
        build_chain()
        upload_status[filename] = {
            "status": "done",
            "msg": f"Successfully indexed {total_chunks} chunks!",
            "chunks": total_chunks
        }
        print(f"✅ Done: {filename} → {total_chunks} chunks")
    except Exception as e:
        upload_status[filename] = {"status": "error", "msg": str(e)}
        print(f"❌ Error indexing {filename}: {e}")

# ─── Routes ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class URLRequest(BaseModel):
    url: str

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".docx", ".doc"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type {ext} not supported. Allowed: PDF, TXT, DOCX")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Return instantly — indexing happens in background (no more 502!)
    upload_status[file.filename] = {"status": "processing", "msg": "Indexing started..."}
    background_tasks.add_task(process_document_background, str(dest), file.filename)

    return {
        "success": True,
        "filename": file.filename,
        "chunks": 0,
        "message": "Uploading in background. Use /upload-status to check progress."
    }

@app.get("/upload-status/{filename:path}")
def get_upload_status(filename: str):
    return upload_status.get(filename, {"status": "unknown", "msg": "Not found"})

@app.post("/add-url")
async def add_url(req: URLRequest, background_tasks: BackgroundTasks):
    def index_url():
        try:
            upload_status[req.url] = {"status": "processing", "msg": "Scraping URL..."}
            loader = WebBaseLoader(req.url)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
            chunks = splitter.split_documents(docs)
            embeddings = get_embeddings()
            embed_in_batches(chunks, embeddings, batch_size=8)
            uploaded_files.append(req.url)
            build_chain()
            upload_status[req.url] = {"status": "done", "chunks": len(chunks), "msg": f"Indexed {len(chunks)} chunks"}
        except Exception as e:
            upload_status[req.url] = {"status": "error", "msg": str(e)}

    upload_status[req.url] = {"status": "processing", "msg": "Starting..."}
    background_tasks.add_task(index_url)
    return {"success": True, "url": req.url, "chunks": 0, "message": "Indexing in background"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    if chat_chain is None:
        try:
            llm = get_llm()
            response = llm.invoke(req.message)
            return {"answer": response.content, "sources": []}
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "exhausted" in err.lower():
                return {"answer": "⚠️ **API quota exceeded.** Your Gemini free tier limit has been reached for today. Please:\n1. Create a new API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n2. Update `GOOGLE_API_KEY` in Railway Variables\n\nFree tier resets at **midnight UTC** (~5:30 AM IST).", "sources": []}
            raise HTTPException(500, f"LLM error: {err}")

    try:
        result = chat_chain.invoke({"question": req.message})
        sources = list({
            doc.metadata.get("source", "")
            for doc in result.get("source_documents", [])
            if doc.metadata.get("source")
        })
        return {"answer": result["answer"], "sources": sources}
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "exhausted" in err.lower():
            return {"answer": "⚠️ **API quota exceeded.** Your Gemini free tier limit has been reached for today. Please create a new API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and update it in Railway Variables.", "sources": []}
        raise HTTPException(500, f"Chat error: {err}")

@app.get("/documents")
def list_documents():
    result = list(uploaded_files)
    for fname, s in upload_status.items():
        if fname not in uploaded_files and s.get("status") == "processing":
            result.append(f"⏳ {fname}")
    return {"files": result}

@app.delete("/documents")
def clear_documents():
    global vectorstore, chat_chain, uploaded_files
    vectorstore = None
    chat_chain = None
    uploaded_files = []
    upload_status.clear()
    memory.clear()
    for f in UPLOAD_DIR.iterdir():
        f.unlink(missing_ok=True)
    return {"success": True, "message": "All documents cleared"}

@app.post("/clear-chat")
def clear_chat():
    memory.clear()
    return {"success": True}

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
