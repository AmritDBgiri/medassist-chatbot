# 🧠 RAG Chatbot — LangChain + Gemini

A production-ready RAG (Retrieval-Augmented Generation) chatbot powered by Google Gemini, LangChain, and FAISS vector store. Upload your documents and chat with them instantly.

## ✨ Features

- **Multi-format ingestion** — PDF, TXT, DOCX files + web URLs
- **Gemini 1.5 Flash** — fast, accurate LLM responses
- **FAISS vector store** — efficient semantic search
- **Conversational memory** — remembers last 6 exchanges
- **Source attribution** — shows which documents were used
- **Fallback chat** — works as general AI even without documents
- **Beautiful UI** — dark theme, drag & drop, markdown rendering

---

## 🚀 Local Setup (5 minutes)

### 1. Clone & install
```bash
git clone <your-repo>
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

Or export it directly:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 3. Run
```bash
python app.py
# Open http://localhost:8000
```

---

## ☁️ Deploy to Railway (Free)

Railway is the easiest way to get this online in minutes.

### Steps:
1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
3. Select your repo
4. Go to **Variables** tab and add:
   ```
   GOOGLE_API_KEY = your_key_here
   PORT = 8000
   ```
5. Railway auto-detects the Dockerfile and deploys 🎉

Your app will be live at `https://your-app.up.railway.app`

---

## ☁️ Deploy to Render (Free tier)

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Runtime**: Docker
   - **Port**: 8000
5. Add environment variable: `GOOGLE_API_KEY`
6. Click **Deploy**

---

## ☁️ Deploy to Heroku

```bash
# Install Heroku CLI, then:
heroku create your-rag-chatbot
heroku config:set GOOGLE_API_KEY=your_key
heroku stack:set container
git push heroku main
```

---

## 🏗 Architecture

```
User → FastAPI Backend
         ├── /upload     → PyPDF/Docx2txt → TextSplitter → FAISS (embed via Gemini)
         ├── /add-url    → WebLoader → TextSplitter → FAISS
         └── /chat       → ConversationalRetrievalChain
                              ├── FAISS retriever (top-4 chunks)
                              ├── ConversationBufferWindowMemory (k=6)
                              └── Gemini 1.5 Flash → Answer + Sources
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend |
| POST | `/upload` | Upload & index a file |
| POST | `/add-url` | Scrape & index a URL |
| POST | `/chat` | Send a message |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents` | Clear all documents |
| POST | `/clear-chat` | Reset conversation memory |

---

## 🔒 Security Notes

- **Never commit your API key** — use environment variables
- Add rate limiting for production (e.g., `slowapi`)
- Add authentication if exposing publicly

---

## 🛠 Customization

**Change the LLM model** in `app.py`:
```python
model="gemini-1.5-pro"  # More powerful, slower
model="gemini-1.5-flash"  # Default: fast & efficient
```

**Adjust chunk size** for better retrieval:
```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # More precise
RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # More context
```

**Change retrieval depth**:
```python
retriever=vectorstore.as_retriever(search_kwargs={"k": 6})  # More sources
```
# medassist-chatbot
