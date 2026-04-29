# 📚 RAG PDF Assistant

A fully local, end-to-end Retrieval-Augmented Generation (RAG) system.  
Upload any PDF → ask questions → get accurate, cited answers. **Zero cloud. Zero cost.**

---

## 🏗 Architecture

```
PDF Upload
    │
    ▼
┌─────────────────┐
│  pdf_processor  │  PyPDF + pdfplumber extraction
│                 │  RecursiveCharacterTextSplitter chunking
└────────┬────────┘
         │  chunks + metadata
         ▼
┌─────────────────┐
│  vector_store   │  HuggingFace sentence-transformers
│                 │  FAISS index (persisted to disk)
└────────┬────────┘
         │
    User Query
         │
         ▼
┌─────────────────┐
│  Retrieval      │  MMR search (top-k diverse chunks)
└────────┬────────┘
         │  context + citations
         ▼
┌─────────────────┐
│  llm_handler    │  Ollama (llama3/mistral/etc.)
│                 │  RAG prompt template
└────────┬────────┘
         │
         ▼
    Streamed Answer
    + Source Citations
```

---

## 🗂 Project Structure

```
rag_pdf_assistant/
├── app.py                  # Streamlit UI (main entry point)
├── setup.sh                # One-shot setup script
├── requirements.txt        # Python dependencies
├── .env.example            # Configuration template
│
├── src/
│   ├── config.py           # Centralized configuration
│   ├── pdf_processor.py    # PDF ingestion + chunking
│   ├── vector_store.py     # FAISS embeddings + retrieval
│   ├── llm_handler.py      # Ollama LLM integration
│   └── rag_pipeline.py     # End-to-end orchestrator
│
├── vectorstore/            # Persisted FAISS indexes (auto-created)
└── data/
    └── cache/              # LLM response cache (auto-created)
```

---

## ⚡ Quick Start

### Option A — Automated Setup (recommended)

```bash
git clone <repo>
cd rag_pdf_assistant
chmod +x setup.sh
./setup.sh
```

### Option B — Manual Setup

#### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

#### 2. Install Ollama

| Platform | Command |
|----------|---------|
| macOS    | `brew install ollama` or download from [ollama.com](https://ollama.com/download) |
| Linux    | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Windows  | Download installer from [ollama.com/download](https://ollama.com/download) |

#### 3. Pull an LLM model

```bash
# Pick one (llama3 recommended, ~4.7GB):
ollama pull llama3
ollama pull mistral          # lighter, faster
ollama pull gemma2           # Google's model
ollama pull phi3             # Microsoft, very lightweight
ollama pull qwen2.5          # Alibaba, multilingual
```

#### 4. Start Ollama server

```bash
ollama serve
# Runs at http://localhost:11434
```

#### 5. Launch the app

```bash
# In a new terminal (with .venv activated):
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🛠 Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Default LLM model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embeddings |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `MAX_PDF_SIZE_MB` | `50` | Max upload size |

---

## 🧩 How It Works

### 1. PDF Processing (`pdf_processor.py`)
- Extracts text page-by-page using **pypdf** (fast) with **pdfplumber** fallback (complex layouts)
- Splits text using `RecursiveCharacterTextSplitter` — tries `\n\n` → `\n` → `. ` → `" "` → `""` in order, preserving semantic units
- Each chunk carries metadata: `source`, `page`, `pdf_hash`, `chunk_id`

### 2. Embedding & Indexing (`vector_store.py`)
- Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90MB)
- Stores vectors in a **FAISS** flat index (cosine similarity)
- Index is saved to disk — same PDFs are never re-embedded

### 3. Retrieval
- Uses **Maximum Marginal Relevance (MMR)** search: balances relevance + diversity
- Avoids returning near-duplicate chunks from the same section
- `lambda_mult=0.7` (70% relevance, 30% diversity)

### 4. LLM Generation (`llm_handler.py`)
- Formats a **RAG prompt** instructing the model to use only provided context
- Calls Ollama via LangChain's `OllamaLLM` with streaming
- Caches responses to disk using `diskcache`

---

## 🎯 Model Recommendations

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `llama3` | 4.7GB | Medium | ⭐⭐⭐⭐ | Best overall |
| `mistral` | 4.1GB | Fast | ⭐⭐⭐⭐ | Balanced |
| `gemma2` | 5.4GB | Medium | ⭐⭐⭐⭐ | Long context |
| `phi3` | 2.3GB | Fast | ⭐⭐⭐ | Low RAM |
| `qwen2.5` | 4.7GB | Medium | ⭐⭐⭐⭐ | Multilingual |

---

## 🚀 Features

- ✅ Multi-PDF support (upload and query across multiple documents)
- ✅ Streaming answers (real-time token output)
- ✅ Source citations with page numbers
- ✅ Chat history
- ✅ FAISS index caching (skip re-indexing same PDFs)
- ✅ LLM response caching
- ✅ MMR retrieval for diverse context
- ✅ Adjustable Top-K slider
- ✅ Model selector (any pulled Ollama model)
- ✅ System status panel

---

## 🔧 Troubleshooting

**"Ollama is not running"**  
→ Run `ollama serve` in a terminal and leave it running.

**"No models found"**  
→ Pull a model: `ollama pull llama3`

**"No text extracted from PDF"**  
→ PDF may be image-based (scanned). Add OCR support with `pytesseract` + `pdf2image`.

**Slow embedding on first run**  
→ Normal — the ~90MB MiniLM model downloads once and is cached locally.

**High memory usage**  
→ Use `phi3` model (2.3GB) and set `CHUNK_SIZE=500` in `.env`.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | RAG orchestration |
| `langchain-ollama` | Ollama LLM wrapper |
| `langchain-huggingface` | HuggingFace embeddings |
| `langchain-community` | FAISS vectorstore |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Text embedding models |
| `pypdf` + `pdfplumber` | PDF text extraction |
| `streamlit` | Web UI |
| `diskcache` | LLM response caching |
| `python-dotenv` | Environment config |

---

## 📄 License

MIT — free for personal and commercial use.
