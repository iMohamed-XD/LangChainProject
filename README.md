# RAG Application — Company Handbook Q&A

A full-stack retrieval-augmented generation (RAG) system that answers questions about a company handbook using Google's Gemini models, LangChain, and a local Chroma vector store. Supports multi-turn conversations via history-aware query reformulation, served through a FastAPI backend and a React + TypeScript frontend.

## Architecture

```
                        ┌─────────────────────────────┐
                        │   React + TS + Tailwind      │
                        │   (frontend/, Vite dev server)│
                        └───────────────┬──────────────┘
                                        │ POST /chat
                                        ▼
                        ┌─────────────────────────────┐
                        │   FastAPI (Backend/app/main.py)│
                        │   - CORS                     │
                        │   - request/response schemas │
                        │   - history truncation        │
                        └───────────────┬──────────────┘
                                        │ app.state.rag_chain
                                        ▼
                        ┌─────────────────────────────┐
                        │   RAG.py — build_rag_chain()  │
                        └───────────────┬──────────────┘
                                        │
company_handbook.md                    │
        │                              │
        ▼                              │
MarkdownHeaderTextSplitter              │
(splits on #, ##, ### and attaches      │
header metadata)                        │
        │                              │
        ▼                              │
RecursiveCharacterTextSplitter          │
(chunk_size=800, chunk_overlap=100)     │
        │                              │
        ▼                              │
GoogleGenerativeAIEmbeddings            │
(embeds each chunk)                     │
        │                              │
        ▼                              │
Chroma vector store                     │
(persisted to Backend/app/data/chroma_db,│
hash-checked on startup — rebuilds only │
if company_handbook.md content changed) │
        │                              │
        ▼                              │
history_aware_retriever ─── rephrases follow-up questions using chat history
        │                              │
        ▼                              │
create_retrieval_chain + create_stuff_documents_chain
        │
        ▼
Answer + source chunks returned as JSON ──► rendered in the React chat UI
```

**Models used:**
- LLM: `gemini-2.5-flash`
- Embeddings: `gemini-embedding-2-preview` (768 dimensions)

## Project Structure

```
.
├── Backend/
│   └── app/
│       ├── main.py                # FastAPI app: /chat endpoint, CORS, lifespan startup
│       ├── RAG.py                 # RAG pipeline: build_rag_chain(), CLI loop (main())
│       ├── RAG Application.ipynb  # original notebook prototype (kept for reference)
│       └── data/
│           ├── company_handbook.md  # source document
│           └── chroma_db/           # generated on first run, persisted vector store
├── frontend/                      # React + TypeScript + Tailwind (Vite)
│   ├── package.json
│   └── ...
├── .venv/                         # Python virtual environment (not committed)
├── .env                           # not committed — holds GOOGLE_API_KEY
├── .gitignore
├── requirements.txt
└── README.md
```

> Note: `Pipfile` / `Pipfile.lock` from an earlier Pipenv setup are no longer used — dependency management is now `venv` + `requirements.txt`. Delete those two files if they're still present.

## Prerequisites

- Python 3.12 (developed/tested on 3.12.10)
- Node.js 18+ and npm (for the Vite/React frontend)
- A Google account (for the Gemini API key)

## 1. Get a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in with your Google account.
3. Click **Create API key**.
4. Select an existing Google Cloud project, or let AI Studio create one for you.
5. Copy the generated key — you won't be able to view it again from this screen, though you can revoke and regenerate it if lost.

**Notes:**
- The free tier has rate limits (requests per minute and per day) that vary by model. Check the [Gemini API rate limits page](https://ai.google.dev/gemini-api/docs/rate-limits) if you hit `429` errors during testing.
- Treat this key like a password. Never commit it to version control (see step 3 below).

## 2. Clone the Project

```bash
git clone https://github.com/iMohamed-XD/LangChainProject
cd LangChainProject
```

## 3. Backend Setup

### 3.1 Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should contain at least:

```
fastapi
uvicorn
langchain
langchain-core
langchain-classic
langchain-community
langchain-google-genai
langchain-text-splitters
langchain-chroma
python-dotenv
jupyter
```

> `langchain_classic` is imported directly in `RAG.py` (`create_history_aware_retriever`, `create_retrieval_chain`, `create_stuff_documents_chain`) — make sure it's listed, since newer LangChain versions moved these chains out of `langchain.chains`.

### 3.3 Configure environment variables

Create a `.env` file in the **project root** (same level as `Backend/`):

```
GOOGLE_API_KEY=your_api_key_here
```

Add it to `.gitignore` if not already there:

```bash
echo ".env" >> .gitignore
```

`RAG.py` loads this via `load_dotenv()` and raises a clear error on startup if it's missing.

### 3.4 Add your source document

Place your handbook at:

```
Backend/app/data/company_handbook.md
```

It must use `#`, `##`, `###` headers — `MarkdownHeaderTextSplitter` relies on these to preserve section context in each chunk.

Example structure:

```markdown
# Company Handbook

## Time Off Policy

### Vacation Days
Full-time employees accrue 15 vacation days per year...

### Sick Leave
...

## Remote Work Policy
...
```

If you rename or relocate the file, update `DOC_PATH` in `load_document()` inside `RAG.py`.

### 3.5 Run the backend

From the project root, with the virtual environment active:

```bash
uvicorn Backend.app.main:app --reload --port 8000
```

On startup, the `lifespan` handler calls `build_rag_chain()` once — this loads the handbook, builds (or reuses) the Chroma index, and wires up the history-aware retrieval chain before the server starts accepting requests. The first run (or any run after the handbook content changes) will rebuild `chroma_db/`, which takes longer than subsequent runs since it re-embeds every chunk.

## 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

This starts the Vite dev server (default `http://localhost:5173`).

## 5. API Reference

### `POST /chat`

**Request body:**

```json
{
  "messages": [
    { "role": "user", "content": "How many vacation days do I get?" },
    { "role": "assistant", "content": "Full-time employees accrue 15 vacation days per year." },
    { "role": "user", "content": "What about part-time employees?" }
  ]
}
```

- `messages` must be non-empty.
- The **last** message must have `role: "user"` — it's treated as the current query; everything before it is chat history.
- History is truncated server-side to the last `MAX_HISTORY_TURNS = 6` turns (12 messages) before being passed to the chain.

**Response body:**

```json
{
  "answer": "Part-time employees accrue vacation days on a pro-rated basis...",
  "sources": [
    "(Vacation Days) Full-time employees accrue 15 vacation days per year..."
  ]
}
```

- If no chunks clear the retriever's relevance bar, `answer` will be `"I couldn't find that information in the handbook."` and `sources` will be an empty list.
- Each entry in `sources` is `(section_header) first 150 chars of chunk...`.

**Error responses:**

| Status | Cause |
|---|---|
| `400` | `messages` is empty, or the last message isn't `role: "user"` |
| `500` | Exception raised inside the RAG chain (e.g. Gemini API error, rate limit) |

## Configuration Reference

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `chunk_size` / `chunk_overlap` | `RAG.py` → `get_chunks()` | 800 / 100 | Character-level chunk sizing after header splitting |
| `search_type` | `RAG.py` → `retriever()` | `mmr` | Retrieval strategy (maximal marginal relevance) |
| `k` / `fetch_k` / `lambda_mult` | `RAG.py` → `retriever()` | 4 / 20 / 0.5 | Retriever result count, candidate pool size, diversity weight |
| `output_dimensionality` | `RAG.py` → `get_embeddings()` | 768 | Embedding vector size |
| `PERSIST_DIR` | `RAG.py` → `vectorstore_and_chain()` | `Backend/app/data/chroma_db` | Where the Chroma index is stored on disk |
| `MAX_HISTORY_TURNS` | `main.py` / `RAG.py` | 6 | Number of chat turns kept before truncation |
| `origins` (CORS) | `main.py` | `["http://localhost:3000"]` | Allowed frontend origin(s) |

## Optional: Standalone CLI Mode

`RAG.py` can still be run directly for local testing without the API/frontend:

```bash
python -m Backend.app.RAG
```

This starts the same interactive loop as before — type a question, get an answer plus numbered source chunks, type `exit`/`stop`/`quit` to end.

## Troubleshooting

**`GOOGLE_API_KEY not found`**
`.env` is missing, misnamed, or not readable from the process's working directory. Confirm it sits at the project root and that you're launching `uvicorn` from there.

**`company_handbook.md not found`**
Either the file isn't at `Backend/app/data/company_handbook.md`, or `DOC_PATH` in `RAG.py` doesn't match the actual filename.

**No answers / "I couldn't find that information in the handbook" for questions you know are covered**
The MMR retriever (`k=4, fetch_k=20`) can return chunks that don't fully cover the answer, or none at all if the query and content diverge semantically. Try rephrasing, or rebuild the vector store if the handbook content changed — a stale `chroma_db` is only invalidated by an MD5 hash check on the full document text, so any edit to `company_handbook.md` triggers an automatic rebuild on next startup. To force a rebuild manually, delete `Backend/app/data/chroma_db/`.

**CORS error in the browser console**
See the CORS note in section 4 — `origins` in `main.py` must match the frontend's actual dev server URL/port.

**Rate limit / `429` errors**
You've hit the Gemini API's free-tier request limits. Wait for the quota window to reset or check usage in [Google AI Studio](https://aistudio.google.com/).

**`ModuleNotFoundError`**
Confirm the virtual environment is activated and `pip install -r requirements.txt` completed without errors. If the error is specifically about `langchain_classic`, add it to `requirements.txt` (see section 3.2).

**`500: RAG chain error` from `/chat`**
Check the uvicorn logs for the underlying exception — usually a Gemini API error (bad key, rate limit) surfacing through `ainvoke()`.
