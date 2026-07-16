# RAG Application — Company Handbook Q&A

A retrieval-augmented generation (RAG) system that answers questions about a company handbook using Google's Gemini models, LangChain, and a local Chroma vector store. Supports multi-turn conversations via history-aware query reformulation.

## Architecture

```
company_handbook.md
        │
        ▼
MarkdownHeaderTextSplitter   (splits on #, ##, ### and attaches header metadata)
        │
        ▼
RecursiveCharacterTextSplitter   (chunk_size=800, chunk_overlap=100)
        │
        ▼
GoogleGenerativeAIEmbeddings   (embeds each chunk)
        │
        ▼
Chroma vector store   (persisted to ./chroma_db)
        │
        ▼
history_aware_retriever   ─── rephrases follow-up questions using chat history
        │
        ▼
create_retrieval_chain + create_stuff_documents_chain   ─── retrieves top-k chunks, feeds them to Gemini as context
        │
        ▼
Answer + source chunks printed to console
```

**Models used:**
- LLM: `gemini-2.5-flash`
- Embeddings: `gemini-embedding-2-preview` (768 dimensions)

## Prerequisites

- Python 3.12 (the notebook environment was built and tested on 3.12.10)
- pip or [Poetry](https://python-poetry.org/) for dependency management
- A Google account (for the Gemini API key)
- Jupyter (to run the `.ipynb` notebook), or an editor with notebook support (VS Code, JupyterLab)

## 1. Get a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in with your Google account.
3. Click **Create API key**.
4. Select an existing Google Cloud project, or let AI Studio create one for you.
5. Copy the generated key — you won't be able to view it again from this screen, though you can revoke and regenerate it if lost.

**Notes:**
- The free tier has rate limits (requests per minute and per day) that vary by model. Check the [Gemini API rate limits page](https://ai.google.dev/gemini-api/docs/rate-limits) if you hit `429` errors during testing.
- Treat this key like a password. Never commit it to version control (see step 4 below).

## 2. Clone / Download the Project

```bash
git clone <your-repo-url>
cd <project-directory>
```

If you don't have a repo yet, just place `RAG_Application.ipynb` in a project folder and work from there.

## 3. Set Up a Virtual Environment

**Using venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

**Using Poetry (alternative):**

```bash
poetry init --python "^3.12"
poetry shell
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
langchain
langchain-core
langchain-community
langchain-google-genai
langchain-text-splitters
langchain-chroma
python-dotenv
jupyter
```

## 5. Configure Environment Variables

Create a `.env` file in the project root (same directory as the notebook):

```
GOOGLE_API_KEY=your_api_key_here
```

Add `.env` to `.gitignore` if this project is under version control:

```bash
echo ".env" >> .gitignore
```

The notebook loads this key via `load_dotenv()` and will raise a clear error on startup if it's missing.

## 6. Add Your Source Document

The notebook expects a markdown file named `company_handbook.md` in the working directory, structured with `#`, `##`, and `###` headers — the splitter uses these headers to preserve section context in each chunk.

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

If your document has a different filename, update `DOC_PATH` in the "Load the Document" cell.

## 7. Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open `RAG_Application.ipynb` and run all cells in order:

1. **Install/Import libraries**
2. **Load the LLM** — validates `GOOGLE_API_KEY` and initializes `gemini-2.5-flash`
3. **Load the Document** — reads and validates `company_handbook.md`
4. **Split the Document** — header-based split, then character-based split, then section tagging
5. **Embed the Document** — initializes the embedding model
6. **Prompt** — defines the QA prompt template
7. **Vector Store** — builds or loads the persisted Chroma index at `./chroma_db`
8. **Retriever** — builds the history-aware retriever and full RAG chain
9. **Invoke the RAG system** — starts an interactive query loop in the notebook's input prompt

## 8. Using the Chat Loop

Once the final cell is running, you'll see:

```
Enter your query (or 'exit' to quit):
```

Type a question about the handbook and press Enter. The system will:
- Reformulate your question using prior chat history (if any)
- Retrieve the most relevant chunks from the vector store
- Generate an answer grounded in those chunks
- Print the answer followed by a numbered preview of the source chunks used

Type `exit`, `stop`, or `quit` to end the session.

## Configuration Reference

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `chunk_size` / `chunk_overlap` | Split cell | 800 / 100 | Character-level chunk sizing after header splitting |
| `score_threshold` | Retriever cell | 0.75 | Minimum relevance score for a chunk to be retrieved |
| `k` | Retriever cell | 4 | Max number of chunks retrieved per query |
| `output_dimensionality` | Embedding cell | 768 | Embedding vector size |
| `PERSIST_DIR` | Vector Store cell | `./chroma_db` | Where the Chroma index is stored on disk |

## Troubleshooting

**`GOOGLE_API_KEY not found`**
`.env` is missing, misnamed, or not in the working directory the notebook was launched from. Confirm with `os.getcwd()` in a scratch cell.

**`company_handbook.md not found`**
Either the file isn't in the working directory, or `DOC_PATH` doesn't match the actual filename.

**No answers / "no relevant content" for questions you know are covered**
The `score_threshold=0.75` retriever can return zero chunks if nothing clears the bar. Try lowering the threshold or rebuilding the vector store if the handbook content changed after the index was first created — a stale `./chroma_db` will keep serving old embeddings. Delete the `chroma_db` folder to force a full rebuild.

**Rate limit / `429` errors**
You've hit the Gemini API's free-tier request limits. Wait for the quota window to reset or check your usage in [Google AI Studio](https://aistudio.google.com/).

**`ModuleNotFoundError`**
Confirm your virtual environment is activated and `pip install -r requirements.txt` completed without errors.

## Project Structure

```
.
├── RAG_Application.ipynb
├── requirements.txt
├── .env                  # not committed — holds GOOGLE_API_KEY
├── company_handbook.md   # your source document
├── chroma_db/            # generated on first run, persisted vector store
└── README.md
```
