from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import hashlib
import shutil


def llm():
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found. Check that a .env file exists in the "
            "working directory and contains GOOGLE_API_KEY=<your_key>."
        )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
    return llm


def load_document():
    DOC_PATH = "../data/company_handbook.md"
    if not os.path.exists(DOC_PATH):
        raise FileNotFoundError(
            f"{DOC_PATH} not found in {os.getcwd()}. "
            "Confirm the file is in the working directory or update DOC_PATH."
        )

    with open(DOC_PATH, encoding="utf-8") as f:
        text = f.read()

    document = [Document(page_content=text, metadata={"source": DOC_PATH})]

    if not document or not document[0].page_content.strip():
        raise ValueError(f"{DOC_PATH} loaded but contains no extractable text.")

    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_chunks = md_splitter.split_text(document[0].page_content)

    if not md_chunks:
        raise ValueError(
            f"{DOC_PATH} has no markdown headers matching {[h[0] for h in headers_to_split_on]}. "
            "Check the file format or headers_to_split_on."
        )
    return document, md_chunks


def get_chunks(md_chunks):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(md_chunks)

    def tag_section(chunk):
        chunk.metadata["section"] = (
            chunk.metadata.get("h3")
            or chunk.metadata.get("h2")
            or chunk.metadata.get("h1")
            or "Unknown"
        )
        return chunk

    chunks = [tag_section(c) for c in chunks]

    # print(f"Number of chunks: {len(chunks)}")
    # lengths = [len(c.page_content) for c in chunks]
    # print(f"Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.0f}")
    return chunks


def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768
    )
    return embeddings


def prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are answering questions about the company handbook.
        Only answer using the retrieved context.
        If the answer is not found, say:
        "I couldn't find that information in the handbook."
        Never invent information.
        Cite the relevant sections whenever possible.:\n\n{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def vectorstore_and_chain(document, embeddings, chunks, qa_prompt, llm):
    PERSIST_DIR = "../data/chroma_db"
    DOC_HASH_FILE = os.path.join(PERSIST_DIR, ".doc_hash")

    current_hash = hashlib.md5(document[0].page_content.encode("utf-8")).hexdigest()

    def _stored_hash():
        if os.path.exists(DOC_HASH_FILE):
            with open(DOC_HASH_FILE) as f:
                return f.read().strip()
        return None

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR) and _stored_hash() == current_hash:
        docsearch = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        docsearch = Chroma.from_documents(
            chunks, embeddings, persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )
        with open(DOC_HASH_FILE, "w") as f:
            f.write(current_hash)
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    return docsearch, combine_docs_chain


def retriever(docsearch, combine_docs_chain, llm):
    retriever = docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
            "rephrase the question to be a standalone question. "
            "Do NOT answer it, just reformulate it if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    return rag_chain


def main():
    llm_instance = llm()
    document, md_chunks = load_document()
    chunks = get_chunks(md_chunks)
    embeddings = get_embeddings()
    qa_prompt = prompt()
    docsearch, combine_docs_chain = vectorstore_and_chain(document, embeddings, chunks, qa_prompt, llm_instance)
    rag_chain = retriever(docsearch, combine_docs_chain, llm_instance)

    chat_history = []
    MAX_HISTORY_TURNS = 6

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() in ("exit", "stop", "quit"):
            break

        try:
            response = rag_chain.invoke({"input": query, "chat_history": chat_history})
        except Exception as e:
            print(f"Error during retrieval/generation: {e}")
            continue

        if not response["context"]:
            print("No relevant content found in the handbook for this query.")
            print("-" * 40)
            continue

        print(response["answer"])
        chat_history.extend([HumanMessage(content=query), AIMessage(content=response["answer"])])
        if len(chat_history) > MAX_HISTORY_TURNS * 2:
            chat_history = chat_history[-MAX_HISTORY_TURNS * 2:]
        print("\nSources used:")
        for i, doc in enumerate(response["context"], 1):
            preview = doc.page_content[:150].replace("\n", " ")
            section = doc.metadata.get("section", "Unknown")
            print(f"[{i}] ({section}) {preview}...")
        print("-" * 40)


if __name__ == "__main__":
    main()


def build_rag_chain():
    """Run the one-time setup (LLM, embeddings, vectorstore) and return
    a ready-to-invoke rag_chain. Call this ONCE, at process startup."""
    llm_instance = llm()
    document, md_chunks = load_document()
    chunks = get_chunks(md_chunks)
    embeddings = get_embeddings()
    qa_prompt = prompt()
    docsearch, combine_docs_chain = vectorstore_and_chain(
        document, embeddings, chunks, qa_prompt, llm_instance
    )
    rag_chain = retriever(docsearch, combine_docs_chain, llm_instance)
    return rag_chain
