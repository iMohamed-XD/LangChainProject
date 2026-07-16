from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Literal
from langchain_core.messages import HumanMessage, AIMessage

from Backend.app.RAG import build_rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, before the app starts accepting requests.
    app.state.rag_chain = build_rag_chain()
    yield
    # (optional) cleanup on shutdown goes here


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Chat(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


MAX_HISTORY_TURNS = 6


@app.post("/chat", response_model=ChatResponse)
async def chat(chat: Chat):
    if not chat.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    *history_msgs, last = chat.messages
    if last.role != "user":
        raise HTTPException(status_code=400, detail="last message must have role='user'")

    # Convert frontend history into LangChain message objects
    chat_history = []
    for m in history_msgs:
        chat_history.append(
            HumanMessage(content=m.content) if m.role == "user"
            else AIMessage(content=m.content)
        )
    if len(chat_history) > MAX_HISTORY_TURNS * 2:
        chat_history = chat_history[-MAX_HISTORY_TURNS * 2:]

    try:
        response = await app.state.rag_chain.ainvoke(
            {"input": last.content, "chat_history": chat_history}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG chain error: {e}")

    if not response["context"]:
        return ChatResponse(answer="I couldn't find that information in the handbook.", sources=[])

    sources = [
        f"({doc.metadata.get('section', 'Unknown')}) {doc.page_content[:150].replace(chr(10), ' ')}..."
        for doc in response["context"]
    ]
    return ChatResponse(answer=response["answer"], sources=sources)
