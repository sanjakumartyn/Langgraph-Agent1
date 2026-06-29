import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import json
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware
from .agent import run_rag_agent
from .database import redis_client
from .tools import ingest_company_to_supabase

app = FastAPI(
    title="Agentic RAG API",
    description="FastAPI + LangGraph conversational Agentic RAG service powered by Supabase and Upstash Redis.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., example="What are Zoho's main business products?")
    session_id: Optional[str] = Field(default=None, example="user-session-123")


class ChatResponse(BaseModel):
    success: bool
    answer: str
    steps: List[str]
    session_id: str


class IngestRequest(BaseModel):
    company_name: str = Field(..., example="Zoho")
    company_website: Optional[str] = Field(default=None, example="https://www.zoho.com")


class IngestResponse(BaseModel):
    success: bool
    data: dict


@app.get("/")
def health_check():
    return {
        "success": True,
        "message": "Agentic RAG Service is running"
    }


@app.post("/rag/chat", response_model=ChatResponse)
async def rag_chat(payload: ChatRequest):
    session_id = payload.session_id or str(uuid.uuid4())
    history_key = f"chat_history:{session_id}"
    
    # 1. Retrieve chat history from Upstash Redis if session exists
    history = []
    if redis_client:
        try:
            history_raw = redis_client.get(history_key)
            if history_raw:
                if isinstance(history_raw, bytes):
                    history_raw = history_raw.decode('utf-8')
                history = json.loads(history_raw)
        except Exception as e:
            print(f"Failed to read from Upstash Redis: {str(e)}")
            
    # Include history in query if present to maintain conversational state
    contextual_query = payload.query
    if history:
        # Format history context
        history_context = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in history[-4:])
        contextual_query = (
            f"Here is the history of our conversation:\n{history_context}\n\n"
            f"Now, answer the new user query: {payload.query}"
        )

    try:
        # 2. Run the Agentic RAG Graph
        result = await run_rag_agent(contextual_query)
        answer = result.get("final_answer", "No answer generated.")
        steps = result.get("steps_taken", [])

        # 3. Save new user & assistant message to Upstash Redis
        if redis_client:
            try:
                history.append({"role": "user", "content": payload.query})
                history.append({"role": "assistant", "content": answer})
                # Cache conversation for 24 hours (86400 seconds)
                redis_client.set(history_key, json.dumps(history), ex=86400)
            except Exception as e:
                print(f"Failed to save to Upstash Redis: {str(e)}")

        return {
            "success": True,
            "answer": answer,
            "steps": steps,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest(payload: IngestRequest):
    try:
        report = await ingest_company_to_supabase(
            company_name=payload.company_name,
            company_website=payload.company_website
        )
        return {
            "success": True,
            "data": report
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
