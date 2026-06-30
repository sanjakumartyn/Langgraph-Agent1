import sys
import asyncio
import os

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import json
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware
from rag_app.agent import run_rag_agent, run_deal_coach_chat
from rag_app.database import get_dynamic_dashboard_metrics, get_neon_connection
from rag_app.pdf_generator import generate_company_proposal_pdf
from rag_app.tools import ingest_company_to_neon

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title="Sales Intelligence — Agentic RAG API",
    description="FastAPI + LangGraph Agentic RAG service with parallel data aggregation, query understanding, and citation-aware synthesis.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class ChatRequest(BaseModel):
    query: str = Field(..., examples=["What are Zoho's main business products?"])
    session_id: Optional[str] = Field(default=None, examples=["user-session-123"])


class ChatResponse(BaseModel):
    success: bool
    answer: str
    final_answer: str
    steps: List[str]
    steps_taken: List[str]
    session_id: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    sources_used: List[Dict[str, str]] = Field(default_factory=list)


class SqlRequest(BaseModel):
    query: str

@app.post("/api/sql")
async def execute_sql_endpoint(payload: SqlRequest):
    """Executes arbitrary SQL on the Neon database (for internal dashboard use)."""
    conn = get_neon_connection(register=False)
    if not conn:
        return {"success": False, "error": "Database connection failed"}
    try:
        with conn.cursor() as cur:
            cur.execute(payload.query)
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                # Convert datetime, uuid, or vector types to strings for JSON serialization if needed
                formatted_rows = []
                for row in rows:
                    formatted_rows.append([str(val) if val is not None else None for val in row])
                conn.commit()
                return {"success": True, "columns": columns, "rows": formatted_rows}
            else:
                conn.commit()
                return {"success": True, "message": f"Query executed successfully. Rows affected: {cur.rowcount}"}
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


class IngestRequest(BaseModel):
    company_name: str = Field(..., examples=["Zoho"])
    company_website: Optional[str] = Field(default=None, examples=["https://www.zoho.com"])


class IngestResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


@app.get("/", response_class=FileResponse)
def serve_ui():
    """Serve the Web Chat UI."""
    if FRONTEND_DIR.exists():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    return {"success": True, "message": "Sales Intelligence Agentic RAG API v2.0"}


@app.get("/health")
def health_check():
    return {
        "success": True,
        "message": "Agentic RAG Service is running",
        "version": "2.0.0"
    }


@app.post("/api/analyze")
async def analyze_company(payload: ChatRequest):
    """Generates a complete SalesIntel dashboard JSON for a given company."""
    session_id = payload.session_id or str(uuid.uuid4())
    
    try:
        # 1. Run the Agentic RAG Graph
        result = await run_rag_agent(payload.query)
        answer = result.get("final_answer", "{}")
        
        # 2. Parse the strict JSON output from the synthesizer
        try:
            dashboard_data = json.loads(answer)
        except json.JSONDecodeError:
            dashboard_data = {
                "company": payload.query,
                "overview": f"Error parsing JSON from agent: {answer}",
                "strategic_fit": 0,
                "needs_prediction": [],
                "meeting_prep": {},
                "solution_mapping": []
            }

        return {
            "success": True,
            "data": dashboard_data,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatMessageRequest(BaseModel):
    query: str
    company: str

@app.post("/api/chat")
async def chat_endpoint(payload: ChatMessageRequest):
    """Stateless chat endpoint for the Deal Coach."""
    try:
        response = await run_deal_coach_chat(payload.query, payload.company)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}


class GenerateRequest(BaseModel):
    company: str

@app.post("/api/generate")
async def generate_endpoint(payload: GenerateRequest):
    """Generates an actual PDF proposal for the given company."""
    company_name = payload.company.strip()
    if not company_name:
        return {"success": False, "error": "Company name required"}
        
    safe_name = company_name.replace(" ", "_")
    filename = f"Proposal_{safe_name}_Q3.pdf"
    
    # Save directly into the frontend directory so the /static/ mount serves it
    file_path = FRONTEND_DIR / filename
    
    try:
        # Generate the PDF synchronously in a thread
        await asyncio.to_thread(
            generate_company_proposal_pdf,
            company_name, 
            str(file_path)
        )
        return {
            "success": True, 
            "filename": filename,
            "url": f"/static/{filename}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.get("/api/internal")
async def internal_dashboard():
    """Returns dynamic dashboard data from MongoDB aggregations."""
    return get_dynamic_dashboard_metrics()


@app.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest(payload: IngestRequest):
    try:
        report = await ingest_company_to_neon(
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
