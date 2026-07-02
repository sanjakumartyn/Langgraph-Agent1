import sys
import asyncio
import os
from typing import Dict, Any
from pathlib import Path

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ── In-memory cache for last analysis result (for PDF generation) ─────────────
# Maps session_id -> {dashboard_data, ai_products, company}
_analysis_cache: Dict[str, Dict[str, Any]] = {}
_latest_by_company: Dict[str, Dict[str, Any]] = {}  # company_name -> last result

app = FastAPI(
    title="SalesIntel AI — Agentic RAG API",
    description="FastAPI + LangGraph Agentic RAG with real AI product matching from MongoDB.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

@app.get("/", response_class=FileResponse)
def serve_ui():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"success": True, "message": "SalesIntel AI Agentic RAG API v3.0", "docs": "/docs"})

@app.get("/health")
def health_check():
    return {
        "success": True,
        "message": "SalesIntel AI Agentic RAG Service is running",
        "version": "3.0.0"
    }

# ── Router Imports ────────────────────────────────────────────────────────────
# Imported here to avoid circular dependencies with caches defined above
from rag_app.api.routers import analysis, chat, dashboard, documents, products

# Include routers without prefix since the original routes define their own prefixes
app.include_router(analysis.router, tags=["Analysis"], prefix="/api")
app.include_router(analysis.router, tags=["Analysis"], prefix="")  # For un-prefixed routes like /company/intelligence

app.include_router(chat.router, tags=["Deal Coach"], prefix="/api")
app.include_router(chat.router, tags=["Deal Coach"], prefix="") # For un-prefixed routes like /company-analysis/deal-coach

app.include_router(dashboard.router, tags=["Dashboard"], prefix="/api")
app.include_router(dashboard.router, tags=["Dashboard"], prefix="")

app.include_router(documents.router, tags=["Documents"], prefix="/api")
app.include_router(documents.router, tags=["Documents"], prefix="")

app.include_router(products.router, tags=["Products"], prefix="/api")
