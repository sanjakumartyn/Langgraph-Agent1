import os
import asyncio
import tempfile
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from rag_app.pdf_generator import generate_company_proposal_pdf
from rag_app.tools import get_ai_products_from_mongodb
from rag_app.api.schemas import GenerateRequest
from rag_app.main import _analysis_cache, _latest_by_company, FRONTEND_DIR

router = APIRouter()

@router.post("/generate")
async def generate_endpoint(payload: GenerateRequest):
    company_name = payload.company.strip()
    if not company_name:
        return {"success": False, "error": "Company name required"}
        
    dashboard_data = None
    ai_products = []
    
    if payload.session_id and payload.session_id in _analysis_cache:
        cached = _analysis_cache[payload.session_id]
        dashboard_data = cached.get("dashboard_data")
        ai_products = cached.get("ai_products", [])
    elif company_name.lower().strip() in _latest_by_company:
        cached = _latest_by_company[company_name.lower().strip()]
        dashboard_data = cached.get("dashboard_data")
        ai_products = cached.get("ai_products", [])
    
    if not ai_products:
        ai_products = get_ai_products_from_mongodb(limit=8)
    
    safe_name = company_name.replace(" ", "_").replace("/", "_")
    filename = f"Proposal_{safe_name}_AI_Solutions.pdf"
    file_path = FRONTEND_DIR / filename
    
    try:
        await asyncio.to_thread(
            generate_company_proposal_pdf,
            company_name,
            str(file_path),
            dashboard_data,
            ai_products,
        )
        return {
            "success": True,
            "filename": filename,
            "url": f"/static/{filename}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/generate-document")
async def generate_document_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    company = body.get("company") or "Company"
    
    return {
        "status": "success",
        "downloads": {
            "pdf": f"/api/company/report/{company}/pdf",
            "docx": f"/api/company/report/{company}/pdf",
            "xlsx": f"/api/company/report/{company}/pdf"
        }
    }

@router.get("/company/report/{company_name}/pdf", include_in_schema=False)
async def download_pdf(company_name: str):
    cache_entry = _latest_by_company.get(company_name.lower().strip())
    dashboard_data = cache_entry.get("dashboard_data") if cache_entry else None
    ai_products = cache_entry.get("ai_products") if cache_entry else None
    
    tmp_path = os.path.join(tempfile.gettempdir(), f"{company_name.replace(' ', '_')}_proposal.pdf")
    generate_company_proposal_pdf(company_name, tmp_path, dashboard_data, ai_products)
    
    return FileResponse(
        tmp_path,
        media_type="application/pdf",
        filename=f"{company_name.replace(' ', '_')}_Proposal.pdf"
    )
