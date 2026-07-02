import json
import uuid
from fastapi import APIRouter, HTTPException
from rag_app.agent import run_rag_agent
from rag_app.tools import ingest_company_to_neon
from rag_app.api.schemas import ChatRequest, IngestRequest, IngestResponse, CompanyIntelligenceRequest
from rag_app.main import _analysis_cache, _latest_by_company

router = APIRouter()

@router.post("/analyze")
async def analyze_company(payload: ChatRequest):
    session_id = payload.session_id or str(uuid.uuid4())
    try:
        result = await run_rag_agent(payload.query, fast_mode=payload.fast_mode)
        raw_answer = result.get("final_answer", "{}")
        
        try:
            dashboard_data = json.loads(raw_answer)
        except json.JSONDecodeError:
            dashboard_data = {
                "company": payload.query,
                "overview": f"Analysis for {payload.query} is being processed. Please retry.",
                "strategic_fit": 70,
                "needs_prediction": [
                    "AI-powered process automation",
                    "Predictive analytics for business decisions",
                    "Enterprise knowledge management"
                ],
                "meeting_prep": {
                    "priorities": ["Digital transformation"],
                    "growth_initiatives": ["AI integration"],
                    "risks": ["Budget constraints"],
                    "buying_signals": ["Exploring AI vendors"],
                    "objections": ["Implementation timeline"],
                    "stakeholders": ["CTO", "VP IT"]
                },
                "solution_mapping": []
            }

        company_name = dashboard_data.get("company", payload.query)
        ai_products = result.get("ai_products", [])
        cache_entry = {
            "dashboard_data": dashboard_data,
            "ai_products": ai_products,
            "company": company_name,
            "steps_taken": result.get("steps_taken", []),
            "sources_used": result.get("sources_used", []),
        }
        _analysis_cache[session_id] = cache_entry
        _latest_by_company[company_name.lower().strip()] = cache_entry

        return {
            "success": True,
            "data": dashboard_data,
            "session_id": session_id,
            "steps_taken": result.get("steps_taken", []),
            "sources_used": result.get("sources_used", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/company/intelligence")
async def company_intelligence(payload: CompanyIntelligenceRequest):
    company_name = payload.company_name.strip()
    session_id = str(uuid.uuid4())

    try:
        result = await run_rag_agent(company_name, fast_mode=payload.fast_mode)
        raw_answer = result.get("final_answer", "{}")

        try:
            dashboard_data = json.loads(raw_answer)
        except json.JSONDecodeError:
            dashboard_data = {
                "company": company_name,
                "overview": f"Analysis for {company_name} is in progress.",
                "strategic_fit": 70,
                "needs_prediction": [],
                "meeting_prep": {},
                "solution_mapping": []
            }

        ai_products = result.get("ai_products", [])
        meeting_prep = dashboard_data.get("meeting_prep", {})
        internal_ctx = result.get("internal_context", {})
        crm_records = internal_ctx.get("crm_records", []) if internal_ctx else []
        industry = crm_records[0].get("industry", "Technology") if crm_records else "Technology"

        front_data = {
            "company": dashboard_data.get("company", company_name),
            "company_name": dashboard_data.get("company", company_name),
            "industry": industry,
            "company_summary": dashboard_data.get("overview", ""),
            "intelligence_overview": dashboard_data.get("overview", ""),
            "confidence_score": dashboard_data.get("strategic_fit", 75),
            "strategic_fit_score": dashboard_data.get("strategic_fit", 75),
            "products_or_services": [],
            "business_priorities": meeting_prep.get("priorities", []),
            "pain_points": meeting_prep.get("risks", []),
            "technology_signals": meeting_prep.get("buying_signals", []),
            "market_news": [],
            "company_history": [],
            "source_evidence": result.get("sources_used", []),
            "ai_needs_prediction": dashboard_data.get("needs_prediction", []),
            "solution_mapping": dashboard_data.get("solution_mapping", []),
            "meeting_preparation": {
                "suggested_discussion_points": meeting_prep.get("priorities", []),
                "potential_objections": meeting_prep.get("objections", []),
                "relevant_case_studies": [f"{cs.get('title', '')} ({cs.get('client', '')})" for cs in internal_ctx.get("related_case_studies", [])] if internal_ctx.get("related_case_studies") else [],
                "stakeholders_to_target": meeting_prep.get("stakeholders", []),
            },
            "executive_qbr": {
                "key_business_priorities": meeting_prep.get("priorities", []),
                "growth_initiatives": meeting_prep.get("growth_initiatives", []),
                "risk_factors": meeting_prep.get("risks", []),
                "buying_signals": meeting_prep.get("buying_signals", []),
            },
            "raw": dashboard_data,
            "steps_taken": result.get("steps_taken", []),
            "sources_used": result.get("sources_used", []),
        }

        cache_entry = {
            "dashboard_data": dashboard_data,
            "ai_products": ai_products,
            "company": company_name,
        }
        _analysis_cache[session_id] = cache_entry
        _latest_by_company[company_name.lower().strip()] = cache_entry

        return {"success": True, "session_id": session_id, "data": front_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest(payload: IngestRequest):
    try:
        report = await ingest_company_to_neon(
            company_name=payload.company_name,
            company_website=payload.company_website
        )
        return {"success": True, "data": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
