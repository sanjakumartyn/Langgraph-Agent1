import json
from fastapi import APIRouter
from rag_app.agent import run_deal_coach_chat
from rag_app.database import get_neon_connection
from rag_app.api.schemas import ChatMessageRequest, SqlRequest, DealCoachRequest
from rag_app.main import _analysis_cache, _latest_by_company

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(payload: ChatMessageRequest):
    try:
        dashboard_data = None
        session_id = payload.session_id
        if session_id and session_id in _analysis_cache:
            dashboard_data = _analysis_cache[session_id].get("dashboard_data")
        elif payload.company:
            cached = _latest_by_company.get(payload.company.lower().strip())
            if cached:
                dashboard_data = cached.get("dashboard_data")
        
        response = await run_deal_coach_chat(
            message=payload.query,
            company=payload.company,
            dashboard_data=dashboard_data,
        )
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/company-analysis/deal-coach")
async def deal_coach_frontend(payload: DealCoachRequest):
    company = (payload.company_name or payload.company or "Unknown").strip()
    message = (payload.message or payload.question or "").strip()
    if not message:
        return {"success": False, "error": "No message provided"}

    dashboard_data = None
    cached = _latest_by_company.get(company.lower().strip())
    if cached:
        dashboard_data = cached.get("dashboard_data")
    elif payload.analysis_context:
        ctx = payload.analysis_context
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except Exception:
                ctx = {}
        dashboard_data = ctx.get("raw") or ctx

    try:
        response = await run_deal_coach_chat(
            message=message,
            company=company,
            dashboard_data=dashboard_data,
        )
        return {
            "success": True,
            "data": {"answer": response, "synthesized_answer": response}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/sql")
async def execute_sql_endpoint(payload: SqlRequest):
    conn = get_neon_connection(register=False)
    if not conn:
        return {"success": False, "error": "Database connection failed"}
    try:
        with conn.cursor() as cur:
            cur.execute(payload.query)
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                formatted_rows = [[str(val) if val is not None else None for val in row] for row in rows]
                conn.commit()
                return {"success": True, "columns": columns, "rows": formatted_rows}
            else:
                conn.commit()
                return {"success": True, "message": f"Query executed. Rows affected: {cur.rowcount}"}
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        conn.close()
