from fastapi import FastAPI, HTTPException

from app.schemas import CompanyRequest, CompanyResponse
from app.graph import run_company_agent

app = FastAPI(
    title="Company Intelligence API",
    description="LangGraph-powered company intelligence web agent",
    version="1.0.0"
)


@app.get("/")
def health_check():
    return {
        "success": True,
        "message": "Company Intelligence API is running"
    }


@app.post("/company/intelligence", response_model=CompanyResponse)
def company_intelligence(payload: CompanyRequest):
    try:
        result = run_company_agent(
            company_name=payload.company_name,
            company_website=payload.company_website
        )

        return {
            "success": True,
            "data": result.get("final_result", {}),
            "errors": result.get("errors", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )