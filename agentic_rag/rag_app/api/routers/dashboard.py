from fastapi import APIRouter
from rag_app.database import get_dynamic_dashboard_metrics

router = APIRouter()

@router.get("/internal")
async def internal_dashboard():
    return get_dynamic_dashboard_metrics()

@router.get("/dashboard", include_in_schema=False)
def dashboard_frontend():
    metrics = get_dynamic_dashboard_metrics()
    stats = metrics.get("stats", {})
    portfolio = metrics.get("portfolio", [])
    top_solutions = metrics.get("top_solutions", [])
    return {
        "success": True,
        "data": {
            "overview": {
                "totalProducts": stats.get("total_products", 0),
                "totalCaseStudies": stats.get("case_studies", 0),
                "activeOpportunities": stats.get("active_opportunities", 0),
                "proposalSuccessRate": stats.get("success_rate", 0),
            },
            "productPortfolio": [
                {"category": p["name"], "count": p["count"], "percentage": p.get("percentage", 0)}
                for p in portfolio
            ],
            "topSellingSolutions": [
                {"productName": s["name"], "opportunities": s["opportunities"], "revenueImpact": s["value"]}
                for s in top_solutions
            ],
            "recentCaseStudies": [],
            "opportunityPipeline": [],
            "crmActivity": {},
            "proposalAnalytics": {},
            "aiRecommendations": [
                "Focus on AI & Generative AI products for highest-growth accounts in the Technology sector.",
                "Leverage recent BFSI case studies to accelerate pipeline in banking.",
                "Prioritize CRM records with 'Evaluation' status — these are closest to closure.",
            ],
        }
    }

@router.get("/companydata/{collection:path}", include_in_schema=False)
async def get_company_collection(collection: str, limit: int = 50, offset: int = 0):
    from rag_app.database import mongo_db
    if mongo_db is None:
        return {"success": False, "error": "MongoDB not connected", "data": {"items": []}}

    collection_map = {
        "product details": "products",
        "case studies": "case studies",
        "opportunity history": "crm records",
        "crm records": "crm records",
        "proposals": "proposal documents",
        "meetings": "past meeting records",
    }
    actual_col = collection_map.get(collection.lower(), collection)

    try:
        col = mongo_db[actual_col]
        total = col.count_documents({})
        items_raw = list(col.find({}, {"_id": 0}).skip(offset).limit(limit))

        items = []
        if actual_col == "products":
            for p in items_raw:
                items.append({
                    "productId": p.get("serviceId", ""),
                    "productName": p.get("serviceName", ""),
                    "category": p.get("category", ""),
                    "description": p.get("description", ""),
                    "technology": p.get("technology", ""),
                    "application": p.get("useCase", ""),
                    "price": p.get("pricing"),
                    "unit": p.get("unit", ""),
                    "businessValue": p.get("businessValue", ""),
                    "targetIndustry": p.get("targetIndustry", ""),
                })
        elif actual_col == "crm records":
            for r in items_raw:
                items.append({
                    "opportunityId": r.get("crmId", ""),
                    "companyName": r.get("company", ""),
                    "industry": r.get("industry", ""),
                    "opportunityStatus": r.get("status", ""),
                    "salesStage": r.get("salesStage", r.get("interactionType", "")),
                    "dealValue": r.get("dealValue", 0),
                    "winProbability": r.get("probability", 70),
                    "assignedSalesRep": r.get("owner", ""),
                    "nextAction": r.get("nextAction", ""),
                    "annualRevenue": r.get("annualRevenue", ""),
                    "headquarters": r.get("headquarters", ""),
                    "contactName": r.get("contactName", ""),
                    "designation": r.get("designation", ""),
                    "associatedService": r.get("associatedService", ""),
                    "interactionDate": r.get("interactionDate", ""),
                    "interactionType": r.get("interactionType", ""),
                    "painPoint": r.get("painPoint", ""),
                    "opportunityStory": r.get("opportunityStory", ""),
                    "whyTheyApproached": r.get("whyTheyApproached", ""),
                    "meetingOutcome": r.get("meetingOutcome", ""),
                    "aiOpportunityBrief": r.get("aiOpportunityBrief", ""),
                })
        else:
            items = items_raw

        pages = (total + limit - 1) // limit if limit > 0 else 1
        return {
            "success": True,
            "data": {
                "items": items,
                "pagination": {"total": total, "limit": limit, "offset": offset, "pages": pages}
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e), "data": {"items": []}}
