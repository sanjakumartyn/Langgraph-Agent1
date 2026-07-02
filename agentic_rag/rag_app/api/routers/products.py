from fastapi import APIRouter
from rag_app.tools import get_ai_products_from_mongodb

router = APIRouter()

@router.get("/products")
async def get_products_endpoint(industry: str = "", limit: int = 20):
    try:
        products = get_ai_products_from_mongodb(industry=industry, limit=limit)
        return {
            "success": True,
            "count": len(products),
            "products": products
        }
    except Exception as e:
        return {"success": False, "error": str(e), "products": []}
