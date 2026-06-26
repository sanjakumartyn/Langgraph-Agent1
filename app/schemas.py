from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CompanyRequest(BaseModel):
    company_name: str = Field(..., example="Asian Paints")
    company_website: Optional[str] = Field(
        default=None,
        example="https://www.asianpaints.com"
    )


class CompanyResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    errors: List[str]