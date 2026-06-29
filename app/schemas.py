from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CompanyRequest(BaseModel):
    company_name: str = Field(..., example="Asian Paints")
    company_website: Optional[str] = Field(
        default=None,
        example="https://www.asianpaints.com"
    )
    force_refresh: Optional[bool] = Field(
        default=False,
        example=False
    )


class CompanyResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    errors: List[str]


class SearchRequest(BaseModel):
    query: str = Field(..., example="AI and cloud technology priorities")
    limit: Optional[int] = Field(default=5, example=5)


class SearchResult(BaseModel):
    company_name: str
    chunk_text: str
    similarity: float


class SearchResponse(BaseModel):
    success: bool
    results: List[SearchResult]
    errors: List[str]