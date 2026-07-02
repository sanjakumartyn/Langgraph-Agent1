from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., examples=["Analyze Amazon for AI opportunities"])
    session_id: Optional[str] = Field(default=None)
    fast_mode: bool = Field(default=True)

class ChatMessageRequest(BaseModel):
    query: str
    company: str
    session_id: Optional[str] = Field(default=None)

class IngestRequest(BaseModel):
    company_name: str = Field(..., examples=["Tesla"])
    company_website: Optional[str] = Field(default=None)

class IngestResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class GenerateRequest(BaseModel):
    company: str
    session_id: Optional[str] = Field(default=None)

class SqlRequest(BaseModel):
    query: str

class CompanyIntelligenceRequest(BaseModel):
    company_name: str
    company_website: Optional[str] = None
    force_refresh: bool = False
    fast_mode: bool = True

class DealCoachRequest(BaseModel):
    company_name: Optional[str] = None
    company: Optional[str] = None
    account_id: Optional[str] = None
    message: Optional[str] = None
    question: Optional[str] = None
    analysis_context: Optional[Any] = None
