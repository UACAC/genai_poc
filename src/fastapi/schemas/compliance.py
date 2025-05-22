# schemas/compliance.py

from typing import List, Optional
from pydantic import BaseModel, Field

class ComplianceCheckRequest(BaseModel):
    data_sample: str
    agent_ids: List[int]

class RAGCheckRequest(BaseModel):
    query_text: str
    collection_name: str
    agent_ids: List[int]

class RAGDebateSequenceRequest(BaseModel):
    session_id: Optional[str] = None
    agent_ids: List[int]
    query_text: str
    collection_name: str

class CreateComplianceAgentRequest(BaseModel):
    name: str
    model_name: str
    system_prompt: str
    user_prompt_template: str

class ComplianceResultSchema(BaseModel):
    compliant: bool = Field(..., description="Whether the content is compliant")
    reason: str = Field(..., description="Explanation for the compliance decision")
    confidence: Optional[float] = Field(default=None, description="Confidence score")