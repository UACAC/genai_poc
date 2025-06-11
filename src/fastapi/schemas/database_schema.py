from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Agent name")
    model_name: str = Field(..., description="Model to use for this agent")
    system_prompt: str = Field(..., min_length=10, description="System prompt defining the agent's role")
    user_prompt_template: str = Field(..., min_length=10, description="User prompt template with {data_sample} placeholder")

class UpdateAgentRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=200, description="Updated agent name")
    model_name: Optional[str] = Field(None, description="Updated model name")
    system_prompt: Optional[str] = Field(None, min_length=10, description="Updated system prompt")
    user_prompt_template: Optional[str] = Field(None, min_length=10, description="Updated user prompt template")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Updated temperature")
    max_tokens: Optional[int] = Field(None, ge=100, le=4000, description="Updated max tokens")
    is_active: Optional[bool] = Field(None, description="Whether agent is active")
    
    @field_validator('user_prompt_template')
    def validate_prompt_template(cls, v):
        if v is not None and '{data_sample}' not in v:
            raise ValueError('User prompt template must contain {data_sample} placeholder')
        return v

class UpdateAgentResponse(BaseModel):
    message: str
    agent_id: int
    agent_name: str
    updated_fields: List[str]

class ComplianceCheckRequest(BaseModel):
    data_sample: str = Field(..., min_length=1, description="Legal content to analyze")
    agent_ids: List[int] = Field(..., min_items=1, description="List of agent IDs to use for analysis")

class RAGCheckRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="Legal query for RAG analysis")
    collection_name: str = Field(..., description="ChromaDB collection name")
    agent_ids: List[int] = Field(..., min_items=1, description="List of agent IDs to use for RAG analysis")

class RAGDebateSequenceRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="Legal content for multi-agent debate")
    collection_name: str = Field(..., description="ChromaDB collection name")
    agent_ids: List[int] = Field(..., min_items=1, description="List of agent IDs for debate sequence")
    session_id: Optional[str] = Field(None, description="Optional session ID for continuing a debate")

# Response models for better API documentation
class AgentResponse(BaseModel):
    agent_id: int
    agent_name: str
    model_name: str
    response: str
    processing_time: Optional[float] = None

class ComplianceCheckResponse(BaseModel):
    agent_responses: Dict[str, str]
    overall_compliance: bool
    session_id: Optional[str] = None
    debate_results: Optional[Dict[str, Any]] = None

class RAGCheckResponse(BaseModel):
    agent_responses: Dict[str, str]
    collection_used: str
    processing_time: Optional[float] = None

class RAGDebateSequenceResponse(BaseModel):
    session_id: str
    debate_chain: List[Dict[str, Any]]
    final_consensus: Optional[str] = None

class CreateAgentResponse(BaseModel):
    message: str
    agent_id: int
    agent_name: str

class GetAgentsResponse(BaseModel):
    agents: List[Dict[str, Any]]
    total_count: int

# Missing schema that agent_service.py expects
class ComplianceResultSchema(BaseModel):
    agent_responses: Dict[str, str]
    overall_compliance: bool
    session_id: Optional[str] = None
    debate_results: Optional[Dict[str, Any]] = None