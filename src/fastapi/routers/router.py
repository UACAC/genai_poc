import uuid
import os
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
import time
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from services.database import ComplianceAgent, DebateSession, ChatHistory, SessionLocal
from services.llm_service import LLMService
from services.agent_service import AgentService
from services.rag_service import RAGService 
from services.generate_docs_service import DocumentService
from services.evaluate_doc_service import EvaluationService
from schemas.database_schema import (
    ComplianceCheckRequest,
    RAGCheckRequest,
    RAGDebateSequenceRequest,
    CreateAgentRequest,
    CreateAgentResponse,
    GetAgentsResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
    EvaluateRequest,
    EvaluateResponse
)
from langchain_core.messages import HumanMessage
from starlette.concurrency import run_in_threadpool
import logging
logger = logging.getLogger("FASTAPI_ROUTER")

router = APIRouter()

llm_service = LLMService()
rag_service = RAGService()
agent_service = AgentService()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatRequest(BaseModel):
    query: str
    model: str
    use_rag: bool = False
    collection_name: Optional[str] = None

@router.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Fixed chat endpoint that handles both OpenAI and Ollama responses."""
    try:
        session_id = str(uuid.uuid4())
        
        if request.use_rag and request.collection_name:
            # RAG mode
            answer, response_time = llm_service.query_model(
                model_name=request.model,
                query=request.query,
                collection_name=request.collection_name,
                query_type="chat",
                session_id=session_id
            )
            return {
                "response": answer,
                "response_time_ms": response_time,
                "session_id": session_id,
                "type": "rag"
            }
        else:
            # Direct LLM mode
            start_time = time.time()
            llm = llm_service.get_llm_service(request.model)
            
            response = llm.invoke([HumanMessage(content=request.query)])
            
            # OllamaLLM returns string directly, ChatOpenAI returns object with .content
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = response
                
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Save chat history
            try:
                history = ChatHistory(
                    user_query=request.query,
                    response=answer,
                    model_used=request.model,
                    query_type="direct",
                    response_time_ms=response_time_ms,
                    langchain_used=True,
                    session_id=session_id
                )
                db.add(history)
                db.commit()
            except Exception as e:
                print(f"Failed to save chat history: {e}")
                db.rollback()
            
            return {
                "response": answer,
                "response_time_ms": response_time_ms,
                "session_id": session_id,
                "type": "direct"
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/chat-history")
def get_chat_history(db: Session = Depends(get_db)):
    records = db.query(ChatHistory).all()
    return [
        {
            "id": record.id,
            "user_query": record.user_query,
            "response": record.response,
            "timestamp": record.timestamp
        } for record in records
    ]

@router.post("/compliance-check")
async def compliance_check(request: ComplianceCheckRequest, db: Session = Depends(get_db)):
    try:
        result = agent_service.run_compliance_check(
            data_sample=request.data_sample,
            agent_ids=request.agent_ids,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag-check")
async def rag_check(request: RAGCheckRequest, db: Session = Depends(get_db)):
    try:
        result = rag_service.run_rag_check( 
            query_text=request.query_text,
            collection_name=request.collection_name,
            agent_ids=request.agent_ids,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag-debate-sequence")
async def rag_debate_sequence(request: RAGDebateSequenceRequest, db: Session = Depends(get_db)):
    try:
        session_id, chain = rag_service.run_rag_debate_sequence(  
            db=db,
            session_id=request.session_id,
            agent_ids=request.agent_ids,
            query_text=request.query_text,
            collection_name=request.collection_name
        )
        return {"session_id": session_id, "debate_chain": chain}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-agent", response_model=CreateAgentResponse)
async def create_agent(request: CreateAgentRequest, db: Session = Depends(get_db)):
    """Create a new compliance agent"""
    try:
        # Check if agent name already exists
        existing_agent = db.query(ComplianceAgent).filter(
            ComplianceAgent.name == request.name
        ).first()
        
        if existing_agent:
            raise HTTPException(
                status_code=400, 
                detail=f"Agent with name '{request.name}' already exists"
            )
        
        # Validate that user_prompt_template contains {data_sample}
        if "{data_sample}" not in request.user_prompt_template:
            raise HTTPException(
                status_code=400,
                detail="User prompt template must contain {data_sample} placeholder"
            )
        
        # Create new agent with only the fields that exist in your database
        new_agent = ComplianceAgent(
            name=request.name,
            model_name=request.model_name,
            system_prompt=request.system_prompt,
            user_prompt_template=request.user_prompt_template,
            temperature=getattr(request, 'temperature', 0.7),
            max_tokens=getattr(request, 'max_tokens', 300),
            created_by=getattr(request, 'created_by', 'streamlit'),
            is_active=True,
            total_queries=0,
            chain_type='basic',
            memory_enabled=False,
            tools_enabled={}
        )
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        print(f"Agent created successfully: ID={new_agent.id}, Name={new_agent.name}")
        
        return CreateAgentResponse(
            message=f"Agent '{request.name}' created successfully",
            agent_id=new_agent.id,
            agent_name=new_agent.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating agent: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {str(e)}"
        )

@router.get("/get-agents", response_model=GetAgentsResponse)
async def get_agents(db: Session = Depends(get_db)):
    """Get all compliance agents"""
    try:
        agents = db.query(ComplianceAgent).all()
        
        # Convert agents to dict format
        agents_data = []
        for agent in agents:
            agent_dict = {
                "id": agent.id,
                "name": agent.name,
                "model_name": agent.model_name,
                "system_prompt": agent.system_prompt,
                "user_prompt_template": agent.user_prompt_template,
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens,
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
                "created_by": agent.created_by,
                "is_active": agent.is_active,
                "total_queries": agent.total_queries,
                "avg_response_time_ms": agent.avg_response_time_ms,
                "success_rate": agent.success_rate,
                "chain_type": agent.chain_type,
                "memory_enabled": agent.memory_enabled,
                "tools_enabled": agent.tools_enabled
            }
            agents_data.append(agent_dict)
        
        print(f"Retrieved {len(agents_data)} agents")
        
        return GetAgentsResponse(
            agents=agents_data,
            total_count=len(agents_data)
        )
        
    except Exception as e:
        print(f"Error getting agents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agents: {str(e)}"
        )

@router.put("/update-agent/{agent_id}", response_model=UpdateAgentResponse)
async def update_agent(agent_id: int, request: UpdateAgentRequest, db: Session = Depends(get_db)):
    """Update an existing compliance agent"""
    try:
        # Find the agent
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Track which fields are being updated
        updated_fields = []
        
        # Update only provided fields
        if request.name is not None:
            # Check if new name already exists (excluding current agent)
            existing_agent = db.query(ComplianceAgent).filter(
                ComplianceAgent.name == request.name,
                ComplianceAgent.id != agent_id
            ).first()
            
            if existing_agent:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent with name '{request.name}' already exists"
                )
            
            agent.name = request.name
            updated_fields.append("name")
        
        if request.model_name is not None:
            agent.model_name = request.model_name
            updated_fields.append("model_name")
        
        if request.system_prompt is not None:
            agent.system_prompt = request.system_prompt
            updated_fields.append("system_prompt")
        
        if request.user_prompt_template is not None:
            agent.user_prompt_template = request.user_prompt_template
            updated_fields.append("user_prompt_template")
        
        if request.temperature is not None:
            agent.temperature = request.temperature
            updated_fields.append("temperature")
        
        if request.max_tokens is not None:
            agent.max_tokens = request.max_tokens
            updated_fields.append("max_tokens")
        
        if request.is_active is not None:
            agent.is_active = request.is_active
            updated_fields.append("is_active")
        
        # Update the updated_at timestamp
        agent.updated_at = datetime.now(timezone.utc)
        updated_fields.append("updated_at")
        
        db.commit()
        db.refresh(agent)
        
        print(f"Agent updated successfully: ID={agent.id}, Name={agent.name}, Updated fields={updated_fields}")
        
        return UpdateAgentResponse(
            message=f"Agent '{agent.name}' updated successfully",
            agent_id=agent.id,
            agent_name=agent.name,
            updated_fields=updated_fields
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating agent: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update agent: {str(e)}"
        )

@router.get("/get-agent/{agent_id}")
async def get_agent_by_id(agent_id: int, db: Session = Depends(get_db)):
    """Get a specific agent by ID"""
    try:
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        agent_dict = {
            "id": agent.id,
            "name": agent.name,
            "model_name": agent.model_name,
            "system_prompt": agent.system_prompt,
            "user_prompt_template": agent.user_prompt_template,
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
            "created_by": agent.created_by,
            "is_active": agent.is_active,
            "total_queries": agent.total_queries,
            "avg_response_time_ms": agent.avg_response_time_ms,
            "success_rate": agent.success_rate,
            "chain_type": agent.chain_type,
            "memory_enabled": agent.memory_enabled,
            "tools_enabled": agent.tools_enabled
        }
        
        return agent_dict
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting agent by ID: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent: {str(e)}"
        )

@router.delete("/delete-agent/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete a compliance agent and handle foreign key relationships"""
    try:
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        agent_name = agent.name
        
        # Delete all debate sessions that reference this agent
        debate_sessions_deleted = db.query(DebateSession).filter(
            DebateSession.compliance_agent_id == agent_id
        ).delete(synchronize_session=False)
        
        # Delete all compliance results that reference this agent
        from services.database import ComplianceResult  # Import if not already imported
        compliance_results_deleted = db.query(ComplianceResult).filter(
            ComplianceResult.agent_id == agent_id
        ).delete(synchronize_session=False)
        
        print(f"Deleting agent {agent_id}: Removed {debate_sessions_deleted} debate sessions and {compliance_results_deleted} compliance results")
        
        # Now delete the agent
        db.delete(agent)
        db.commit()
        
        print(f"Agent deleted successfully: ID={agent_id}, Name={agent_name}")
        
        return {
            "message": f"Agent '{agent_name}' deleted successfully",
            "agent_id": agent_id,
            "agent_name": agent_name,
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            "cleanup_info": {
                "debate_sessions_deleted": debate_sessions_deleted,
                "compliance_results_deleted": compliance_results_deleted
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting agent: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete agent: {str(e)}"
        )

@router.patch("/toggle-agent-status/{agent_id}")
async def toggle_agent_status(agent_id: int, db: Session = Depends(get_db)):
    """Toggle agent active/inactive status"""
    try:
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Toggle the status
        agent.is_active = not agent.is_active
        agent.updated_at = datetime.now(timezone.utc)
        
        db.commit()
        db.refresh(agent)
        
        status = "activated" if agent.is_active else "deactivated"
        print(f"Agent {status}: ID={agent.id}, Name={agent.name}")
        
        return {
            "message": f"Agent '{agent.name}' {status} successfully",
            "agent_id": agent.id,
            "agent_name": agent.name,
            "is_active": agent.is_active,
            "updated_at": agent.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error toggling agent status: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle agent status: {str(e)}"
        )

@router.get("/health")
async def health_check():
    try:
        health = llm_service.health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/collections/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Get information about a specific collection"""
    try:
        info = rag_service.query_collection_info(collection_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from services.database import (
    get_session_history, get_session_details, SessionType, AnalysisType
)

@router.get("/session-history")
async def get_agent_session_history(
    limit: int = 50, 
    session_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get recent agent session history"""
    try:
        # Convert string to enum if provided
        type_filter = None
        if session_type:
            try:
                type_filter = SessionType(session_type)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid session_type. Valid options: {[t.value for t in SessionType]}"
                )
        
        history = get_session_history(limit=limit, session_type=type_filter)
        
        return {
            "sessions": history,
            "total_returned": len(history),
            "available_session_types": [t.value for t in SessionType],
            "available_analysis_types": [t.value for t in AnalysisType]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session-details/{session_id}")
async def get_agent_session_details(session_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific session"""
    try:
        details = get_session_details(session_id)
        
        if not details:
            raise HTTPException(
                status_code=404,
                detail=f"Session with ID {session_id} not found"
            )
        
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent-performance/{agent_id}")
async def get_agent_performance_metrics(agent_id: int, db: Session = Depends(get_db)):
    """Get performance metrics for a specific agent"""
    try:
        # Check if agent exists
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Get agent response statistics
        from services.database import AgentResponse
        
        total_responses = db.query(AgentResponse).filter(
            AgentResponse.agent_id == agent_id
        ).count()
        
        if total_responses == 0:
            return {
                "agent_id": agent_id,
                "agent_name": agent.name,
                "total_responses": 0,
                "message": "No response data available for this agent"
            }
        
        responses = db.query(AgentResponse).filter(
            AgentResponse.agent_id == agent_id
        ).all()
        
        # Response time statistics
        response_times = [r.response_time_ms for r in responses if r.response_time_ms]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # RAG usage statistics
        rag_responses = [r for r in responses if r.rag_used]
        rag_usage_rate = len(rag_responses) / total_responses if total_responses > 0 else 0
        
        # Processing method breakdown
        method_counts = {}
        for response in responses:
            method = response.processing_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Recent activity (last 10 responses)
        recent_responses = db.query(AgentResponse).filter(
            AgentResponse.agent_id == agent_id
        ).order_by(AgentResponse.created_at.desc()).limit(10).all()
        
        recent_activity = []
        for response in recent_responses:
            recent_activity.append({
                "session_id": response.session_id,
                "created_at": response.created_at,
                "processing_method": response.processing_method,
                "response_time_ms": response.response_time_ms,
                "rag_used": response.rag_used,
                "documents_found": response.documents_found,
                "response_preview": response.response_text[:200] + "..." if len(response.response_text) > 200 else response.response_text
            })
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "model_name": agent.model_name,
            "performance_metrics": {
                "total_responses": total_responses,
                "avg_response_time_ms": round(avg_response_time, 2),
                "rag_usage_rate": round(rag_usage_rate * 100, 2),  # as percentage
                "processing_method_breakdown": method_counts
            },
            "recent_activity": recent_activity,
            "agent_info": {
                "created_at": agent.created_at,
                "total_queries": agent.total_queries,
                "success_rate": agent.success_rate,
                "is_active": agent.is_active
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session-analytics")
async def get_session_analytics(days: int = 7, db: Session = Depends(get_db)):
    """Get analytics about agent sessions over the specified number of days"""
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import func
        from services.database import AgentSession, AgentResponse
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Session counts by type
        session_counts = db.query(
            AgentSession.session_type,
            func.count(AgentSession.id).label('count')
        ).filter(
            AgentSession.created_at >= start_date
        ).group_by(AgentSession.session_type).all()
        
        # Session counts by analysis type
        analysis_counts = db.query(
            AgentSession.analysis_type,
            func.count(AgentSession.id).label('count')
        ).filter(
            AgentSession.created_at >= start_date
        ).group_by(AgentSession.analysis_type).all()
        
        # Average response times
        avg_response_time = db.query(
            func.avg(AgentSession.total_response_time_ms)
        ).filter(
            AgentSession.created_at >= start_date,
            AgentSession.total_response_time_ms.isnot(None)
        ).scalar()
        
        # Most active agents
        active_agents = db.query(
            AgentResponse.agent_id,
            ComplianceAgent.name,
            func.count(AgentResponse.id).label('response_count')
        ).join(
            ComplianceAgent, AgentResponse.agent_id == ComplianceAgent.id
        ).filter(
            AgentResponse.created_at >= start_date
        ).group_by(
            AgentResponse.agent_id, ComplianceAgent.name
        ).order_by(
            func.count(AgentResponse.id).desc()
        ).limit(10).all()
        
        # RAG usage statistics
        total_responses = db.query(AgentResponse).filter(
            AgentResponse.created_at >= start_date
        ).count()
        
        rag_responses = db.query(AgentResponse).filter(
            AgentResponse.created_at >= start_date,
            AgentResponse.rag_used == True
        ).count()
        
        rag_usage_rate = (rag_responses / total_responses * 100) if total_responses > 0 else 0
        
        return {
            "analytics_period": {
                "days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "session_statistics": {
                "by_session_type": {sc.session_type.value: sc.count for sc in session_counts},
                "by_analysis_type": {ac.analysis_type.value: ac.count for ac in analysis_counts},
                "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else 0
            },
            "agent_activity": [
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "response_count": agent.response_count
                }
                for agent in active_agents
            ],
            "rag_statistics": {
                "total_responses": total_responses,
                "rag_responses": rag_responses,
                "rag_usage_rate": round(rag_usage_rate, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    

doc_service = DocumentService(
    rag_service=rag_service,
    agent_service=agent_service,
    llm_service=llm_service,
    chroma_url=os.getenv("CHROMA_URL", "http://localhost:8020"),
    agent_api_url=os.getenv("FASTAPI_URL", "http://localhost:9020")
)


class GenerateRequest(BaseModel):
    template_collection:  str
    template_doc_ids:     Optional[List[str]]   = None
    source_collections:   Optional[List[str]]   = None
    source_doc_ids:       Optional[List[str]]   = None
    agent_ids:            List[int]
    use_rag:              bool                  = True
    top_k:                int                   = 5
    doc_title:            str                   = None

@router.post("/generate_documents")
async def generate_documents(req: GenerateRequest):
    logger.info("Received /generate_documents ⇒ %s", req)
    docs = await run_in_threadpool(
        doc_service.generate_documents,
        req.template_collection,
        req.template_doc_ids,
        req.source_collections,
        req.source_doc_ids,
        req.agent_ids,
        req.use_rag,
        req.top_k,
        req.doc_title
    )
    return {"documents": docs}



eval_svc = EvaluationService(
    rag=rag_service,
    llm=llm_service
)

@router.post("/evaluate_doc", response_model=EvaluateResponse)
async def evaluate_doc(req: EvaluateRequest, db: Session = Depends(get_db)):
    try:
        # generate a session_id so you can track history
        doc_session_id = str(uuid.uuid4())

        answer, rt_ms = eval_svc.evaluate_document(
            document_id     = req.document_id,
            collection_name = req.collection_name,
            prompt          = req.prompt,
            top_k           = req.top_k,
            model_name      = req.model_name,
            session_id      = doc_session_id,      
        )
        # Save chat history
        try:
            eval_history = ChatHistory(
                user_query=req.prompt,
                response=answer,
                model_used=req.model_name,
                query_type="rag",
                response_time_ms=rt_ms,
                langchain_used=True,
                session_id=doc_session_id
            )
            db.add(eval_history)
            db.commit()
        except Exception as e:
            print(f"Failed to save evaluation history: {e}")
            db.rollback()

        return EvaluateResponse(
            document_id     = req.document_id,
            collection_name = req.collection_name,
            prompt          = req.prompt,
            model_name      = req.model_name,
            response        = answer,
            response_time_ms= rt_ms,
            session_id      = doc_session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
