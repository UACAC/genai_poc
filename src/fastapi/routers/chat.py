import uuid
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from services.database import SessionLocal, ComplianceAgent, DebateSession, ChatHistory
from services.llm_service import LLMService
from services.rag_service import EnhancedRAGService
from services.agent_service import AgentService
from services.rag_agent_service import RAGAgentService

router = APIRouter()

llm_service = LLMService()
rag_service = EnhancedRAGService()
agent_service = AgentService()
rag_agent_service = RAGAgentService()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatRequest(BaseModel):
    query: str

class RAGQueryRequest(BaseModel):
    query: str
    collection_name: str

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

@router.post("/chat-gpt4")
async def chat_gpt4(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        response = llm_service.query_gpt4(request.query)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-llama")
async def chat_llama(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        response = llm_service.query_llama(request.query)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-mistral")
async def chat_mistral(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        response = llm_service.query_mistral(request.query)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-gemma")
async def chat_gemma(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        response = llm_service.query_gemma(request.query)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-gpt4-rag")
async def chat_rag_gpt4(request: RAGQueryRequest, db: Session = Depends(get_db)):
    try:
        response = rag_service.query_gpt(request.query, request.collection_name)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"500: RAG query failed: {str(e)}")

@router.post("/chat-rag-llama")
async def chat_rag_llama(request: RAGQueryRequest, db: Session = Depends(get_db)):
    try:
        response = rag_service.query_llama(request.query, request.collection_name)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"500: RAG query failed: {str(e)}")

@router.post("/chat-rag-mistral")
async def chat_rag_mistral(request: RAGQueryRequest, db: Session = Depends(get_db)):
    try:
        response = rag_service.query_mistral(request.query, request.collection_name)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"500: RAG query failed: {str(e)}")

@router.post("/chat-rag-gemma")
async def chat_rag_gemma(request: RAGQueryRequest, db: Session = Depends(get_db)):
    try:
        response = rag_service.query_gemma(request.query, request.collection_name)
        db.add(ChatHistory(user_query=request.query, response=response))
        db.commit()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"500: RAG query failed: {str(e)}")

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
        result = rag_agent_service.run_rag_check(
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
        session_id, chain = rag_agent_service.run_rag_debate_sequence(
            db=db,
            session_id=request.session_id,
            agent_ids=request.agent_ids,
            query_text=request.query_text,
            collection_name=request.collection_name
        )
        return {"session_id": session_id, "debate_chain": chain}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
