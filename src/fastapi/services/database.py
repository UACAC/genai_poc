import os
import time

from typing import Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy import Index

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey,
    Text, Float, Boolean, JSON, Enum
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime, timezone
import enum


# Database connection setup - handles both direct DATABASE_URL and component-based setup
def get_database_url() -> str:
    """
    Get database URL from environment variables.
    Supports both direct DATABASE_URL and individual components.
    Updated to work with AWS Secrets Manager variable names.
    """
    # Option 1: Direct DATABASE_URL (preferred for simplicity)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        print("Using direct DATABASE_URL from environment")
        return database_url
    
    # Option 2: Component-based setup (AWS Secrets Manager style)
    db_username = os.getenv("DB_USERNAME", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    
    # Try both DB_ENDPOINT (AWS style) and DB_HOST (Docker style)
    db_endpoint = os.getenv("DB_ENDPOINT")
    db_host = os.getenv("DB_HOST")
    
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME") or os.getenv("DBNAME") or os.getenv("POSTGRES_DB", "rag_memory")
    
    # Debug: Print what we found (without password)
    print(f"Database config found:")
    print(f"  DB_USERNAME: {db_username}")
    print(f"  DB_PASSWORD: {'***' if db_password else 'NOT SET'}")
    print(f"  DB_ENDPOINT: {os.getenv('DB_ENDPOINT', 'NOT SET')}")
    print(f"  DB_HOST: {os.getenv('DB_HOST', 'NOT SET')}")
    print(f"  DB_PORT: {db_port}")
    print(f"  DB_NAME: {db_name}")
    print(f"  DBNAME: {os.getenv('DBNAME', 'NOT SET')}")
    
    if db_endpoint:
        if ':' in db_endpoint:
            parts = db_endpoint.rsplit(':', 1)
            potential_host = parts[0]
            potential_port = parts[1]
            
            try:
                int(potential_port)
                db_host = potential_host
                if os.getenv("DB_PORT") is None:
                    db_port = potential_port
                print(f"Detected port in endpoint: {potential_port}, using host: {db_host}")
            except ValueError:
                db_host = db_endpoint
                print(f"No valid port in endpoint, using full endpoint as host: {db_host}")
        else:
            db_host = db_endpoint
            print(f"No port in endpoint, using as host: {db_host}")
    
    missing = []
    if not db_username:
        missing.append("DB_USERNAME")
    if not db_password:
        missing.append("DB_PASSWORD") 
    if not db_host:
        missing.append("DB_HOST or DB_ENDPOINT")
    if not db_name:
        missing.append("DB_NAME or DBNAME")
    
    if missing:
        print(f"ERROR: Database configuration incomplete. Missing: {', '.join(missing)}")
        
        print("All DB_* environment variables:")
        for key, value in os.environ.items():
            if key.startswith('DB_'):
                print(f"  {key}: {'***' if 'PASSWORD' in key else value}")
        
        raise ValueError(f"Database configuration incomplete. Missing: {', '.join(missing)}")
    
    constructed_url = f"postgresql://{db_username}:{db_password}@{db_host}/{db_name}"
    print(f"Constructed DATABASE_URL: postgresql://{db_username}:***@{db_host}/{db_name}")
    return constructed_url

DATABASE_URL = get_database_url()

for i in range(5):
    try:
        engine = create_engine(DATABASE_URL)
        engine.connect().close()
        break
    except OperationalError:
        print(f"Database not ready, retrying {i+1}/5...")
        time.sleep(5)
else:
    raise Exception("Could not connect to the database after 5 attempts")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Tables
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_query = Column(Text)
    response = Column(Text)
    model_used = Column(String)
    collection_name = Column(String)
    query_type = Column(String)
    response_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    session_id = Column(String, index=True)
    langchain_used = Column(Boolean, default=False)
    source_documents = Column(JSON)

class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), index=True , onupdate=datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

class ComplianceAgent(Base):
    __tablename__ = "compliance_agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    system_prompt = Column(Text, nullable=False)
    user_prompt_template = Column(Text, nullable=False)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=300)
    use_structured_output = Column(Boolean, default=False)
    output_schema = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), index=True , onupdate=datetime.now(timezone.utc))
    created_by = Column(String)
    is_active = Column(Boolean, default=True)
    total_queries = Column(Integer, default=0)
    avg_response_time_ms = Column(Float)
    success_rate = Column(Float)
    chain_type = Column(String, default='basic')
    memory_enabled = Column(Boolean, default=False)
    tools_enabled = Column(JSON)

class ComplianceSequence(Base):
    __tablename__ = "compliance_sequence"
    id = Column(Integer, primary_key=True, index=True)
    compliance_agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    sequence_order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    compliance_agent = relationship("ComplianceAgent", back_populates="sequences")

ComplianceAgent.sequences = relationship(
    "ComplianceSequence", order_by=ComplianceSequence.sequence_order,
    back_populates="compliance_agent"
)

class DebateSession(Base):
    __tablename__ = "debate_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    compliance_agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    debate_order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    status = Column(String, default='active')
    initial_data = Column(Text)
    agent_response = Column(Text)
    response_time_ms = Column(Integer)
    langchain_used = Column(Boolean, default=False)
    compliance_agent = relationship("ComplianceAgent", back_populates="debate_sessions")

ComplianceAgent.debate_sessions = relationship(
    "DebateSession", order_by=DebateSession.debate_order,
    back_populates="compliance_agent"
)

class ComplianceResult(Base):
    __tablename__ = "compliance_results"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    data_sample = Column(Text, nullable=False)
    # compliant = Column(Boolean)
    confidence_score = Column(Float)
    reason = Column(Text)
    raw_response = Column(Text)
    processing_method = Column(String)
    response_time_ms = Column(Integer)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    agent = relationship("ComplianceAgent")
    

class SessionType(enum.Enum):
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT_DEBATE = "multi_agent_debate"
    RAG_ANALYSIS = "rag_analysis"
    RAG_DEBATE = "rag_debate"
    COMPLIANCE_CHECK = "compliance_check"

class AnalysisType(enum.Enum):
    DIRECT_LLM = "direct_llm"
    RAG_ENHANCED = "rag_enhanced"
    HYBRID = "hybrid"

# Enhanced AgentSession model to track all types of agent interactions
class AgentSession(Base):
    __tablename__ = "agent_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    session_type = Column(Enum(SessionType), nullable=False)
    analysis_type = Column(Enum(AnalysisType), nullable=False)
    
    # Input data
    user_query = Column(Text, nullable=False)
    collection_name = Column(String, nullable=True)  # For RAG sessions
    
    # Session metadata
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    completed_at = Column(DateTime, nullable=True)
    total_response_time_ms = Column(Integer, nullable=True)
    
    # Session status
    status = Column(String, default='active')  # active, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Results summary
    overall_result = Column(JSON, nullable=True)  # Summary of all agent responses
    agent_count = Column(Integer, default=0)
    
    # Relationships
    agent_responses = relationship("AgentResponse", back_populates="session", cascade="all, delete-orphan")

# Individual agent responses within a session
class AgentResponse(Base):
    __tablename__ = "agent_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(Integer, ForeignKey("compliance_agents.id", ondelete="CASCADE"), nullable=False)
    
    # Response details
    response_text = Column(Text, nullable=False)
    processing_method = Column(String, nullable=False)  # langchain, rag_enhanced, direct_llm, etc.
    response_time_ms = Column(Integer, nullable=True)
    
    # Sequence information (for debates)
    sequence_order = Column(Integer, nullable=True)  # Order in debate sequence
    
    # RAG information
    rag_used = Column(Boolean, default=False)
    documents_found = Column(Integer, default=0)
    rag_context = Column(Text, nullable=True)  # The retrieved context
    
    # Compliance/Analysis results
    # compliant = Column(Boolean, nullable=True)
    confidence_score = Column(Float, nullable=True)
    analysis_summary = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    model_used = Column(String, nullable=False)
    
    # Relationships
    session = relationship("AgentSession", back_populates="agent_responses")
    agent = relationship("ComplianceAgent")

# Utilities
def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def update_agent_performance(agent_id: int, response_time_ms: int, success: bool):
    db = SessionLocal()
    try:
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        if agent:
            agent.total_queries += 1
            if agent.avg_response_time_ms is None:
                agent.avg_response_time_ms = response_time_ms
            else:
                total_time = agent.avg_response_time_ms * (agent.total_queries - 1) + response_time_ms
                agent.avg_response_time_ms = total_time / agent.total_queries

            if agent.success_rate is None:
                agent.success_rate = 1.0 if success else 0.0
            else:
                total_successes = agent.success_rate * (agent.total_queries - 1) + (1 if success else 0)
                agent.success_rate = total_successes / agent.total_queries

            db.commit()
    except Exception as e:
        print(f"Performance update error: {e}")
        db.rollback()
    finally:
        db.close()

# def log_compliance_result(agent_id: int, data_sample: str, compliant: Optional[bool],
#                             confidence_score: Optional[float], reason: str,
#                             raw_response: str, processing_method: str,
#                             response_time_ms: int, model_used: str,
#                             session_id: Optional[str] = None):
def log_compliance_result(agent_id: int, data_sample: str,
                            confidence_score: Optional[float], reason: str,
                            raw_response: str, processing_method: str,
                            response_time_ms: int, model_used: str,
                            session_id: Optional[str] = None):
    db = SessionLocal()
    try:
        result = ComplianceResult(
            session_id=session_id,
            agent_id=agent_id,
            data_sample=data_sample,
            # compliant=compliant,
            confidence_score=confidence_score,
            reason=reason,
            raw_response=raw_response,
            processing_method=processing_method,
            response_time_ms=response_time_ms,
            model_used=model_used
        )
        db.add(result)
        db.commit()
        # update_agent_performance(agent_id, response_time_ms, compliant is not None)
        update_agent_performance(agent_id, response_time_ms)
    except Exception as e:
        print(f"Log result error: {e}")
        db.rollback()
    finally:
        db.close()

def log_agent_session(session_id: str, session_type: SessionType, analysis_type: AnalysisType, 
                        user_query: str, collection_name: str = None) -> None:
    """Log the start of an agent session"""
    db = SessionLocal()
    try:
        session = AgentSession(
            session_id=session_id,
            session_type=session_type,
            analysis_type=analysis_type,
            user_query=user_query,
            collection_name=collection_name,
            status='active'
        )
        db.add(session)
        db.commit()
    except Exception as e:
        print(f"Error logging agent session: {e}")
        db.rollback()
    finally:
        db.close()

# def log_agent_response(session_id: str, agent_id: int, response_text: str, 
#                         processing_method: str, response_time_ms: int, model_used: str,
#                         sequence_order: int = None, rag_used: bool = False, 
#                         documents_found: int = 0, rag_context: str = None,
#                         compliant: bool = None, confidence_score: float = None,
#                         analysis_summary: str = None) -> None:
def log_agent_response(session_id: str, agent_id: int, response_text: str, 
                        processing_method: str, response_time_ms: int, model_used: str,
                        sequence_order: int = None, rag_used: bool = False, 
                        documents_found: int = 0, rag_context: str = None,
                        confidence_score: float = None,
                        analysis_summary: str = None) -> None:
    """Log an individual agent response"""
    db = SessionLocal()
    try:
        response = AgentResponse(
            session_id=session_id,
            agent_id=agent_id,
            response_text=response_text,
            processing_method=processing_method,
            response_time_ms=response_time_ms,
            sequence_order=sequence_order,
            rag_used=rag_used,
            documents_found=documents_found,
            rag_context=rag_context,
            # compliant=compliant,
            confidence_score=confidence_score,
            analysis_summary=analysis_summary,
            model_used=model_used
        )
        db.add(response)
        db.commit()
    except Exception as e:
        print(f"Error logging agent response: {e}")
        db.rollback()
    finally:
        db.close()

def complete_agent_session(session_id: str, overall_result: dict, agent_count: int, 
                            total_response_time_ms: int = None, status: str = 'completed',
                            error_message: str = None) -> None:
    """Mark an agent session as completed and log summary"""
    db = SessionLocal()
    try:
        session = db.query(AgentSession).filter(AgentSession.session_id == session_id).first()
        if session:
            session.completed_at = datetime.now(timezone.utc)
            session.overall_result = overall_result
            session.agent_count = agent_count
            session.total_response_time_ms = total_response_time_ms
            session.status = status
            session.error_message = error_message
            db.commit()
    except Exception as e:
        print(f"Error completing agent session: {e}")
        db.rollback()
    finally:
        db.close()

def get_session_history(limit: int = 50, session_type: SessionType = None):
    """Get recent agent session history"""
    db = SessionLocal()
    try:
        query = db.query(AgentSession).order_by(AgentSession.created_at.desc())
        
        if session_type:
            query = query.filter(AgentSession.session_type == session_type)
            
        sessions = query.limit(limit).all()
        
        result = []
        for session in sessions:
            session_data = {
                "session_id": session.session_id,
                "session_type": session.session_type.value,
                "analysis_type": session.analysis_type.value,
                "user_query": session.user_query[:200] + "..." if len(session.user_query) > 200 else session.user_query,
                "collection_name": session.collection_name,
                "created_at": session.created_at,
                "completed_at": session.completed_at,
                "status": session.status,
                "agent_count": session.agent_count,
                "total_response_time_ms": session.total_response_time_ms
            }
            result.append(session_data)
        
        return result
    except Exception as e:
        print(f"Error getting session history: {e}")
        return []
    finally:
        db.close()

def get_session_details(session_id: str):
    """Get detailed information about a specific session"""
    db = SessionLocal()
    try:
        session = db.query(AgentSession).filter(AgentSession.session_id == session_id).first()
        if not session:
            return None
            
        # Get all agent responses for this session
        responses = db.query(AgentResponse).filter(
            AgentResponse.session_id == session_id
        ).order_by(AgentResponse.sequence_order.asc(), AgentResponse.created_at.asc()).all()
        
        session_data = {
            "session_info": {
                "session_id": session.session_id,
                "session_type": session.session_type.value,
                "analysis_type": session.analysis_type.value,
                "user_query": session.user_query,
                "collection_name": session.collection_name,
                "created_at": session.created_at,
                "completed_at": session.completed_at,
                "status": session.status,
                "error_message": session.error_message,
                "overall_result": session.overall_result,
                "agent_count": session.agent_count,
                "total_response_time_ms": session.total_response_time_ms
            },
            "agent_responses": []
        }
        
        for response in responses:
            response_data = {
                "agent_id": response.agent_id,
                "agent_name": response.agent.name if response.agent else "Unknown",
                "response_text": response.response_text,
                "processing_method": response.processing_method,
                "response_time_ms": response.response_time_ms,
                "sequence_order": response.sequence_order,
                "rag_used": response.rag_used,
                "documents_found": response.documents_found,
                # "compliant": response.compliant,
                "confidence_score": response.confidence_score,
                "model_used": response.model_used,
                "created_at": response.created_at
            }
            session_data["agent_responses"].append(response_data)
        
        return session_data
    except Exception as e:
        print(f"Error getting session details: {e}")
        return None
    finally:
        db.close()

# Composite indexes for better query performance
Index('idx_chat_session_timestamp', ChatHistory.session_id, ChatHistory.timestamp)
Index('idx_chat_model_type', ChatHistory.model_used, ChatHistory.query_type)
Index('idx_compliance_session_agent', ComplianceResult.session_id, ComplianceResult.agent_id)
Index('idx_compliance_agent_created', ComplianceResult.agent_id, ComplianceResult.created_at)