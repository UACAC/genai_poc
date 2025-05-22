import os
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey,
    Text, Float, Boolean, JSON
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")

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
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String, index=True)
    langchain_used = Column(Boolean, default=False)
    source_documents = Column(JSON)

class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
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
    created_at = Column(DateTime, default=datetime.utcnow)
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
    created_at = Column(DateTime, default=datetime.utcnow)
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
    compliant = Column(Boolean)
    confidence_score = Column(Float)
    reason = Column(Text)
    raw_response = Column(Text)
    processing_method = Column(String)
    response_time_ms = Column(Integer)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
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

def log_compliance_result(agent_id: int, data_sample: str, compliant: Optional[bool],
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
            compliant=compliant,
            confidence_score=confidence_score,
            reason=reason,
            raw_response=raw_response,
            processing_method=processing_method,
            response_time_ms=response_time_ms,
            model_used=model_used
        )
        db.add(result)
        db.commit()
        update_agent_performance(agent_id, response_time_ms, compliant is not None)
    except Exception as e:
        print(f"Log result error: {e}")
        db.rollback()
    finally:
        db.close()
