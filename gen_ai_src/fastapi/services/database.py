import os
import time
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Pull DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Retry connecting to the database if it's not ready
for i in range(5):
    try:
        engine = create_engine(DATABASE_URL)
        connection = engine.connect()
        connection.close()
        break  
    except OperationalError:
        print(f"Database not ready, retrying {i+1}/5...")
        time.sleep(5)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_query = Column(String)
    response = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow())

class Agent(Base):
    """Table to store general agent personas."""
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)  # Persona name
    model_name = Column(String, nullable=False)  # Model used (e.g., GPT-4)
    description = Column(String)  # Optional description

class ComplianceAgent(Base):
    """Table to store compliance agents with system and user prompts."""
    __tablename__ = "compliance_agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)  # Compliance Agent Name
    model_name = Column(String, nullable=False)  # Model used (e.g., GPT-4)
    system_prompt = Column(String, nullable=False)  # System role definition
    user_prompt_template = Column(String, nullable=False)  # User prompt template

class ComplianceSequence(Base):
    """Table to define the order of compliance checks."""
    __tablename__ = "compliance_sequence"
    id = Column(Integer, primary_key=True, index=True)
    compliance_agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    sequence_order = Column(Integer, nullable=False)  # Order in which compliance agents check compliance
    compliance_agent = relationship("ComplianceAgent", back_populates="sequences")

ComplianceAgent.sequences = relationship("ComplianceSequence", order_by=ComplianceSequence.sequence_order, back_populates="compliance_agent")

class DebateSession(Base):
    """Table to manage multi-agent debates with compliance agents."""
    __tablename__ = "debate_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    compliance_agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    debate_order = Column(Integer, nullable=False)  # Order in which the agent speaks
    compliance_agent = relationship("ComplianceAgent", back_populates="debate_sessions")

ComplianceAgent.debate_sessions = relationship("DebateSession", order_by=DebateSession.debate_order, back_populates="compliance_agent")

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
