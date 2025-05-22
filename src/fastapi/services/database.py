import os
import time
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean, JSON
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional, Dict, Any

# Pull DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Retry connecting to the database if it's not ready
for i in range(5):
    try:
        engine = create_engine(DATABASE_URL)
        connection = engine.connect()
        connection.close()
        print("Database connection successful!")
        break  
    except OperationalError:
        print(f"Database not ready, retrying {i+1}/5...")
        time.sleep(5)
else:
    raise Exception("Could not connect to database after 5 attempts")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_query = Column(Text)  # Changed to Text for longer queries
    response = Column(Text)    # Changed to Text for longer responses
    model_used = Column(String, nullable=True)  # Track which model was used
    collection_name = Column(String, nullable=True)  # For RAG queries
    query_type = Column(String, nullable=True)  # 'chat', 'rag', 'compliance', etc.
    response_time_ms = Column(Integer, nullable=True)  # Track response time
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String, nullable=True, index=True)  # Group related queries
    
    # Additional metadata for LangChain integration
    langchain_used = Column(Boolean, default=False)
    source_documents = Column(JSON, nullable=True)  # Store RAG source info
    
class Agent(Base):
    """Table to store general agent personas."""
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    description = Column(Text)  # Changed to Text for longer descriptions
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ComplianceAgent(Base):
    """Enhanced table to store compliance agents with system and user prompts."""
    __tablename__ = "compliance_agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    system_prompt = Column(Text, nullable=False)  # Changed to Text
    user_prompt_template = Column(Text, nullable=False)  # Changed to Text
    
    # Enhanced fields for LangChain support
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=300)
    use_structured_output = Column(Boolean, default=False)
    output_schema = Column(JSON, nullable=True)  # Store Pydantic schema
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Performance tracking
    total_queries = Column(Integer, default=0)
    avg_response_time_ms = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)
    
    # Agent configuration for LangChain
    chain_type = Column(String, default='basic')  # 'basic', 'structured', 'rag'
    memory_enabled = Column(Boolean, default=False)
    tools_enabled = Column(JSON, nullable=True)  # List of enabled tools

class ComplianceSequence(Base):
    """Table to define the order of compliance checks."""
    __tablename__ = "compliance_sequence"
    
    id = Column(Integer, primary_key=True, index=True)
    compliance_agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    sequence_order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    compliance_agent = relationship("ComplianceAgent", back_populates="sequences")

# Add the reverse relationship
ComplianceAgent.sequences = relationship(
    "ComplianceSequence", 
    order_by=ComplianceSequence.sequence_order, 
    back_populates="compliance_agent"
)

class DebateSession(Base):
    """Enhanced table to manage multi-agent debates with compliance agents."""
    __tablename__ = "debate_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    compliance_agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    debate_order = Column(Integer, nullable=False)
    
    # Enhanced debate tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='active')  # 'active', 'completed', 'failed'
    
    # Debate metadata
    initial_data = Column(Text, nullable=True)  # Store the original data being debated
    agent_response = Column(Text, nullable=True)  # Store agent's response
    response_time_ms = Column(Integer, nullable=True)
    langchain_used = Column(Boolean, default=False)
    
    compliance_agent = relationship("ComplianceAgent", back_populates="debate_sessions")

# Add the reverse relationship
ComplianceAgent.debate_sessions = relationship(
    "DebateSession", 
    order_by=DebateSession.debate_order, 
    back_populates="compliance_agent"
)

class ComplianceResult(Base):
    """New table to store detailed compliance check results."""
    __tablename__ = "compliance_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    agent_id = Column(Integer, ForeignKey("compliance_agents.id"), nullable=False)
    
    # Input data
    data_sample = Column(Text, nullable=False)
    
    # Results
    compliant = Column(Boolean, nullable=True)
    confidence_score = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    raw_response = Column(Text, nullable=True)
    
    # Metadata
    processing_method = Column(String, nullable=True)  # 'legacy', 'langchain', 'structured'
    response_time_ms = Column(Integer, nullable=True)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agent = relationship("ComplianceAgent")

class RAGCollection(Base):
    """New table to track RAG collections and their metadata."""
    __tablename__ = "rag_collections"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Collection metadata
    document_count = Column(Integer, default=0)
    embedding_model = Column(String, nullable=True)
    chunk_size = Column(Integer, nullable=True)
    chunk_overlap = Column(Integer, nullable=True)
    
    # Performance metrics
    total_queries = Column(Integer, default=0)
    avg_retrieval_time_ms = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_indexed_at = Column(DateTime, nullable=True)
    
    is_active = Column(Boolean, default=True)

class RAGQuery(Base):
    """New table to track RAG queries and their performance."""
    __tablename__ = "rag_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("rag_collections.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    
    # Results
    response = Column(Text, nullable=True)
    source_documents = Column(JSON, nullable=True)
    retrieval_score = Column(Float, nullable=True)
    
    # Performance
    retrieval_time_ms = Column(Integer, nullable=True)
    generation_time_ms = Column(Integer, nullable=True)
    total_time_ms = Column(Integer, nullable=True)
    
    # Model info
    model_used = Column(String, nullable=False)
    langchain_used = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    collection = relationship("RAGCollection")

class SystemMetrics(Base):
    """New table to track system performance metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String, nullable=True)
    
    # Context
    component = Column(String, nullable=True)  # 'agent', 'rag', 'debate', 'system'
    details = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

# Database utility functions
def init_db():
    """Create all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise

def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_default_data():
    """Create some default data for testing."""
    db = SessionLocal()
    try:
        # Check if we already have data
        if db.query(ComplianceAgent).count() > 0:
            print("Default data already exists, skipping creation.")
            return
        
        # Create default RAG collection
        default_collection = RAGCollection(
            name="default",
            description="Default collection for testing",
            embedding_model="multi-qa-mpnet-base-dot-v1"
        )
        db.add(default_collection)
        
        # Create a sample compliance agent
        sample_agent = ComplianceAgent(
            name="Privacy Compliance Agent",
            model_name="gpt4",
            system_prompt="You are a privacy compliance expert. Analyze data for privacy violations.",
            user_prompt_template="Analyze this data for privacy compliance: {input}",
            created_by="system"
        )
        db.add(sample_agent)
        
        db.commit()
        print("Default data created successfully!")
        
    except Exception as e:
        print(f"Error creating default data: {e}")
        db.rollback()
    finally:
        db.close()

def update_agent_performance(agent_id: int, response_time_ms: int, success: bool):
    """Update agent performance metrics."""
    db = SessionLocal()
    try:
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        if agent:
            agent.total_queries += 1
            
            # Update average response time
            if agent.avg_response_time_ms is None:
                agent.avg_response_time_ms = response_time_ms
            else:
                # Calculate rolling average
                total_time = agent.avg_response_time_ms * (agent.total_queries - 1) + response_time_ms
                agent.avg_response_time_ms = total_time / agent.total_queries
            
            # Update success rate
            if agent.success_rate is None:
                agent.success_rate = 1.0 if success else 0.0
            else:
                # Calculate rolling success rate
                total_successes = agent.success_rate * (agent.total_queries - 1) + (1 if success else 0)
                agent.success_rate = total_successes / agent.total_queries
            
            db.commit()
    except Exception as e:
        print(f"Error updating agent performance: {e}")
        db.rollback()
    finally:
        db.close()

def log_compliance_result(
    agent_id: int,
    data_sample: str,
    compliant: Optional[bool],
    confidence_score: Optional[float],
    reason: str,
    raw_response: str,
    processing_method: str,
    response_time_ms: int,
    model_used: str,
    session_id: Optional[str] = None
):
    """Log a compliance check result."""
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
        
        # Update agent performance
        update_agent_performance(agent_id, response_time_ms, compliant is not None)
        
    except Exception as e:
        print(f"Error logging compliance result: {e}")
        db.rollback()
    finally:
        db.close()

def get_agent_analytics(agent_id: int, days: int = 30) -> Dict[str, Any]:
    """Get analytics for a specific agent."""
    db = SessionLocal()
    try:
        from datetime import timedelta
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get basic agent info
        agent = db.query(ComplianceAgent).filter(ComplianceAgent.id == agent_id).first()
        if not agent:
            return {"error": "Agent not found"}
        
        # Get results in the time period
        results = db.query(ComplianceResult).filter(
            ComplianceResult.agent_id == agent_id,
            ComplianceResult.created_at >= start_date
        ).all()
        
        # Calculate metrics
        total_checks = len(results)
        compliant_checks = len([r for r in results if r.compliant is True])
        non_compliant_checks = len([r for r in results if r.compliant is False])
        unclear_checks = len([r for r in results if r.compliant is None])
        
        avg_response_time = sum([r.response_time_ms for r in results if r.response_time_ms]) / max(1, len([r for r in results if r.response_time_ms]))
        
        # Get method distribution
        method_distribution = {}
        for result in results:
            method = result.processing_method or "unknown"
            method_distribution[method] = method_distribution.get(method, 0) + 1
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "period_days": days,
            "total_checks": total_checks,
            "compliant_checks": compliant_checks,
            "non_compliant_checks": non_compliant_checks,
            "unclear_checks": unclear_checks,
            "compliance_rate": compliant_checks / max(1, total_checks),
            "avg_response_time_ms": avg_response_time,
            "method_distribution": method_distribution,
            "overall_performance": {
                "total_queries": agent.total_queries,
                "avg_response_time_ms": agent.avg_response_time_ms,
                "success_rate": agent.success_rate
            }
        }
        
    except Exception as e:
        return {"error": f"Error getting agent analytics: {e}"}
    finally:
        db.close()

# Initialize database on import
if __name__ == "__main__":
    init_db()
    create_default_data()