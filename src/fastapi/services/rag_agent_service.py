import os
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Tuple
from services.database import SessionLocal, ComplianceAgent, DebateSession
from services.llm_service import LLMService
from services.rag_service import RAGService

# Try to import enhanced services
try:
    from services.rag_service import EnhancedRAGService
    from services.agent_service import EnhancedAgentService
    ENHANCED_SERVICES_AVAILABLE = True
except ImportError:
    ENHANCED_SERVICES_AVAILABLE = False

# Initialize services
llm_service = LLMService()
rag_service = RAGService()

# Initialize enhanced services if available
if ENHANCED_SERVICES_AVAILABLE:
    enhanced_rag_service = EnhancedRAGService()
    enhanced_agent_service = EnhancedAgentService()

class EnhancedRAGAgentService:
    """
    Enhanced RAG Agent Service that leverages LangChain for better retrieval
    and agent interactions while maintaining backward compatibility.
    """

    def __init__(self):
        self.compliance_agents = []
        self.use_enhanced_services = ENHANCED_SERVICES_AVAILABLE
        
        # Choose which services to use
        if self.use_enhanced_services:
            self.rag_service = enhanced_rag_service
            self.agent_service = enhanced_agent_service
        else:
            self.rag_service = rag_service
            self.agent_service = None

    def load_selected_compliance_agents(self, agent_ids: List[int]):
        """Load the specified compliance agents from DB with enhanced chain creation."""
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = (
                session.query(ComplianceAgent)
                .filter(ComplianceAgent.id.in_(agent_ids))
                .all()
            )
            for agent in agents:
                agent_config = {
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name.lower(),
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                }
                
                # Create RAG chains if enhanced services are available
                if self.use_enhanced_services:
                    try:
                        # Create a specialized RAG chain for this agent
                        agent_config["has_enhanced_chain"] = True
                    except Exception as e:
                        print(f"Failed to create enhanced chain for agent {agent.name}: {e}")
                        agent_config["has_enhanced_chain"] = False
                else:
                    agent_config["has_enhanced_chain"] = False
                
                self.compliance_agents.append(agent_config)
        finally:
            session.close()

    def load_debate_agents(self, session_id: str):
        """Load debate agents for a specific session with enhanced chain creation."""
        session = SessionLocal()
        try:
            # Query for debate session info, ordered by debate_order
            debate_sessions = (
                session.query(DebateSession)
                .filter(DebateSession.session_id == session_id)
                .order_by(DebateSession.debate_order)
                .all()
            )
            
            # Get the agent IDs
            agent_ids = [ds.compliance_agent_id for ds in debate_sessions]
            
            # Query for the agents
            agents = (
                session.query(ComplianceAgent)
                .filter(ComplianceAgent.id.in_(agent_ids))
                .all()
            )
            
            # Create a mapping from agent ID to agent data
            agent_map = {agent.id: agent for agent in agents}
            
            # Assemble the agents in the correct order
            debate_agents = []
            for ds in debate_sessions:
                agent = agent_map.get(ds.compliance_agent_id)
                if agent:
                    agent_config = {
                        "id": agent.id,
                        "name": agent.name,
                        "model_name": agent.model_name.lower(),
                        "system_prompt": agent.system_prompt,
                        "user_prompt_template": agent.user_prompt_template,
                        "debate_order": ds.debate_order
                    }
                    
                    # Add enhanced chain if available
                    if self.use_enhanced_services:
                        try:
                            agent_config["has_enhanced_chain"] = True
                        except Exception:
                            agent_config["has_enhanced_chain"] = False
                    else:
                        agent_config["has_enhanced_chain"] = False
                    
                    debate_agents.append(agent_config)
            
            return debate_agents
        finally:
            session.close()

    def run_rag_check(self, query_text: str, collection_name: str, agent_ids: List[int], db: Session):
        """
        Enhanced RAG compliance check with better error handling and logging.
        Uses enhanced services when available, falls back to legacy.
        """
        start_time = time.time()
        
        # Load the specified RAG agents
        self.load_selected_compliance_agents(agent_ids)

        # Run the checks in parallel
        rag_results = self.run_parallel_rag_checks(query_text, collection_name, db)

        # Determine overall compliance
        bool_vals = [res["compliant"] for res in rag_results.values() if res["compliant"] is not None]
        all_compliant = bool_vals and all(bool_vals)

        processing_time = (time.time() - start_time) * 1000

        if all_compliant:
            return {
                "overall_compliance": True,
                "details": rag_results,
                "processing_time_ms": processing_time,
                "enhanced_services_used": self.use_enhanced_services,
                "agents_processed": len(agent_ids)
            }
        else:
            # Create a new session for the debate
            session_id = str(uuid.uuid4())

            # Save these agents in the DebateSession table
            for idx, agent_info in enumerate(self.compliance_agents):
                db.add(DebateSession(
                    session_id=session_id,
                    compliance_agent_id=agent_info["id"],
                    debate_order=idx + 1
                ))
            db.commit()

            # Run the RAG-based debate
            debate_results = self.run_rag_debate(session_id, query_text, collection_name, db)

            return {
                "overall_compliance": False,
                "details": rag_results,
                "debate_results": debate_results,
                "session_id": session_id,
                "processing_time_ms": processing_time,
                "enhanced_services_used": self.use_enhanced_services,
                "agents_processed": len(agent_ids)
            }

    def run_parallel_rag_checks(self, query_text: str, collection_name: str, db: Session):
        """Enhanced parallel RAG checks with better error handling."""
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_rag, agent, query_text, collection_name, db): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Handle individual agent failures gracefully
                    results[idx] = {
                        "agent_id": self.compliance_agents[idx]["id"],
                        "agent_name": self.compliance_agents[idx]["name"],
                        "compliant": None,
                        "reason": f"Error during processing: {str(e)}",
                        "raw_text": f"Error: {str(e)}",
                        "method": "error"
                    }
        return results

    def verify_rag(self, agent: Dict[str, Any], query_text: str, collection_name: str, db: Session):
        """
        Enhanced RAG verification with LangChain support and better error handling.
        """
        start_time = time.time()
        model_name = agent["model_name"]
        
        try:
            # Try enhanced RAG service first if available
            if self.use_enhanced_services:
                try:
                    raw_text = self.rag_service.query(model_name, query_text, collection_name)
                    method = "enhanced_langchain"
                except Exception as e:
                    print(f"Enhanced RAG failed for {agent['name']}: {e}")
                    # Fall back to legacy
                    raw_text = self._verify_rag_legacy(agent, query_text, collection_name)
                    method = "legacy_fallback"
            else:
                raw_text = self._verify_rag_legacy(agent, query_text, collection_name)
                method = "legacy"
            
            # Parse the response
            compliant, reason = self._parse_compliance_response(raw_text)
            
            response_time = (time.time() - start_time) * 1000
            
            result = {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "compliant": compliant,
                "reason": reason,
                "raw_text": raw_text,
                "method": method,
                "response_time_ms": response_time
            }
            
            # Log result if enhanced database features are available
            try:
                from services.database import log_compliance_result
                log_compliance_result(
                    agent_id=agent["id"],
                    data_sample=query_text[:500],  # Truncate for storage
                    compliant=compliant,
                    confidence_score=None,
                    reason=reason,
                    raw_response=raw_text,
                    processing_method=method,
                    response_time_ms=int(response_time),
                    model_used=model_name
                )
            except Exception:
                pass  # Logging is optional
            
            return result
            
        except Exception as e:
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "compliant": None,
                "reason": f"Error during RAG verification: {str(e)}",
                "raw_text": f"Error: {str(e)}",
                "method": "error",
                "response_time_ms": (time.time() - start_time) * 1000
            }
    
    def _verify_rag_legacy(self, agent: Dict[str, Any], query_text: str, collection_name: str) -> str:
        """Legacy RAG verification method."""
        model_name = agent["model_name"]

        if model_name == "gpt4" or model_name == "gpt-4":
            return rag_service.query_gpt(query_text, collection_name)
        elif model_name == "llama" or model_name == "llama3":
            return rag_service.query_llama(query_text, collection_name)
        elif model_name == "mistral":
            return rag_service.query_mistral(query_text, collection_name)
        elif model_name == "gemma":
            return rag_service.query_gemma(query_text, collection_name)
        else:
            return f"Error: Model '{agent['model_name']}' not recognized."
    
    def _parse_compliance_response(self, raw_text: str) -> Tuple[Optional[bool], str]:
        """Parse compliance response into boolean and reason."""
        lines = raw_text.split("\n", 1)
        first_line = lines[0].lower()

        if "yes" in first_line:
            compliant = True
        elif "no" in first_line:
            compliant = False
        else:
            compliant = None

        reason = lines[1].strip() if len(lines) > 1 else ""
        return compliant, reason

    def run_rag_debate(self, session_id: str, query_text: str, collection_name: str, db: Session):
        """
        Enhanced RAG debate with better error handling and performance tracking.
        """
        debate_agents = self.load_debate_agents(session_id)
        results = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.debate_rag_compliance, 
                    agent, 
                    query_text, 
                    collection_name, 
                    db
                ): agent["name"]
                for agent in debate_agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    results[agent_name] = future.result()
                except Exception as e:
                    results[agent_name] = f"Error during debate: {str(e)}"

        return results

    def debate_rag_compliance(self, agent: Dict[str, Any], query_text: str, collection_name: str, db: Session):
        """
        Enhanced debate RAG compliance with LangChain support.
        """
        model_name = agent["model_name"]
        
        # Enhanced debate prompt
        debate_prompt = f"""
        Agent {agent['name']} is participating in a compliance debate.
        
        Query under review: {query_text}
        
        Based on the retrieved information, do you find this compliant? 
        Answer 'Yes' or 'No' on the first line, then provide your detailed reasoning.
        """

        try:
            # Try enhanced service first
            if self.use_enhanced_services:
                try:
                    return self.rag_service.query(model_name, debate_prompt, collection_name)
                except Exception:
                    pass  # Fall back to legacy
            
            # Legacy approach
            if model_name == "gpt4" or model_name == "gpt-4":
                return rag_service.query_gpt(debate_prompt, collection_name)
            elif model_name == "llama" or model_name == "llama3":
                return rag_service.query_llama(debate_prompt, collection_name)
            elif model_name == "mistral":
                return rag_service.query_mistral(debate_prompt, collection_name)
            elif model_name == "gemma":
                return rag_service.query_gemma(debate_prompt, collection_name)
            else:
                return f"Error: Model '{agent['model_name']}' not recognized."
                
        except Exception as e:
            return f"Error during debate: {str(e)}"

    def run_rag_debate_sequence(self, db: Session, session_id: Optional[str], 
                               agent_ids: List[int], query_text: str, 
                               collection_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enhanced sequential RAG debate with conversation memory and better context management.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Clear prior debate records
        db.query(DebateSession).filter(DebateSession.session_id == session_id).delete()
        db.commit()

        # Insert agents in order
        for idx, agent_id in enumerate(agent_ids):
            db.add(DebateSession(
                session_id=session_id,
                compliance_agent_id=agent_id,
                debate_order=idx + 1
            ))
        db.commit()

        # Load debate agents from the database
        debate_agents = self.load_debate_agents(session_id)

        debate_chain = []
        context = f"Original user query:\n{query_text}\n"
        
        # Track debate performance
        total_start_time = time.time()

        for agent in debate_agents:
            agent_start_time = time.time()
            
            # Create contextual prompt for sequential debate
            contextual_query = f"{context}\n\nAgent {agent['name']}, considering the above discussion, evaluate compliance."
            
            try:
                agent_response = self.debate_rag_compliance(agent, contextual_query, collection_name, db)
                
                agent_response_time = (time.time() - agent_start_time) * 1000
                
                debate_entry = {
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "response": agent_response,
                    "debate_order": agent["debate_order"],
                    "response_time_ms": agent_response_time,
                    "method": "enhanced" if self.use_enhanced_services else "legacy"
                }
                
                # Store debate response in database if enhanced features available
                try:
                    from services.database import SessionLocal
                    temp_db = SessionLocal()
                    debate_session = temp_db.query(DebateSession).filter(
                        DebateSession.session_id == session_id,
                        DebateSession.compliance_agent_id == agent["id"]
                    ).first()
                    
                    if debate_session:
                        debate_session.agent_response = agent_response[:1000]  # Truncate for storage
                        debate_session.response_time_ms = int(agent_response_time)
                        debate_session.langchain_used = self.use_enhanced_services
                        temp_db.commit()
                    temp_db.close()
                except Exception:
                    pass  # Database update is optional
                
            except Exception as e:
                debate_entry = {
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "response": f"Error: {str(e)}",
                    "debate_order": agent["debate_order"],
                    "response_time_ms": (time.time() - agent_start_time) * 1000,
                    "method": "error"
                }
                agent_response = f"Error: {str(e)}"

            debate_chain.append(debate_entry)
            context += f"\n---\nAgent {agent['name']} responded:\n{agent_response}\n"

        total_processing_time = (time.time() - total_start_time) * 1000
        
        # Add summary information
        if debate_chain:
            debate_chain.append({
                "summary": {
                    "total_agents": len(debate_agents),
                    "total_processing_time_ms": total_processing_time,
                    "enhanced_services_used": self.use_enhanced_services,
                    "session_id": session_id
                }
            })

        return session_id, debate_chain

class RAGAgentService:
    """
    This service parallels your AgentService, but uses retrieval (RAG).
    Each agent is expected to do a RAG query (with LLaMA, Mistral, or Gemma) 
    and then respond in a "Yes" or "No" + explanation format.
    """

    def __init__(self):
        self.compliance_agents = []

    def load_selected_compliance_agents(self, agent_ids):
        """Load the specified compliance agents from DB."""
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = (
                session.query(ComplianceAgent)
                .filter(ComplianceAgent.id.in_(agent_ids))
                .all()
            )
            for agent in agents:
                self.compliance_agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name.lower(),  # Normalize model names
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                })
        finally:
            session.close()

    def load_debate_agents(self, session_id):
        """Load debate agents for a specific session."""
        session = SessionLocal()
        try:
            # Query for debate session info, ordered by debate_order
            debate_sessions = (
                session.query(DebateSession)
                .filter(DebateSession.session_id == session_id)
                .order_by(DebateSession.debate_order)
                .all()
            )
            
            # Get the agent IDs
            agent_ids = [ds.compliance_agent_id for ds in debate_sessions]
            
            # Query for the agents
            agents = (
                session.query(ComplianceAgent)
                .filter(ComplianceAgent.id.in_(agent_ids))
                .all()
            )
            
            # Create a mapping from agent ID to agent data
            agent_map = {agent.id: agent for agent in agents}
            
            # Assemble the agents in the correct order
            debate_agents = []
            for ds in debate_sessions:
                agent = agent_map.get(ds.compliance_agent_id)
                if agent:
                    debate_agents.append({
                        "id": agent.id,
                        "name": agent.name,
                        "model_name": agent.model_name.lower(),
                        "system_prompt": agent.system_prompt,
                        "user_prompt_template": agent.user_prompt_template,
                        "debate_order": ds.debate_order
                    })
            
            return debate_agents
        finally:
            session.close()

    def run_rag_check(self, query_text: str, collection_name: str, agent_ids: list[int], db: Session):
        """
        1) Runs parallel RAG checks (one per agent).
        2) If all say "Yes," returns a final result. Otherwise, spawns a new 
        debate session & runs them in the same manner, returning the debate results.
        """
        # 1) Load the specified RAG agents
        self.load_selected_compliance_agents(agent_ids)

        # 2) Run the checks in parallel
        rag_results = self.run_parallel_rag_checks(query_text, collection_name, db)

        # 3) Determine overall compliance
        bool_vals = [res["compliant"] for res in rag_results.values() if res["compliant"] is not None]
        all_compliant = bool_vals and all(bool_vals)

        if all_compliant:
            return {
                "overall_compliance": True,
                "details": rag_results
            }
        else:
            # 4) Create a new session for the debate
            session_id = str(uuid.uuid4())

            # Save these agents in the DebateSession table
            for idx, agent_info in enumerate(self.compliance_agents):
                db.add(DebateSession(
                    session_id=session_id,
                    compliance_agent_id=agent_info["id"],
                    debate_order=idx + 1
                ))
            db.commit()

            # 5) Run the RAG-based debate
            debate_results = self.run_rag_debate(session_id, query_text, collection_name, db)

            return {
                "overall_compliance": False,
                "details": rag_results,
                "debate_results": debate_results,
                "session_id": session_id
            }

    def run_parallel_rag_checks(self, query_text: str, collection_name: str, db: Session):
        """Calls each agent in parallel, collecting yes/no answers from RAG queries."""
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_rag, agent, query_text, collection_name, db): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def verify_rag(self, agent: dict, query_text: str, collection_name: str, db: Session):
        """
        1) Perform a RAG query using either LLaMA, Mistral, or Gemma.
        2) Parse the result into a "Yes"/"No" + explanation. 
        """
        model_name = agent["model_name"]

        if model_name == "llama" or model_name == "llama3":
            raw_text = rag_service.query_llama(query_text, collection_name)
        elif model_name == "mistral":
            raw_text = rag_service.query_mistral(query_text, collection_name)
        elif model_name == "gemma":
            raw_text = rag_service.query_gemma(query_text, collection_name)
        else:
            raw_text = f"Error: Model '{agent['model_name']}' not recognized."

        # The agent might produce any text; we assume they say "Yes" or "No" 
        # on the first line, plus an explanation after.
        lines = raw_text.split("\n", 1)
        first_line = lines[0].lower()

        if "yes" in first_line:
            compliant = True
        elif "no" in first_line:
            compliant = False
        else:
            compliant = None

        reason = lines[1].strip() if len(lines) > 1 else ""

        return {
            "compliant": compliant,
            "reason": reason,
            "raw_text": raw_text
        }

    def run_rag_debate(self, session_id: str, query_text: str, collection_name: str, db: Session):
        """
        Runs a debate session where multiple agents evaluate the query.
        """
        debate_agents = self.load_debate_agents(session_id)
        results = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.debate_rag_compliance, agent, query_text, collection_name, db): agent["name"]
                for agent in debate_agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                results[agent_name] = future.result()

        return results

    def debate_rag_compliance(self, agent: dict, query_text: str, collection_name: str, db: Session):
        """
        Runs an additional retrieval-based check during the debate phase.
        """
        model_name = agent["model_name"]

        if model_name == "llama" or model_name == "llama3":
            raw_text = rag_service.query_llama(query_text, collection_name)
        elif model_name == "mistral":
            raw_text = rag_service.query_mistral(query_text, collection_name)
        elif model_name == "gemma":
            raw_text = rag_service.query_gemma(query_text, collection_name)
        else:
            raw_text = f"Error: Model '{agent['model_name']}' not recognized."

        return raw_text

    def run_rag_debate_sequence(self, db: Session, session_id: str | None, agent_ids: list[int], query_text: str, collection_name: str):
        """
        Runs a sequential debate session with multiple agents.
        """

        if not session_id:
            session_id = str(uuid.uuid4())

        # Clear prior debate records
        db.query(DebateSession).filter(DebateSession.session_id == session_id).delete()
        db.commit()

        # Insert agents in order
        for idx, agent_id in enumerate(agent_ids):
            db.add(DebateSession(
                session_id=session_id,
                compliance_agent_id=agent_id,
                debate_order=idx + 1
            ))
        db.commit()

        # Load debate agents from the database
        debate_agents = self.load_debate_agents(session_id)

        debate_chain = []
        context = f"Original user query:\n{query_text}\n"

        for agent in debate_agents:
            agent_response = self.debate_rag_compliance(agent, context, collection_name, db)

            debate_chain.append({
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "response": agent_response
            })

            context += f"\n---\nAgent {agent['name']} responded:\n{agent_response}\n"

        return session_id, debate_chain