import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Database imports
from sqlalchemy.orm import Session
from services.database import SessionLocal, ComplianceAgent, ComplianceSequence, DebateSession
from services.llm_service import LLMService

# LangChain imports
try:
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain.output_parsers import PydanticOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Initialize legacy service
llm_service = LLMService()

class ComplianceResult(BaseModel):
    """Structured output for compliance results."""
    compliant: bool = Field(description="Whether the content is compliant")
    reason: str = Field(description="Explanation for the compliance decision")
    confidence: Optional[float] = Field(description="Confidence score (0-1)", default=None)

class EnhancedAgentService:
    """Enhanced service for managing compliance checks and debates using LangChain."""

    def __init__(self):
        self.compliance_agents = []
        self.use_langchain = LANGCHAIN_AVAILABLE
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain_components()
        
    def _initialize_langchain_components(self):
        """Initialize LangChain LLMs and components."""
        self.llms = {}
        
        # Initialize OpenAI if available
        openai_api_key = os.getenv("OPEN_AI_API_KEY")
        if openai_api_key:
            self.llms["gpt-4"] = ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=openai_api_key,
                temperature=0.7
            )
            self.llms["gpt4"] = self.llms["gpt-4"]  # Alias
        
        # Initialize Ollama models
        self.llms["llama3"] = Ollama(model="llama3", temperature=0.7)
        self.llms["llama"] = self.llms["llama3"]  # Alias
        self.llms["mistral"] = Ollama(model="mistral", temperature=0.7)
        self.llms["gemma"] = Ollama(model="gemma", temperature=0.7)
        
        # Initialize output parser for structured responses
        self.compliance_parser = PydanticOutputParser(pydantic_object=ComplianceResult)
    
    def create_agent_chain(self, agent_config: Dict[str, Any]):
        """Create a LangChain chain for an agent."""
        if not self.use_langchain:
            return None
            
        model_name = agent_config["model_name"].lower()
        llm = self.llms.get(model_name)
        
        if not llm:
            raise ValueError(f"Model '{model_name}' not supported")
        
        # Create compliance checking prompt
        compliance_prompt = ChatPromptTemplate.from_messages([
            ("system", agent_config["system_prompt"]),
            ("human", agent_config["user_prompt_template"]),
            ("human", "Respond with 'Yes' or 'No' on the first line, followed by your explanation.")
        ])
        
        # Create the chain
        chain = compliance_prompt | llm | StrOutputParser()
        return chain
    
    def create_structured_agent_chain(self, agent_config: Dict[str, Any]):
        """Create a LangChain chain with structured output."""
        if not self.use_langchain:
            return None
            
        model_name = agent_config["model_name"].lower()
        llm = self.llms.get(model_name)
        
        if not llm:
            raise ValueError(f"Model '{model_name}' not supported")
        
        # Create structured prompt with output format instructions
        format_instructions = self.compliance_parser.get_format_instructions()
        
        structured_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{agent_config['system_prompt']}\n\n{format_instructions}"),
            ("human", agent_config["user_prompt_template"])
        ])
        
        # Create chain with structured output
        chain = structured_prompt | llm | self.compliance_parser
        return chain

    def load_selected_compliance_agents(self, agent_ids: List[int]):
        """Load only the selected compliance agents from the database."""
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = session.query(ComplianceAgent).filter(ComplianceAgent.id.in_(agent_ids)).all()
            for agent in agents:
                agent_config = {
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name.lower(),
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                }
                
                # Create LangChain chain if available
                if self.use_langchain:
                    try:
                        agent_config["chain"] = self.create_agent_chain(agent_config)
                        agent_config["structured_chain"] = self.create_structured_agent_chain(agent_config)
                    except Exception as e:
                        # Fallback to legacy if chain creation fails
                        agent_config["chain"] = None
                        agent_config["structured_chain"] = None
                
                self.compliance_agents.append(agent_config)
        finally:
            session.close()

    def run_compliance_check(self, data_sample: str, agent_ids: List[int], db: Session):
        """
        Enhanced compliance check with LangChain support.
        Falls back to legacy implementation if LangChain is not available.
        """
        # Load the specified agents
        self.load_selected_compliance_agents(agent_ids)

        # Run parallel checks
        compliance_results = self.run_parallel_compliance_checks(data_sample)

        # Determine overall compliance
        bool_vals = [res["compliant"] for res in compliance_results.values() if res["compliant"] is not None]
        all_compliant = bool_vals and all(bool_vals)

        if all_compliant:
            return {
                "overall_compliance": True,
                "details": compliance_results,
                "langchain_used": self.use_langchain
            }
        else:
            session_id = str(uuid.uuid4())

            # Store agents in the DebateSession table
            for idx, agent_info in enumerate(self.compliance_agents):
                db.add(DebateSession(
                    session_id=session_id,
                    compliance_agent_id=agent_info["id"],
                    debate_order=idx + 1
                ))
            db.commit()

            # Run debate
            debate_results = self.run_debate(session_id, data_sample)

            return {
                "overall_compliance": False,
                "details": compliance_results,
                "debate_results": debate_results,
                "session_id": session_id,
                "langchain_used": self.use_langchain
            }

    def run_parallel_compliance_checks(self, data_sample: str):
        """Run compliance checks in parallel with LangChain or legacy support."""
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_compliance, agent, data_sample): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def verify_compliance(self, agent: Dict[str, Any], data_sample: str):
        """
        Enhanced compliance verification with LangChain support.
        Falls back to legacy if LangChain chain is not available.
        """
        # Try LangChain approach first
        if self.use_langchain and agent.get("chain"):
            try:
                return self._verify_compliance_langchain(agent, data_sample)
            except Exception as e:
                # Fall back to legacy on error
                print(f"LangChain verification failed for agent {agent['name']}: {e}")
        
        # Use legacy approach
        return self._verify_compliance_legacy(agent, data_sample)
    
    def _verify_compliance_langchain(self, agent: Dict[str, Any], data_sample: str):
        """Verify compliance using LangChain."""
        # Try structured output first
        if agent.get("structured_chain"):
            try:
                result = agent["structured_chain"].invoke({"input": data_sample})
                return {
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "compliant": result.compliant,
                    "reason": result.reason,
                    "confidence": result.confidence,
                    "raw_text": f"Compliant: {result.compliant}\nReason: {result.reason}",
                    "method": "langchain_structured"
                }
            except Exception:
                # Fall back to regular chain
                pass
        
        # Use regular chain
        if agent.get("chain"):
            raw_text = agent["chain"].invoke({"input": data_sample})
            compliant, reason = self._parse_compliance_response(raw_text)
            
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "compliant": compliant,
                "reason": reason,
                "raw_text": raw_text,
                "method": "langchain"
            }
        
        raise Exception("No LangChain chain available")
    
    def _verify_compliance_legacy(self, agent: Dict[str, Any], data_sample: str):
        """Legacy compliance verification method."""
        model_name = agent["model_name"]

        if model_name == "gpt-4" or model_name == "gpt4":
            raw_text = llm_service.query_gpt4(prompt=data_sample).strip()
        elif model_name == "llama" or model_name == "llama3":
            raw_text = llm_service.query_llama(prompt=data_sample).strip()
        elif model_name == "mistral":
            raw_text = llm_service.query_mistral(prompt=data_sample).strip()
        elif model_name == "gemma":
            raw_text = llm_service.query_gemma(prompt=data_sample).strip()
        else:
            raw_text = f"Error: Model '{model_name}' not recognized."

        compliant, reason = self._parse_compliance_response(raw_text)

        return {
            "agent_id": agent["id"],
            "agent_name": agent["name"],
            "compliant": compliant,
            "reason": reason,
            "raw_text": raw_text,
            "method": "legacy"
        }
    
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

    def run_debate(self, session_id: str, data_sample: str):
        """Run a debate session with enhanced LangChain support."""
        debate_agents = self.load_debate_agents(session_id)

        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.debate_compliance, agent, data_sample): agent["name"]
                for agent in debate_agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                results[agent_name] = future.result()

        return results

    def load_debate_agents(self, session_id: str):
        """Load debate agents with LangChain chain creation."""
        session = SessionLocal()
        debate_agents = []
        try:
            debate_records = (
                session.query(DebateSession)
                .filter(DebateSession.session_id == session_id)
                .order_by(DebateSession.debate_order)
                .all()
            )
            for record in debate_records:
                agent = record.compliance_agent
                agent_config = {
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name.lower(),
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                }
                
                # Create LangChain chain for debate
                if self.use_langchain:
                    try:
                        agent_config["chain"] = self.create_agent_chain(agent_config)
                    except Exception:
                        agent_config["chain"] = None
                
                debate_agents.append(agent_config)
        finally:
            session.close()
        return debate_agents

    def debate_compliance(self, agent: Dict[str, Any], data_sample: str):
        """Enhanced debate compliance with LangChain support."""
        debate_prompt = (
            f"Agent {agent['name']} is evaluating this data again:\n"
            f"{data_sample}\n\n"
            "Do you find this compliant? Answer 'Yes' or 'No' and explain why."
        )

        # Try LangChain approach
        if self.use_langchain and agent.get("chain"):
            try:
                return agent["chain"].invoke({"input": debate_prompt})
            except Exception as e:
                print(f"LangChain debate failed for agent {agent['name']}: {e}")

        # Fall back to legacy
        model_name = agent["model_name"]

        if model_name == "gpt-4" or model_name == "gpt4":
            response_text = llm_service.query_gpt4(prompt=debate_prompt)
        elif model_name == "llama" or model_name == "llama3":
            response_text = llm_service.query_llama(prompt=debate_prompt)
        elif model_name == "mistral":
            response_text = llm_service.query_mistral(prompt=debate_prompt)
        elif model_name == "gemma":
            response_text = llm_service.query_gemma(prompt=debate_prompt)
        else:
            return f"Error: Model '{model_name}' not recognized."

        return response_text.strip()

    def run_debate_sequence(self, db: Session, session_id: Optional[str], 
                           agent_ids: List[int], data_sample: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Enhanced sequential debate with LangChain conversation memory."""
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

        # Load debate agents
        debate_agents = self.load_debate_agents(session_id)

        # Create conversation memory for LangChain
        memory = None
        if self.use_langchain and LANGCHAIN_AVAILABLE:
            memory = ConversationBufferMemory()
            memory.chat_memory.add_user_message(f"Initial data: {data_sample}")

        debate_chain = []
        context = f"Initial data:\n{data_sample}\n"

        for agent in debate_agents:
            # Enhanced debate with context
            if self.use_langchain and agent.get("chain") and memory:
                try:
                    # Use conversation context
                    enhanced_prompt = f"{context}\n\nConsidering the above discussion, evaluate the compliance."
                    agent_response = agent["chain"].invoke({"input": enhanced_prompt})
                    memory.chat_memory.add_ai_message(f"Agent {agent['name']}: {agent_response}")
                except Exception:
                    # Fall back to legacy
                    agent_response = self.debate_compliance(agent, context)
            else:
                agent_response = self.debate_compliance(agent, context)

            debate_chain.append({
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "response": agent_response,
                "method": "langchain" if (self.use_langchain and agent.get("chain")) else "legacy"
            })

            context += f"\n---\nAgent {agent['name']} responded:\n{agent_response}\n"

        return session_id, debate_chain


# Legacy class for backward compatibility
class AgentService(EnhancedAgentService):
    """Legacy AgentService that extends EnhancedAgentService for backward compatibility."""
    
    def __init__(self):
        super().__init__()
        # Force legacy mode for this class
        self.use_langchain = False
    
    def verify_compliance(self, agent: Dict[str, Any], data_sample: str):
        """Override to use only legacy verification."""
        return self._verify_compliance_legacy(agent, data_sample)
    
    def debate_compliance(self, agent: Dict[str, Any], data_sample: str):
        """Override to use only legacy debate."""
        model_name = agent["model_name"]

        debate_prompt = (
            f"Agent {agent['name']} is evaluating this data again:\n"
            f"{data_sample}\n\n"
            "Do you find this compliant? Answer 'Yes' or 'No' and explain why."
        )

        if model_name == "gpt-4" or model_name == "gpt4":
            response_text = llm_service.query_gpt4(prompt=debate_prompt)
        elif model_name == "llama" or model_name == "llama3":
            response_text = llm_service.query_llama(prompt=debate_prompt)
        elif model_name == "mistral":
            response_text = llm_service.query_mistral(prompt=debate_prompt)
        elif model_name == "gemma":
            response_text = llm_service.query_gemma(prompt=debate_prompt)
        else:
            return f"Error: Model '{model_name}' not recognized."

        return response_text.strip()