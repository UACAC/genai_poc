import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from services.database import SessionLocal, ComplianceAgent, DebateSession
from services.llm_service import LLMService
from services.rag_service import RAGService

# We'll use the existing LLMService (which holds a reference to RAGService)
llm_service = LLMService()
rag_service = RAGService()

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