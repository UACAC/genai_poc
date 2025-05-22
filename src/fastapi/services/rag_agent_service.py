import os
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy.orm import Session
from services.database import ComplianceAgent, DebateSession
from services.rag_service import RAGService, EnhancedRAGService
from services.llm_service import LLMService

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize services
rag_service = RAGService()
llm_service = LLMService()
enhanced_rag_service = EnhancedRAGService()

class EnhancedRAGAgentService:
    """
    Enhanced RAG Agent Service that supports LangChain-based agents and retrieval.
    """

    def __init__(self):
        self.compliance_agents = []
        self.rag_service = enhanced_rag_service
        self.agent_service = llm_service

    def load_selected_compliance_agents(self, db: Session, agent_ids: List[int]):
        """Load specified compliance agents with enhanced metadata."""
        self.compliance_agents = []
        agents = db.query(ComplianceAgent).filter(ComplianceAgent.id.in_(agent_ids)).all()
        for agent in agents:
            self.compliance_agents.append({
                "id": agent.id,
                "name": agent.name,
                "model_name": agent.model_name.lower(),
                "system_prompt": agent.system_prompt,
                "user_prompt_template": agent.user_prompt_template
            })

    def run_rag_check(self, query_text: str, collection_name: str, agent_ids: List[int], db: Session):
        """Run parallel RAG compliance checks."""
        self.load_selected_compliance_agents(db, agent_ids)
        rag_results = self.run_parallel_rag_checks(query_text, collection_name, db)

        bool_vals = [res["compliant"] for res in rag_results.values() if res["compliant"] is not None]
        all_compliant = bool_vals and all(bool_vals)

        if all_compliant:
            return {"overall_compliance": True, "details": rag_results}

        session_id = str(uuid.uuid4())
        for idx, agent in enumerate(self.compliance_agents):
            db.add(DebateSession(
                session_id=session_id,
                compliance_agent_id=agent["id"],
                debate_order=idx + 1
            ))
        db.commit()

        debate_results = self.run_rag_debate(session_id, query_text, collection_name, db)
        return {
            "overall_compliance": False,
            "details": rag_results,
            "debate_results": debate_results,
            "session_id": session_id
        }

    def run_parallel_rag_checks(self, query_text: str, collection_name: str, db: Session) -> Dict[int, Any]:
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_rag, agent, query_text, collection_name): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def verify_rag(self, agent: Dict[str, Any], query_text: str, collection_name: str) -> Dict[str, Any]:
        model = agent["model_name"]
        try:
            raw_text = self.rag_service.query(model, query_text, collection_name)
        except Exception as e:
            raw_text = f"Error querying model: {str(e)}"

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
        agents = self.load_debate_agents(db, session_id)
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.debate_rag_compliance, agent, query_text, collection_name): agent["name"]
                for agent in agents
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        return results

    def load_debate_agents(self, db: Session, session_id: str) -> List[Dict[str, Any]]:
        debate_records = (
            db.query(DebateSession)
            .filter(DebateSession.session_id == session_id)
            .order_by(DebateSession.debate_order)
            .all()
        )
        return [{
            "id": record.compliance_agent.id,
            "name": record.compliance_agent.name,
            "model_name": record.compliance_agent.model_name.lower(),
            "system_prompt": record.compliance_agent.system_prompt,
            "user_prompt_template": record.compliance_agent.user_prompt_template
        } for record in debate_records]

    def debate_rag_compliance(self, agent: Dict[str, Any], query_text: str, collection_name: str) -> str:
        prompt = (
            f"Agent {agent['name']} is evaluating the following query again:\n"
            f"{query_text}\n\nDo you find this compliant? Answer 'Yes' or 'No' and explain."
        )
        try:
            return self.rag_service.query(agent["model_name"], prompt, collection_name)
        except Exception as e:
            return f"Error during debate: {str(e)}"

# Legacy fallback
class RAGAgentService(EnhancedRAGAgentService):
    def __init__(self):
        super().__init__()
        self.rag_service = RAGService()
