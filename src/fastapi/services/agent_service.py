import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy.orm import Session
from services.database import SessionLocal, ComplianceAgent, DebateSession
from services.llm_service import LLMService

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from schemas.compliance import ComplianceResultSchema


class ComplianceResult(BaseModel):
    compliant: bool = Field(description="Whether the content is compliant")
    reason: str = Field(description="Explanation for the compliance decision")
    confidence: Optional[float] = Field(default=None, description="Confidence score")

class AgentService:
    """AgentService using LangChain for compliance verification and debate."""

    def __init__(self):
        self.compliance_agents: List[Dict[str, Any]] = []
        self.llms = self._initialize_llms()
        self.compliance_parser = PydanticOutputParser(pydantic_object=ComplianceResultSchema)

    def _initialize_llms(self) -> Dict[str, Any]:
        openai_api_key = os.getenv("OPEN_AI_API_KEY")
        llms = {}
        if openai_api_key:
            llms["gpt4"] = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0.7)
        llms["llama3"] = Ollama(model="llama3", temperature=0.7)
        llms["mistral"] = Ollama(model="mistral", temperature=0.7)
        llms["gemma"] = Ollama(model="gemma", temperature=0.7)
        return llms

    def load_selected_compliance_agents(self, agent_ids: List[int]) -> None:
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = session.query(ComplianceAgent).filter(ComplianceAgent.id.in_(agent_ids)).all()
            for agent in agents:
                model_name = agent.model_name.lower()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", agent.system_prompt),
                    ("human", agent.user_prompt_template)
                ])
                chain = prompt | self.llms.get(model_name) | StrOutputParser()
                structured_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"{agent.system_prompt}\n\n{self.compliance_parser.get_format_instructions()}"),
                    ("human", agent.user_prompt_template)
                ])
                structured_chain = structured_prompt | self.llms.get(model_name) | self.compliance_parser
                self.compliance_agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": model_name,
                    "chain": chain,
                    "structured_chain": structured_chain
                })
        finally:
            session.close()

    def run_compliance_check(self, data_sample: str, agent_ids: List[int], db: Session) -> Dict[str, Any]:
        self.load_selected_compliance_agents(agent_ids)
        results = self.run_parallel_checks(data_sample)
        compliant = all(res.get("compliant") for res in results.values())

        if compliant:
            return {"overall_compliance": True, "details": results}

        session_id = str(uuid.uuid4())
        for idx, agent in enumerate(self.compliance_agents):
            db.add(DebateSession(session_id=session_id, compliance_agent_id=agent["id"], debate_order=idx + 1))
        db.commit()
        debate_results = self.run_debate(session_id, data_sample)
        return {
            "overall_compliance": False,
            "details": results,
            "debate_results": debate_results,
            "session_id": session_id
        }

    def run_parallel_checks(self, data_sample: str) -> Dict[int, Dict[str, Any]]:
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._invoke_chain, agent, data_sample): i for i, agent in enumerate(self.compliance_agents)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def _invoke_chain(self, agent: Dict[str, Any], data_sample: str) -> Dict[str, Any]:
        try:
            result = agent["structured_chain"].invoke({"input": data_sample})
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "compliant": result.compliant,
                "reason": result.reason,
                "confidence": result.confidence,
                "method": "langchain_structured"
            }
        except Exception:
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

    def _parse_compliance_response(self, raw_text: str) -> Tuple[Optional[bool], str]:
        lines = raw_text.split("\n", 1)
        first_line = lines[0].lower()
        if "yes" in first_line:
            return True, lines[1].strip() if len(lines) > 1 else ""
        elif "no" in first_line:
            return False, lines[1].strip() if len(lines) > 1 else ""
        return None, raw_text

    def run_debate(self, session_id: str, data_sample: str) -> Dict[str, str]:
        agents = self._load_debate_agents(session_id)
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._debate, agent, data_sample): agent["name"] for agent in agents}
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        return results

    def _load_debate_agents(self, session_id: str) -> List[Dict[str, Any]]:
        session = SessionLocal()
        debate_agents = []
        try:
            records = session.query(DebateSession).filter(DebateSession.session_id == session_id).order_by(DebateSession.debate_order).all()
            for record in records:
                agent = record.compliance_agent
                model_name = agent.model_name.lower()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", agent.system_prompt),
                    ("human", agent.user_prompt_template)
                ])
                chain = prompt | self.llms.get(model_name) | StrOutputParser()
                debate_agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": model_name,
                    "chain": chain
                })
        finally:
            session.close()
        return debate_agents

    def _debate(self, agent: Dict[str, Any], data_sample: str) -> str:
        prompt = (
            f"Agent {agent['name']} is reviewing: {data_sample}. Do you find this compliant? Answer 'Yes' or 'No' and explain."
        )
        try:
            return agent["chain"].invoke({"input": prompt})
        except Exception as e:
            return f"Error during debate: {str(e)}"
