import os
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from services.database import (
    SessionLocal, ComplianceAgent, DebateSession, log_compliance_result,log_agent_response, 
    log_agent_session, log_agent_response, complete_agent_session,
    SessionType, AnalysisType
)
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field 
from langchain.output_parsers import PydanticOutputParser
from schemas.database_schema import ComplianceResultSchema

class ComplianceResult(BaseModel):
    # compliant: bool = Field(description="Whether the content is compliant")
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
            llms["gpt-4"] = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, temperature=0.7)
        llms["llama3"] = OllamaLLM(model="llama3", temperature=0.7)
        # llms["mistral"] = OllamaLLM(model="mistral", temperature=0.7)
        # llms["gemma"] = OllamaLLM(model="gemma", temperature=0.7)
        return llms

    def load_selected_compliance_agents(self, agent_ids: List[int]) -> None:
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = session.query(ComplianceAgent).filter(ComplianceAgent.id.in_(agent_ids)).all()
            for agent in agents:
                model_name = agent.model_name.lower().strip()
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
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        session_type = SessionType.MULTI_AGENT_DEBATE if len(agent_ids) > 1 else SessionType.SINGLE_AGENT
        log_agent_session(
            session_id=session_id,
            session_type=session_type,
            analysis_type=AnalysisType.DIRECT_LLM,
            user_query=data_sample
        )
        
        self.load_selected_compliance_agents(agent_ids)
        results = self.run_parallel_checks(data_sample, session_id, db)
        # compliant = all(res.get("compliant") for res in results.values())

        # if compliant:
        #     total_time = int((time.time() - start_time) * 1000)
        #     complete_agent_session(
        #         session_id=session_id,
        #         overall_result={"overall_compliance": True, "details": results},
        #         agent_count=len(agent_ids),
        #         total_response_time_ms=total_time,
        #         status='completed'
        #     )
            
        #     return {"overall_compliance": True, "details": results, "session_id": session_id}

        for idx, agent in enumerate(self.compliance_agents):
            db.add(DebateSession(session_id=session_id, compliance_agent_id=agent["id"], debate_order=idx + 1))
        db.commit()
        
        debate_results = self.run_debate(session_id, data_sample, db)
        
        total_time = int((time.time() - start_time) * 1000)
        complete_agent_session(
            session_id=session_id,
            overall_result={
                # "overall_compliance": False,
                "details": results,
                "debate_results": debate_results
            },
            agent_count=len(agent_ids),
            total_response_time_ms=total_time,
            status='completed'
        )
        
        return {
            # "overall_compliance": False,
            "details": results,
            "debate_results": debate_results,
            "session_id": session_id
        }

    def run_parallel_checks(self, data_sample: str, session_id: str, db: Session) -> Dict[int, Dict[str, Any]]:
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._invoke_chain, agent, data_sample, session_id, db): i for i, agent in enumerate(self.compliance_agents)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def _invoke_chain(self, agent: Dict[str, Any], data_sample: str, session_id: str, db: Session) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            result = agent["structured_chain"].invoke({"data_sample": data_sample})
            response_time_ms = int((time.time() - start_time) * 1000)
            
            log_agent_response(
                session_id=session_id,
                agent_id=agent["id"],
                response_text=str(result),
                processing_method="langchain_structured",
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                # compliant=result.compliant,
                confidence_score=result.confidence,
                analysis_summary=result.reason
            )
            
            log_compliance_result(
                agent_id=agent["id"],
                data_sample=data_sample,
                # compliant=result.compliant,
                confidence_score=result.confidence,
                reason=result.reason,
                raw_response=str(result),
                processing_method="langchain_structured",
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                session_id=session_id
            )
            
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                # "compliant": result.compliant,
                "reason": result.reason,
                "confidence": result.confidence,
                "method": "langchain_structured"
            }
        except Exception:
            raw_text = agent["chain"].invoke({"data_sample": data_sample})
            response_time_ms = int((time.time() - start_time) * 1000)
            # compliant, reason = self._parse_compliance_response(raw_text)
            reason = raw_text
            log_agent_response(
                session_id=session_id,
                agent_id=agent["id"],
                response_text=raw_text,
                processing_method="langchain_fallback",
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                # compliant=compliant,
                analysis_summary=reason
            )
            
            log_compliance_result(
                agent_id=agent["id"],
                data_sample=data_sample,
                # compliant=compliant,
                confidence_score=None,
                reason=reason,
                raw_response=raw_text,
                processing_method="langchain",
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                session_id=session_id
            )
            
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                # "compliant": compliant,
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

    def run_debate(self, session_id: str, data_sample: str, db: Session) -> Dict[str, str]:
        agents = self._load_debate_agents(session_id)
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._debate, agent, data_sample, session_id, db, idx+1): agent["name"] 
                for idx, agent in enumerate(agents)
            }
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

    def _debate(self, agent: Dict[str, Any], data_sample: str, session_id: str, db: Session, sequence_order: int = None) -> str:
        start_time = time.time()
        
        try:
            response = agent["chain"].invoke({"data_sample": data_sample})
            response_time_ms = int((time.time() - start_time) * 1000)
            
            log_agent_response(
                session_id=session_id,
                agent_id=agent["id"],
                response_text=response,
                processing_method="debate",
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                sequence_order=sequence_order
            )
            
            log_compliance_result(
                agent_id=agent["id"],
                data_sample=data_sample,
                # compliant=None,
                confidence_score=None,
                reason="",
                raw_response=response,
                processing_method="debate",
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                session_id=session_id
            )
            return response
        except Exception as e:
            return f"Error during debate: {str(e)}"
