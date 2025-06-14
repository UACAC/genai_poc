import os
import uuid
import time
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from services.llm_utils import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from services.database import (
    SessionLocal,
    ComplianceAgent,
    DebateSession,
    ChatHistory,
    log_compliance_result
)

class LLMService:
    def __init__(self):
        self.chromadb_dir = os.getenv("CHROMADB_PERSIST_DIRECTORY", "/app/chroma_db_data")
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.n_results = int(os.getenv("N_RESULTS", "3"))
        self.openai_api_key = os.getenv("OPEN_AI_API_KEY")
        self.compliance_agents = []

    def get_llm_service(self, model_name: str):
        model_name = model_name.lower()
        if model_name in ["gpt4", "gpt-4", "gpt-3.5-turbo"]:
            return get_llm(model_name=model_name)
        elif model_name in ["llama", "llama3"]:
            return get_llm(model_name=model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def get_retriever(self, collection_name: str):
        db = Chroma(persist_directory=self.chromadb_dir, collection_name=collection_name, embedding_function=self.embedding_function)
        return db.as_retriever(search_kwargs={"k": self.n_results})

    def query_model(self, model_name: str, query: str, collection_name: str, query_type: str = "rag", session_id: Optional[str] = None) -> str:
        retriever = self.get_retriever(collection_name)
        llm = self.get_llm_service(model_name)

        prompt = ChatPromptTemplate.from_template(
            "{context}\n\nQuestion: {input}"
        )
        
        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

        start_time = time.time()
        result = chain.invoke({"input": query})
        response_time_ms = int((time.time() - start_time) * 1000)

        # Save chat history
        session = SessionLocal()
        try:
            history = ChatHistory(
                user_query=query,
                response=result.get("answer", "No response generated."),
                model_used=model_name,
                collection_name=collection_name,
                query_type=query_type,
                response_time_ms=response_time_ms,
                langchain_used=True,
                session_id=session_id,
                source_documents=[doc.page_content for doc in result.get("context", [])] if result.get("context") else []
            )
            session.add(history)
            session.commit()
        except Exception as e:
            print(f"Failed to save chat history: {e}")
            session.rollback()
        finally:
            session.close()

        return result.get("answer", "No response generated."), response_time_ms

    def run_parallel_rag_checks(self, query_text: str, collection_name: str, db: Session, session_id: Optional[str] = None):
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_rag, agent, query_text, collection_name, db, session_id): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def verify_rag(self, agent: Dict[str, Any], query_text: str, collection_name: str, db: Session, session_id: Optional[str] = None):
        model_name = agent["model_name"]
        raw_text, response_time_ms = self.query_model(model_name, query_text, collection_name, query_type="rag", session_id=session_id)
        # lines = raw_text.split("\n", 1)
        
        # first_line = lines[0].lower()

        # if "yes" in first_line:
        #     compliant = True
        # elif "no" in first_line:
        #     compliant = False
        # else:
        #     compliant = None

        # reason = lines[1].strip() if len(lines) > 1 else ""

        log_compliance_result(
            agent_id=agent["id"],
            data_sample=query_text,
            # compliant=compliant,
            confidence_score=None,
            reason=reason,
            raw_response=raw_text,
            processing_method="rag",
            response_time_ms=response_time_ms,
            model_used=model_name,
            session_id=session_id
        )

        # return {"compliant": compliant, "reason": reason, "raw_text": raw_text}
        return {"raw_text": raw_text}

    def run_rag_check(self, query_text: str, collection_name: str, agent_ids: List[int], db: Session):
        self.load_selected_compliance_agents(agent_ids)
        session_id = str(uuid.uuid4())
        rag_results = self.run_parallel_rag_checks(query_text, collection_name, db, session_id=session_id)
        # bool_vals = [res["compliant"] for res in rag_results.values() if res["compliant"] is not None]
        # all_compliant = bool_vals and all(bool_vals)

        # if all_compliant:
        #     return {"overall_compliance": True, "details": rag_results}
        # else:
        for idx, agent_info in enumerate(self.compliance_agents):
            db.add(DebateSession(session_id=session_id, compliance_agent_id=agent_info["id"], debate_order=idx + 1))
        db.commit()
        debate_results = self.run_rag_debate(session_id, query_text, collection_name, db)
        return {"overall_compliance": False, "details": rag_results, "debate_results": debate_results, "session_id": session_id}

    def run_rag_debate(self, session_id: str, query_text: str, collection_name: str, db: Session):
        debate_agents = self.load_debate_agents(session_id)
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.debate_rag_compliance, agent, query_text, collection_name, db, session_id): agent["name"]
                for agent in debate_agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                results[agent_name] = future.result()
        return results

    def debate_rag_compliance(self, agent: Dict[str, Any], query_text: str, collection_name: str, db: Session, session_id: Optional[str] = None):
        model_name = agent["model_name"]
        response, response_time_ms = self.query_model(model_name, query_text, collection_name, query_type="debate", session_id=session_id)

        log_compliance_result(
            agent_id=agent["id"],
            data_sample=query_text,
            # compliant=None,
            confidence_score=None,
            reason="",  # optionally parse reason if needed
            raw_response=response,
            processing_method="debate",
            response_time_ms=response_time_ms,
            model_used=model_name,
            session_id=session_id
        )

        return response

    def run_rag_debate_sequence(self, db: Session, session_id: Optional[str], agent_ids: List[int], query_text: str, collection_name: str):
        if not session_id:
            session_id = str(uuid.uuid4())
        db.query(DebateSession).filter(DebateSession.session_id == session_id).delete()
        db.commit()
        for idx, agent_id in enumerate(agent_ids):
            db.add(DebateSession(session_id=session_id, compliance_agent_id=agent_id, debate_order=idx + 1))
        db.commit()
        debate_agents = self.load_debate_agents(session_id)
        debate_chain = []
        context = f"Original user query:\n{query_text}\n"
        for agent in debate_agents:
            agent_response = self.debate_rag_compliance(agent, context, collection_name, db, session_id=session_id)
            debate_chain.append({"agent_id": agent["id"], "agent_name": agent["name"], "response": agent_response})
            context += f"\n---\nAgent {agent['name']} responded:\n{agent_response}\n"
        return session_id, debate_chain

    def load_selected_compliance_agents(self, agent_ids: List[int]):
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = session.query(ComplianceAgent).filter(ComplianceAgent.id.in_(agent_ids)).all()
            for agent in agents:
                self.compliance_agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name.lower(),
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                })
        finally:
            session.close()

    def load_debate_agents(self, session_id: str):
        session = SessionLocal()
        try:
            debate_sessions = session.query(DebateSession).filter(DebateSession.session_id == session_id).order_by(DebateSession.debate_order).all()
            agent_ids = [ds.compliance_agent_id for ds in debate_sessions]
            agents = session.query(ComplianceAgent).filter(ComplianceAgent.id.in_(agent_ids)).all()
            agent_map = {agent.id: agent for agent in agents}
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

    def health_check(self):
        """Check service health and model availability."""
        health_status = {
            "status": "healthy",
            "chromadb_status": "connected",
            "models": {},
            "timestamp": time.time()
        }
        
        try:
            # Test ChromaDB connection
            test_db = Chroma(
                persist_directory=self.chromadb_dir, 
                collection_name="health_check",
                embedding_function=self.embedding_function
            )
            health_status["chromadb_status"] = "connected"
        except Exception as e:
            health_status["chromadb_status"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Test available models
        for model in ["gpt-4", "gpt-3.5-turbo", "llama3"]:
            try:
                llm = self.get_llm_service(model)
                health_status["models"][model] = "available"
            except Exception as e:
                health_status["models"][model] = f"unavailable: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status
