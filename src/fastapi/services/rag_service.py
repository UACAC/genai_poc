import os
from typing import Optional, List
from sqlalchemy.orm import Session

from services.llm_service import LLMService

class RAGService:
    def __init__(self):
        self.llm_service = LLMService()

    def query_llama(self, prompt: str, collection_name: Optional[str] = None) -> str:
        return self.llm_service.query_llama(prompt)

    def query_mistral(self, prompt: str, collection_name: Optional[str] = None) -> str:
        return self.llm_service.query_mistral(prompt)

    def query_gemma(self, prompt: str, collection_name: Optional[str] = None) -> str:
        return self.llm_service.query_gemma(prompt)

    def query_gpt(self, prompt: str, collection_name: Optional[str] = None, db: Optional[Session] = None) -> str:
        return self.llm_service.query_gpt4(prompt)


class EnhancedRAGService:
    """
    LangChain-enabled RAG service with support for multiple models
    and more flexible interaction.
    """
    def __init__(self):
        self.llm_service = LLMService()

    def query_llama(self, prompt: str, collection_name: str) -> str:
        return self.llm_service.query_with_langchain("llama3", prompt)

    def query_mistral(self, prompt: str, collection_name: str) -> str:
        return self.llm_service.query_with_langchain("mistral", prompt)

    def query_gemma(self, prompt: str, collection_name: str) -> str:
        return self.llm_service.query_with_langchain("gemma", prompt)

    def query_gpt(self, prompt: str, collection_name: str, db: Optional[Session] = None) -> str:
        return self.llm_service.query_with_langchain("gpt-4", prompt)

    def run_rag_check(self, query_text: str, collection_name: str, agent_ids: List[int], db: Session):
        from services.rag_agent_service import EnhancedRAGAgentService
        rag_agent = EnhancedRAGAgentService()
        return rag_agent.run_rag_check(query_text, collection_name, agent_ids, db)

    def run_rag_debate_sequence(self, db: Session, session_id: Optional[str], agent_ids: List[int], query_text: str, collection_name: str):
        from services.rag_agent_service import EnhancedRAGAgentService
        rag_agent = EnhancedRAGAgentService()
        return rag_agent.run_rag_debate_sequence(db, session_id, agent_ids, query_text, collection_name)
