from typing import List, Optional
from pydantic import Field
from services.rag_service import RAGService
from services.llm_service import LLMService

class EvaluationService:
    def __init__(self, rag: RAGService, llm: LLMService):
        self.rag = rag
        self.llm = llm

    def evaluate_document(
        self,
        document_id: str,
        collection_name: str,
        prompt: str,
        top_k: Optional[int] = Field(5),
        model_name: Optional[str] = Field(...)
    ) -> str:
        # 1) RAG‐fetch the most relevant chunks of your document
        chunks, _ = self.rag.get_relevant_documents(document_id, collection_name)
        context = "\n\n".join(chunks[:top_k])

        # 2) stitch your user’s prompt onto those chunks
        full_prompt = f"""
Here’s the relevant context from document `{document_id}`:

{context}

---  
Now: {prompt}
""".strip()

        # 3) call your LLMService just like you do elsewhere
        answer, _ = self.llm.query_model(
            model_name=model_name or "gpt-4",
            query=full_prompt,
            collection_name=collection_name,
            query_type="rag"
        )
        return answer
