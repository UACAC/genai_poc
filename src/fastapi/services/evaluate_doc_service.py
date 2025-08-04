from typing import List, Optional
from pydantic import Field
from services.rag_service import RAGService
from services.llm_service import LLMService
import uuid

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
        model_name: Optional[str] = Field(...),
        session_id: str = None,
    ):
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

        # 3) choose model and call LLMService
        chosen = model_name or "gpt-4"

        answer, rt_ms = self.llm.query_model(
            model_name=model_name,
            query=full_prompt,
            collection_name=collection_name,
            query_type="rag",
            session_id=session_id
        )
        # 4) generate a session ID just like your chat flow
        session_id = str(uuid.uuid4())
        

        return answer, rt_ms
