# services/document_service.py
import io
import base64
import requests
from typing import List, Optional
from docx import Document
from services.rag_service import RAGService
from services.agent_service import AgentService

class DocumentService:
    def __init__(
        self,
        rag_service: RAGService,
        agent_service: AgentService,
        chroma_url: str,
        agent_api_url: str,
    ):
        self.rag = rag_service
        self.agent = agent_service
        self.chroma_url = chroma_url.rstrip("/")
        self.agent_api = agent_api_url.rstrip("/")

    def _fetch_templates(self, collection: str) -> List[str]:
        resp = requests.get(f"{self.chroma_url}/documents",
                            params={"collection_name": collection})
        resp.raise_for_status()
        return resp.json().get("documents", [])

    def _retrieve_context(self, tmpl: str, sources: List[str], top_k: int) -> str:
        pieces = []
        for coll in sources:
            docs, ok = self.rag.get_relevant_documents(tmpl, coll)
            if ok:
                pieces += docs[:top_k]
        return "\n\n".join(pieces)

    def _invoke_agent(
        self,
        prompt: str,
        agent_id: int,
        use_rag: bool,
        collection: Optional[str]
    ) -> str:
        payload = {"agent_ids": [agent_id]}
        if use_rag:
            payload.update(query_text=prompt, collection_name=collection)
            url = f"{self.agent_api}/rag-check"
            result = requests.post(url, json=payload).json()
            # RAG returns {"agent_responses": {agent_name: response,...}}
            return next(iter(result["agent_responses"].values()))
        else:
            payload.update(data_sample=prompt)
            url = f"{self.agent_api}/compliance-check"
            result = requests.post(url, json=payload).json()
            # Compliance returns {"details": {idx: {...}}}
            first = result["details"][0] if isinstance(result["details"], list) else next(result["details"].values())
            return first["reason"]

    def generate_documents(
        self,
        template_collection: str,
        source_collections: Optional[List[str]],
        agent_ids: List[int],
        use_rag: bool = True,
        top_k: int = 5
    ) -> List[dict]:
        templates = self._fetch_templates(template_collection)
        out = []

        for i, tmpl in enumerate(templates):
            # 1) build the final prompt
            if use_rag and source_collections:
                ctx = self._retrieve_context(tmpl, source_collections, top_k)
                prompt = tmpl.replace("{context}", ctx) if "{context}" in tmpl else f"{tmpl}\n\nContext:\n{ctx}"
            else:
                prompt = tmpl

            # 2) call each agent
            for aid in agent_ids:
                analysis = self._invoke_agent(prompt, aid, use_rag, source_collections[0] if source_collections else None)

                # 3) build an in-memory DOCX
                doc = Document()
                title = f"tmpl_{i}_agt_{aid}"
                doc.add_heading(title, level=1)
                doc.add_paragraph(analysis)
                buf = io.BytesIO()
                doc.save(buf)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                out.append({"title": title, "docx_b64": b64})

        return out
