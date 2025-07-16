# services/document_service.py
import io
import uuid
import base64
import requests
from typing import List, Optional
from docx import Document
from services.rag_service import RAGService
from services.agent_service import AgentService
from services.llm_service import LLMService
from services.database import SessionLocal, ComplianceAgent

class DocumentService:
    def __init__(
        self,
        rag_service: RAGService,
        agent_service: AgentService,
        llm_service: LLMService,
        chroma_url: str,
        agent_api_url: str,
    ):
        self.rag = rag_service
        self.agent = agent_service
        self.llm = llm_service
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
        collection: str,
        ) -> str:
        """
        Use your LLMService.query_model to do a RAG-enabled completion.
        We assume that in your DB you can look up the ComplianceAgent
        to get its `model_name` (e.g. "gpt-4").
        """
        # load the agent metadata so we know which model to call
        # (you can cache this lookup if you want)
        
        session = SessionLocal()
        try:
            agent = session.query(ComplianceAgent).get(agent_id)
            model_name = agent.model_name.lower()
        finally:
            session.close()

        # ask your LLMService to do RAG over `collection` + `prompt`
        answer, _ = self.llm.query_model(
            model_name=model_name,
            query=prompt,
            collection_name=collection,
            query_type="rag",
        )
        return answer


    def generate_documents(
        self,
        template_collection: str,
        template_doc_ids:    Optional[List[str]]    = None,
        source_collections:  Optional[List[str]]    = None,
        source_doc_ids:      Optional[List[str]]    = None,
        agent_ids:           List[int]              = [],
        use_rag:             bool                   = True,
        top_k:               int                    = 5,
    ) -> List[dict]:
        # --- 1) RAG‐retrieve template skeletons ---
        templates = []
        if use_rag and template_doc_ids:
            for tid in template_doc_ids:
                docs, ok = self.rag.get_relevant_documents(tid, template_collection)
                if ok:
                    templates.append("\n\n".join(docs[:top_k]))
        else:
            templates = self._fetch_templates(template_collection)

        # --- 2) RAG‐retrieve source requirements ---
        sources = []
        if use_rag and source_collections and source_doc_ids:
            for coll, sid in zip(source_collections, source_doc_ids):
                docs, ok = self.rag.get_relevant_documents(sid, coll)
                if ok:
                    sources.append("\n\n".join(docs[:top_k]))

        out = []
        for i, tmpl in enumerate(templates):
            # --- 3) Build the prompt ---
            prompt = "\n\n".join(filter(None, [
                "TEMPLATE SKELETON:",
                tmpl,
                "SOURCE REQUIREMENTS:",
                "\n\n".join(sources),
                "Please fill out the above template using the source requirements and produce a complete test plan."
            ]))

            # --- 4) For each agent, invoke exactly one RAG→LLM chain ---
            # we'll collect all outputs in turn; you could also just pick one agent.
            analyses = []
            # make sure we load all agents just once per call
            self.agent.load_selected_compliance_agents(agent_ids)

            for aid in agent_ids:
                # pick the matching metadata dict
                agent_meta = next(a for a in self.agent.compliance_agents if a["id"] == aid)

                # invoke the unified direct-LLM path (which uses that agent's system+user prompts)
                db = SessionLocal()
                try:
                    result = self.agent._invoke_chain(
                        agent=agent_meta,
                        data_sample=prompt,
                        session_id=str(uuid.uuid4()),
                        db=db
                    )
                    analyses.append(result["reason"])
                finally:
                    db.close()

            # join multiple agent outputs if you like, or just pick the first
            analysis = "\n\n---\n\n".join(analyses)

            # --- 5) Pack into a DOCX ---
            doc = Document()
            title = f"tmpl_{i}"
            doc.add_heading(title, level=1)
            doc.add_paragraph(analysis)
            buf = io.BytesIO()
            doc.save(buf)
            out.append({
                "title":   title,
                "docx_b64": base64.b64encode(buf.getvalue()).decode("utf-8")
            })

        return out

