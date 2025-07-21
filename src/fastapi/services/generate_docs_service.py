# services/document_service.py
import io
import uuid
import base64
import requests
from typing import List, Optional, Union
from docx import Document
import fitz
import markdown
import os
from bs4 import BeautifulSoup
from services.rag_service import RAGService
from services.agent_service import AgentService
from services.llm_service import LLMService
from services.database import SessionLocal, ComplianceAgent
class TemplateParser:
    @staticmethod
    def extract_headings_from_docx(path: str) -> List[str]:
        doc = Document(path)
        return [p.text for p in doc.paragraphs
                if p.style.name.startswith("Heading") and p.text.strip()]

    @staticmethod
    def extract_headings_from_pdf(path: str, size_threshold: float = 16.0) -> List[str]:
        doc = fitz.open(path)
        headings = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0: continue
                for line in block["lines"]:
                    size = line["spans"][0]["size"]
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if size >= size_threshold and text:
                        headings.append(text)
        return headings

    @staticmethod
    def extract_headings_from_html(html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        headings = []
        for level in range(1,7):
            for tag in soup.find_all(f"h{level}"):
                text = tag.get_text(strip=True)
                if text:
                    headings.append(text)
        return headings

    @staticmethod
    def extract_headings_from_markdown(md: str) -> List[str]:
        lines = md.splitlines()
        return [l.lstrip("# ").strip() for l in lines
                if l.startswith("#")]

    @classmethod
    def extract(cls, path_or_str: str, is_path: bool=True) -> List[str]:
        ext = os.path.splitext(path_or_str if is_path else "")[1].lower()
        if ext == ".docx":
            return cls.extract_headings_from_docx(path_or_str)
        if ext == ".pdf":
            return cls.extract_headings_from_pdf(path_or_str)
        if ext in {".html", ".htm"}:
            html = open(path_or_str).read() if is_path else path_or_str
            return cls.extract_headings_from_html(html)
        if ext == ".md" or (not is_path and "\n" in path_or_str):
            md = open(path_or_str).read() if is_path else path_or_str
            return cls.extract_headings_from_markdown(md)
        raise ValueError(f"Unsupported template format: {ext!r}")
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
        to get its `model_name`
        """
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
        template_paths:       List[str],
        source_collections:   List[str],
        source_doc_ids:       List[str],
        agent_ids:            List[int],
        top_k:                int = 5
    ) -> List[dict]:
        out = []
        # load your agents once
        self.agent.load_selected_compliance_agents(agent_ids)

        for tpl_path in template_paths:
            headings = TemplateParser.extract(tpl_path, is_path=True)
            doc = Document()
            doc.add_heading(os.path.basename(tpl_path), level=1)

            for heading in headings:
                # 1) RAG‐retrieve source context for this section
                ctx = []
                for coll, sid in zip(source_collections, source_doc_ids):
                    docs, ok = self.rag.get_relevant_documents(sid, coll)
                    if ok:
                        ctx += docs[:top_k]
                context = "\n\n".join(ctx)

                # 2) build your per‐section prompt
                prompt = f"""### Section: {heading}

Using the following material, write the content for this section of the test plan:

{context}
"""

                # 3) call your agent’s RAG+LLM chain
                agent_meta = next(a for a in self.agent.compliance_agents
                                  if a["id"] == agent_ids[0])
                db = SessionLocal()
                try:
                    result = self.agent._invoke_chain(
                        agent=agent_meta,
                        data_sample=prompt,
                        session_id=str(uuid.uuid4()),
                        db=db
                    )
                finally:
                    db.close()

                # 4) insert into the docx
                doc.add_heading(heading, level=2)
                doc.add_paragraph(result["reason"])

            # 5) serialize back to base64
            buf = io.BytesIO()
            doc.save(buf)
            out.append({
                "title": os.path.splitext(os.path.basename(tpl_path))[0],
                "docx_b64": base64.b64encode(buf.getvalue()).decode()
            })

        return out