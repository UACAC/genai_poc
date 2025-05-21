import os
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from services.database import SessionLocal, ComplianceAgent, DebateSession
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

class RAGService:
    def __init__(self):
        """Initialize ChromaDB, embedding model, and LLM clients."""
        # Configuration for ChromaDB
        self.chromadb_api = os.getenv("CHROMADB_API", "http://chromadb:8020")
        self.n_results = int(os.getenv("N_RESULTS", "3"))
        self.ollama_url = "http://llama:11434/api/generate"  # Correct port
        
        # OpenAI setup
        self.openai_api_key = os.getenv("OPEN_AI_API_KEY")
        if self.openai_api_key:
            self.openai_client = ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=self.openai_api_key
            )
        
        # Load the embedding model (produces 768-d vectors)
        self.embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        
        # For debate functionality
        self.compliance_agents = []

    # ---- CHROMADB INTERACTION METHODS ----
    
    def query_chromadb(self, query_text: str, collection_name: str, n_results: int = None) -> dict:
        """Encodes the query and retrieves relevant document chunks from ChromaDB."""
        if n_results is None:
            n_results = self.n_results

        query_embedding = self.embedding_model.encode([query_text]).tolist()
        payload = {
            "collection_name": collection_name,
            "query_embeddings": query_embedding,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        response = requests.post(f"{self.chromadb_api}/documents/query", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error querying ChromaDB: {response.text}")

    def build_context(self, query_result: dict) -> str:
        """Builds a single context string from retrieved document sections."""
        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        if not documents:
            return ""

        context_parts = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            doc_name = meta.get("document_name", "Unknown") if meta else "Unknown"
            context_parts.append(f"[{doc_name} | Score: {round(dist, 4)}] {doc}")
        return "\n\n".join(context_parts)

    # ---- MODEL INTERACTION METHODS ----
    
    def query_ollama(self, model_name: str, query_text: str, context: str) -> str:
        """Calls the Ollama API to generate a response using LLaMA, Mistral, or Gemma."""

        available_models = ["llama3", "mistral", "gemma"]
        if model_name not in available_models:
            return f"Error: Unsupported model '{model_name}'. Available: {available_models}"

        prompt = (
            "You are a helpful assistant that answers questions based on provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query_text}\n\n"
            "Answer:"
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 300
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama ({model_name}): {e}")
            return f"Error: Could not connect to Ollama for model '{model_name}'."

    # ---- RAG QUERY METHODS ----
    
    def query_gpt(self, query_text: str, collection_name: str, db=None) -> str:
        """Performs a RAG query using GPT-4."""
        if not self.openai_api_key:
            return "Error: OpenAI API key not set."
            
        query_result = self.query_chromadb(query_text, collection_name)
        context = self.build_context(query_result)

        if not context:
            return "No relevant context found in the database."

        prompt = (
            "You are a helpful assistant that answers questions based on provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query_text}\n\n"
            "Answer:"
        )
        
        response = self.openai_client.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def query_model(self, model_name: str, query_text: str, collection_name: str) -> str:
        """Performs a RAG query using the specified model via Ollama."""
        query_result = self.query_chromadb(query_text, collection_name)
        context = self.build_context(query_result)

        if not context:
            return "No relevant context found in the database."

        return self.query_ollama(model_name, query_text, context)

    def query_llama(self, query_text: str, collection_name: str) -> str:
        """Performs a RAG query using LLaMA via Ollama."""
        return self.query_model("llama3", query_text, collection_name)

    def query_mistral(self, query_text: str, collection_name: str) -> str:
        """Performs a RAG query using Mistral via Ollama."""
        return self.query_model("mistral", query_text, collection_name)

    def query_gemma(self, query_text: str, collection_name: str) -> str:
        """Performs a RAG query using Gemma via Ollama."""
        return self.query_model("gemma", query_text, collection_name)

    # ---- AGENT MANAGEMENT METHODS ----
    
    def load_selected_compliance_agents(self, agent_ids):
        """Load the specified compliance agents from DB."""
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = (
                session.query(ComplianceAgent)
                .filter(ComplianceAgent.id.in_(agent_ids))
                .all()
            )
            for agent in agents:
                self.compliance_agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name.lower(),  # Normalize model names
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                })
        finally:
            session.close()

    def load_debate_agents(self, session_id):
        """Load debate agents for a specific session."""
        session = SessionLocal()
        try:
            # Query for debate session info, ordered by debate_order
            debate_sessions = (
                session.query(DebateSession)
                .filter(DebateSession.session_id == session_id)
                .order_by(DebateSession.debate_order)
                .all()
            )
            
            # Get the agent IDs
            agent_ids = [ds.compliance_agent_id for ds in debate_sessions]
            
            # Query for the agents
            agents = (
                session.query(ComplianceAgent)
                .filter(ComplianceAgent.id.in_(agent_ids))
                .all()
            )
            
            # Create a mapping from agent ID to agent data
            agent_map = {agent.id: agent for agent in agents}
            
            # Assemble the agents in the correct order
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

    # ---- RAG COMPLIANCE CHECK METHODS ----
    
    def run_rag_check(self, query_text: str, collection_name: str, agent_ids: list[int], db: Session):
        """
        1) Runs parallel RAG checks (one per agent).
        2) If all say "Yes," returns a final result. Otherwise, spawns a new 
        debate session & runs them in the same manner, returning the debate results.
        """
        # 1) Load the specified RAG agents
        self.load_selected_compliance_agents(agent_ids)

        # 2) Run the checks in parallel
        rag_results = self.run_parallel_rag_checks(query_text, collection_name, db)

        # 3) Determine overall compliance
        bool_vals = [res["compliant"] for res in rag_results.values() if res["compliant"] is not None]
        all_compliant = bool_vals and all(bool_vals)

        if all_compliant:
            return {
                "overall_compliance": True,
                "details": rag_results
            }
        else:
            # 4) Create a new session for the debate
            session_id = str(uuid.uuid4())

            # Save these agents in the DebateSession table
            for idx, agent_info in enumerate(self.compliance_agents):
                db.add(DebateSession(
                    session_id=session_id,
                    compliance_agent_id=agent_info["id"],
                    debate_order=idx + 1
                ))
            db.commit()

            # 5) Run the RAG-based debate
            debate_results = self.run_rag_debate(session_id, query_text, collection_name, db)

            return {
                "overall_compliance": False,
                "details": rag_results,
                "debate_results": debate_results,
                "session_id": session_id
            }

    def run_parallel_rag_checks(self, query_text: str, collection_name: str, db: Session):
        """Calls each agent in parallel, collecting yes/no answers from RAG queries."""
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_rag, agent, query_text, collection_name, db): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def verify_rag(self, agent: dict, query_text: str, collection_name: str, db: Session):
        """
        1) Perform a RAG query using either GPT-4, LLaMA, Mistral, or Gemma.
        2) Parse the result into a "Yes"/"No" + explanation. 
        """
        model_name = agent["model_name"]

        if model_name == "gpt4" or model_name == "gpt-4":
            raw_text = self.query_gpt(query_text, collection_name, db)
        elif model_name == "llama" or model_name == "llama3":
            raw_text = self.query_llama(query_text, collection_name)
        elif model_name == "mistral":
            raw_text = self.query_mistral(query_text, collection_name)
        elif model_name == "gemma":
            raw_text = self.query_gemma(query_text, collection_name)
        else:
            raw_text = f"Error: Model '{agent['model_name']}' not recognized."

        # The agent might produce any text; we assume they say "Yes" or "No" 
        # on the first line, plus an explanation after.
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

    # ---- DEBATE METHODS ----
    
    def run_rag_debate(self, session_id: str, query_text: str, collection_name: str, db: Session):
        """
        Runs a debate session where multiple agents evaluate the query.
        """
        debate_agents = self.load_debate_agents(session_id)
        results = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.debate_rag_compliance, agent, query_text, collection_name, db): agent["name"]
                for agent in debate_agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                results[agent_name] = future.result()

        return results

    def debate_rag_compliance(self, agent: dict, query_text: str, collection_name: str, db: Session):
        """
        Runs an additional retrieval-based check during the debate phase.
        """
        model_name = agent["model_name"].lower()

        if model_name == "gpt4" or model_name == "gpt-4":
            raw_text = self.query_gpt(query_text, collection_name, db)
        elif model_name == "llama" or model_name == "llama3":
            raw_text = self.query_llama(query_text, collection_name)
        elif model_name == "mistral":
            raw_text = self.query_mistral(query_text, collection_name)
        elif model_name == "gemma":
            raw_text = self.query_gemma(query_text, collection_name)
        else:
            raw_text = f"Error: Model '{agent['model_name']}' not recognized."

        return raw_text

    def run_rag_debate_sequence(self, db: Session, session_id: str | None, agent_ids: list[int], query_text: str, collection_name: str):
        """
        Runs a sequential debate session with multiple agents.
        """

        if not session_id:
            session_id = str(uuid.uuid4())

        # Clear prior debate records
        db.query(DebateSession).filter(DebateSession.session_id == session_id).delete()
        db.commit()

        # Insert agents in order
        for idx, agent_id in enumerate(agent_ids):
            db.add(DebateSession(
                session_id=session_id,
                compliance_agent_id=agent_id,
                debate_order=idx + 1
            ))
        db.commit()

        # Load debate agents from the database
        debate_agents = self.load_debate_agents(session_id)

        debate_chain = []
        context = f"Original user query:\n{query_text}\n"

        for agent in debate_agents:
            agent_response = self.debate_rag_compliance(agent, context, collection_name, db)

            debate_chain.append({
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "response": agent_response
            })

            context += f"\n---\nAgent {agent['name']} responded:\n{agent_response}\n"

        return session_id, debate_chain