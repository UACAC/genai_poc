import os
import uuid
import time
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from chromadb import Client
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from services.llm_utils import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from services.database import (
    SessionLocal, 
    ComplianceAgent, 
    DebateSession, 
    log_compliance_result,
    log_agent_response,
    log_agent_session,
    complete_agent_session,
    SessionType,
    AnalysisType
)

class RAGService:
    def __init__(self):
        # FIXED: Use the same API endpoint as your Streamlit app
        self.chromadb_api_url = os.getenv("CHROMA_URL", "http://localhost:8020")
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.n_results = int(os.getenv("N_RESULTS", "5"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
        self.compliance_agents = []
        
        print(f"    RAG Service API Config:")
        print(f"    ChromaDB API URL: {self.chromadb_api_url}")
        
        # Test connection
        self.test_connection()
        
    def test_connection(self):
        """Test connection to ChromaDB API"""
        try:
            response = requests.get(f"{self.chromadb_api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"ChromaDB API connection successful")
                return True
            else:
                print(f"ChromaDB API returned status: {response.status_code}")
                return False
        except Exception as e:
            print(f"ChromaDB API connection failed: {e}")
            return False
        
    def list_available_collections(self) -> List[str]:
        """List all available collections via API"""
        try:
            response = requests.get(f"{self.chromadb_api_url}/collections", timeout=10)
            response.raise_for_status()
            data = response.json()
            collections = data.get("collections", [])
            print(f"Available collections (via API): {collections}")
            return collections
        except Exception as e:
            print(f"Error listing collections via API: {e}")
            return []

        
    def verify_uploaded_documents(self):
        """Verify RAG service can see documents uploaded via Streamlit"""
        print("VERIFYING RAG SERVICE CAN ACCESS UPLOADED DOCUMENTS")
        print("=" * 60)
        
        try:
            collections = self.list_available_collections()
            
            if not collections:
                print("No collections found!")
                return False
            
            found_documents = False
            for collection_name in collections:
                print(f"\nChecking collection: '{collection_name}'")
                
                collection = self.chroma_client.get_collection(collection_name)
                doc_count = collection.count()
                print(f"   Document count: {doc_count}")
                
                if doc_count > 0:
                    found_documents = True
                    print(f"    Found {doc_count} documents!")
                    
                    # Show sample documents
                    sample = collection.peek(2)
                    docs = sample.get('documents', [])
                    ids = sample.get('ids', [])
                    
                    for i, (doc_id, doc) in enumerate(zip(ids, docs)):
                        print(f"        Doc {i+1} ID: {doc_id}")
                        print(f"        Preview: {doc[:150]}...")
                    
                    # Test different queries
                    test_queries = ["document", "legal", "court", "rule", "motion", "case"]
                    
                    print(f"\nTesting queries on '{collection_name}':")
                    for query in test_queries:
                        docs, found = self.get_relevant_documents(query, collection_name)
                        if found:
                            print(f"Query '{query}': Found {len(docs)} documents")
                            return True
                        else:
                            print(f"Query '{query}': No matches")
            
            if not found_documents:
                print("No documents found in any collection")
                
            return found_documents
            
        except Exception as e:
            print(f"Error verifying documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_relevant_documents(self, query: str, collection_name: str) -> Tuple[List[str], bool]:
        """Get relevant documents via FastAPI"""
        try:
            print(f"🔍 RAG Query (API): '{query[:100]}...' in collection: '{collection_name}'")
            
            # Check if collection exists and has documents
            try:
                response = requests.get(
                    f"{self.chromadb_api_url}/documents", 
                    params={"collection_name": collection_name},
                    timeout=10
                )
                
                if response.status_code == 404:
                    print(f"Collection '{collection_name}' not found")
                    # List available collections for debugging
                    available = self.list_available_collections()
                    print(f"Available collections: {available}")
                    return [], False
                
                response.raise_for_status()
                docs_data = response.json()
                
            except Exception as e:
                print(f"Error checking collection: {e}")
                return [], False
            
            # Check if collection has documents
            doc_count = len(docs_data.get("documents", []))
            print(f"📊 Collection '{collection_name}' has {doc_count} documents")
            
            if doc_count == 0:
                print(f"No documents in collection")
                return [], False
            
            # Generate query embedding
            query_embedding = self.embedding_function.embed_query(query)
            
            # Query via API
            query_payload = {
                "collection_name": collection_name,
                "query_embeddings": [query_embedding],
                "n_results": min(self.n_results, doc_count),
                "include": ["documents", "metadatas", "distances"]
            }
            
            try:
                response = requests.post(
                    f"{self.chromadb_api_url}/documents/query", 
                    json=query_payload,
                    timeout=30
                )
                response.raise_for_status()
                query_results = response.json()
                
            except Exception as e:
                print(f"Error querying documents: {e}")
                return [], False
            
            # Extract results
            documents = query_results.get("documents", [[]])[0]
            distances = query_results.get("distances", [[]])[0]
            
            if not documents:
                print(f"Query returned no results")
                return [], False
            
            print(f"Query returned {len(documents)} results:")
            for i, (doc, distance) in enumerate(zip(documents, distances)):
                similarity = 1.0 - distance
                print(f"  Result {i+1}: Similarity={similarity:.4f}")
                print(f"    Preview: {doc[:100]}...")
            
            # Apply similarity threshold
            filtered_docs = []
            for doc, distance in zip(documents, distances):
                similarity = 1.0 - distance
                if similarity >= self.similarity_threshold:
                    filtered_docs.append(doc)
            
            print(f"After threshold filter ({self.similarity_threshold}): {len(filtered_docs)} docs")
            
            # If no docs pass threshold, use top results anyway
            if not filtered_docs and documents:
                print(f"No docs passed threshold, using top {min(3, len(documents))} results")
                filtered_docs = documents[:3]
            
            if filtered_docs:
                print(f"Successfully retrieved {len(filtered_docs)} relevant documents")
                return filtered_docs, True
            else:
                print(f"No relevant documents found")
                return [], False
            
        except Exception as e:
            print(f"Error in API RAG: {e}")
            import traceback
            traceback.print_exc()
            return [], False
        
    def verify_uploaded_documents(self):
        """Verify RAG service can see documents uploaded via Streamlit"""
        print("VERIFYING RAG SERVICE CAN ACCESS UPLOADED DOCUMENTS (VIA API)")
        print("=" * 70)
        
        # Test API connection first
        if not self.test_connection():
            print("Cannot connect to ChromaDB API")
            return False
        
        # List collections
        collections = self.list_available_collections()
        
        if not collections:
            print("No collections found!")
            print("Make sure you've uploaded documents via your Streamlit interface first")
            return False
        
        # Test each collection
        found_documents = False
        for collection_name in collections:
            print(f"\nTesting collection: '{collection_name}'")
            
            # Get documents in this collection
            try:
                response = requests.get(
                    f"{self.chromadb_api_url}/documents", 
                    params={"collection_name": collection_name},
                    timeout=10
                )
                response.raise_for_status()
                docs_data = response.json()
                
                doc_count = len(docs_data.get("documents", []))
                print(f"   Document count: {doc_count}")
                
                if doc_count > 0:
                    found_documents = True
                    print(f"Found {doc_count} documents!")
                    
                    # Show sample
                    docs = docs_data.get('documents', [])
                    ids = docs_data.get('ids', [])
                    
                    for i, (doc_id, doc) in enumerate(zip(ids[:2], docs[:2])):
                        print(f"Doc {i+1} ID: {doc_id}")
                        print(f"Preview: {doc[:150]}...")
                    
                    # Test a query
                    test_queries = ["document", "legal", "court", "analysis"]
                    for query in test_queries:
                        docs_result, found = self.get_relevant_documents(query, collection_name)
                        if found:
                            print(f"   ✅ Test query '{query}': Found {len(docs_result)} documents")
                            return True
                        
            except Exception as e:
                print(f"Error accessing collection: {e}")
        

    def get_llm_service(self, model_name: str):
        """Get LLM service for the specified model"""
        model_name = model_name.lower()
        if model_name in ["gpt4", "gpt-4", "gpt-3.5", "gpt-3.5-turbo"]:
            return get_llm(model_name=model_name)
        elif model_name in ["llama", "llama3"]:
            return get_llm(model_name=model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def process_agent_with_rag(self, agent: Dict[str, Any], query_text: str, collection_name: str, session_id: str, db: Session) -> Dict[str, Any]:
        """Process query with agent using RAG via API"""
        start_time = time.time()
        
        try:
            print(f"🤖 Processing with agent: {agent['name']} using model: {agent['model_name']}")
            
            # Try to get relevant documents via API
            relevant_docs, docs_found = self.get_relevant_documents(query_text, collection_name)
            
            # Get LLM for this agent
            llm = self.get_llm_service(agent["model_name"])
            
            if docs_found and relevant_docs:
                print(f"✅ Using RAG mode with {len(relevant_docs)} documents")
                
                # Create context from retrieved documents
                context = "\n\n---DOCUMENT SEPARATOR---\n\n".join(relevant_docs)
                
                # Enhanced RAG prompt
                enhanced_content = f"""KNOWLEDGE BASE CONTEXT:
{context}

USER QUERY: {query_text}

INSTRUCTIONS: 
1. Carefully analyze the provided knowledge base context above
2. Use information from the context to inform your analysis of the user query
3. If the context contains relevant information, cite or reference it in your response
4. If the context is not directly relevant, acknowledge this and proceed with your general knowledge
5. Provide a comprehensive analysis that combines context information with your expertise"""
                
                formatted_user_prompt = agent["user_prompt_template"].replace("{data_sample}", enhanced_content)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", agent["system_prompt"]),
                    ("human", formatted_user_prompt)
                ])
                
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({})
                
                response_time_ms = int((time.time() - start_time) * 1000)
                processing_method = f"rag_enhanced_{agent['model_name']}"
                
                # rag_info = f"\n\n---\n**RAG Information**: Used {len(relevant_docs)} relevant documents from collection '{collection_name}' with {agent['model_name']} model."
                rag_info = f"\n\n---\n**RAG Information**: Used {len(relevant_docs)} relevant documents from collection '{collection_name}' with {agent['model_name']} model.\n\n**Retrieved Document Previews:**\n"

                for i, doc in enumerate(relevant_docs, 1):
                    preview = doc[:800] + "..." if len(doc) > 800 else doc
                    rag_info += f"\nSource {i} Preview:\n{preview}\n"

                final_response = response + rag_info
                
            else:
                print(f"Using Direct LLM mode - no relevant documents found")
                
                formatted_user_prompt = agent["user_prompt_template"].replace("{data_sample}", query_text)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", agent["system_prompt"]),
                    ("human", formatted_user_prompt)
                ])
                
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({})
                
                response_time_ms = int((time.time() - start_time) * 1000)
                processing_method = f"direct_{agent['model_name']}"
                
                direct_info = f"\n\n---\n**Direct LLM Information**: No relevant documents found in collection '{collection_name}'. Used {agent['model_name']} model directly."
                final_response = response + direct_info
            
            # Log the response (remove the problematic rag_context for now)
            log_agent_response(
                session_id=session_id,
                agent_id=agent["id"],
                response_text=final_response,
                processing_method=processing_method,
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                rag_used=docs_found,
                documents_found=len(relevant_docs) if docs_found else 0,
                rag_context=context if docs_found and relevant_docs else None
            )
            
            log_compliance_result(
                agent_id=agent["id"],
                data_sample=query_text,
                confidence_score=None,
                reason="RAG analysis completed",
                raw_response=final_response,
                processing_method=processing_method,
                response_time_ms=response_time_ms,
                model_used=agent["model_name"],
                session_id=session_id
            )
            
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "response": final_response,
                "processing_method": processing_method,
                "response_time_ms": response_time_ms,
                "rag_used": docs_found,
                "documents_found": len(relevant_docs) if docs_found else 0
            }
            
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            error_response = f"Error processing with agent {agent['name']}: {str(e)}"
            print(f"Error in process_agent_with_rag: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "response": error_response,
                "processing_method": "error",
                "response_time_ms": response_time_ms,
                "rag_used": False,
                "documents_found": 0
            }


    def run_rag_check(self, query_text: str, collection_name: str, agent_ids: List[int], db: Session) -> Dict[str, Any]:
        """Run RAG check with multiple agents and enhanced logging"""
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        session_type = SessionType.MULTI_AGENT_DEBATE if len(agent_ids) > 1 else SessionType.RAG_ANALYSIS
        log_agent_session(
            session_id=session_id,
            session_type=session_type,
            analysis_type=AnalysisType.RAG_ENHANCED,
            user_query=query_text,
            collection_name=collection_name
        )
        
        self.load_selected_compliance_agents(agent_ids)
        
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_agent_with_rag, agent, query_text, collection_name, session_id, db): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results[result["agent_name"]] = result["response"]
        
        total_time = int((time.time() - start_time) * 1000)
        complete_agent_session(
            session_id=session_id,
            overall_result={
                "agent_responses": results,
                "collection_used": collection_name
            },
            agent_count=len(agent_ids),
            total_response_time_ms=total_time,
            status='completed'
        )
        
        return {
            "agent_responses": results,
            "collection_used": collection_name,
            "session_id": session_id,
            "processing_time": total_time
        }

    def run_rag_debate_sequence(self, db: Session, session_id: Optional[str], agent_ids: List[int], query_text: str, collection_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Run RAG debate sequence with multiple agents and enhanced logging"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        log_agent_session(
            session_id=session_id,
            session_type=SessionType.RAG_DEBATE,
            analysis_type=AnalysisType.RAG_ENHANCED,
            user_query=query_text,
            collection_name=collection_name
        )
        
        # Clear existing debate sessions
        db.query(DebateSession).filter(DebateSession.session_id == session_id).delete()
        db.commit()
        
        # Create new debate sessions
        for idx, agent_id in enumerate(agent_ids):
            db.add(DebateSession(session_id=session_id, compliance_agent_id=agent_id, debate_order=idx + 1))
        db.commit()
        
        # Load debate agents
        debate_agents = self.load_debate_agents(session_id)
        
        debate_chain = []
        cumulative_context = f"Original user query: {query_text}\n\n"
        
        for i, agent in enumerate(debate_agents):
            print(f"Debate Round {i+1}: Agent {agent['name']}")
            
            # For the first agent, use original query. For subsequent agents, use cumulative context
            current_input = cumulative_context if i > 0 else query_text
            
            # Process with RAG
            result = self.process_agent_with_rag(agent, current_input, collection_name, session_id, db)
            
            agent_response_id = self._update_agent_response_sequence_order(session_id, agent["id"], i + 1)
            
            debate_chain.append({
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "response": result["response"],
                "processing_method": result["processing_method"],
                "response_time_ms": result["response_time_ms"],
                "rag_used": result["rag_used"],
                "documents_found": result["documents_found"],
                "sequence_order": i + 1
            })
            
            cumulative_context += f"--- Agent {agent['name']} Analysis ---\n{result['response']}\n\n"
        
        total_time = int((time.time() - start_time) * 1000)
        complete_agent_session(
            session_id=session_id,
            overall_result={
                "debate_chain": debate_chain,
                "collection_used": collection_name
            },
            agent_count=len(agent_ids),
            total_response_time_ms=total_time,
            status='completed'
        )
        
        return session_id, debate_chain

    def load_selected_compliance_agents(self, agent_ids: List[int]):
        """Load selected compliance agents"""
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

    def load_debate_agents(self, session_id: str) -> List[Dict[str, Any]]:
        """Load debate agents in order"""
        session = SessionLocal()
        try:
            debate_sessions = session.query(DebateSession).filter(
                DebateSession.session_id == session_id
            ).order_by(DebateSession.debate_order).all()
            
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
    
    def _update_agent_response_sequence_order(self, session_id: str, agent_id: int, sequence_order: int):
        """Update the sequence order for an agent response in a debate"""
        db = SessionLocal()
        try:
            # Find the most recent response for this agent in this session
            from services.database import AgentResponse
            response = db.query(AgentResponse).filter(
                AgentResponse.session_id == session_id,
                AgentResponse.agent_id == agent_id
            ).order_by(AgentResponse.created_at.desc()).first()
            
            if response:
                response.sequence_order = sequence_order
                db.commit()
                return response.id
        except Exception as e:
            print(f"Error updating sequence order: {e}")
            db.rollback()
        finally:
            db.close()
        return None
