import os
import requests
import time
from typing import Optional, Dict, Any, List
from services.rag_service import RAGService

# LangChain imports
try:
    from langchain.schema import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain.chains import LLMChain
    from langchain.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LLMService:
    def __init__(self):
        """Initialize LLMs (LLaMA, Mistral, Gemma) and the RAG service with LangChain support."""
        self.openai_api_key = os.getenv("OPEN_AI_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        # Make API keys optional for better flexibility
        if not self.openai_api_key:
            print("Warning: OpenAI API key not found. GPT-4 functionality will be disabled.")
        if not self.huggingface_api_key:
            print("Warning: Hugging Face API key not found. Some features may be limited.")
        
        # Initialize services
        self.rag_service = RAGService()
        self.ollama_url = "http://llama:11434/api/generate"
        
        # Initialize LangChain components if available
        self.langchain_llms = {}
        self.use_langchain = LANGCHAIN_AVAILABLE
        
        if self.use_langchain:
            self._initialize_langchain_llms()
        
        # Initialize OpenAI client (keeping your existing approach)
        if self.openai_api_key:
            if LANGCHAIN_AVAILABLE:
                self.openai_client = ChatOpenAI(
                    model_name="gpt-4o",
                    openai_api_key=self.openai_api_key,
                    temperature=0.7
                )
            else:
                # Fallback initialization if LangChain not available
                self.openai_client = None
    
    def _initialize_langchain_llms(self):
        """Initialize LangChain LLM instances for consistent interface."""
        try:
            # Initialize OpenAI models
            if self.openai_api_key:
                self.langchain_llms["gpt-4"] = ChatOpenAI(
                    model_name="gpt-4o",
                    openai_api_key=self.openai_api_key,
                    temperature=0.7
                )
                self.langchain_llms["gpt4"] = self.langchain_llms["gpt-4"]  # Alias
            
            # Initialize Ollama models with LangChain
            self.langchain_llms["llama3"] = Ollama(
                model="llama3",
                temperature=0.7,
                base_url="http://llama:11434"
            )
            self.langchain_llms["llama"] = self.langchain_llms["llama3"]  # Alias
            
            self.langchain_llms["mistral"] = Ollama(
                model="mistral", 
                temperature=0.7,
                base_url="http://llama:11434"
            )
            
            self.langchain_llms["gemma"] = Ollama(
                model="gemma",
                temperature=0.7,
                base_url="http://llama:11434"
            )
            
            print("LangChain LLMs initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing LangChain LLMs: {e}")
            self.use_langchain = False
    
    def get_llm(self, model_name: str):
        """Get LangChain LLM instance by model name."""
        if not self.use_langchain:
            return None
        return self.langchain_llms.get(model_name.lower())
    
    def create_chain(self, model_name: str, prompt_template: str, **kwargs):
        """Create a LangChain chain with the specified model and prompt."""
        if not self.use_langchain:
            raise Exception("LangChain not available")
        
        llm = self.get_llm(model_name)
        if not llm:
            raise ValueError(f"Model '{model_name}' not available")
        
        # Create prompt template
        if isinstance(llm, ChatOpenAI):
            # Use ChatPromptTemplate for chat models
            prompt = ChatPromptTemplate.from_template(prompt_template)
        else:
            # Use PromptTemplate for completion models
            prompt = PromptTemplate.from_template(prompt_template)
        
        # Create and return chain
        chain = prompt | llm | StrOutputParser()
        return chain
    
    def query_with_langchain(self, model_name: str, prompt: str, **kwargs) -> str:
        """Query using LangChain with enhanced error handling and metrics."""
        if not self.use_langchain:
            raise Exception("LangChain not available")
        
        llm = self.get_llm(model_name)
        if not llm:
            raise ValueError(f"Model '{model_name}' not available in LangChain")
        
        start_time = time.time()
        
        try:
            if isinstance(llm, ChatOpenAI):
                # Use chat interface for OpenAI models
                if model_name.lower() in ["gpt-4", "gpt4"]:
                    with get_openai_callback() as cb:
                        response = llm.invoke([HumanMessage(content=prompt)])
                        # Could log token usage: cb.total_tokens, cb.total_cost
                        return response.content.strip()
                else:
                    response = llm.invoke([HumanMessage(content=prompt)])
                    return response.content.strip()
            else:
                # Use invoke for other models
                response = llm.invoke(prompt)
                return response.strip()
                
        except Exception as e:
            print(f"LangChain query failed for {model_name}: {e}")
            # Fall back to legacy method
            return self._query_legacy(model_name, prompt)
        finally:
            response_time = (time.time() - start_time) * 1000
            # Could log performance metrics here
    
    def _query_legacy(self, model_name: str, prompt: str) -> str:
        """Legacy query method as fallback."""
        model_name = model_name.lower()
        
        if model_name in ["gpt-4", "gpt4"]:
            if self.openai_client:
                return self.query_gpt4(prompt)
            else:
                return "Error: OpenAI API key not configured"
        elif model_name in ["llama", "llama3"]:
            return self.query_ollama("llama3", prompt)
        elif model_name == "mistral":
            return self.query_ollama("mistral", prompt)
        elif model_name == "gemma":
            return self.query_ollama("gemma", prompt)
        else:
            return f"Error: Model '{model_name}' not recognized"
    
    def query_ollama(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        Enhanced Ollama query with better error handling and configuration.
        Supported model names: 'llama3', 'mistral', 'gemma'.
        """
        if model_name not in ["llama3", "mistral", "gemma"]:
            return f"Error: Unsupported model '{model_name}'. Use 'llama3', 'mistral', or 'gemma'."
        
        # Extract parameters with defaults
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 300)
        stream = kwargs.get('stream', False)
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error querying {model_name}: {e}")
            return f"Error: Could not connect to Ollama for model '{model_name}'. Details: {str(e)}"
    
    def query_llama(self, prompt: str, **kwargs) -> str:
        """Query Llama3 with enhanced options."""
        if self.use_langchain:
            try:
                return self.query_with_langchain("llama3", prompt, **kwargs)
            except Exception:
                pass  # Fall back to legacy
        return self.query_ollama("llama3", prompt, **kwargs)
        
    def query_mistral(self, prompt: str, **kwargs) -> str:
        """Query Mistral with enhanced options."""
        if self.use_langchain:
            try:
                return self.query_with_langchain("mistral", prompt, **kwargs)
            except Exception:
                pass  # Fall back to legacy
        return self.query_ollama("mistral", prompt, **kwargs)
        
    def query_gemma(self, prompt: str, **kwargs) -> str:
        """Query Gemma with enhanced options."""
        if self.use_langchain:
            try:
                return self.query_with_langchain("gemma", prompt, **kwargs)
            except Exception:
                pass  # Fall back to legacy
        return self.query_ollama("gemma", prompt, **kwargs)
    
    def query_gpt4(self, prompt: str, **kwargs) -> str:
        """Query GPT-4 with enhanced options and fallback."""
        if not self.openai_api_key:
            return "Error: OpenAI API key not configured"
        
        if self.use_langchain:
            try:
                return self.query_with_langchain("gpt-4", prompt, **kwargs)
            except Exception as e:
                print(f"LangChain GPT-4 query failed: {e}")
                # Fall back to direct method
        
        # Fallback to your existing method
        if self.openai_client:
            try:
                response = self.openai_client.invoke([HumanMessage(content=prompt)])
                return response.content.strip()
            except Exception as e:
                return f"Error querying GPT-4: {str(e)}"
        else:
            return "Error: OpenAI client not initialized"
    
    def query_model(self, model_name: str, prompt: str, **kwargs) -> str:
        """Universal model query method."""
        model_name = model_name.lower()
        
        # Try LangChain first if available
        if self.use_langchain and model_name in self.langchain_llms:
            try:
                return self.query_with_langchain(model_name, prompt, **kwargs)
            except Exception:
                pass  # Fall back to legacy
        
        # Legacy methods
        if model_name in ["gpt-4", "gpt4"]:
            return self.query_gpt4(prompt, **kwargs)
        elif model_name in ["llama", "llama3"]:
            return self.query_llama(prompt, **kwargs)
        elif model_name == "mistral":
            return self.query_mistral(prompt, **kwargs)
        elif model_name == "gemma":
            return self.query_gemma(prompt, **kwargs)
        else:
            return f"Error: Model '{model_name}' not recognized"
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        models = []
        
        if self.openai_api_key:
            models.extend(["gpt-4", "gpt4"])
        
        # Test Ollama connectivity
        try:
            response = requests.get("http://llama:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models.extend(["llama3", "llama", "mistral", "gemma"])
        except:
            pass  # Ollama not available
        
        return models
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all LLM services."""
        health = {
            "langchain_available": self.use_langchain,
            "services": {}
        }
        
        # Check OpenAI
        if self.openai_api_key:
            try:
                test_response = self.query_gpt4("Hello", max_tokens=5)
                health["services"]["openai"] = {
                    "status": "healthy" if "Error" not in test_response else "error",
                    "response": test_response[:50] + "..." if len(test_response) > 50 else test_response
                }
            except Exception as e:
                health["services"]["openai"] = {"status": "error", "error": str(e)}
        else:
            health["services"]["openai"] = {"status": "not_configured"}
        
        # Check Ollama
        try:
            response = requests.get("http://llama:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                health["services"]["ollama"] = {
                    "status": "healthy",
                    "available_models": [m.get("name", "") for m in models]
                }
            else:
                health["services"]["ollama"] = {"status": "error", "message": "Ollama API not responding"}
        except Exception as e:
            health["services"]["ollama"] = {"status": "error", "error": str(e)}
        
        return health
    
    def test_all_models(self, test_prompt: str = "Hello, respond with 'OK'") -> Dict[str, str]:
        """Test all available models with a simple prompt."""
        results = {}
        available_models = self.get_available_models()
        
        for model in available_models:
            try:
                response = self.query_model(model, test_prompt, max_tokens=10)
                results[model] = "OK" if "OK" in response or "Hello" in response else f"Unexpected: {response[:30]}"
            except Exception as e:
                results[model] = f"Error: {str(e)}"
        
        return results