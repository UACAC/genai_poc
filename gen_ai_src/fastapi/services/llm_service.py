import os
import requests
from services.rag_service import RAGService
from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

class LLMService:
    def __init__(self):
        """Initialize LLMs (LLaMA, Mistral, Gemma) and the RAG service."""
        self.openai_api_key = os.getenv("OPEN_AI_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPEN_AI_API_KEY in .env.")
        if not self.huggingface_api_key:
            raise ValueError("Hugging Face API key not found. Set HUGGINGFACE_API_KEY in .env.")
        
        # Initialize OpenAI GPT-4 via LangChain
        self.openai_client = ChatOpenAI(
            model_name="gpt-4o",  # your custom model name if relevant
            openai_api_key=self.openai_api_key
        )
        
        self.rag_service = RAGService()
        self.ollama_url = "http://llama:11434/api/generate"  # Fixed port to match your docker-compose
        
    def query_ollama(self, model_name, prompt):
        """
        Query the Ollama container running all models.
        Supported model names: 'llama3', 'mistral', 'gemma'.
        """
        if model_name not in ["llama3", "mistral", "gemma"]:
            return f"Error: Unsupported model '{model_name}'. Use 'llama3', 'mistral', or 'gemma'."
        
        payload = {
            "model": model_name,  # Ensures correct model is used
            "prompt": prompt,
            "stream": False,  # Set to True if streaming responses are needed
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error querying {model_name}: {e}")
            return f"Error: Could not connect to Ollama for model '{model_name}'."
            
    def query_llama(self, prompt):
        """Query Llama3 via Ollama."""
        return self.query_ollama("llama3", prompt)
        
    def query_mistral(self, prompt):
        """Query Mistral via Ollama."""
        return self.query_ollama("mistral", prompt)
        
    def query_gemma(self, prompt):
        """Query Gemma via Ollama."""
        return self.query_ollama("gemma", prompt)
    
    def query_gpt4(self, prompt):
        """Query GPT-4 using OpenAI API (no retrieval)."""
        response = self.openai_client.invoke([HumanMessage(content=prompt)])
        return response.content.strip()