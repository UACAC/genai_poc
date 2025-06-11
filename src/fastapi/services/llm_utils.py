import os
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

def get_llm(model_name: str):
    """
    Unified LLM loader for OpenAI and Ollama models.
    """
    model_name = model_name.lower()
    openai_api_key = os.getenv("OPEN_AI_API_KEY")
    ollama_host = os.getenv("LLM_OLLAMA_HOST", "http://ollama:11434")
    
    # OpenAI models
    if model_name in ["gpt4", "gpt-4",  "gpt-3.5-turbo"]:
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key, temperature=0.7)
    
    # Ollama models - handle both simple names and versioned names
    elif model_name in ["llama", "llama3", "llama3:8b"]:
        actual_model = "llama3" if model_name in ["llama", "llama3"] else model_name
        return OllamaLLM(model=actual_model, base_url=ollama_host, temperature=0.7)
    
    # elif model_name in ["mistral"]:
    #     return OllamaLLM(model="mistral", base_url=ollama_host, temperature=0.7)
    
    # elif model_name in ["gemma"]:
    #     return OllamaLLM(model="gemma", base_url=ollama_host, temperature=0.7)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")