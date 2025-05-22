# services/llm_service.py

import os
import time
import requests
from typing import Optional, Dict, Any, List

from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback

LANGCHAIN_ENABLED = True

class LLMService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPEN_AI_API_KEY")
        self.langchain_enabled = LANGCHAIN_ENABLED and bool(self.openai_api_key)

        # Support both internal (Docker) and external (localhost) FastAPI URLs
        self.ollama_host = os.getenv("OLLAMA_HOST", "llama")
        self.ollama_url = f"http://{self.ollama_host}:11434/api/generate"

        self.models = {}
        if self.langchain_enabled:
            self._init_langchain_models()

    def _init_langchain_models(self):
        try:
            base_url = f"http://{self.ollama_host}:11434"
            self.models = {
                "gpt-4": ChatOpenAI(model_name="gpt-4o", openai_api_key=self.openai_api_key),
                "llama3": Ollama(model="llama3", base_url=base_url),
                "mistral": Ollama(model="mistral", base_url=base_url),
                "gemma": Ollama(model="gemma", base_url=base_url),
            }
        except Exception as e:
            print(f"LangChain init failed: {e}")
            self.langchain_enabled = False

    def query_gpt4(self, prompt: str) -> str:
        return self.query_model("gpt-4", prompt)

    def query_llama(self, prompt: str) -> str:
        return self.query_model("llama3", prompt)

    def query_mistral(self, prompt: str) -> str:
        return self.query_model("mistral", prompt)

    def query_gemma(self, prompt: str) -> str:
        return self.query_model("gemma", prompt)

    def query_model(self, model_name: str, prompt: str, **kwargs) -> str:
        model_name = model_name.lower()
        try:
            if self.langchain_enabled and model_name in self.models:
                return self._query_langchain(model_name, prompt)
        except Exception as e:
            print(f"[LangChain Fallback] {model_name}: {e}")

        return self._query_legacy(model_name, prompt)

    def query_with_langchain(self, model_name: str, prompt: str) -> str:
        return self.query_model(model_name, prompt)

    def _query_langchain(self, model_name: str, prompt: str) -> str:
        model = self.models[model_name]
        if isinstance(model, ChatOpenAI):
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        return model.invoke(prompt).strip()

    def _query_legacy(self, model_name: str, prompt: str) -> str:
        if model_name not in ["llama3", "mistral", "gemma"]:
            return f"Error: Legacy querying only supported for Ollama models. Model '{model_name}' not found."

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            return f"Error querying Ollama for '{model_name}': {e}"

    def get_supported_models(self) -> List[str]:
        return list(self.models.keys()) if self.langchain_enabled else ["llama3", "mistral", "gemma"]

    def health_check(self) -> Dict[str, Any]:
        results = {
            "langchain": self.langchain_enabled,
            "openai_configured": bool(self.openai_api_key)
        }
        if self.ollama_url:
            try:
                r = requests.get(f"http://{self.ollama_host}:11434/api/tags")
                results["ollama"] = r.status_code == 200
            except Exception as e:
                results["ollama"] = f"Error: {e}"
        return results
