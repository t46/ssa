"""LLM Client abstraction for multiple providers."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import anthropic
import openai
import ollama
from dotenv import load_dotenv


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    def generate_response_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Generate streaming response from messages.
        
        Returns an iterator that yields text chunks.
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude client implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Anthropic client."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"Environment variable {config['api_key_env']} not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config = config
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Anthropic Claude."""
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "system":
                # Anthropic doesn't support system messages in the same way
                # We'll prepend it to the first user message
                if anthropic_messages and anthropic_messages[0]["role"] == "user":
                    anthropic_messages[0]["content"] = f"{msg['content']}\n\n{anthropic_messages[0]['content']}"
        
        # Merge kwargs with config defaults
        request_kwargs = {
            "model": self.config["model"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            **kwargs
        }
        
        response = self.client.messages.create(
            messages=anthropic_messages,
            **request_kwargs
        )
        
        # Extract text content
        response_text = getattr(response, "content", None)
        if isinstance(response_text, list) and hasattr(response_text[0], "text"):
            return response_text[0].text
        return str(response_text)
    
    def generate_response_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Generate streaming response using Anthropic Claude."""
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "system":
                # Anthropic doesn't support system messages in the same way
                # We'll prepend it to the first user message
                if anthropic_messages and anthropic_messages[0]["role"] == "user":
                    anthropic_messages[0]["content"] = f"{msg['content']}\n\n{anthropic_messages[0]['content']}"
        
        # Merge kwargs with config defaults
        request_kwargs = {
            "model": self.config["model"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            **kwargs
        }
        
        # Return a unified iterator that yields text chunks
        class UnifiedAnthropicStream:
            def __init__(self, anthropic_stream):
                self.anthropic_stream = anthropic_stream
            
            def __iter__(self):
                with self.anthropic_stream as stream:
                    for text in stream.text_stream:
                        yield text
        
        anthropic_stream = self.client.messages.stream(
            messages=anthropic_messages,
            **request_kwargs
        )
        return UnifiedAnthropicStream(anthropic_stream)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config["model"]


class OpenAIClient(LLMClient):
    """OpenAI GPT client implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI client."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"Environment variable {config['api_key_env']} not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.config = config
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI GPT."""
        # Merge kwargs with config defaults
        request_kwargs = {
            "model": self.config["model"],
            **kwargs
        }
        
        # Only add token parameters if they exist in config and are not None
        if "max_tokens" in self.config and self.config["max_tokens"] is not None:
            request_kwargs["max_tokens"] = self.config["max_tokens"]
        if "max_completion_tokens" in self.config and self.config["max_completion_tokens"] is not None:
            request_kwargs["max_completion_tokens"] = self.config["max_completion_tokens"]
        
        # Only add temperature if it's in config and not None
        if "temperature" in self.config and self.config["temperature"] is not None:
            request_kwargs["temperature"] = self.config["temperature"]
        
        response = self.client.chat.completions.create(
            messages=messages,
            **request_kwargs
        )
        
        return response.choices[0].message.content
    
    def generate_response_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Generate streaming response using OpenAI GPT."""
        # Merge kwargs with config defaults
        request_kwargs = {
            "model": self.config["model"],
            **kwargs
        }
        
        # Only add token parameters if they exist in config and are not None
        if "max_tokens" in self.config and self.config["max_tokens"] is not None:
            request_kwargs["max_tokens"] = self.config["max_tokens"]
        if "max_completion_tokens" in self.config and self.config["max_completion_tokens"] is not None:
            request_kwargs["max_completion_tokens"] = self.config["max_completion_tokens"]
        
        # Only add temperature if it's in config and not None
        if "temperature" in self.config and self.config["temperature"] is not None:
            request_kwargs["temperature"] = self.config["temperature"]
        
        # For OpenAI, use non-streaming response to avoid organization verification issues
        response = self.client.chat.completions.create(
            messages=messages,
            stream=False,
            **request_kwargs
        )
        
        # Return a unified iterator that yields text chunks
        class UnifiedOpenAIStream:
            def __init__(self, content):
                self.content = content
            
            def __iter__(self):
                # Return a single chunk with the full content
                yield self.content
        
        return UnifiedOpenAIStream(response.choices[0].message.content)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config["model"]


class OllamaClient(LLMClient):
    """Ollama client implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama client."""
        self.config = config
        # Ollama doesn't require API keys, it connects to local Ollama server
        # We can optionally check if Ollama server is running
        try:
            ollama.list()  # Test connection to Ollama server
        except Exception as e:
            raise ValueError(f"Failed to connect to Ollama server: {e}")
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Ollama."""
        # Convert messages to Ollama format
        # Ollama expects a single prompt string, so we'll combine all messages
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        # Merge kwargs with config defaults
        request_kwargs = {
            "model": self.config["model"],
            "prompt": prompt,
            **kwargs
        }
        
        # Only add parameters if they exist in config and are not None
        if "temperature" in self.config and self.config["temperature"] is not None:
            request_kwargs["options"] = {"temperature": self.config["temperature"]}
        
        response = ollama.generate(**request_kwargs)
        return response['response']
    
    def generate_response_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Generate streaming response using Ollama."""
        # Convert messages to Ollama format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        # Merge kwargs with config defaults
        request_kwargs = {
            "model": self.config["model"],
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        # Only add parameters if they exist in config and are not None
        if "temperature" in self.config and self.config["temperature"] is not None:
            request_kwargs["options"] = {"temperature": self.config["temperature"]}
        
        # Return a unified iterator that yields text chunks
        class UnifiedOllamaStream:
            def __init__(self, ollama_stream):
                self.ollama_stream = ollama_stream
            
            def __iter__(self):
                for chunk in self.ollama_stream:
                    if 'response' in chunk:
                        yield chunk['response']
        
        ollama_stream = ollama.generate(**request_kwargs)
        return UnifiedOllamaStream(ollama_stream)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config["model"]


class LLMClientFactory:
    """Factory class for creating LLM clients."""
    
    def __init__(self, config_path: str = "config/llm_config.yaml"):
        """Initialize factory with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_client(self, provider: Optional[str] = None) -> LLMClient:
        """Create LLM client for the specified provider."""
        # Load environment variables from .env file
        load_dotenv()
        
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", self.config["default_provider"])
        
        provider = provider.lower()
        
        if provider == "anthropic":
            return AnthropicClient(self.config["anthropic"])
        elif provider == "openai":
            return OpenAIClient(self.config["openai"])
        elif provider == "ollama":
            return OllamaClient(self.config["ollama"])
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: anthropic, openai, ollama")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return ["anthropic", "openai", "ollama"]
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        provider = provider.lower()
        if provider not in self.config:
            raise ValueError(f"Provider {provider} not found in configuration")
        return self.config[provider]
