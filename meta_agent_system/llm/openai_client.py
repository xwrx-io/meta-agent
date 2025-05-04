import os
from typing import Dict, List, Any, Optional
import json
from openai import OpenAI
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import OPENAI_API_KEY, DEFAULT_MODEL
import hashlib
import pickle

logger = get_logger(__name__)

class OpenAIClient:
    """Client for OpenAI's models"""
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model: The model to use (default from settings)
            api_key: OpenAI API key (defaults to config or environment variable)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        if not api_key and not OPENAI_API_KEY:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        logger.info(f"Initialized OpenAI client with model: {model}")
        
    def _get_cache_key(self, prompt, system_message, temperature, max_tokens):
        """Generate a cache key for the given request parameters"""
        # Create a string representation of the request
        request_str = f"{prompt}|{system_message}|{temperature}|{max_tokens}"
        
        # Hash the request string to get a cache key
        cache_key = hashlib.md5(request_str.encode('utf-8')).hexdigest()
        return cache_key

    def _get_cache_path(self, cache_key):
        """Get the path to the cache file for the given key"""
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(__file__), "../.cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Return path to cache file
        return os.path.join(cache_dir, f"{cache_key}.pkl")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's API with caching.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The generated text
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        use_cache = kwargs.get("use_cache", True)
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, system_message, temperature, max_tokens)
            cache_path = self._get_cache_path(cache_key)
            
            # If cache file exists, load and return response
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        cached_response = pickle.load(f)
                    return cached_response
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {str(e)}")
        
        # Generate response from API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Save to cache if enabled
            if use_cache:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(content, f)
                except Exception as e:
                    logger.warning(f"Failed to save to cache: {str(e)}")
            
            return content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            return f"Error: {str(e)}"
    
    def structured_generate(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output using OpenAI's function calling capability.
        
        Args:
            prompt: The prompt to send to the model
            output_schema: JSON schema defining the expected output structure
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The structured output as a dictionary
        """
        temperature = kwargs.get("temperature", 0.7)
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                functions=[
                    {
                        "name": "generate_structured_output",
                        "description": "Generate a structured output based on the user's request",
                        "parameters": output_schema
                    }
                ],
                function_call={"name": "generate_structured_output"},
                temperature=temperature
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.arguments:
                return json.loads(function_call.arguments)
            else:
                return {"error": "No structured output generated"}
        except Exception as e:
            logger.error(f"Error generating structured output with OpenAI: {str(e)}")
            return {"error": f"Error: {str(e)}"}
