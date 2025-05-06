import os
from typing import Dict, Any, Optional
import json
from openai import OpenAI
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import OPENAI_API_KEY, DEFAULT_MODEL

logger = get_logger(__name__)

class OpenAIClient:
    """Simple client for OpenAI's models"""
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        self.model = model
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        if not api_key and not OPENAI_API_KEY:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        logger.info(f"Initialized OpenAI client with model: {model}")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI's API."""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        
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
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error: {str(e)}"
    
    def structured_generate(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured output using OpenAI's function calling."""
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
                        "description": "Generate structured output based on the user's request",
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
            logger.error(f"Error generating structured output: {str(e)}")
            return {"error": f"Error: {str(e)}"}
