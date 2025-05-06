import os
from typing import Dict, Any, Optional
import json
from openai import OpenAI
from datetime import datetime
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import OPENAI_API_KEY, DEFAULT_MODEL, RESULTS_DIR

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
        
        # Create LLM logs directory
        self.logs_file = os.path.join(RESULTS_DIR, "llm_interaction_logs.json")
        self.init_logs_file()
        
    def init_logs_file(self):
        """Initialize the logs file if it doesn't exist"""
        if not os.path.exists(self.logs_file):
            with open(self.logs_file, 'w') as f:
                json.dump([], f)
    
    def log_interaction(self, expert_name: str, prompt: str, response: str, metadata: Dict[str, Any] = None):
        """Log an interaction with the LLM"""
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "expert": expert_name,
            "model": self.model,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        
        # Read existing logs
        try:
            with open(self.logs_file, 'r') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
        
        # Append new log and save
        logs.append(log_entry)
        with open(self.logs_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
        # Also create/append to human-readable text log
        text_log_file = os.path.join(RESULTS_DIR, "llm_interactions.txt")
        with open(text_log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {log_entry['timestamp']}\n")
            f.write(f"EXPERT: {expert_name}\n")
            f.write(f"MODEL: {self.model}\n")
            f.write(f"\n--- PROMPT ---\n")
            f.write(f"{prompt}\n")
            f.write(f"\n--- RESPONSE ---\n")
            f.write(f"{response}\n")
            f.write(f"\n{'='*80}\n")
        
    def generate(self, prompt: str, expert_name: str = "Unknown", **kwargs) -> str:
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
            
            response_text = response.choices[0].message.content
            
            # Log the interaction
            self.log_interaction(
                expert_name=expert_name,
                prompt=prompt,
                response=response_text,
                metadata={
                    "system_message": system_message,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "usage": response.usage.model_dump() if hasattr(response, "usage") else {}
                }
            )
            
            return response_text
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error generating text: {error_msg}")
            
            # Log the failed interaction
            self.log_interaction(
                expert_name=expert_name,
                prompt=prompt,
                response=error_msg,
                metadata={
                    "error": True,
                    "system_message": system_message,
                    "temperature": temperature
                }
            )
            
            return error_msg
    
    def structured_generate(self, prompt: str, output_schema: Dict[str, Any], expert_name: str = "Unknown", **kwargs) -> Dict[str, Any]:
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
                result = json.loads(function_call.arguments)
                
                # Log the interaction
                self.log_interaction(
                    expert_name=expert_name,
                    prompt=prompt,
                    response=json.dumps(result, indent=2),
                    metadata={
                        "structured": True,
                        "system_message": system_message,
                        "temperature": temperature,
                        "schema": output_schema,
                        "usage": response.usage.model_dump() if hasattr(response, "usage") else {}
                    }
                )
                
                return result
            else:
                error_result = {"error": "No structured output generated"}
                
                # Log the interaction with error
                self.log_interaction(
                    expert_name=expert_name,
                    prompt=prompt,
                    response=json.dumps(error_result, indent=2),
                    metadata={
                        "structured": True,
                        "error": True,
                        "system_message": system_message,
                        "temperature": temperature,
                        "schema": output_schema
                    }
                )
                
                return error_result
        except Exception as e:
            error_result = {"error": f"Error: {str(e)}"}
            logger.error(f"Error generating structured output: {str(e)}")
            
            # Log the failed interaction
            self.log_interaction(
                expert_name=expert_name,
                prompt=prompt,
                response=json.dumps(error_result, indent=2),
                metadata={
                    "structured": True,
                    "error": True,
                    "system_message": system_message,
                    "temperature": temperature,
                    "schema": output_schema
                }
            )
            
            return error_result
