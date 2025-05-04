from typing import Dict, Any, List, Optional
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
import json

logger = get_logger(__name__)

class ExpertFactory:
    """
    Factory for creating expert agents dynamically.
    
    This class creates new expert agents with specialized capabilities
    based on the requirements of the meta agent system.
    """
    def __init__(self, llm_client: OpenAIClient):
        """
        Initialize the expert factory.
        
        Args:
            llm_client: LLM client for generating agent behavior
        """
        self.llm_client = llm_client
        logger.info("Initialized expert factory")
    
    def create_expert(self, 
                     name: str, 
                     expertise: str, 
                     capabilities: List[str], 
                     task_description: str) -> Optional[ExpertAgent]:
        """
        Create a new expert agent with specialized capabilities.
        
        Args:
            name: Name of the expert agent
            expertise: Description of the expertise required
            capabilities: List of task types this agent can handle
            task_description: Description of the task this agent needs to solve
            
        Returns:
            Newly created expert agent
        """
        logger.info(f"Creating expert agent '{name}' with expertise: {expertise}")
        
        # Generate system prompt based on expertise and task description
        system_prompt = self._generate_system_prompt(name, expertise, capabilities, task_description)
        
        # Create a behavior function that uses the LLM
        def expert_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
            # Construct a prompt based on the task
            task_description = task.get("description", "")
            task_data = task.get("data", {})
            
            # Create a prompt that includes task details
            prompt = f"""
Task Description: {task_description}

Task Data: {json.dumps(task_data, indent=2)}

Please analyze this task and provide a solution based on your expertise.
"""
            
            # Get response from LLM
            llm_response = self.llm_client.generate(
                prompt=prompt, 
                system_message=system_prompt,
                temperature=0.7
            )
            
            # Try to parse structured output if available
            try:
                # Check if the response contains JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', llm_response)
                if json_match:
                    result = json.loads(json_match.group(0))
                    return result
            except:
                # If parsing fails, return the raw response
                pass
            
            # Return the raw response if no structured output could be parsed
            return {
                "status": "success",
                "result": llm_response
            }
        
        # Create and return the expert agent
        return ExpertAgent(
            name=name,
            capabilities=capabilities,
            behavior=expert_behavior,
            description=expertise
        )
    
    def _generate_system_prompt(self, 
                              name: str, 
                              expertise: str, 
                              capabilities: List[str], 
                              task_description: str) -> str:
        """
        Generate a system prompt for an expert agent.
        
        Args:
            name: Name of the expert agent
            expertise: Description of the expertise required
            capabilities: List of task types this agent can handle
            task_description: Description of the task this agent needs to solve
            
        Returns:
            System prompt string
        """
        # Create a prompt for the LLM to generate a system prompt for the expert
        meta_prompt = f"""
You are creating a system prompt for an AI assistant that will act as an expert agent specialized in {expertise}.

The agent name is: {name}
The agent's capabilities include: {', '.join(capabilities)}
The task the agent needs to solve is: {task_description}

Please write a detailed system prompt that will guide the AI assistant to:
1. Understand its role as an expert in {expertise}
2. Know how to approach problems related to {', '.join(capabilities)}
3. Provide responses in a structured, helpful format
4. Be thorough and precise in its analysis

The system prompt should be comprehensive, instructive, and set clear expectations for the AI assistant.
"""
        
        # Generate the system prompt using the LLM
        generated_prompt = self.llm_client.generate(
            prompt=meta_prompt,
            system_message="You are an expert at creating precise and effective AI system prompts.",
            temperature=0.7
        )
        
        return generated_prompt
