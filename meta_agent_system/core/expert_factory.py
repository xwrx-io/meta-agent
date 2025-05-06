from typing import Dict, Any, List, Optional
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
import json
import os

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
        self.created_experts = {}  # Track created experts by name
        self.logger = get_logger(__name__)
        self.logger.info("Initialized expert factory")
    
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

    def create_experts_from_recommendations(self, recommendations_file):
        """Create expert agents from recommendations in a file."""
        self.logger.info(f"Creating experts from recommendations file: {recommendations_file}")
        
        new_experts = []
        
        # Load recommendations from file
        try:
            with open(recommendations_file, 'r') as f:
                recommendations = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading recommendations from {recommendations_file}: {str(e)}")
            return []
        
        if not recommendations or not isinstance(recommendations, dict) or "recommended_experts" not in recommendations:
            self.logger.warning(f"Invalid recommendations format: {recommendations}")
            return []
        
        # Create each recommended expert
        for expert_data in recommendations.get("recommended_experts", []):
            try:
                name = expert_data.get("name")
                capabilities = expert_data.get("capabilities", [])
                system_prompt = expert_data.get("system_prompt", "")
                
                if not name or not capabilities:
                    self.logger.warning(f"Missing required fields in expert recommendation: {expert_data}")
                    continue
                
                # Check if we already have an expert with this name
                if name in self.created_experts:
                    self.logger.info(f"Expert {name} already exists, skipping creation")
                    continue
                
                # Log the new expert creation with its system prompt
                self.logger.info(f"Creating new expert: {name}")
                self.logger.info(f"Capabilities: {capabilities}")
                self.logger.info(f"System prompt: {system_prompt}")
                
                # Create a dynamic expert agent
                new_expert = self.create_dynamic_expert(name, capabilities, system_prompt)
                if new_expert:
                    new_experts.append(new_expert)
                    # Add to created experts map
                    self.created_experts[name] = new_expert
                    self.logger.info(f"Successfully created expert: {name}")
            except Exception as e:
                self.logger.error(f"Error creating expert from recommendation: {str(e)}")
        
        self.logger.info(f"Created {len(new_experts)} new expert agents")
        return new_experts
    
    def create_dynamic_expert(self, name, capabilities, system_prompt):
        """
        Create a new dynamic expert agent with custom capabilities and system prompt.
        
        Args:
            name: Name of the expert
            capabilities: List of capabilities this expert has
            system_prompt: Custom system prompt for the expert
            
        Returns:
            Expert agent object
        """
        self.logger.info(f"Creating dynamic expert: {name}")
        
        def dynamic_expert_behavior(task):
            """Custom behavior function for the dynamic expert"""
            # Parse the task
            task_description = task.get("description", "")
            task_data = task.get("data", {})
            context = task.get("context", {})
            
            # Construct a prompt based on the system_prompt and task
            prompt = f"""
Task: {task_description}

Context: {json.dumps(context, indent=2)}

Data: {json.dumps(task_data, indent=2)}

Based on your specialized expertise as {name}, please analyze this task and provide your recommendations.
"""
            # Generate a response using the system prompt
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    system_message=system_prompt,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                return {
                    "status": "success",
                    "agent_name": name,
                    "result": {
                        "analysis": response,
                        "recommendations": self._extract_recommendations(response),
                        "message": f"Analyzed task using {name} expertise"
                    }
                }
            except Exception as e:
                self.logger.error(f"Error in dynamic expert {name}: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        # Create the expert agent
        return ExpertAgent(
            name=name,
            capabilities=capabilities,
            behavior=dynamic_expert_behavior,
            description=f"Dynamically created expert with expertise in: {', '.join(capabilities)}"
        )
    
    def _extract_recommendations(self, text):
        """Extract structured recommendations from text response"""
        try:
            # Look for JSON pattern in the text
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group(0))
            
            # If no JSON found, create a simple structure
            recommendations = []
            for line in text.split('\n'):
                if line.strip().startswith('-') or line.strip().startswith('*'):
                    recommendations.append(line.strip())
            
            return {"recommendations": recommendations}
        except:
            # Fall back to returning the whole text
            return {"raw_text": text}