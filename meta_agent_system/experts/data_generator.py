from typing import Dict, Any, List
import json
import random
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR

logger = get_logger(__name__)

def create_data_generator(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a data generator expert agent.
    
    This expert generates sample data based on JSON schemas.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for data generation
    """
    system_prompt = """
You are a Data Generation Expert. Your job is to create realistic, diverse sample data based on JSON schemas.

For credit card applications, you need to generate:
- 10 applications that should be approved
- 10 applications that should be declined

The approval/rejection should be based on sensible, realistic criteria such as:
- Income level relative to requested credit
- Credit score
- Debt-to-income ratio
- Employment stability
- Credit history
- Existing debt

Make sure to create diverse applications with different combinations of these factors. 
The rules for approval should be implicit in the data but not explicitly stated.

Format your response as a JSON array of application objects that conform to the schema.
"""
    
    def data_generation_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior function for data generation"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        context = task.get("context", {})
        
        # Get schema from context if available
        schema = None
        for key, value in context.items():
            if isinstance(value, dict) and value.get("agent_name") == "Schema Designer":
                if "schema" in value.get("result", {}):
                    schema = value["result"]["schema"]
                    break
        
        if not schema:
            return {
                "status": "error",
                "error": "No schema found in context. Schema Designer must run first."
            }
        
        # Construct a prompt for data generation
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

JSON Schema:
{json.dumps(schema, indent=2)}

Please generate 20 diverse credit card applications based on this schema:
- 10 applications that should be approved
- 10 applications that should be declined

The approval/rejection should be based on realistic criteria, but DO NOT include an 'approved' field in the data.
The meta agent will need to discover the approval rules by analyzing the applications.

For this task, you should consider factors like:
- Income level relative to requested credit
- Credit score
- Debt-to-income ratio
- Employment stability
- Credit history
- Existing debt

Generate diverse applications with different combinations of these factors.
Respond with a JSON array containing all 20 application objects.
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=4000
        )
        
        # Extract JSON array from the response
        try:
            # Look for JSON array pattern
            import re
            json_match = re.search(r'\[[\s\S]*\]', llm_response)
            if json_match:
                applications = json.loads(json_match.group(0))
            else:
                # If no JSON array found, try parsing the entire output
                applications = json.loads(llm_response)
            
            if not isinstance(applications, list):
                raise ValueError("Expected a list of applications")
            
            # Create a hidden record of which applications should be approved
            approved_indices = list(range(10))  # First 10 are approved
            hidden_approvals = {i: (i in approved_indices) for i in range(len(applications))}
            
            # Save applications to files
            os.makedirs(APPLICATIONS_DIR, exist_ok=True)
            
            # Save each application to a separate file
            application_files = []
            for i, application in enumerate(applications):
                file_path = os.path.join(APPLICATIONS_DIR, f"application_{i+1}.json")
                with open(file_path, 'w') as f:
                    json.dump(application, f, indent=2)
                application_files.append(file_path)
            
            # Save hidden approvals to a separate file
            approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
            with open(approvals_file, 'w') as f:
                json.dump(hidden_approvals, f, indent=2)
            
            return {
                "status": "success",
                "applications": applications,
                "application_files": application_files,
                "num_applications": len(applications),
                "message": f"Generated {len(applications)} credit card applications"
            }
        except Exception as e:
            logger.error(f"Error generating applications: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to generate applications: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Data Generator",
        capabilities=["data_generation"],
        behavior=data_generation_behavior,
        description="Generates sample data based on JSON schemas"
    )
