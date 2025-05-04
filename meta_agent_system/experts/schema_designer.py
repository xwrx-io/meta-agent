from typing import Dict, Any
import json
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.utils.helpers import save_json
from meta_agent_system.config.settings import SCHEMA_DIR
import os

logger = get_logger(__name__)

def create_schema_designer(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a schema designer expert agent.
    
    This expert creates comprehensive JSON schemas for data structures.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for schema design
    """
    system_prompt = """
You are a Schema Design Expert. Your job is to create comprehensive JSON schemas that accurately represent complex data structures.

For credit card applications, consider including fields such as:
- Personal information (name, address, contact details)
- Financial information (income, employment, existing debt)
- Credit history (credit score, payment history)
- Requested credit details (card type, credit limit)
- Additional relevant information

Your schema should:
1. Be comprehensive and include all relevant fields
2. Use appropriate data types for each field
3. Include descriptions for each field
4. Specify required fields
5. Include any validation rules (min/max values, patterns, etc.)

Format your response as a valid JSON Schema object that can be used for validation.
"""
    
    def schema_design_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior function for schema design"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        
        # Check if schema already exists
        schema_file = os.path.join(SCHEMA_DIR, "credit_card_application_schema.json")
        if os.path.exists(schema_file):
            logger.info(f"Schema file already exists at {schema_file}, skipping generation")
            return {
                "status": "success",
                "result": {
                    "schema_file": schema_file,
                    "message": "Reused existing schema file"
                }
            }
        
        # Construct a prompt for schema design
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

Requirements:
{json.dumps(task_data.get('requirements', []), indent=2)}

Please create a comprehensive JSON schema for a credit card application. 
This schema should include all relevant fields that might be used to make approval decisions.

Respond with a complete, valid JSON Schema document.
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=3000
        )
        
        # Extract JSON schema from the response
        try:
            # Look for JSON object pattern
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                schema = json.loads(json_match.group(0))
            else:
                # If no JSON object found, try parsing the entire output
                schema = json.loads(llm_response)
            
            # Save the schema to file
            os.makedirs(SCHEMA_DIR, exist_ok=True)
            schema_file = os.path.join(SCHEMA_DIR, "credit_card_application_schema.json")
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2)
            
            return {
                "status": "success",
                "schema": schema,
                "schema_file": schema_file,
                "message": "Created JSON schema for credit card applications"
            }
        except Exception as e:
            logger.error(f"Error parsing schema: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to parse schema: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Schema Designer",
        capabilities=["schema_design"],
        behavior=schema_design_behavior,
        description="Creates comprehensive JSON schemas for data structures"
    )
