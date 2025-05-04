from typing import Dict, Any, List
import json
import random
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, SCHEMA_DIR
import re

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
                    logger.info("Found schema in context")
                    break
        
        # If not found in context, try to load from file
        if not schema:
            schema_files = []
            schema_dir = os.path.join(os.path.dirname(APPLICATIONS_DIR), "schema")
            if os.path.exists(schema_dir):
                schema_files = [f for f in os.listdir(schema_dir) if f.endswith('.json')]
            
            if schema_files:
                latest_schema = sorted(schema_files)[-1]
                schema_file = os.path.join(schema_dir, latest_schema)
                try:
                    with open(schema_file, 'r') as f:
                        schema = json.load(f)
                    logger.info(f"Loaded schema from file: {schema_file}")
                except Exception as e:
                    logger.error(f"Error loading schema from file: {str(e)}")
        
        if not schema:
            logger.error("No schema found in context or files")
            return {
                "status": "error",
                "error": "No schema found in context or files. Schema Designer must run first."
            }
        
        # Check if applications already exist
        if os.path.exists(APPLICATIONS_DIR):
            application_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                              if f.startswith("application_") and f.endswith(".json")]
            
            # Check if we have enough applications (at least 20)
            if len(application_files) >= 20 and os.path.exists(os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")):
                logger.info(f"Found {len(application_files)} existing applications, skipping generation")
                
                # Load a few applications to return in the result
                applications = []
                for i in range(min(3, len(application_files))):
                    try:
                        with open(os.path.join(APPLICATIONS_DIR, application_files[i]), 'r') as f:
                            applications.append(json.load(f))
                    except Exception as e:
                        logger.error(f"Error loading application: {str(e)}")
                
                return {
                    "status": "success",
                    "result": {
                        "applications_dir": APPLICATIONS_DIR,
                        "applications": applications,
                        "message": f"Reused {len(application_files)} existing applications"
                    }
                }
        
        # Instead of generating all 20 applications at once, generate them in smaller batches
        all_applications = []
        
        # Generate 5 applications at a time (4 batches of 5)
        for batch in range(4):
            is_approved_batch = batch < 2  # First 2 batches are approved (10 applications)
            
            # Create a simpler schema with just the essential fields for the LLM
            simplified_schema = simplify_schema(schema)
            
            # Construct a focused prompt for generating just 5 applications
            prompt = f"""
Generate 5 realistic credit card applications that should be {'APPROVED' if is_approved_batch else 'DECLINED'}.

Follow these specific criteria for {'approved' if is_approved_batch else 'declined'} applications:
- {'Approved applications typically have: higher credit scores (700+), higher income, stable employment, low debt' if is_approved_batch else 'Declined applications typically have: lower credit scores (below 650), lower income, unstable employment, high debt'}
- Be diverse in the specific values but consistent with the approval pattern
- Follow the schema exactly

Schema (simplified for key fields):
{json.dumps(simplified_schema, indent=2)}

Respond ONLY with a valid JSON array of 5 application objects. No explanations or additional text.
Example format:
[
  {{
    "personalInformation": {{ ... }},
    "financialInformation": {{ ... }},
    "creditHistory": {{ ... }},
    ...
  }},
  // 4 more applications
]
"""
            
            # Get response from LLM with more precise parameters
            llm_response = llm_client.generate(
                prompt=prompt,
                system_message="You are a data generation expert. Your job is to create realistic JSON data that strictly follows the provided schema. Output ONLY valid JSON without any extra text.",
                temperature=0.7,
                max_tokens=3000,
                use_cache=False  # Disable caching to ensure fresh response
            )
            
            # Extract JSON array from the response
            try:
                # Clean the response thoroughly
                cleaned_response = llm_response.strip()
                
                # Remove any markdown code block markers
                cleaned_response = re.sub(r'^```json\s*', '', cleaned_response)
                cleaned_response = re.sub(r'^```\s*', '', cleaned_response)
                cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
                
                # Try to extract JSON array using regex
                json_match = re.search(r'\[[\s\S]*\]', cleaned_response)
                if json_match:
                    batch_applications = json.loads(json_match.group(0))
                else:
                    # If regex fails, try the whole thing
                    batch_applications = json.loads(cleaned_response)
                
                if not isinstance(batch_applications, list):
                    raise ValueError("Expected a list of applications")
                
                logger.info(f"Successfully generated batch {batch+1} with {len(batch_applications)} applications")
                
                # Expand any simplified fields back to full schema format
                batch_applications = [expand_to_full_schema(app, schema) for app in batch_applications]
                
                # Add to our collection
                all_applications.extend(batch_applications)
                
            except Exception as e:
                logger.error(f"Error generating batch {batch+1}: {str(e)}")
                logger.error(f"Raw response: {llm_response[:500]}...")
                
                # Try with a direct API call to structured generation instead
                try:
                    logger.info(f"Trying structured generation for batch {batch+1}")
                    output_schema = {
                        "type": "array",
                        "items": simplified_schema
                    }
                    
                    structured_response = llm_client.structured_generate(
                        prompt=f"Generate 5 realistic {'approved' if is_approved_batch else 'declined'} credit card applications based on the criteria",
                        output_schema=output_schema
                    )
                    
                    if isinstance(structured_response, list):
                        # Expand simplified fields
                        batch_applications = [expand_to_full_schema(app, schema) for app in structured_response]
                        all_applications.extend(batch_applications)
                        logger.info(f"Successfully generated batch {batch+1} with structured call")
                    else:
                        logger.error(f"Structured generation returned non-list: {structured_response}")
                        return {
                            "status": "error",
                            "error": f"Failed to generate applications: {str(e)}",
                            "raw_output": llm_response[:500]
                        }
                except Exception as e2:
                    logger.error(f"Error in structured generation fallback: {str(e2)}")
                    return {
                        "status": "error",
                        "error": f"Failed to generate applications: {str(e)} and {str(e2)}",
                        "raw_output": llm_response[:500]
                    }
        
        # Now save all applications
        os.makedirs(APPLICATIONS_DIR, exist_ok=True)
        
        # Save each application to a separate file
        application_files = []
        for i, application in enumerate(all_applications):
            file_path = os.path.join(APPLICATIONS_DIR, f"application_{i+1}.json")
            with open(file_path, 'w') as f:
                json.dump(application, f, indent=2)
            logger.info(f"Saved application to {file_path}")
            application_files.append(file_path)
        
        # Save hidden approvals to a separate file
        approved_indices = list(range(10))  # First 10 are approved
        hidden_approvals = {i: (i in approved_indices) for i in range(len(all_applications))}
        approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
        with open(approvals_file, 'w') as f:
            json.dump(hidden_approvals, f, indent=2)
        logger.info(f"Saved hidden approvals to {approvals_file}")
        
        return {
            "status": "success",
            "applications": all_applications,
            "application_files": application_files,
            "num_applications": len(all_applications),
            "message": f"Generated {len(all_applications)} credit card applications"
        }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Data Generator",
        capabilities=["data_generation"],
        behavior=data_generation_behavior,
        description="Generates sample data based on JSON schemas"
    )

def simplify_schema(full_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Create a simplified schema with just essential fields"""
    # Start with a basic structure
    simplified = {
        "type": "object",
        "properties": {
            "personalInformation": {
                "type": "object",
                "properties": {
                    "firstName": {"type": "string"},
                    "lastName": {"type": "string"},
                    "dateOfBirth": {"type": "string", "format": "date"}
                }
            },
            "financialInformation": {
                "type": "object",
                "properties": {
                    "annualIncome": {"type": "number"},
                    "employmentStatus": {"type": "string"},
                    "existingDebt": {"type": "number"}
                }
            },
            "creditHistory": {
                "type": "object",
                "properties": {
                    "creditScore": {"type": "integer"},
                    "paymentHistory": {"type": "string"}
                }
            },
            "requestedCreditDetails": {
                "type": "object",
                "properties": {
                    "cardType": {"type": "string"},
                    "creditLimit": {"type": "number"}
                }
            }
        }
    }
    
    return simplified

def expand_to_full_schema(simplified_app: Dict[str, Any], full_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a simplified application to conform to the full schema"""
    # Start with the simplified application
    expanded_app = simplified_app
    
    # Fill in any missing required fields from the schema
    if "properties" in full_schema:
        for section_name, section_schema in full_schema["properties"].items():
            if section_name not in expanded_app:
                expanded_app[section_name] = {}
            
            if "properties" in section_schema:
                for field_name, field_schema in section_schema["properties"].items():
                    # If field is missing in section, add a default value
                    if field_name not in expanded_app[section_name]:
                        field_type = field_schema.get("type", "string")
                        
                        # Generate a default value based on the field type
                        if field_type == "string":
                            # Check if there's an enum to pick from
                            if "enum" in field_schema:
                                expanded_app[section_name][field_name] = field_schema["enum"][0]
                            else:
                                expanded_app[section_name][field_name] = f"Default {field_name}"
                        elif field_type == "number" or field_type == "integer":
                            expanded_app[section_name][field_name] = 0
                        elif field_type == "boolean":
                            expanded_app[section_name][field_name] = True
                        elif field_type == "object" and field_name not in expanded_app[section_name]:
                            expanded_app[section_name][field_name] = {}
    
    return expanded_app
