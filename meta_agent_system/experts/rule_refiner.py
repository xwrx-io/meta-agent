from typing import Dict, Any, List
import json
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, RESULTS_DIR

logger = get_logger(__name__)

def create_rule_refiner(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a rule refinement expert agent.
    
    This expert improves rules based on validation feedback.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for rule refinement
    """
    system_prompt = """
You are a Rule Refinement Expert. Your job is to improve existing rules based on validation feedback.

For credit card application rules, you need to:
1. Analyze validation results to identify where rules are failing
2. Refine rule thresholds and conditions to improve accuracy
3. Add new rules to cover edge cases
4. Remove or modify rules that cause errors
5. Ensure the ruleset as a whole is coherent and comprehensive

Your goal is to achieve 100% accuracy in classifying all applications correctly.
Be precise in your refinements, and explain the reasoning behind each change.

Format your improved ruleset as a structured JSON object that can be used for further validation.
"""
    
    def rule_refinement_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior function for rule refinement"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        context = task.get("context", {})
        
        # Get current ruleset, validation results, and applications
        ruleset = None
        validation_results = None
        applications = []
        
        # Find the most recent rule extraction result
        for key, value in reversed(list(context.items())):
            if isinstance(value, dict) and value.get("agent_name") == "Rule Extractor":
                if "result" in value and "ruleset" in value["result"]:
                    ruleset = value["result"]["ruleset"]
                    break
        
        # Find the most recent validation result
        for key, value in reversed(list(context.items())):
            if isinstance(value, dict) and value.get("agent_name") == "Validator":
                if "result" in value and "validation_results" in value["result"]:
                    validation_results = value["result"]["validation_results"]
                    break
        
        # Get applications
        for key, value in context.items():
            if isinstance(value, dict) and value.get("agent_name") == "Data Generator":
                if "applications" in value.get("result", {}):
                    applications = value["result"]["applications"]
                    break
        
        if not ruleset:
            return {
                "status": "error",
                "error": "No ruleset found in context. Rule Extractor must run first."
            }
        
        if not validation_results:
            return {
                "status": "error",
                "error": "No validation results found in context. Validator must run first."
            }
        
        if not applications:
            # Try to load applications from files if they exist
            if os.path.exists(APPLICATIONS_DIR):
                application_files = [os.path.join(APPLICATIONS_DIR, f) for f in os.listdir(APPLICATIONS_DIR) 
                                   if f.startswith("application_") and f.endswith(".json")]
                
                applications = []
                for file_path in application_files:
                    try:
                        with open(file_path, 'r') as f:
                            applications.append(json.load(f))
                    except Exception as e:
                        logger.error(f"Error loading application from {file_path}: {str(e)}")
        
        if not applications:
            return {
                "status": "error",
                "error": "No applications found in context or files. Data Generator must run first."
            }
        
        # Construct a prompt for rule refinement
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

Current Ruleset:
{json.dumps(ruleset, indent=2)}

Validation Results:
{json.dumps(validation_results, indent=2)}

Example Applications (first 3):
{json.dumps(applications[:3], indent=2)}

Please refine the ruleset based on the validation results to improve its accuracy. 
The goal is to achieve 100% accuracy in classifying all applications correctly.

For each rule you modify or add:
1. Explain why the change is needed
2. How it addresses a specific issue found in validation
3. How it improves the overall accuracy

Your refined ruleset should:
1. Fix any misclassifications identified in the validation
2. Address any edge cases that were missed
3. Refine thresholds or conditions for better precision
4. Remove or modify rules that cause errors
5. Maintain a coherent overall structure

Return the complete refined ruleset as a JSON object that can be used for further validation.
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=3000
        )
        
        # Extract JSON ruleset from the response
        try:
            # Look for JSON object pattern
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                refined_ruleset = json.loads(json_match.group(0))
            else:
                # If no JSON object found, try parsing the entire output
                refined_ruleset = json.loads(llm_response)
            
            # Save the refined ruleset to file
            os.makedirs(RESULTS_DIR, exist_ok=True)
            ruleset_file = os.path.join(RESULTS_DIR, "refined_credit_card_approval_rules.json")
            with open(ruleset_file, 'w') as f:
                json.dump(refined_ruleset, f, indent=2)
            
            # Extract explanation from the response
            explanation = llm_response
            try:
                # Remove the JSON part for better readability
                json_str = json_match.group(0) if json_match else json.dumps(refined_ruleset)
                explanation = llm_response.replace(json_str, "").strip()
            except:
                pass
            
            return {
                "status": "success",
                "ruleset": refined_ruleset,
                "explanation": explanation,
                "ruleset_file": ruleset_file,
                "message": "Refined rules for credit card application approval"
            }
        except Exception as e:
            logger.error(f"Error parsing refined ruleset: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to parse refined ruleset: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Rule Refiner",
        capabilities=["rule_refinement", "rule_optimization"],
        behavior=rule_refinement_behavior,
        description="Improves rules based on validation feedback"
    )
