from typing import Dict, Any, List
import json
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, RESULTS_DIR

logger = get_logger(__name__)

def create_validator(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a validation expert agent.
    
    This expert validates results against criteria and tests ruleset accuracy.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for validation
    """
    system_prompt = """
You are a Validation Expert. Your job is to test rules against data to verify their accuracy and effectiveness.

For credit card applications, you need to:
1. Apply the ruleset to all applications
2. Track which applications are approved/declined by the rules
3. Check for inconsistencies or edge cases
4. Evaluate the overall accuracy of the ruleset
5. Identify any patterns that the rules missed

Be thorough and precise in your validation. Test every rule against every application.
When reporting results, provide specific examples of successes and failures.
"""
    
    def validation_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior function for validation"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        context = task.get("context", {})
        
        # Get ruleset and applications from context if available
        ruleset = None
        applications = []
        
        for key, value in context.items():
            if isinstance(value, dict) and value.get("agent_name") == "Rule Extractor":
                if "ruleset" in value.get("result", {}):
                    ruleset = value["result"]["ruleset"]
            
            if isinstance(value, dict) and value.get("agent_name") == "Data Generator":
                if "applications" in value.get("result", {}):
                    applications = value["result"]["applications"]
        
        if not ruleset:
            # Try to load ruleset from file if it exists
            ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
            if os.path.exists(ruleset_file):
                try:
                    with open(ruleset_file, 'r') as f:
                        ruleset = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading ruleset from {ruleset_file}: {str(e)}")
        
        if not ruleset:
            return {
                "status": "error",
                "error": "No ruleset found in context or files. Rule Extractor must run first."
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
        
        # Get the hidden approvals for validation
        hidden_approvals = {}
        hidden_approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
        if os.path.exists(hidden_approvals_file):
            try:
                with open(hidden_approvals_file, 'r') as f:
                    hidden_approvals = json.load(f)
            except Exception as e:
                logger.error(f"Error loading hidden approvals from {hidden_approvals_file}: {str(e)}")
                
        # Construct a prompt for validation
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

Ruleset:
{json.dumps(ruleset, indent=2)}

Applications Data:
{json.dumps(applications, indent=2)}

Please validate the ruleset by applying it to all credit card applications.
For each application, determine whether it would be approved or declined based on the rules.

Track your results and evaluate the accuracy of the ruleset. Your validation should:
1. Apply each rule consistently to all applications
2. Determine the final approval/decline decision for each application
3. Identify any inconsistencies, edge cases, or applications that don't fit the rules
4. Calculate the overall accuracy of the ruleset

Format your response as a JSON object that includes:
1. The decision for each application (approved/declined)
2. Any applications where the rules give unexpected or inconsistent results
3. Overall accuracy metrics
4. Suggestions for improving the ruleset if it's not 100% accurate
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=3000
        )
        
        # Extract JSON results from the response
        try:
            # Look for JSON object pattern
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                validation_results = json.loads(json_match.group(0))
            else:
                # If no JSON object found, try parsing the entire output
                validation_results = json.loads(llm_response)
            
            # Validate against hidden approvals if available
            accuracy = 0
            if hidden_approvals:
                correct_count = 0
                total_count = 0
                
                decisions = validation_results.get("decisions", {})
                for app_id, approved in hidden_approvals.items():
                    if app_id in decisions:
                        app_decision = decisions[app_id].lower() == "approved"
                        if app_decision == approved:
                            correct_count += 1
                        total_count += 1
                
                if total_count > 0:
                    accuracy = correct_count / total_count
            
            # Add accuracy to validation results
            validation_results["calculated_accuracy"] = accuracy if accuracy > 0 else None
            
            # Save the validation results to file
            os.makedirs(RESULTS_DIR, exist_ok=True)
            validation_file = os.path.join(RESULTS_DIR, "validation_results.json")
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            # Determine success based on accuracy
            success = accuracy >= 1.0 if accuracy > 0 else False
            
            return {
                "status": "success" if success else "partial_success",
                "validation_results": validation_results,
                "accuracy": accuracy if accuracy > 0 else None,
                "success": success,
                "validation_file": validation_file,
                "message": "Validated ruleset against applications",
                "ruleset_accuracy": f"{accuracy * 100:.2f}%" if accuracy > 0 else "Unknown"
            }
        except Exception as e:
            logger.error(f"Error parsing validation results: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to parse validation results: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Validator",
        capabilities=["validation"],
        behavior=validation_behavior,
        description="Validates results against criteria and tests ruleset accuracy"
    )
