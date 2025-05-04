from typing import Dict, Any, List
import json
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR

logger = get_logger(__name__)

def create_rule_extractor(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a rule extractor expert agent.
    
    This expert extracts rules from data patterns and analysis.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for rule extraction
    """
    system_prompt = """
You are a Rule Extraction Expert. Your job is to formulate clear, specific rules based on data analysis.

For credit card applications, you need to:
1. Extract specific rules that determine approval or rejection
2. Define clear thresholds and conditions
3. Organize rules in a logical structure (if-then format)
4. Ensure rules are comprehensive and cover all cases
5. Prioritize rules by importance/impact

Your rules should:
- Be precise and testable
- Have clear conditions and outcomes
- Cover all possible cases
- Be expressed in a format that can be implemented programmatically
- Be organized in a logical hierarchy if applicable

Format your rules in a structured JSON format that can be used for validation.
"""
    
    def rule_extraction_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior function for rule extraction"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        context = task.get("context", {})
        
        # Get data analysis from context if available
        analysis = ""
        applications = []
        
        for key, value in context.items():
            if isinstance(value, dict) and value.get("agent_name") == "Data Analyzer":
                if "analysis" in value.get("result", {}):
                    analysis = value["result"]["analysis"]
            
            if isinstance(value, dict) and value.get("agent_name") == "Data Generator":
                if "applications" in value.get("result", {}):
                    applications = value["result"]["applications"]
        
        if not analysis:
            return {
                "status": "error",
                "error": "No analysis found in context. Data Analyzer must run first."
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
        
        # Construct a prompt for rule extraction
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

Data Analysis:
{analysis}

Sample Applications Data (first 3 for reference):
{json.dumps(applications[:3], indent=2)}

Based on the data analysis, please extract specific rules for credit card application approval.
Formulate clear, precise rules that determine whether an application should be approved or declined.

Your rules should:
1. Specify exact thresholds for numeric values (e.g., credit score > 700)
2. Define specific conditions for categorical values
3. Include any compound conditions (e.g., income > 50000 AND debt_to_income < 0.3)
4. Be comprehensive enough to classify all applications
5. Be organized in a logical structure

Format your rules as a JSON object that includes:
1. A list of rules, each with conditions and an outcome (approve/decline)
2. The priority or order of rule application if important
3. Any default outcome if no rules match

This ruleset will be used to validate all applications, so it needs to be precise and comprehensive.
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=3000
        )
        
        # Extract JSON rules from the response
        try:
            # Look for JSON object pattern
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                ruleset = json.loads(json_match.group(0))
            else:
                # If no JSON object found, try parsing the entire output
                ruleset = json.loads(llm_response)
            
            # Save the ruleset to file
            os.makedirs(RESULTS_DIR, exist_ok=True)
            ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
            with open(ruleset_file, 'w') as f:
                json.dump(ruleset, f, indent=2)
            
            return {
                "status": "success",
                "ruleset": ruleset,
                "ruleset_file": ruleset_file,
                "message": "Extracted rules for credit card application approval"
            }
        except Exception as e:
            logger.error(f"Error parsing ruleset: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to parse ruleset: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Rule Extractor",
        capabilities=["rule_extraction"],
        behavior=rule_extraction_behavior,
        description="Extracts rules from data patterns and analysis"
    )
