from typing import Dict, Any, List
import json
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR
import re

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
        analysis_found = False
        
        # First try to get analysis from context
        for key, value in context.items():
            if isinstance(value, dict) and value.get("agent_name") == "Data Analyzer":
                if "result" in value and "analysis" in value["result"]:
                    analysis = value["result"]["analysis"]
                    analysis_found = True
                    logger.info("Found analysis in context")
                    break
        
        # If we couldn't find analysis in the context, try to load it from file
        if not analysis_found:
            analysis_file = os.path.join(RESULTS_DIR, "credit_card_analysis.json")
            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, 'r') as f:
                        analysis_data = json.load(f)
                        analysis = analysis_data.get("analysis", "")
                        if analysis:
                            analysis_found = True
                            logger.info(f"Loaded analysis from file: {analysis_file}")
                except Exception as e:
                    logger.error(f"Error loading analysis from file: {str(e)}")
            
            # Try text file as backup
            if not analysis_found:
                text_file = os.path.join(RESULTS_DIR, "credit_card_analysis.txt")
                if os.path.exists(text_file):
                    try:
                        with open(text_file, 'r') as f:
                            analysis = f.read()
                            if analysis:
                                analysis_found = True
                                logger.info(f"Loaded analysis from text file: {text_file}")
                    except Exception as e:
                        logger.error(f"Error loading analysis from text file: {str(e)}")
        
        if not analysis_found:
            logger.error("No analysis found in context or files")
            return {
                "status": "error",
                "error": "No analysis found in context or files. Data Analyzer must run first."
            }
        
        # Get applications if needed for context
        applications = []
        for key, value in context.items():
            if isinstance(value, dict) and value.get("agent_name") == "Data Generator":
                if "applications" in value.get("result", {}):
                    applications = value["result"]["applications"]
                    logger.info(f"Found {len(applications)} applications in context")
                    break
        
        # If no applications in context, try loading from files
        if not applications:
            if os.path.exists(APPLICATIONS_DIR):
                application_files = [os.path.join(APPLICATIONS_DIR, f) for f in os.listdir(APPLICATIONS_DIR) 
                                   if f.startswith("application_") and f.endswith(".json")]
                
                if application_files:
                    applications = []
                    for file_path in application_files:
                        try:
                            with open(file_path, 'r') as f:
                                application = json.load(f)
                                applications.append(application)
                        except Exception as e:
                            logger.error(f"Error loading application from {file_path}: {str(e)}")
        
        # Construct a prompt for rule extraction
        prompt = f"""
Task Description: {task_description}

Based on the analysis of credit card applications, extract clear rules that determine whether an application is approved or rejected.

Analysis of Applications:
{analysis}

Please formulate specific rules with clear thresholds that can be used to classify applications. 

Your response MUST be a valid JSON object with the following structure:
{{
  "rules": [
    {{
      "field": "creditHistory.creditScore",
      "condition": ">=",
      "value": 700,
      "importance": "high"
    }},
    {{
      "field": "financialInformation.annualIncome",
      "condition": ">=",
      "value": 50000,
      "importance": "medium"
    }},
    // More rules as needed
  ],
  "logic": "all",  // Can be "all" (AND) or "any" (OR)
  "description": "Human-readable description of the ruleset"
}}

Create rules that would achieve 100% accuracy on the 20 sample applications. Do not include any explanatory text or markdown formatting in your response, only the valid JSON object.
"""
        
        # Include sample applications in the prompt if available
        if applications:
            prompt += "\n\nHere are a few sample applications for reference:\n"
            sample_apps = applications[:2] + applications[10:12]  # 2 approved, 2 declined
            prompt += json.dumps(sample_apps, indent=2)
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent JSON output
            max_tokens=3000
        )
        
        # Extract JSON rules from the response
        try:
            # Clean up response
            cleaned_response = llm_response.strip()
            cleaned_response = re.sub(r'^```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'^```\s*', '', cleaned_response)
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
            
            # Look for JSON object pattern
            json_match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if json_match:
                ruleset = json.loads(json_match.group(0))
            else:
                # If no JSON object found, try parsing the entire output
                ruleset = json.loads(cleaned_response)
            
            # Save the ruleset to file
            os.makedirs(RESULTS_DIR, exist_ok=True)
            ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
            with open(ruleset_file, 'w') as f:
                json.dump(ruleset, f, indent=2)
            logger.info(f"Saved ruleset to {ruleset_file}")
            
            return {
                "status": "success",
                "result": {
                    "ruleset": ruleset,
                    "ruleset_file": ruleset_file,
                    "message": "Extracted rules for credit card application approval"
                }
            }
        except Exception as e:
            logger.error(f"Error parsing ruleset: {str(e)}")
            logger.error(f"Raw LLM response: {llm_response}")
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
