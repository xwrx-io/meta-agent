from typing import Dict, Any, List
import json
import os
import re
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR

logger = get_logger(__name__)

def create_rule_extractor(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a rule extractor expert agent.
    
    This expert extracts rules from data analysis.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for rule extraction
    """
    system_prompt = """
You are a Rule Extraction Expert. Your job is to analyze data patterns and extract clear, actionable rules that explain those patterns.

For credit card applications, you need to:
1. Identify the key factors that determine approval or rejection
2. Define specific thresholds for those factors (e.g., minimum credit score)
3. Create rules that combine these factors effectively
4. Avoid creating rules that memorize specific cases rather than capturing general patterns
5. Think about both simple threshold rules and more complex combinations of factors

Your rules should be clear, specific, and generalizable to new applications.
Format your response as a structured set of rules that can be used to make approval decisions.
"""
    
    def rule_extraction_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified behavior function for rule extraction"""
        context = task.get("context", {})
        
        # Get applications 
        applications = get_applications_from_context_or_file(context)
        hidden_approvals = get_hidden_approvals()
        
        if not applications or not hidden_approvals:
            return {"status": "error", "message": "Missing applications or approvals"}
        
        # Prepare approved and declined applications for LLM analysis
        approved_apps = []
        declined_apps = []
        
        for i, app in enumerate(applications):
            app_id = str(i + 1)  # Applications are 1-indexed
            if app_id in hidden_approvals:
                if hidden_approvals[app_id]:
                    approved_apps.append(app)
                else:
                    declined_apps.append(app)
        
        # Get misclassified applications
        misclass_file = os.path.join(RESULTS_DIR, "persistent_misclassifications.json")
        persistent_misclassifications = {}
        if os.path.exists(misclass_file):
            try:
                with open(misclass_file, 'r') as f:
                    persistent_misclassifications = json.load(f)
            except Exception as e:
                logger.error(f"Error loading persistent misclassifications: {str(e)}")
        
        # Create a simpler prompt focused on what worked and what didn't
        prompt = f"""
You are a Credit Card Rule Expert. Create rules that fix the most commonly misclassified applications.

INCORRECTLY CLASSIFIED APPLICATIONS:
{json.dumps([app for app_id, app in persistent_misclassifications.items()], indent=2)}

IMPORTANT: 
1. Create a ruleset with exactly 3 rules, each designed to fix specific misclassifications
2. Each rule should be precise and focused on a specific pattern
3. Use nested logic (AND/OR combinations) for complex patterns

CRITICAL: Use these tier-based fields for more reliable rules:
- creditHistory.creditTier (values: Excellent, Very Good, Good, Poor, Very Poor)
- financialInformation.incomeTier (values: Very High, High, Medium, Low)
- financialInformation.debtTier (values: Very Low, Low, Medium, High)

Return ONLY a JSON object in this format:
{{
  "logic": "any",
  "rules": [
    // 3 rules, each can have nested logic using tier fields
  ],
  "description": "Brief explanation"
}}
"""
        
        # Get LLM's analysis and rules
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.1  # Low temperature for more deterministic output
        )
        
        # Extract JSON ruleset from response
        try:
            # Look for JSON pattern
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                ruleset = json.loads(json_match.group(0))
            else:
                # Try parsing the entire response as JSON
                ruleset = json.loads(llm_response)
            
            # Save the ruleset
            ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
            with open(ruleset_file, 'w') as f:
                json.dump(ruleset, f, indent=2)
            
            logger.info(f"LLM-generated ruleset saved to {ruleset_file}")
            
            return {
                "status": "success",
                    "ruleset": ruleset,
                    "ruleset_file": ruleset_file,
                "message": "Generated rules based on LLM analysis"
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {"status": "error", "message": f"Error parsing LLM response: {str(e)}"}
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Rule Extractor",
        capabilities=["rule_extraction", "data_analysis"],
        behavior=rule_extraction_behavior,
        description="Extracts rules from data analysis"
    )

def get_applications_from_context_or_file(context):
    """Get applications from context or file"""
    applications = []
    for key, value in context.items():
        if isinstance(value, dict) and value.get("agent_name") == "Data Generator":
            if "applications" in value.get("result", {}):
                applications = value["result"]["applications"]
                logger.info(f"Found {len(applications)} applications in context")
                break
    
    # If no applications in context, try loading from files
    if not applications and os.path.exists(APPLICATIONS_DIR):
        application_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                          if f.startswith("application_") and f.endswith(".json")]
        
        if application_files:
            applications = []
            for f in application_files:
                file_path = os.path.join(APPLICATIONS_DIR, f)
                try:
                    with open(file_path, 'r') as file:
                        application = json.load(file)
                        applications.append(application)
                except Exception as e:
                    logger.error(f"Error loading application from {file_path}: {str(e)}")
    
    return applications

def get_hidden_approvals():
    """Get hidden approvals from file"""
    hidden_approvals = {}
    hidden_approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
    if os.path.exists(hidden_approvals_file):
        try:
            with open(hidden_approvals_file, 'r') as f:
                hidden_approvals = json.load(f)
                logger.info(f"Loaded hidden approvals with {len(hidden_approvals)} entries")
        except Exception as e:
            logger.error(f"Error loading hidden approvals: {str(e)}")
    
    return hidden_approvals

def apply_ruleset(ruleset, application):
    """Apply a ruleset to an application"""
    logic = ruleset.get("logic", "all")
    
    if logic == "scoring":
        # Apply scoring rules
        total_score = 0
        for rule in ruleset.get("rules", []):
            if rule.get("type") == "range":
                # Get field value
                value = get_nested_value(application, rule.get("field"))
                # Find matching range
                for range_def in rule.get("ranges", []):
                    min_val = range_def.get("min", 0)
                    max_val = range_def.get("max", float('inf'))
                    if min_val <= value <= max_val:
                        total_score += range_def.get("score", 0)
                        break
            elif rule.get("type") == "categorical":
                # Get field value
                value = get_nested_value(application, rule.get("field"))
                # Look up in categories
                if value in rule.get("categories", {}):
                    total_score += rule.get("categories", {}).get(value, 0)
        
        # Compare to threshold
        return total_score >= ruleset.get("threshold", 60)
    
    # Standard rule logic
    rules = ruleset.get("rules", [])
    if not rules:
        return False
    
    # Apply each rule
    results = []
    for rule in rules:
        field = rule.get("field", "")
        condition = rule.get("condition", "")
        rule_value = rule.get("value", "")
        
        # Get field value from application
        app_value = get_nested_value(application, field)
        
        # Evaluate condition
        result = False
        try:
            if condition == "==": result = app_value == rule_value
            elif condition == "!=": result = app_value != rule_value
            elif condition == ">=": result = float(app_value) >= float(rule_value)
            elif condition == "<=": result = float(app_value) <= float(rule_value)
            elif condition == ">": result = float(app_value) > float(rule_value)
            elif condition == "<": result = float(app_value) < float(rule_value)
        except:
            result = False
        
        results.append(result)
    
    # Apply logic
    if logic == "all":
        return all(results)
    elif logic == "any":
        return any(results)
    
    return False

def get_nested_value(obj, path):
    """Get a value from a nested object using a dot path"""
    if not path:
        return None
        
    parts = path.split('.')
    value = obj
    
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    
    return value
