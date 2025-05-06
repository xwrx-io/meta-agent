from typing import Dict, Any, List
import json
import os
import re
import time
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, RESULTS_DIR

logger = get_logger(__name__)

def create_rule_refiner(llm_client: OpenAIClient) -> ExpertAgent:
    """Create a rule refinement expert agent that learns from examples."""
    system_prompt = """
You are a Credit Card Approval Rule Expert. Your goal is to discover the exact approval rules by analyzing examples.

YOUR PRIMARY GOAL is to identify the patterns that separate approved and declined applications.
Focus on finding clear rules that achieve 100% accuracy on the examples.

I'll provide:
1. Current ruleset
2. Approved applications
3. Declined applications
4. Misclassified applications (where current rules are wrong)

You'll create a new ruleset that focuses on the MOST IMPORTANT FACTORS: 
- creditTier (Very Poor, Poor, Good, Very Good, Excellent)
- incomeTier (Low, Medium, High, Very High)
- debtTier (Very Low, Low, Medium, High)

Look for SIMPLE, CLEAR PATTERNS in the data!
"""
    
    def rule_refinement_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize rules based on clear examples and diagnostic feedback."""
        iteration = task.get("data", {}).get("iteration", 0)
        
        # Load data
        applications = load_applications()
        current_ruleset = load_ruleset()
        diagnostics = load_diagnostics()
        hidden_approvals = load_hidden_approvals()
        
        if not applications or not hidden_approvals:
            logger.error("Missing application data")
            return {"status": "error", "message": "Missing application data"}
        
        # Separate applications into approved, declined, and misclassified
        approved_examples = []
        declined_examples = []
        misclassified = []
        
        for idx, app in enumerate(applications):
            app_id = idx + 1
            key = str(app_id)
            should_approve = hidden_approvals.get(key, False)
            
            # Extract key data from application
            app_data = {
                "id": app_id,
                "credit_tier": app.get("creditHistory", {}).get("creditTier"),
                "credit_score": app.get("creditHistory", {}).get("creditScore"),
                "income_tier": app.get("financialInformation", {}).get("incomeTier"),
                "income": app.get("financialInformation", {}).get("annualIncome"),
                "debt_tier": app.get("financialInformation", {}).get("debtTier"),
                "debt": app.get("financialInformation", {}).get("existingDebt"),
                "payment_history": app.get("creditHistory", {}).get("paymentHistory")
            }
            
            # Find if current rules classify this correctly
            is_misclassified = False
            current_result = None
            
            for eval in diagnostics.get("rule_evaluations", []):
                if eval.get("application_id") == app_id:
                    current_result = eval.get("actual")
                    is_misclassified = eval.get("expected") != eval.get("actual")
                    break
            
            if is_misclassified:
                misclassified.append({
                    **app_data,
                    "should_approve": should_approve,
                    "current_result": current_result
                })
            elif should_approve:
                approved_examples.append(app_data)
            else:
                declined_examples.append(app_data)
        
        # Create prompt with clear application examples and feedback
        prompt = create_teaching_prompt(current_ruleset, approved_examples, declined_examples, misclassified, iteration)
        
        # Get LLM response
        logger.info(f"Getting rules for iteration {iteration}")
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.2,
            use_cache=False
        )
        
        # Process LLM response
        try:
            # Extract JSON from response
            improved_ruleset = extract_ruleset(llm_response, iteration)
            
            # Save the ruleset
            save_ruleset_file(improved_ruleset)
            
            return {
                "status": "success",
                "ruleset": improved_ruleset,
                "message": f"Created new ruleset for iteration {iteration}"
            }
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            
            # Create a simple fallback ruleset
            fallback_ruleset = create_fallback_ruleset(iteration)
            save_ruleset_file(fallback_ruleset)
            
            return {
                "status": "error",
                "ruleset": fallback_ruleset,
                "message": f"Used fallback ruleset due to error: {str(e)}"
            }
    
    def load_applications():
        """Load all applications."""
        applications = []
        if os.path.exists(APPLICATIONS_DIR):
            app_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                        if f.startswith("application_") and f.endswith(".json")]
            
            app_files.sort(key=lambda f: int(f.replace("application_", "").replace(".json", "")))
            
            for f in app_files:
                try:
                    with open(os.path.join(APPLICATIONS_DIR, f), 'r') as file:
                        applications.append(json.load(file))
                except Exception as e:
                    logger.error(f"Error loading {f}: {str(e)}")
        
        return applications
    
    def load_ruleset():
        """Load current ruleset."""
        ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
        if os.path.exists(ruleset_file):
            try:
                with open(ruleset_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading ruleset: {str(e)}")
        
        return {}
    
    def load_diagnostics():
        """Load validation diagnostics."""
        diagnostics_file = os.path.join(RESULTS_DIR, "validation_diagnostics.json")
        if os.path.exists(diagnostics_file):
            try:
                with open(diagnostics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading diagnostics: {str(e)}")
        
        return {}
    
    def load_hidden_approvals():
        """Load hidden approval decisions."""
        approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
        if os.path.exists(approvals_file):
            try:
                with open(approvals_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading hidden approvals: {str(e)}")
        
        return {}
    
    def create_teaching_prompt(current_ruleset, approved, declined, misclassified, iteration):
        """Create a clear, educational prompt with examples."""
        # Format examples nicely
        approved_examples = format_examples(approved[:3], "APPROVED")
        declined_examples = format_examples(declined[:3], "DECLINED")
        misclassified_examples = ""
        
        for app in misclassified[:5]:
            misclassified_examples += f"""
APPLICATION #{app['id']}:
- Credit Tier: {app['credit_tier']} (Score: {app['credit_score']})
- Income Tier: {app['income_tier']} (${app['income']})
- Debt Tier: {app['debt_tier']} (${app['debt']})
- Current result: {'APPROVED' if app['current_result'] else 'DECLINED'}
- SHOULD BE: {'APPROVED' if app['should_approve'] else 'DECLINED'}
"""
        
        # Create a teaching prompt that explains patterns
        return f"""
# Credit Card Approval Rule Discovery - Iteration {iteration}

Your goal is to discover the exact rules determining credit card approvals.

## Current Ruleset (Accuracy: Not Perfect)
```json
{json.dumps(current_ruleset, indent=2)}
```

## CORRECTLY CLASSIFIED EXAMPLES:

### APPROVED APPLICATIONS:
{approved_examples}

### DECLINED APPLICATIONS:
{declined_examples}

## MISCLASSIFIED APPLICATIONS (Current rules get these wrong):
{misclassified_examples}

## YOUR TASK:
1. Study the examples carefully
2. Identify clear patterns that separate approvals from declines
3. Focus especially on the misclassified applications
4. Create rules based on creditTier, incomeTier, and debtTier where possible
5. Use a simple logical structure - either ANY of several conditions OR ALL conditions must be met

## RULE FORMAT:
Return ONLY a JSON object with this structure:
```json
{{
  "logic": "any",  // Use "any" or "all" at the top level
  "rules": [
    {{
      "field": "creditHistory.creditTier",
      "condition": "equals",
      "threshold": "Excellent"
    }},
    {{
      "logic": "all",  // You can nest logic groups
      "rules": [
        {{
          "field": "financialInformation.incomeTier",
          "condition": "in",
          "values": ["High", "Very High"]
        }},
        {{
          "field": "financialInformation.debtTier",
          "condition": "equals",
          "threshold": "Low"
        }}
      ]
    }}
  ],
  "description": "Brief explanation of your rule strategy"
}}
```

RESPOND WITH ONLY THE JSON RULESET - NO OTHER TEXT.
"""
    
    def format_examples(examples, decision):
        """Format application examples nicely."""
        result = ""
        for app in examples:
            result += f"""
APPLICATION #{app['id']}:
- Credit Tier: {app['credit_tier']} (Score: {app['credit_score']})
- Income Tier: {app['income_tier']} (${app['income']})
- Debt Tier: {app['debt_tier']} (${app['debt']})
- Result: {decision}
"""
        return result
    
    def extract_ruleset(llm_response, iteration):
        """Extract JSON ruleset from LLM response."""
        # Find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if not json_match:
            logger.error(f"No JSON found in response: {llm_response[:100]}...")
            raise ValueError("No valid JSON found in LLM response")
        
        extracted_json = json_match.group(0)
        
        try:
            # Parse the JSON
            ruleset = json.loads(extracted_json)
            
            # Basic validation
            if "rules" not in ruleset or not isinstance(ruleset.get("rules"), list):
                raise ValueError("Invalid ruleset - missing rules array")
            
            if not ruleset.get("rules"):
                raise ValueError("Rules array cannot be empty")
            
            # Set defaults if missing
            if "logic" not in ruleset:
                ruleset["logic"] = "all"
            
            if "description" not in ruleset:
                ruleset["description"] = "Generated credit card approval rules"
            
            # Add metadata
            ruleset["timestamp"] = int(time.time())
            ruleset["iteration"] = iteration
            
            return ruleset
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {extracted_json[:100]}...")
            
            # Try to fix JSON issues
            fixed_json = re.sub(r'//.*', '', extracted_json)  # Remove comments
            fixed_json = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', fixed_json)  # Fix keys
            fixed_json = fixed_json.replace("'", "\"")  # Replace single quotes
            
            try:
                ruleset = json.loads(fixed_json)
                ruleset["timestamp"] = int(time.time())
                ruleset["iteration"] = iteration
                return ruleset
            except:
                raise ValueError("Could not parse JSON response")
    
    def save_ruleset_file(ruleset):
        """Save ruleset to file with verification."""
        ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
        
        with open(ruleset_file, 'w') as f:
            json.dump(ruleset, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        logger.info(f"Saved ruleset with {len(ruleset.get('rules', []))} rules")
    
    def create_fallback_ruleset(iteration):
        """Create a simple fallback ruleset."""
        return {
            "logic": "any",
            "rules": [
                {
                    "field": "creditHistory.creditTier", 
                    "condition": "equals",
                    "threshold": "Excellent"
                },
                {
                    "logic": "all",
                    "rules": [
                        {
                            "field": "financialInformation.incomeTier",
                            "condition": "equals",
                            "threshold": "Very High"
                        },
                        {
                            "field": "creditHistory.creditTier",
                            "condition": "in",
                            "values": ["Good", "Very Good"]
                        }
                    ]
                },
                {
                    "logic": "all",
                    "rules": [
                        {
                            "field": "financialInformation.debtTier",
                            "condition": "equals",
                            "threshold": "Very Low"
                        },
                        {
                            "field": "creditHistory.creditTier",
                            "condition": "in",
                            "values": ["Good", "Very Good"]
                        }
                    ]
                }
            ],
            "description": "Fallback ruleset focused on credit tier, income tier, and debt tier",
            "timestamp": int(time.time()),
            "iteration": iteration
        }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Rule Refiner",
        capabilities=["rule_refinement", "rule_optimization"],
        behavior=rule_refinement_behavior,
        description="Discovers credit card approval rules from examples"
    )
