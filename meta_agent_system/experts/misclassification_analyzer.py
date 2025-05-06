import json
import os
from typing import Dict, Any, List
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, RESULTS_DIR
from meta_agent_system.llm.openai_client import OpenAIClient

logger = get_logger(__name__)

def analyze_misclassifications(llm_client=None):
    """Analyze misclassified applications in depth to provide targeted feedback"""
    # Initialize OpenAI client if not passed in
    if llm_client is None:
        llm_client = OpenAIClient()
    
    # Load necessary data
    with open(os.path.join(RESULTS_DIR, "validation_diagnostics.json"), 'r') as f:
        diagnostics = json.load(f)
    
    with open(os.path.join(RESULTS_DIR, "persistent_misclassifications.json"), 'r') as f:
        persistent_misclassifications = json.load(f)
    
    # Get applications from files
    applications = []
    application_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                       if f.startswith("application_") and f.endswith(".json")]
    
    for f in application_files:
        app_id = f.replace("application_", "").replace(".json", "")
        with open(os.path.join(APPLICATIONS_DIR, f), 'r') as file:
            app = json.load(file)
            app["id"] = app_id
            applications.append(app)
    
    # Index applications by ID for quick lookup
    app_dict = {app["id"]: app for app in applications}
    
    # Get the rule evaluations
    rule_evaluations = diagnostics.get("rule_evaluations", [])
    ruleset = diagnostics.get("ruleset", {})
    
    # Identify incorrect evaluations
    incorrect_evaluations = [eval for eval in rule_evaluations if not eval.get("correct", False)]
    
    # Detailed analysis of each misclassified application
    detailed_analysis = []
    
    for eval in incorrect_evaluations:
        app_id = str(eval.get("application_id"))
        app = app_dict.get(app_id)
        
        if not app:
            continue
            
        # Get application details
        credit_tier = app["creditHistory"]["creditTier"]
        payment_history = app["creditHistory"]["paymentHistory"]
        income_tier = app["financialInformation"]["incomeTier"]
        debt_tier = app["financialInformation"]["debtTier"]
        employment = app["financialInformation"]["employmentStatus"]
        
        # Get rule evaluation details
        rule_evals = eval.get("rule_evaluations", [])
        
        # Determine why this application was misclassified
        rules_failed = []
        rules_passed = []
        
        for rule_eval in rule_evals:
            if rule_eval.get("passed", False):
                rules_passed.append(rule_eval)
            else:
                rules_failed.append(rule_eval)
        
        # Find similar applications that were correctly classified
        expected_approval = eval.get("expected")
        similar_apps = []
        
        for other_app in applications:
            if other_app["id"] == app_id:
                continue
                
            # Check if this app has similar characteristics
            other_credit_tier = other_app["creditHistory"]["creditTier"]
            other_payment_history = other_app["creditHistory"]["paymentHistory"]
            other_income_tier = other_app["financialInformation"]["incomeTier"]
            other_debt_tier = other_app["financialInformation"]["debtTier"]
            
            # Calculate similarity score (simple version)
            similarity_score = 0
            if credit_tier == other_credit_tier:
                similarity_score += 0.3
            if payment_history == other_payment_history:
                similarity_score += 0.2
            if income_tier == other_income_tier:
                similarity_score += 0.3
            if debt_tier == other_debt_tier:
                similarity_score += 0.2
            
            # Find correct classifications with high similarity
            other_eval = next((e for e in rule_evaluations if e.get("application_id") == int(other_app["id"])), None)
            if other_eval and other_eval.get("correct") and other_eval.get("expected") == expected_approval and similarity_score > 0.5:
                similar_apps.append({
                    "id": other_app["id"],
                    "credit_tier": other_credit_tier,
                    "payment_history": other_payment_history, 
                    "income_tier": other_income_tier,
                    "debt_tier": other_debt_tier,
                    "similarity_score": similarity_score
                })
        
        # Sort similar apps by similarity score
        similar_apps = sorted(similar_apps, key=lambda x: x["similarity_score"], reverse=True)[:3]
        
        # Create detailed analysis for this application
        analysis = {
            "application_id": app_id,
            "expected": "approve" if expected_approval else "decline",
            "actual": "approve" if eval.get("actual") else "decline",
            "application_details": {
                "credit_tier": credit_tier,
                "payment_history": payment_history,
                "income_tier": income_tier,
                "debt_tier": debt_tier,
                "employment": employment
            },
            "rule_analysis": {
                "rules_failed": len(rules_failed),
                "rules_passed": len(rules_passed),
                "reason": generate_failure_reason(app, rules_failed, ruleset)
            },
            "similar_correctly_classified": similar_apps,
            "suggested_rule_improvement": suggest_rule_improvement(app, expected_approval, similar_apps)
        }
        
        # Add persistence information
        if app_id in persistent_misclassifications:
            analysis["persistence"] = {
                "count": persistent_misclassifications[app_id].get("misclassification_count", 0),
                "iterations": persistent_misclassifications[app_id].get("iterations", [])
            }
        
        # Get deeper LLM analysis if significant misclassification
        if app_id in persistent_misclassifications and persistent_misclassifications[app_id].get("misclassification_count", 0) > 1:
            # Create a prompt for the LLM
            prompt = f"""
            I need a detailed analysis of why this credit card application is being misclassified.
            
            Application details:
            - Credit tier: {credit_tier}
            - Payment history: {payment_history}
            - Income tier: {income_tier}
            - Debt tier: {debt_tier}
            - Employment: {employment}
            
            This application is being persistently misclassified. It should be {analysis['expected']} but is being {analysis['actual']}.
            
            Rule analysis:
            - Rules failed: {len(rules_failed)}
            - Rules passed: {len(rules_passed)}
            
            Please provide a detailed explanation of potential rule improvements that could fix this misclassification.
            """
            
            system_message = "You are a Credit Card Approval Expert that helps identify patterns and recommends rule improvements."
            
            # Get LLM analysis
            llm_analysis = llm_client.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,
                expert_name="Misclassification Analyzer"
            )
            
            # Add LLM analysis to the detailed analysis
            analysis["llm_analysis"] = llm_analysis
        
        detailed_analysis.append(analysis)
    
    # Save detailed analysis
    with open(os.path.join(RESULTS_DIR, "detailed_misclassification_analysis.json"), 'w') as f:
        json.dump(detailed_analysis, f, indent=2)
    
    return detailed_analysis

def generate_failure_reason(app, failed_rules, ruleset):
    """Generate an explanation of why the rules failed for this application"""
    if not failed_rules:
        return "All individual rules passed but overall logic failed"
    
    reasons = []
    top_level_logic = ruleset.get("logic", "all")
    
    for rule_eval in failed_rules:
        if rule_eval.get("type") == "nested":
            reasons.append(f"A nested rule group with {rule_eval.get('logic', 'unknown')} logic failed")
        elif "field" in rule_eval:
            field = rule_eval.get("field")
            app_value = rule_eval.get("application_value")
            
            if field.endswith("Tier"):
                reasons.append(f"The application's {field} of '{app_value}' didn't meet the rule criteria")
            elif field.endswith("paymentHistory"):
                reasons.append(f"The application's payment history of '{app_value}' didn't meet the rule criteria")
            else:
                threshold = rule_eval.get("threshold")
                condition = rule_eval.get("condition")
                if threshold is not None and condition:
                    reasons.append(f"The value {app_value} for {field} failed the condition {condition} {threshold}")
    
    if top_level_logic == "all" and reasons:
        return f"Application was rejected because ALL rules must pass, but: {'; '.join(reasons)}"
    elif top_level_logic == "any" and len(reasons) == len(failed_rules):
        return f"Application was rejected because at least one rule must pass, but none did: {'; '.join(reasons)}"
    
    return "; ".join(reasons)

def suggest_rule_improvement(app, expected_approval, similar_apps):
    """Suggest how rules could be improved to correctly classify this application"""
    if expected_approval:
        # Application should be approved but was declined
        if similar_apps:
            similar = similar_apps[0]
            return f"Consider adding a rule combination that approves applications with {app['creditHistory']['creditTier']} credit tier, " \
                  f"{app['financialInformation']['incomeTier']} income tier, and {app['financialInformation']['debtTier']} debt tier. " \
                  f"Similar application #{similar['id']} was correctly approved with {similar['credit_tier']} credit, " \
                  f"{similar['income_tier']} income, and {similar['debt_tier']} debt."
        else:
            return f"Create a special case rule to approve applications with exactly: " \
                  f"{app['creditHistory']['creditTier']} credit tier, {app['creditHistory']['paymentHistory']} payment history, " \
                  f"{app['financialInformation']['incomeTier']} income tier, and {app['financialInformation']['debtTier']} debt tier."
    else:
        # Application should be declined but was approved
        return f"Add restrictions to prevent approval of applications with combination of " \
              f"{app['creditHistory']['creditTier']} credit tier, {app['financialInformation']['incomeTier']} income tier, " \
              f"and {app['financialInformation']['debtTier']} debt tier." 