from typing import Dict, Any
import json
import os
from datetime import datetime
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, RESULTS_DIR

logger = get_logger(__name__)

def get_nested_value(obj, path):
    """Get a value from a nested object using a dot path."""
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

def evaluate_rule(rule, application):
    """Evaluate a single rule against an application."""
    # Handle nested rule groups
    if "rules" in rule and "logic" in rule:
        sub_results = [evaluate_rule(sub_rule, application) for sub_rule in rule["rules"]]
        if rule["logic"].lower() == "all":
            return all(sub_results)
        elif rule["logic"].lower() == "any":
            return any(sub_results)
        return False
    
    # Handle special rule types
    if rule.get("type") == "ratio":
        numerator = get_nested_value(application, rule.get("numerator_field", ""))
        denominator = get_nested_value(application, rule.get("denominator_field", ""))
        
        if numerator is None or denominator is None or denominator == 0:
            return False
        
        ratio = numerator / denominator
        threshold = rule.get("threshold", 0)
        condition = rule.get("condition", "less_than")
        
        if condition == "less_than":
            return ratio < threshold
        elif condition == "greater_than":
            return ratio > threshold
        return False
    
    if rule.get("type") == "range":
        field = rule.get("field", "")
        app_value = get_nested_value(application, field)
        min_val = rule.get("min", float('-inf'))
        max_val = rule.get("max", float('inf'))
        
        if app_value is None:
            return False
        
        return min_val <= app_value <= max_val
    
    # Handle standard conditions
    field = rule.get("field", "")
    condition = rule.get("condition", rule.get("operator", ""))
    threshold = rule.get("threshold", rule.get("value"))
    values = rule.get("values", [])
    
    # Support value as array
    if isinstance(rule.get("value"), list):
        values = rule.get("value", [])
    
    app_value = get_nested_value(application, field)
    
    if app_value is None:
        return False
    
    # Evaluate conditions
    if condition in ["in", "contains"] and isinstance(values, list):
        return app_value in values
    elif condition in ["equal", "equal_to", "equals", "=="]:
        return app_value == threshold
    elif condition in ["not_equal", "not_equal_to", "!="]:
        return app_value != threshold
    elif condition in ["greater_than", ">"]:
        return app_value > threshold
    elif condition in ["less_than", "<"]:
        return app_value < threshold
    elif condition in ["greater_than_or_equal", ">="]:
        return app_value >= threshold
    elif condition in ["less_than_or_equal", "<="]:
        return app_value <= threshold
    elif condition == "not_in":
        return app_value not in values
    
    logger.warning(f"Unrecognized rule format: {rule}")
    return False

def create_validator(llm_client: OpenAIClient) -> ExpertAgent:
    """Create a validation expert agent."""
    
    def validation_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ruleset against applications with clear diagnostics"""
        iteration = task.get("data", {}).get("iteration", 0)
        
        # Load data
        data = load_validation_data()
        applications = data["applications"]
        ruleset = data["ruleset"]
        hidden_approvals = data["hidden_approvals"]
        
        if not applications or not ruleset or not hidden_approvals:
            logger.error("Missing required data for validation")
            return {"status": "error", "message": "Missing required data"}
        
        # Log validation stats
        logger.info(f"===== VALIDATION DEBUGGING =====")
        logger.info(f"Ruleset logic: {ruleset.get('logic')}")
        logger.info(f"Rule count: {len(ruleset.get('rules', []))}")
        logger.info(f"Applications count: {len(applications)}")
        logger.info(f"Hidden approvals count: {len(hidden_approvals)}")
        
        # Evaluate each application
        results = evaluate_all_applications(applications, ruleset, hidden_approvals)
        
        # Calculate accuracy
        total_count = len(applications)
        correct_count = results["correct_count"]
        calculated_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        # Save diagnostics
        save_validation_results(ruleset, results, calculated_accuracy, iteration)
        
        # Log summary
        logger.info(f"Approved count: {results['approved_count']}, Declined count: {results['declined_count']}")
        logger.info(f"Correctly classified: {correct_count} out of {total_count}")
        logger.info(f"===== END DEBUGGING =====")
        
        # Identify edge cases
        edge_cases = identify_edge_cases(results["evaluations"], ruleset)
        logger.info(f"Identified {len(edge_cases)} edge cases")
        
        return {
            "status": "success" if calculated_accuracy == 100 else "partial_success",
            "accuracy": calculated_accuracy,
            "success": calculated_accuracy == 100,
            "message": f"Validated ruleset with {calculated_accuracy:.2f}% accuracy",
            "previous_accuracy": get_previous_accuracy()
        }
    
    def load_validation_data():
        """Load all validation data in one function."""
        data = {}
        
        # Load applications
        applications = []
        if os.path.exists(APPLICATIONS_DIR):
            app_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                        if f.startswith("application_") and f.endswith(".json")]
            
            app_files.sort(key=lambda f: int(f.replace("application_", "").replace(".json", "")))
            
            for f in app_files:
                file_path = os.path.join(APPLICATIONS_DIR, f)
                try:
                    with open(file_path, 'r') as file:
                        applications.append(json.load(file))
                except Exception as e:
                    logger.error(f"Error loading application from {file_path}: {str(e)}")
        
        data["applications"] = applications
        
        # Load ruleset
        ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
        if os.path.exists(ruleset_file):
            try:
                with open(ruleset_file, 'r') as f:
                    data["ruleset"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading ruleset: {str(e)}")
                data["ruleset"] = {}
        else:
            data["ruleset"] = {}
            
        # Load hidden approvals
        hidden_approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
        if os.path.exists(hidden_approvals_file):
            try:
                with open(hidden_approvals_file, 'r') as f:
                    data["hidden_approvals"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading hidden approvals: {str(e)}")
                data["hidden_approvals"] = {}
        else:
            data["hidden_approvals"] = {}
            
        return data
    
    def evaluate_all_applications(applications, ruleset, hidden_approvals):
        """Evaluate all applications against ruleset."""
        all_evaluations = []
        correct_count = 0
        approved_count = 0
        declined_count = 0
        misclassified_apps = []
        
        for idx, application in enumerate(applications):
            app_id = idx + 1  # 1-indexed
            key = str(app_id)
            expected_approval = hidden_approvals.get(key, None)
            
            if expected_approval is None:
                continue
            
            # Evaluate the ruleset for this application
            logic_type = ruleset.get("logic", "all").lower()
            rules = ruleset.get("rules", [])
            rule_results = []
            rule_evaluations = []
            
            for i, rule in enumerate(rules):
                rule_passed = evaluate_rule(rule, application)
                rule_results.append(rule_passed)
                
                # Record evaluation details
                rule_evaluations.append({
                    "rule_index": i,
                    "passed": rule_passed
                })
            
            # Apply ruleset logic
            approved = False
            if logic_type == "any":
                approved = any(rule_results)
            else:  # all logic
                approved = all(rule_results)
            
            # Update counters
            if approved:
                approved_count += 1
            else:
                declined_count += 1
            
            # Check correctness
            is_correct = (approved == expected_approval)
            if is_correct:
                correct_count += 1
            else:
                misclassified_apps.append({
                    "application_id": app_id,
                    "expected": expected_approval,
                    "actual": approved
                })
            
            # Record complete evaluation for diagnostics
            all_evaluations.append({
                "application_id": app_id,
                "expected": expected_approval,
                "actual": approved,
                "correct": is_correct,
                "rule_evaluations": rule_evaluations,
                "application_data": extract_app_data(application)
            })
        
        return {
            "evaluations": all_evaluations,
            "correct_count": correct_count,
            "approved_count": approved_count,
            "declined_count": declined_count,
            "misclassified": misclassified_apps
        }
    
    def extract_app_data(application):
        """Extract key application data for diagnostics."""
        return {
            "credit_score": get_nested_value(application, "creditHistory.creditScore"),
            "credit_tier": get_nested_value(application, "creditHistory.creditTier"),
            "payment_history": get_nested_value(application, "creditHistory.paymentHistory"),
            "annual_income": get_nested_value(application, "financialInformation.annualIncome"),
            "income_tier": get_nested_value(application, "financialInformation.incomeTier"),
            "existing_debt": get_nested_value(application, "financialInformation.existingDebt"),
            "debt_ratio": get_nested_value(application, "financialInformation.debtRatio"),
            "debt_tier": get_nested_value(application, "financialInformation.debtTier"),
            "employment_status": get_nested_value(application, "financialInformation.employmentStatus")
        }
    
    def save_validation_results(ruleset, results, accuracy, iteration):
        """Save all validation results to files."""
        # Save validation results
        validation_results = {
            "accuracy": accuracy,
            "results": [
                {
                    "application_id": eval.get("application_id"),
                    "decision": "approved" if eval.get("actual") else "declined",
                    "correct": eval.get("correct")
                } for eval in results["evaluations"]
            ],
            "misclassified_applications": results["misclassified"]
        }
        
        validation_file = os.path.join(RESULTS_DIR, "validation_results.json")
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Save detailed diagnostics
        diagnostics_file = os.path.join(RESULTS_DIR, "validation_diagnostics.json")
        with open(diagnostics_file, 'w') as f:
            json.dump({
                "ruleset": ruleset,
                "rule_evaluations": results["evaluations"]
            }, f, indent=2)
        
        # Update validation history
        update_validation_history(accuracy, len(ruleset.get("rules", [])), iteration)
        
        # Update persistent misclassifications
        update_persistent_misclassifications(results["evaluations"], iteration)
    
    def update_validation_history(accuracy, rule_count, iteration):
        """Update the validation history file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(RESULTS_DIR, "validation_history.json")
        validation_history = []
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    validation_history = json.load(f)
            except Exception:
                pass
        
        validation_history.append({
            "timestamp": timestamp,
            "iteration": iteration,
            "accuracy": accuracy,
            "rule_count": rule_count
        })
        
        with open(history_file, 'w') as f:
            json.dump(validation_history, f, indent=2)
    
    def update_persistent_misclassifications(evaluations, iteration):
        """Track persistently misclassified applications."""
        file_path = os.path.join(RESULTS_DIR, "persistent_misclassifications.json")
        persistent = {}
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    persistent = json.load(f)
            except Exception:
                pass
        
        for eval in evaluations:
            app_id = str(eval.get("application_id"))
            is_correct = eval.get("correct", False)
            
            if not is_correct:
                if app_id not in persistent:
                    persistent[app_id] = {
                        "application_id": app_id,
                        "expected": eval.get("expected"),
                        "misclassification_count": 1,
                        "iterations": [iteration]
                    }
                else:
                    persistent[app_id]["misclassification_count"] += 1
                    persistent[app_id]["iterations"].append(iteration)
        
        with open(file_path, 'w') as f:
            json.dump(persistent, f, indent=2)
    
    def identify_edge_cases(evaluations, ruleset):
        """Identify applications that are edge cases."""
        edge_cases = []
        
        for eval in evaluations:
            rule_evals = eval.get("rule_evaluations", [])
            app_id = eval.get("application_id")
            
            # For "any" logic, if exactly one rule passed, it's an edge case
            if ruleset.get("logic") == "any":
                passed_count = sum(1 for r in rule_evals if r.get("passed", False))
                if passed_count == 1:
                    edge_cases.append({
                        "application_id": app_id,
                        "type": "edge_case_single_rule_pass"
                    })
            
            # For "all" logic, if exactly one rule failed, it's an edge case
            elif ruleset.get("logic") == "all":
                failed_count = sum(1 for r in rule_evals if not r.get("passed", False))
                if failed_count == 1:
                    edge_cases.append({
                        "application_id": app_id,
                        "type": "edge_case_single_rule_fail"
                    })
        
        edge_case_file = os.path.join(RESULTS_DIR, "edge_cases.json")
        with open(edge_case_file, 'w') as f:
            json.dump(edge_cases, f, indent=2)
        
        return edge_cases
    
    def get_previous_accuracy():
        """Get the accuracy from previous iteration."""
        history_file = os.path.join(RESULTS_DIR, "validation_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    if history and len(history) > 1:
                        return history[-2].get("accuracy", 0)
            except Exception:
                pass
        return 0
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Validator",
        capabilities=["validation"],
        behavior=validation_behavior,
        description="Validates results against criteria and tests ruleset accuracy"
    )
