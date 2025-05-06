import os
import sys
import json
from dotenv import load_dotenv
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.utils.helpers import save_json, get_timestamp, ensure_directory_exists
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR
from meta_agent_system.experts.validator import create_validator
from meta_agent_system.experts.rule_analyzer import create_rule_analyzer
from meta_agent_system.experts.rule_refiner import create_rule_refiner
from meta_agent_system.utils.visualization_helper import generate_accuracy_visualization
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
from meta_agent_system.experts.misclassification_analyzer import analyze_misclassifications

# Configure logging
logger = get_logger(__name__)

def main():
    """Main entry point for credit card rule discovery system"""
    # Load environment variables
    load_dotenv()
    
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    print("\n=== Credit Card Rule Discovery System ===\n")
    print("Initializing system...")
    
    # Initialize OpenAI client
    openai_client = OpenAIClient()
    
    # Create our core experts
    validator = create_validator(openai_client)
    rule_analyzer = create_rule_analyzer(openai_client)
    rule_refiner = create_rule_refiner(openai_client)
    
    # Ensure results directory exists
    ensure_directory_exists(RESULTS_DIR)
    
    # Define maximum iterations
    max_iterations = 10
    
    print("\nStarting rule discovery process...\n")
    
    # Initial ruleset - starting point
    initial_ruleset = {
        "logic": "all",
        "rules": [
            {
                "field": "creditHistory.creditScore",
                "condition": "greater_than",
                "threshold": 650
            },
            {
                "field": "financialInformation.annualIncome",
                "condition": "greater_than",
                "threshold": 30000
            },
            {
                "field": "creditHistory.paymentHistory",
                "condition": "in",
                "values": ["Excellent", "Good"]
            }
        ],
        "description": "Initial ruleset based on common credit card approval factors",
        "timestamp": int(time.time()),
        "iteration": 0
    }
    
    # Save initial ruleset
    ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
    with open(ruleset_file, 'w') as f:
        json.dump(initial_ruleset, f, indent=2)
    
    # Initialize variables to track progress
    current_accuracy = 0
    best_accuracy = 0
    best_ruleset = initial_ruleset
    best_iteration = 0
    consecutive_decreases = 0
    iteration = 0
    
    # Main iteration loop
    while current_accuracy < 100 and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Step 1: Validate current ruleset
        print("Validating current ruleset...")
        validation_result = validator.execute({
            "description": "Validate credit card approval rules",
            "data": {"iteration": iteration}
        })
        
        current_accuracy = validation_result.get("accuracy", 0)
        print(f"Current accuracy: {current_accuracy:.2f}%")
        
        # Update best accuracy if improved
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_iteration = iteration
            best_ruleset = None  # Will load from file
            
            # Save a backup of this best ruleset
            try:
                with open(ruleset_file, 'r') as f:
                    best_ruleset = json.load(f)
                
                best_ruleset_file = os.path.join(RESULTS_DIR, f"best_ruleset_iteration_{iteration}.json")
                with open(best_ruleset_file, 'w') as f:
                    json.dump(best_ruleset, f, indent=2)
                    
                print(f"New best accuracy: {best_accuracy:.2f}%")
            except Exception as e:
                logger.error(f"Error saving best ruleset: {str(e)}")
        else:
            consecutive_decreases += 1
        
        # New step: Perform detailed misclassification analysis
        print("Analyzing misclassifications in depth...")
        detailed_analysis = analyze_misclassifications()
        print(f"Found {len(detailed_analysis)} misclassified applications to focus on")
        
        # If we've reached 100% accuracy, break the loop
        if current_accuracy == 100:
            print("\nPerfect accuracy achieved! Ruleset correctly classifies all applications.")
            break
        
        # Step 2: Analyze applications to find patterns
        print("Analyzing application patterns...")
        analysis_result = rule_analyzer.execute({
            "description": "Analyze credit card applications for patterns",
            "data": {"iteration": iteration}
        })
        
        # Step 3: Refine rules based on validation and analysis
        print("Refining ruleset...")
        refinement_result = rule_refiner.execute({
            "description": "Refine credit card approval rules",
            "data": {"iteration": iteration, "consecutive_decreases": consecutive_decreases}
        })
        
        # Print refinement summary
        ruleset = refinement_result.get("ruleset", {})
        
        # Check if the ruleset has nested rules (better logging)
        nested_rule_count = 0
        for rule in ruleset.get("rules", []):
            if "rules" in rule and isinstance(rule.get("rules"), list):
                nested_rule_count += 1
        
        # Better reporting of rule complexity
        if nested_rule_count > 0:
            print(f"Rules refined. New ruleset has {len(ruleset.get('rules', []))} rules " +
                  f"with '{ruleset.get('logic', 'all')}' logic and {nested_rule_count} nested rule groups.")
        else:
            print(f"Rules refined. New ruleset has {len(ruleset.get('rules', []))} rules " +
                  f"with '{ruleset.get('logic', 'all')}' logic.")
        
        # Reset consecutive decreases counter if we saw an improvement
        if current_accuracy > validation_result.get("previous_accuracy", 0):
            consecutive_decreases = 0
        
        # Optional: Add a small pause between iterations
        time.sleep(1)
    
    # Summarize results
    print("\n=== Rule Discovery Summary ===")
    print(f"Iterations completed: {iteration}")
    print(f"Final accuracy: {current_accuracy:.2f}%")
    print(f"Best accuracy: {best_accuracy:.2f}% (iteration {best_iteration})")
    
    # Generate visualization
    viz_file = generate_accuracy_visualization()
    if viz_file:
        print(f"Accuracy visualization saved to: {viz_file}")
    
    # Load final ruleset - use best ruleset if available and better than final
    final_ruleset = {}
    if best_ruleset and best_accuracy > current_accuracy:
        final_ruleset = best_ruleset
        # Also save it as the final ruleset
        with open(ruleset_file, 'w') as f:
            json.dump(best_ruleset, f, indent=2)
        print(f"Restored best ruleset from iteration {best_iteration} (accuracy: {best_accuracy:.2f}%)")
    else:
        try:
            with open(ruleset_file, 'r') as f:
                final_ruleset = json.load(f)
        except Exception as e:
            logger.error(f"Error loading final ruleset: {str(e)}")
    
    # Print final ruleset
    print("\nFinal Credit Card Approval Rules:")
    print(f"Logic: {final_ruleset.get('logic', 'all').upper()}")
    print(f"Rule count: {len(final_ruleset.get('rules', []))}")
    
    for i, rule in enumerate(final_ruleset.get("rules", [])):
        print_rule(rule)
    
    # Print ruleset description
    print("\nRuleset rationale:")
    print(final_ruleset.get("description", "No description provided"))
    
    print("\nRule discovery process complete!")

def explore_applications():
    """Utility function to explore the application data"""
    if not os.path.exists(APPLICATIONS_DIR):
        print("Applications directory not found.")
        return
    
    application_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                        if f.startswith("application_") and f.endswith(".json")]
    
    print(f"Found {len(application_files)} applications.")
    
    # Load hidden approvals
    hidden_approvals = {}
    hidden_approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
    if os.path.exists(hidden_approvals_file):
        try:
            with open(hidden_approvals_file, 'r') as f:
                hidden_approvals = json.load(f)
        except Exception as e:
            print(f"Error loading hidden approvals: {str(e)}")
    
    # Count approved and declined
    approved_count = sum(1 for value in hidden_approvals.values() if value)
    declined_count = sum(1 for value in hidden_approvals.values() if not value)
    
    print(f"Applications: {approved_count} approved, {declined_count} declined")
    
    # Print first application as example
    if application_files:
        sample_file = os.path.join(APPLICATIONS_DIR, application_files[0])
        try:
            with open(sample_file, 'r') as f:
                sample_app = json.load(f)
            
            print("\nSample application structure:")
            print(json.dumps(sample_app, indent=2))
        except Exception as e:
            print(f"Error loading sample application: {str(e)}")

def print_rule(rule, indent=0):
    """Print a rule with proper indentation for nested rules"""
    indent_str = "  " * indent
    
    # Handle nested rules
    if "rules" in rule and "logic" in rule:
        print(f"{indent_str}Rule Group ({rule['logic'].upper()}):")
        for sub_rule in rule["rules"]:
            print_rule(sub_rule, indent + 1)
        return
    
    # Handle ratio rules
    if rule.get("type") == "ratio":
        print(f"{indent_str}Rule: {rule.get('numerator_field')} to {rule.get('denominator_field')} ratio {rule.get('condition')} {rule.get('threshold')}")
        return
    
    # Handle range rules
    if rule.get("type") == "range":
        print(f"{indent_str}Rule: {rule.get('field')} between {rule.get('min')} and {rule.get('max')}")
        return
    
    # Handle standard rules
    field = rule.get("field", "")
    condition = rule.get("condition", "")
    threshold = rule.get("threshold")
    values = rule.get("values", [])
    
    rule_str = f"{indent_str}Rule: {field} {condition}"
    if threshold is not None:
        rule_str += f" {threshold}"
    elif values:
        rule_str += f" {values}"
    
    print(rule_str)

if __name__ == "__main__":
    main()
