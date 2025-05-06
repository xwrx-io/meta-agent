import os
import sys
import json
import time
from dotenv import load_dotenv
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.utils.helpers import save_json, get_timestamp, ensure_directory_exists
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR
from meta_agent_system.experts.validator import create_validator
from meta_agent_system.experts.rule_analyzer import create_rule_analyzer
from meta_agent_system.experts.rule_refiner import create_rule_refiner
from meta_agent_system.utils.visualization_helper import generate_accuracy_visualization
import matplotlib.pyplot as plt
from datetime import datetime
from meta_agent_system.experts.misclassification_analyzer import analyze_misclassifications
from meta_agent_system.core.summary_generator import generate_summary

# Configure logging
logger = get_logger(__name__)

def main():
    """Credit card rule discovery system main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    print("\n=== Credit Card Rule Discovery System ===\n")
    print("Initializing system...")
    
    # Initialize OpenAI client
    openai_client = OpenAIClient()
    
    # Create expert components
    validator = create_validator(openai_client)
    rule_analyzer = create_rule_analyzer(openai_client)
    rule_refiner = create_rule_refiner(openai_client)
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initial ruleset
    initial_ruleset = {
        "logic": "any",
        "rules": [
            {
                "field": "creditHistory.creditTier",
                "condition": "equals",
                "threshold": "Excellent"
            },
            {
                "field": "financialInformation.incomeTier",
                "condition": "equals",
                "threshold": "Very High"
            },
            {
                "field": "financialInformation.debtTier",
                "condition": "equals",
                "threshold": "Very Low"
            }
        ],
        "description": "Initial ruleset using basic tier factors",
        "timestamp": int(time.time()),
        "iteration": 0
    }
    
    # Save initial ruleset
    ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
    with open(ruleset_file, 'w') as f:
        json.dump(initial_ruleset, f, indent=2)
    
    # Track progress
    current_accuracy = 0
    best_accuracy = 0
    best_ruleset = initial_ruleset
    best_iteration = 0
    max_iterations = 10
    iteration = 0
    
    print("\nStarting rule discovery process...\n")
    
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
            
            # Save best ruleset
            with open(ruleset_file, 'r') as f:
                best_ruleset = json.load(f)
            
            best_ruleset_file = os.path.join(RESULTS_DIR, f"best_ruleset_iteration_{iteration}.json")
            with open(best_ruleset_file, 'w') as f:
                json.dump(best_ruleset, f, indent=2)
                
            print(f"New best accuracy: {best_accuracy:.2f}%")
        
        # If we've reached 100% accuracy, break the loop
        if current_accuracy == 100:
            print("\nPerfect accuracy achieved!")
            break
        
        # Step 2: Analyze patterns
        print("Analyzing application patterns...")
        rule_analyzer.execute({
            "description": "Analyze credit card applications for patterns",
            "data": {"iteration": iteration}
        })
        
        # Step 3: Refine rules
        print("Refining ruleset...")
        refinement_result = rule_refiner.execute({
            "description": "Refine credit card approval rules",
            "data": {"iteration": iteration}
        })
        
        # Report on new ruleset
        ruleset = refinement_result.get("ruleset", {})
        nested_rule_count = sum(1 for rule in ruleset.get("rules", []) 
                               if isinstance(rule, dict) and "rules" in rule)
        
        print(f"Rules refined. New ruleset has {len(ruleset.get('rules', []))} rules " +
              f"with '{ruleset.get('logic', 'all')}' logic and {nested_rule_count} nested rule groups.")
    
    # Summarize results
    print("\n=== Rule Discovery Complete ===")
    print(f"Iterations completed: {iteration}")
    print(f"Final accuracy: {current_accuracy:.2f}%")
    print(f"Best accuracy: {best_accuracy:.2f}% (iteration {best_iteration})")
    
    # Generate accuracy visualization
    viz_file = generate_accuracy_visualization()
    if viz_file:
        print(f"Accuracy visualization saved to: {viz_file}")
    
    # Use best ruleset if better than final
    if best_accuracy > current_accuracy:
        with open(ruleset_file, 'w') as f:
            json.dump(best_ruleset, f, indent=2)
        print(f"Restored best ruleset from iteration {best_iteration}")
    
    final_ruleset = best_ruleset if best_accuracy > current_accuracy else ruleset
    
    # Generate comprehensive summary report with ASCII chart
    print("\nGenerating comprehensive summary report...")
    summary_file = generate_summary(
        openai_client,
        best_accuracy, 
        best_iteration, 
        current_accuracy, 
        iteration, 
        final_ruleset
    )
    print(f"\nDetailed summary report saved to: {summary_file}")
    
    print("\nRule discovery process complete!")

def print_rules(rules, indent=0):
    """Print rules in a readable format with indentation for nested rules"""
    for rule in rules:
        prefix = "  " * indent
        
        if "rules" in rule:
            # Nested rule group
            print(f"{prefix}Rule Group ({rule.get('logic', 'all').upper()}):")
            print_rules(rule.get("rules", []), indent + 1)
        elif "field" in rule:
            # Standard rule
            field = rule.get("field", "").split(".")[-1]  # Just the field name
            condition = rule.get("condition", "")
            
            if "threshold" in rule:
                value = rule.get("threshold")
                print(f"{prefix}• {field} {condition} {value}")
            elif "values" in rule:
                values = rule.get("values", [])
                print(f"{prefix}• {field} {condition} {values}")

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

if __name__ == "__main__":
    main()
