import os
import sys
import json
import time
import argparse
from dotenv import load_dotenv
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.utils.helpers import save_json, get_timestamp, ensure_directory_exists
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR
from meta_agent_system.experts.validator import create_validator
from meta_agent_system.experts.rule_analyzer import create_rule_analyzer
from meta_agent_system.experts.rule_refiner import create_rule_refiner
from meta_agent_system.utils.visualization_helper import generate_accuracy_visualization
from meta_agent_system.experts.expertise_recommender import create_expertise_recommender
import matplotlib.pyplot as plt
from datetime import datetime
from meta_agent_system.experts.misclassification_analyzer import analyze_misclassifications
from meta_agent_system.core.summary_generator import generate_summary
import re
import textwrap
from colorama import Fore, Back, Style, init
from meta_agent_system.core.expert_manager import ExpertManager

# Initialize colorama
init()

# Configure logging
logger = get_logger(__name__)

def get_initial_ruleset_from_scratch(openai_client):
    """Generate a minimal ruleset from scratch with minimal accuracy"""
    print("Generating minimal ruleset from scratch...")
    
    # Use "any" logic with a non-existent field
    # This should result in all applications being declined (since the rule will fail)
    # If most applications should be approved, this will start with low accuracy
    return {
        "logic": "any",  # Changed from "all" to "any"
        "rules": [
            {
                "field": "customerInfo.nonExistentField",
                "condition": "equals",
                "threshold": "SomeValue"
            }
        ],
        "description": "Initial minimal ruleset with no prior knowledge",
        "timestamp": int(time.time()),
        "iteration": 0
    }

def get_default_initial_ruleset():
    """Return the default hardcoded initial ruleset"""
    return {
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

def main():
    """Credit card rule discovery system main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Credit Card Rule Discovery System")
    parser.add_argument('--from-scratch', action='store_true', 
                        help='Start with a ruleset generated from scratch instead of using the default')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum number of iterations to run (default: 10)')
    args = parser.parse_args()
    
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
    expertise_recommender = create_expertise_recommender(openai_client)
    
    # Create expert manager for dynamic experts
    expert_manager = ExpertManager(openai_client)
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initial ruleset (default or from scratch)
    if args.from_scratch:
        initial_ruleset = get_initial_ruleset_from_scratch(openai_client)
        print("Starting with LLM-generated ruleset")
    else:
        initial_ruleset = get_default_initial_ruleset()
        print("Starting with default ruleset")
    
    # Save initial ruleset
    ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
    with open(ruleset_file, 'w') as f:
        json.dump(initial_ruleset, f, indent=2)
    
    # Track progress
    current_accuracy = 0
    best_accuracy = 0
    best_ruleset = initial_ruleset
    best_iteration = 0
    max_iterations = args.max_iterations
    iteration = 0
    
    print(f"\nStarting rule discovery process (max {max_iterations} iterations)...\n")
    
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
        
        # Run expertise recommender ONLY after first iteration
        if iteration == 1:
            print("\n=== Expertise Recommendations ===")
            print("Analyzing expertise needs based on first iteration results...")
            
            # Call expertise recommender with validation results
            expertise_result = expertise_recommender.execute({
                "description": "Identify needed expertise based on validation results",
                "data": {
                    "validation_result": validation_result,
                    "iteration": iteration,
                    "current_accuracy": current_accuracy
                }
            })
            
            # Print recommendations to terminal
            if expertise_result.get("status") == "success":
                recommendations = expertise_result.get("recommendations", [])
                recommendations_file = expertise_result.get("recommendations_file", "")
                
                print(f"\n{Fore.CYAN}Identified {len(recommendations)} potential expert types that could help:{Style.RESET_ALL}\n")
                print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
                
                for i, expert in enumerate(recommendations, 1):
                    expert_name = expert.get('name', 'Unnamed Expert')
                    capabilities = expert.get('capabilities', [])
                    system_prompt = expert.get('system_prompt', 'No system prompt provided')
                    
                    # Format with clear visual separation and structure
                    print(f"\n{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}EXPERT {i}: {expert_name}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
                    
                    # Format capabilities as a bulleted list
                    print(f"\n{Fore.GREEN}CAPABILITIES:{Style.RESET_ALL}")
                    for capability in capabilities:
                        print(f"{Fore.WHITE}  • {capability}{Style.RESET_ALL}")
                    
                    # Format system prompt with proper indentation and JSON formatting
                    print(f"\n{Fore.GREEN}SYSTEM PROMPT:{Style.RESET_ALL}")
                    
                    # Improved JSON extraction and formatting
                    # First, extract all text before any JSON
                    json_start = system_prompt.find('{"')
                    if json_start == -1:
                        json_start = system_prompt.find('REQUIRED JSON')
                        if json_start != -1:
                            # Find the actual JSON after the REQUIRED JSON text
                            json_text = system_prompt[json_start:]
                            # Look for opening brace
                            brace_start = json_text.find('{')
                            if brace_start != -1:
                                prefix = system_prompt[:json_start + brace_start]
                                json_text = json_text[brace_start:]
                                
                                # Find matching closing brace
                                brace_count = 0
                                for i, char in enumerate(json_text):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_str = json_text[:i+1]
                                            suffix = json_text[i+1:]
                                            break
                                
                                # Format the parts
                                if prefix.strip():
                                    wrapped_prefix = textwrap.fill(prefix.strip(), width=76, 
                                                          initial_indent="  ", subsequent_indent="  ")
                                    print(f"{Fore.WHITE}{wrapped_prefix}{Style.RESET_ALL}")
                                
                                # Try to parse and pretty-print the JSON
                                try:
                                    # Replace smart quotes with straight quotes
                                    json_str = json_str.replace('"', '"').replace('"', '"')
                                    parsed_json = json.loads(json_str)
                                    pretty_json = json.dumps(parsed_json, indent=4)
                                    
                                    # Print the JSON with syntax highlighting
                                    print(f"\n{Fore.MAGENTA}  RESPONSE FORMAT:{Style.RESET_ALL}")
                                    for line in pretty_json.split('\n'):
                                        indented_line = "    " + line
                                        # Highlight keys in cyan
                                        highlighted = re.sub(r'(".*?"):', f"{Fore.CYAN}\\1{Fore.RESET}:", indented_line)
                                        # Highlight values in white
                                        highlighted = re.sub(r': (".*?")(,?)', f": {Fore.WHITE}\\1{Fore.RESET}\\2", highlighted)
                                        print(highlighted)
                                    print()
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, print as is
                                    print(f"{Fore.WHITE}  {json_str}{Style.RESET_ALL}")
                                
                                if suffix.strip():
                                    wrapped_suffix = textwrap.fill(suffix.strip(), width=76, 
                                                          initial_indent="  ", subsequent_indent="  ")
                                    print(f"{Fore.WHITE}{wrapped_suffix}{Style.RESET_ALL}")
                            else:
                                # If can't find JSON structure, print normally
                                wrapped = textwrap.fill(system_prompt, width=76, 
                                               initial_indent="  ", subsequent_indent="  ")
                                print(f"{Fore.WHITE}{wrapped}{Style.RESET_ALL}")
                        else:
                            # If no JSON mentioned, print normally
                            wrapped = textwrap.fill(system_prompt, width=76, 
                                           initial_indent="  ", subsequent_indent="  ")
                            print(f"{Fore.WHITE}{wrapped}{Style.RESET_ALL}")
                    else:
                        # There is JSON at the beginning
                        prefix = system_prompt[:json_start]
                        json_text = system_prompt[json_start:]
                        
                        # Format as above
                        if prefix.strip():
                            wrapped_prefix = textwrap.fill(prefix.strip(), width=76, 
                                               initial_indent="  ", subsequent_indent="  ")
                            print(f"{Fore.WHITE}{wrapped_prefix}{Style.RESET_ALL}")
                        
                        # Find end of JSON
                        brace_count = 0
                        for i, char in enumerate(json_text):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = json_text[:i+1]
                                    suffix = json_text[i+1:]
                                    break
                        
                        # Try to parse and pretty-print the JSON
                        try:
                            parsed_json = json.loads(json_str)
                            pretty_json = json.dumps(parsed_json, indent=4)
                            
                            # Print with highlighting
                            print(f"\n{Fore.MAGENTA}  RESPONSE FORMAT:{Style.RESET_ALL}")
                            for line in pretty_json.split('\n'):
                                indented_line = "    " + line
                                # Highlight keys in cyan
                                highlighted = re.sub(r'(".*?"):', f"{Fore.CYAN}\\1{Fore.RESET}:", indented_line)
                                # Highlight values in white
                                highlighted = re.sub(r': (".*?")(,?)', f": {Fore.WHITE}\\1{Fore.RESET}\\2", highlighted)
                                print(highlighted)
                            print()
                        except json.JSONDecodeError:
                            # If JSON parsing fails, print as is
                            print(f"{Fore.WHITE}  {json_str}{Style.RESET_ALL}")
                        
                        if suffix.strip():
                            wrapped_suffix = textwrap.fill(suffix.strip(), width=76, 
                                               initial_indent="  ", subsequent_indent="  ")
                            print(f"{Fore.WHITE}{wrapped_suffix}{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
                print(f"\n{Fore.GREEN}Detailed expertise recommendations saved to:{Style.RESET_ALL} {Fore.WHITE}{recommendations_file}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
                
                # Create dynamic experts based on recommendations
                print(f"\n{Fore.CYAN}Creating specialized experts based on recommendations...{Style.RESET_ALL}")
                dynamic_experts = expert_manager.create_experts_from_recommendations(recommendations)
                print(f"{Fore.GREEN}Created {len(dynamic_experts)} specialized experts to assist with rule discovery{Style.RESET_ALL}")
                
                print("\nContinuing with rule discovery process...\n")
        
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
        
        # Step 2.5: Gather expert insights
        expert_insights = []
        if iteration > 1:  # Only use dynamic experts after they've been created
            print("Gathering specialized insights from domain experts...")
            current_ruleset = {}
            with open(ruleset_file, 'r') as f:
                current_ruleset = json.load(f)
            
            expert_insights = expert_manager.gather_expert_insights(
                iteration=iteration,
                current_ruleset=current_ruleset,
                validation_result=validation_result
            )
            
            if expert_insights:
                print(f"{Fore.GREEN}Received insights from {len(expert_insights)} specialized experts{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}No expert insights available for this iteration{Style.RESET_ALL}")
        
        # Step 3: Refine rules with expert insights
        print("Refining ruleset...")
        refinement_result = rule_refiner.execute({
            "description": "Refine credit card approval rules",
            "data": {
                "iteration": iteration,
                "expert_insights": expert_insights
            }
        })
        
        # Report on new ruleset
        ruleset = refinement_result.get("ruleset", {})
        nested_rule_count = sum(1 for rule in ruleset.get("rules", []) 
                               if isinstance(rule, dict) and "rules" in rule)
        
        print(f"Rules refined. New ruleset has {len(ruleset.get('rules', []))} rules " +
              f"with '{ruleset.get('logic', 'all')}' logic and {nested_rule_count} nested rule groups.")
        
        # After updating best_accuracy, record expert contributions:
        if current_accuracy > best_accuracy:
            # Existing code to update best_accuracy, best_iteration, etc.
            
            # Record expert contributions if any
            if expert_insights:
                expert_manager.record_expert_contribution(
                    iteration=iteration,
                    current_accuracy=current_accuracy,
                    previous_accuracy=best_accuracy,  # Previous best
                    expert_insights=expert_insights
                )
            
            best_accuracy = current_accuracy
            best_iteration = iteration
            # Rest of existing code
        
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
        final_ruleset,
        expert_manager  # Pass the expert manager
    )
    print(f"\nDetailed summary report saved to: {summary_file}")
    
    # At the end of the program, after generating the summary report:
    if expert_manager.dynamic_experts:
        print("\n")
        print(expert_manager.get_experts_summary())
    
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
