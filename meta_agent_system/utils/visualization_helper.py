import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from meta_agent_system.config.settings import RESULTS_DIR
from meta_agent_system.utils.logger import get_logger

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

def generate_accuracy_visualization():
    """Generate visualization of accuracy improvement over iterations"""
    # Load validation history
    history_file = os.path.join(RESULTS_DIR, "validation_history.json")
    if not os.path.exists(history_file):
        logger.error("Validation history file not found")
        return None
    
    try:
        with open(history_file, 'r') as f:
            validation_history = json.load(f)
        
        # Extract iteration numbers and accuracy values
        iterations = [entry.get("iteration", i) for i, entry in enumerate(validation_history)]
        accuracy_values = [entry.get("accuracy", 0) for entry in validation_history]
        
        # Create accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, accuracy_values, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Baseline (50%)')
        plt.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Perfect (100%)')
        
        plt.title('Accuracy Improvement Over Iterations', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set y-axis limits
        plt.ylim(0, 105)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"accuracy_improvement_{timestamp}.png")
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Generated accuracy visualization: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating accuracy visualization: {str(e)}")
        return None

def generate_feature_comparison(applications, hidden_approvals):
    """Generate visualizations comparing features between approved and declined applications"""
    if not applications or not hidden_approvals:
        logger.error("Missing data for feature comparison visualization")
        return None
    
    # Define key features to visualize
    numeric_features = [
        "creditHistory.creditScore",
        "financialInformation.annualIncome", 
        "financialInformation.existingDebt"
    ]
    
    categorical_features = [
        "creditHistory.paymentHistory",
        "financialInformation.employmentStatus"
    ]
    
    # Split applications into approved and declined
    approved_apps = []
    declined_apps = []
    
    for idx, app in enumerate(applications):
        app_id = idx + 1  # 1-indexed
        key = str(app_id)
        expected_approval = hidden_approvals.get(key, None)
        
        if expected_approval:
            approved_apps.append(app)
        else:
            declined_apps.append(app)
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(len(numeric_features) + len(categorical_features), 1, 
                           figsize=(10, 4 * (len(numeric_features) + len(categorical_features))))
    
    # If there's only one subplot, wrap it in a list
    if len(numeric_features) + len(categorical_features) == 1:
        axs = [axs]
    
    # Visualize numeric features
    for i, feature in enumerate(numeric_features):
        # Extract values
        approved_values = [get_nested_value(app, feature) for app in approved_apps]
        declined_values = [get_nested_value(app, feature) for app in declined_apps]
        
        # Remove None values
        approved_values = [v for v in approved_values if v is not None]
        declined_values = [v for v in declined_values if v is not None]
        
        # Create histogram
        ax = axs[i]
        ax.hist(approved_values, alpha=0.5, label='Approved', color='green')
        ax.hist(declined_values, alpha=0.5, label='Declined', color='red')
        
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature.split('.')[-1])
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Visualize categorical features
    for i, feature in enumerate(categorical_features):
        ax = axs[i + len(numeric_features)]
        
        # Extract values
        approved_values = [get_nested_value(app, feature) for app in approved_apps]
        declined_values = [get_nested_value(app, feature) for app in declined_apps]
        
        # Count occurrences of each value
        all_values = list(set(approved_values + declined_values))
        approved_counts = [approved_values.count(val) for val in all_values]
        declined_counts = [declined_values.count(val) for val in all_values]
        
        # Set up bar positions
        x = np.arange(len(all_values))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, approved_counts, width, label='Approved', color='green')
        ax.bar(x + width/2, declined_counts, width, label='Declined', color='red')
        
        ax.set_title(f'Distribution of {feature}')
        ax.set_xticks(x)
        ax.set_xticklabels(all_values)
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"feature_comparison_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Generated feature comparison visualization: {output_file}")
    return output_file

def generate_rule_evaluation_visual(rule_evaluations):
    """Generate visualization of rule outcomes across applications"""
    if not rule_evaluations:
        logger.error("Missing data for rule evaluation visualization")
        return None
    
    # Count applications
    n_applications = len(rule_evaluations)
    
    # Get the maximum number of rules from any evaluation
    max_rules = max(len(eval.get("rule_evaluations", [])) for eval in rule_evaluations)
    
    # Create a binary matrix of rule outcomes (rows=applications, columns=rules)
    outcome_matrix = np.zeros((n_applications, max_rules))
    
    # Fill matrix with 1 for passed rules, 0 for failed
    for i, eval in enumerate(rule_evaluations):
        app_rules = eval.get("rule_evaluations", [])
        for j, rule in enumerate(app_rules):
            outcome_matrix[i, j] = 1 if rule.get("passed", False) else 0
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(outcome_matrix, cmap='RdYlGn', aspect='auto')
    
    # Add color bar
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_ticklabels(['Failed', 'Passed'])
    
    # Add gridlines
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Add labels
    plt.title('Rule Evaluation Results by Application', fontsize=16)
    plt.xlabel('Rule Index', fontsize=14)
    plt.ylabel('Application ID', fontsize=14)
    
    # Add application correctness indication
    app_correct = ['Correct' if eval.get("correct", False) else 'Incorrect' for eval in rule_evaluations]
    app_expected = ['Approved' if eval.get("expected", False) else 'Declined' for eval in rule_evaluations]
    
    # Add labels to y-axis
    plt.yticks(range(n_applications), [f"{i+1}: {app_expected[i]} ({app_correct[i]})" 
                                      for i in range(n_applications)])
    
    # Add rule descriptions if available
    if rule_evaluations and rule_evaluations[0].get("rule_evaluations"):
        rule_descriptions = []
        for i in range(max_rules):
            if rule_evaluations[0].get("rule_evaluations") and i < len(rule_evaluations[0].get("rule_evaluations")):
                rule = rule_evaluations[0].get("rule_evaluations")[i]
                field = rule.get("field", "").split(".")[-1]
                condition = rule.get("condition", "")
                threshold = rule.get("threshold", "")
                values = rule.get("values", [])
                
                if threshold:
                    desc = f"{field} {condition} {threshold}"
                elif values:
                    desc = f"{field} {condition} {values}"
                else:
                    desc = f"{field} {condition}"
                
                rule_descriptions.append(desc)
            else:
                rule_descriptions.append(f"Rule {i+1}")
        
        plt.xticks(range(max_rules), rule_descriptions, rotation=45, ha='right')
    else:
        plt.xticks(range(max_rules), [f"Rule {i+1}" for i in range(max_rules)])
    
    # Save visualization
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"rule_evaluation_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Generated rule evaluation visualization: {output_file}")
    return output_file 