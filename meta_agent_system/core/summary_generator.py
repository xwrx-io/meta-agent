import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate
from colorama import Fore, Back, Style, init
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR

# Initialize colorama
init()

logger = get_logger(__name__)

def generate_summary(openai_client, best_accuracy, best_iteration, final_accuracy, iterations_completed, final_ruleset):
    """
    Generate a comprehensive summary of the credit card rule discovery process
    
    Args:
        openai_client: OpenAI client for generating explanations
        best_accuracy: Best accuracy achieved
        best_iteration: Iteration where best accuracy was achieved
        final_accuracy: Final accuracy after all iterations
        iterations_completed: Total number of iterations run
        final_ruleset: The final ruleset used
        
    Returns:
        None, but prints summary to console and saves to file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(RESULTS_DIR, f"run_summary_{timestamp}.txt")
    
    # Create summary sections
    stats_summary = generate_statistics_summary(best_accuracy, best_iteration, final_accuracy, iterations_completed)
    ruleset_summary = generate_ruleset_summary(final_ruleset)
    applications_summary = generate_applications_summary(openai_client, final_ruleset)
    improvement_graph = generate_accuracy_visualization()
    
    # Combine all sections for saving to file (without colors)
    file_summary = f"""
{'='*80}
                CREDIT CARD APPROVAL RULE DISCOVERY - SUMMARY REPORT
{'='*80}

{stats_summary}

{ruleset_summary}

{applications_summary}

Accuracy Improvement Visualization: {improvement_graph}

{'='*80}
                               END OF REPORT
{'='*80}
"""
    
    # Create a colorized version for terminal display
    terminal_header = f"""
{Fore.CYAN}{'='*80}{Style.RESET_ALL}
{Fore.YELLOW}                CREDIT CARD APPROVAL RULE DISCOVERY - SUMMARY REPORT{Style.RESET_ALL}
{Fore.CYAN}{'='*80}{Style.RESET_ALL}
"""
    
    terminal_footer = f"""
{Fore.CYAN}{'='*80}{Style.RESET_ALL}
{Fore.YELLOW}                               END OF REPORT{Style.RESET_ALL}
{Fore.CYAN}{'='*80}{Style.RESET_ALL}
"""
    
    # Print colorized summary to terminal
    print(terminal_header)
    
    # Print each section with colors already included
    print(stats_summary)
    print(ruleset_summary)
    print(applications_summary)
    
    # Print the visualization info
    print(f"{Fore.GREEN}Accuracy Improvement Visualization:{Style.RESET_ALL} {Fore.WHITE}{improvement_graph}{Style.RESET_ALL}")
    
    print(terminal_footer)
    
    # Save the plain-text version to file
    with open(summary_file, 'w') as f:
        f.write(file_summary)
    
    logger.info(f"Generated comprehensive summary: {summary_file}")
    return summary_file

def generate_statistics_summary(best_accuracy, best_iteration, final_accuracy, iterations_completed):
    """Generate key statistics summary with ASCII chart"""
    # Load validation history for the ASCII chart
    try:
        with open(os.path.join(RESULTS_DIR, "validation_history.json"), 'r') as f:
            history = json.load(f)
        
        # Try the complex chart first, if it fails, fall back to the simple one
        try:
            ascii_chart = generate_colored_ascii_chart(history)
        except Exception as e:
            logger.warning(f"Complex ASCII chart failed: {str(e)}, falling back to simple chart")
            ascii_chart = generate_colored_simple_ascii_chart(history)
            
    except Exception as e:
        logger.error(f"Error generating ASCII chart: {str(e)}")
        ascii_chart = f"{Fore.RED}ASCII chart generation failed{Style.RESET_ALL}"
    
    # Get the colorized validation history table
    validation_table = get_colored_validation_history_table()
    
    return f"""
{Fore.GREEN}PERFORMANCE STATISTICS{Style.RESET_ALL}
{Fore.CYAN}{'-'*21}{Style.RESET_ALL}
{Fore.WHITE}Total Iterations: {Fore.YELLOW}{iterations_completed}{Style.RESET_ALL}
{Fore.WHITE}Final Accuracy: {Fore.YELLOW}{final_accuracy:.2f}%{Style.RESET_ALL}
{Fore.WHITE}Best Accuracy: {Fore.YELLOW}{best_accuracy:.2f}%{Style.RESET_ALL} (iteration {Fore.YELLOW}{best_iteration}{Style.RESET_ALL})
{Fore.WHITE}Learning Rate: {Fore.YELLOW}{(best_accuracy / best_iteration):.2f}%{Style.RESET_ALL} per iteration

{Fore.GREEN}Validation History:{Style.RESET_ALL}
{validation_table}

{Fore.GREEN}Accuracy Improvement Chart:{Style.RESET_ALL}
{ascii_chart}
"""

def get_colored_validation_history_table():
    """Get validation history as a formatted table with colors"""
    try:
        with open(os.path.join(RESULTS_DIR, "validation_history.json"), 'r') as f:
            history = json.load(f)
            
        if not history:
            return f"{Fore.RED}No validation history available{Style.RESET_ALL}"
        
        # Create a table with iteration, accuracy, and rule count
        table_data = []
        for entry in history:
            iteration = entry.get("iteration", "?")
            accuracy = entry.get("accuracy", 0)
            rule_count = entry.get("rule_count", 0)
            
            # Add data without colors for tabulate to handle alignment properly
            table_data.append([iteration, f"{accuracy:.2f}%", rule_count])
        
        # Get the table as text with proper alignment
        table = tabulate(table_data, headers=["Iteration", "Accuracy", "Rule Count"], tablefmt="simple")
        
        # Now colorize the output line by line
        colored_lines = []
        lines = table.split('\n')
        
        # Color the header row (first line)
        if len(lines) > 0:
            colored_lines.append(f"{Fore.CYAN}{lines[0]}{Style.RESET_ALL}")
        
        # Color the separator line (second line)
        if len(lines) > 1:
            colored_lines.append(f"{Fore.CYAN}{lines[1]}{Style.RESET_ALL}")
        
        # Process data rows
        for i in range(2, len(lines)):
            line = lines[i]
            # Find the position of column breaks based on the header row
            col_positions = []
            header = lines[0]
            for j in range(len(header)):
                if j > 0 and header[j-1] == ' ' and header[j] != ' ':
                    col_positions.append(j)
            
            if len(col_positions) >= 2:  # We need at least 2 column positions
                # Split the line into columns based on positions
                iteration_part = line[:col_positions[0]].strip()
                accuracy_part = line[col_positions[0]:col_positions[1]].strip()
                rule_count_part = line[col_positions[1]:].strip()
                
                # Apply colors
                colored_iteration = f"{Fore.YELLOW}{iteration_part}{Style.RESET_ALL}"
                
                # Color accuracy based on value
                accuracy_value = float(accuracy_part.replace('%', ''))
                if accuracy_value >= 95:
                    colored_accuracy = f"{Fore.GREEN}{accuracy_part}{Style.RESET_ALL}"
                elif accuracy_value >= 80:
                    colored_accuracy = f"{Fore.YELLOW}{accuracy_part}{Style.RESET_ALL}"
                else:
                    colored_accuracy = f"{Fore.WHITE}{accuracy_part}{Style.RESET_ALL}"
                
                colored_rule_count = f"{Fore.WHITE}{rule_count_part}{Style.RESET_ALL}"
                
                # Rebuild the line with proper spacing and colors
                rebuilt_line = (
                    colored_iteration.ljust(col_positions[0]) + 
                    colored_accuracy.ljust(col_positions[1] - col_positions[0]) + 
                    colored_rule_count
                )
                colored_lines.append(rebuilt_line)
            else:
                # Fallback if we can't determine column positions
                colored_lines.append(f"{Fore.WHITE}{line}{Style.RESET_ALL}")
        
        return '\n'.join(colored_lines)
    
    except Exception as e:
        logger.error(f"Error generating validation history table: {str(e)}")
        return f"{Fore.RED}Error generating validation history table: {str(e)}{Style.RESET_ALL}"

def generate_ruleset_summary(ruleset):
    """Generate a nice display of the final ruleset with colors"""
    rules_text = format_colored_rules_text(ruleset.get("rules", []))
    
    return f"""
{Fore.GREEN}FINAL RULESET{Style.RESET_ALL}
{Fore.CYAN}{'-'*12}{Style.RESET_ALL}
{Fore.WHITE}Logic: {Fore.YELLOW}{ruleset.get('logic', 'all').upper()}{Style.RESET_ALL}
{Fore.WHITE}Rule Count: {Fore.YELLOW}{len(ruleset.get('rules', []))}{Style.RESET_ALL}

{rules_text}

{Fore.GREEN}Ruleset Rationale:{Style.RESET_ALL}
{Fore.WHITE}{ruleset.get('description', 'No description provided')}{Style.RESET_ALL}
"""

def format_colored_rules_text(rules, indent=0):
    """Format rules as readable text with indentation and colors"""
    result = []
    
    for rule in rules:
        prefix = "  " * indent
        
        if "rules" in rule:
            # Nested rule group
            result.append(f"{Fore.YELLOW}{prefix}Rule Group ({rule.get('logic', 'all').upper()}):{Style.RESET_ALL}")
            # Get nested rules text and add each line with proper indentation
            nested_rules = format_colored_rules_text(rule.get("rules", []), indent + 1)
            for line in nested_rules.split('\n'):
                if line.strip():  # Only add non-empty lines
                    result.append(line)
        elif "field" in rule:
            # Standard rule
            field = rule.get("field", "").split(".")[-1]  # Just the field name
            condition = rule.get("condition", "")
            
            if "threshold" in rule:
                value = rule.get("threshold")
                result.append(f"{Fore.CYAN}{prefix}• {field}{Style.RESET_ALL} {Fore.WHITE}{condition}{Style.RESET_ALL} {Fore.GREEN}{value}{Style.RESET_ALL}")
            elif "values" in rule:
                values = rule.get("values", [])
                result.append(f"{Fore.CYAN}{prefix}• {field}{Style.RESET_ALL} {Fore.WHITE}{condition}{Style.RESET_ALL} {Fore.GREEN}{values}{Style.RESET_ALL}")
    
    return "\n".join(result)

def generate_applications_summary(openai_client, final_ruleset):
    """Generate a summary of applications with approval rationales using LLM"""
    # Load applications and results
    apps_with_results = load_applications_with_results()
    
    if not apps_with_results:
        return f"{Fore.RED}No application data available for summary{Style.RESET_ALL}"
    
    # Get LLM-generated rationales for each application
    apps_with_rationales = generate_application_rationales(openai_client, apps_with_results, final_ruleset)
    
    # Create a neat table with colors
    table_data = []
    for app in apps_with_rationales:
        table_data.append([
            app["id"],
            app["name"],
            "APPROVED" if app["approved"] else "DECLINED",
            "Correct" if app["correct"] else "Incorrect",
            app["rationale"]
        ])
    
    # Get basic table as text
    table = tabulate(
        table_data, 
        headers=["App ID", "Applicant", "Decision", "Classification", "Rationale"],
        tablefmt="grid"
    )
    
    # Colorize the table
    lines = table.split('\n')
    colored_lines = []
    
    header_processed = False
    
    for line in lines:
        if '|' not in line:
            # Grid lines without data, keep as is with cyan color
            colored_lines.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
        elif not header_processed:
            # This is the header line, color it differently
            parts = line.split('|')
            colored_parts = [Fore.CYAN + parts[0]]
            
            for part in parts[1:-1]:  # Skip first and last (they're grid edges)
                if part.strip():
                    colored_parts.append(f"{Fore.YELLOW}{part}{Style.RESET_ALL}")
                else:
                    colored_parts.append(part)
            
            colored_parts.append(Fore.CYAN + parts[-1] + Style.RESET_ALL)
            colored_lines.append('|'.join(colored_parts))
            header_processed = True
        else:
            # Data lines, color based on content
            if "APPROVED" in line:
                # Color approved rows
                parts = line.split('|')
                colored_parts = [Fore.CYAN + parts[0]]
                
                for i, part in enumerate(parts[1:-1]):
                    if i == 2:  # Decision column
                        colored_parts.append(f"{Fore.GREEN}{part}{Style.RESET_ALL}")
                    elif i == 3:  # Classification column
                        if "Correct" in part:
                            colored_parts.append(f"{Fore.GREEN}{part}{Style.RESET_ALL}")
                        else:
                            colored_parts.append(f"{Fore.RED}{part}{Style.RESET_ALL}")
                    else:
                        colored_parts.append(f"{Fore.WHITE}{part}{Style.RESET_ALL}")
                
                colored_parts.append(Fore.CYAN + parts[-1] + Style.RESET_ALL)
                colored_lines.append('|'.join(colored_parts))
            elif "DECLINED" in line:
                # Color declined rows
                parts = line.split('|')
                colored_parts = [Fore.CYAN + parts[0]]
                
                for i, part in enumerate(parts[1:-1]):
                    if i == 2:  # Decision column
                        colored_parts.append(f"{Fore.RED}{part}{Style.RESET_ALL}")
                    elif i == 3:  # Classification column
                        if "Correct" in part:
                            colored_parts.append(f"{Fore.GREEN}{part}{Style.RESET_ALL}")
                        else:
                            colored_parts.append(f"{Fore.RED}{part}{Style.RESET_ALL}")
                    else:
                        colored_parts.append(f"{Fore.WHITE}{part}{Style.RESET_ALL}")
                
                colored_parts.append(Fore.CYAN + parts[-1] + Style.RESET_ALL)
                colored_lines.append('|'.join(colored_parts))
            else:
                # Other lines (separators)
                colored_lines.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
    
    colored_table = '\n'.join(colored_lines)
    
    return f"""
{Fore.GREEN}APPLICATION DECISIONS WITH RATIONALES{Style.RESET_ALL}
{Fore.CYAN}{'-'*35}{Style.RESET_ALL}
{colored_table}
"""

def load_applications_with_results():
    """Load applications with validation results"""
    try:
        # Load applications
        applications = []
        app_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                    if f.startswith("application_") and f.endswith(".json")]
        
        app_files.sort(key=lambda f: int(f.replace("application_", "").replace(".json", "")))
        
        for f in app_files:
            app_id = int(f.replace("application_", "").replace(".json", ""))
            file_path = os.path.join(APPLICATIONS_DIR, f)
            
            with open(file_path, 'r') as file:
                app_data = json.load(file)
                applications.append({
                    "id": app_id,
                    "name": app_data.get("personalDetails", {}).get("name", f"Applicant {app_id}"),
                    "data": app_data
                })
        
        # Load validation results
        with open(os.path.join(RESULTS_DIR, "validation_results.json"), 'r') as f:
            validation_results = json.load(f)
        
        # Combine applications with results
        for app in applications:
            for result in validation_results.get("results", []):
                if result.get("application_id") == app["id"]:
                    app["approved"] = result.get("decision") == "approved"
                    app["correct"] = result.get("correct", False)
                    break
        
        return applications
    
    except Exception as e:
        logger.error(f"Error loading applications with results: {str(e)}")
        return []

def generate_application_rationales(openai_client, applications, ruleset):
    """Use LLM to generate approval/decline rationales for applications"""
    
    # Create a batch of applications to analyze (limit to 10 for efficiency)
    batch = applications[:10] if len(applications) > 10 else applications
    
    # Create a prompt explaining the ruleset and applications
    prompt = f"""
I need explanations for why these credit card applications were approved or declined.

The approval rules are:
```json
{json.dumps(ruleset, indent=2)}
```

Please explain in a short, clear sentence why each application was approved or declined 
based on these rules. Focus on the specific rule or criteria that determined the decision.

Here are the applications:
"""

    # Add application details to the prompt
    for app in batch:
        data = app["data"]
        credit_info = data.get("creditHistory", {})
        financial_info = data.get("financialInformation", {})
        
        prompt += f"""
Application #{app['id']} ({app['name']}): {'APPROVED' if app['approved'] else 'DECLINED'}
- Credit Tier: {credit_info.get('creditTier')}
- Credit Score: {credit_info.get('creditScore')}
- Payment History: {credit_info.get('paymentHistory')}
- Income Tier: {financial_info.get('incomeTier')}
- Annual Income: ${financial_info.get('annualIncome')}
- Debt Tier: {financial_info.get('debtTier')}
- Existing Debt: ${financial_info.get('existingDebt')}
- Employment Status: {financial_info.get('employmentStatus')}
"""

    prompt += """
For each application, provide a rationale like:
Application #X: [Decision rationale in one clear sentence]
"""

    # Get LLM response
    try:
        response = openai_client.generate(
            prompt=prompt,
            system_message="You are a Credit Card Analyst who explains application decisions clearly and concisely.",
            temperature=0.3,
            expert_name="Summary Generator"
        )
        
        # Parse LLM response to extract rationales
        rationales = parse_rationales(response, batch)
        
        # Add rationales to applications
        for app in applications:
            app_id = app["id"]
            if app_id in rationales:
                app["rationale"] = rationales[app_id]
            else:
                # For applications beyond the batch size, provide a default rationale
                logic = ruleset.get("logic", "all")
                tier_text = ""
                
                if app["approved"]:
                    if logic == "any":
                        tier_text = f"met at least one approval criteria"
                    else:
                        tier_text = f"met all required criteria"
                else:
                    if logic == "any":
                        tier_text = f"didn't meet any approval criteria"
                    else:
                        tier_text = f"failed to meet all required criteria"
                
                app["rationale"] = f"Application {tier_text} based on credit, income, and debt profile."
        
        return applications
    
    except Exception as e:
        logger.error(f"Error generating application rationales: {str(e)}")
        
        # Provide default rationales if LLM fails
        for app in applications:
            app["rationale"] = "Decision based on application profile and ruleset criteria."
        
        return applications

def parse_rationales(llm_response, applications):
    """Parse LLM response to extract rationales for each application"""
    rationales = {}
    
    # Extract application IDs
    app_ids = [app["id"] for app in applications]
    
    # Simple parsing logic - look for lines starting with "Application #"
    lines = llm_response.split("\n")
    for line in lines:
        line = line.strip()
        
        # Check for application reference
        for app_id in app_ids:
            if line.startswith(f"Application #{app_id}:") or line.startswith(f"Application {app_id}:"):
                # Extract the rationale part after the colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    rationales[app_id] = parts[1].strip()
                break
    
    # Ensure we have a rationale for each application
    for app in applications:
        if app["id"] not in rationales:
            rationales[app["id"]] = "Decision based on evaluation of application criteria."
    
    return rationales

def generate_accuracy_visualization():
    """Generate visualization of accuracy improvements over iterations"""
    # Load validation history
    try:
        with open(os.path.join(RESULTS_DIR, "validation_history.json"), 'r') as f:
            validation_history = json.load(f)
        
        if not validation_history:
            logger.warning("No validation history available for visualization")
            return "No visualization available (empty validation history)"
        
        # Extract data
        iterations = [entry.get("iteration", i) for i, entry in enumerate(validation_history)]
        accuracies = [entry.get("accuracy", 0) for entry in validation_history]
        rule_counts = [entry.get("rule_count", 0) for entry in validation_history]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot accuracy on top subplot
        ax1.plot(iterations, accuracies, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Target Accuracy')
        
        # Add annotations for accuracy values
        for i, accuracy in enumerate(accuracies):
            ax1.annotate(f"{accuracy:.1f}%", 
                       (iterations[i], accuracy),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center')
        
        ax1.set_title('Credit Card Approval Rule Accuracy Improvement', fontsize=16)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        ax1.legend()
        
        # Plot rule count on bottom subplot
        ax2.bar(iterations, rule_counts, color='orange', alpha=0.7)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Rule Count', fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add annotations for rule counts
        for i, count in enumerate(rule_counts):
            ax2.annotate(f"{count}", 
                       (iterations[i], count),
                       textcoords="offset points", 
                       xytext=(0,5), 
                       ha='center')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = os.path.join(RESULTS_DIR, f"accuracy_improvement_{timestamp}.png")
        plt.savefig(viz_file)
        plt.close()
        
        logger.info(f"Generated accuracy visualization: {viz_file}")
        return viz_file
    
    except Exception as e:
        logger.error(f"Error generating accuracy visualization: {str(e)}")
        return "Error generating visualization"

def generate_colored_ascii_chart(history_data):
    """Generate a colored ASCII chart of accuracy over iterations"""
    if not history_data:
        return f"{Fore.RED}No data available for chart{Style.RESET_ALL}"
    
    # Extract data
    iterations = [entry.get("iteration", i+1) for i, entry in enumerate(history_data)]
    accuracies = [entry.get("accuracy", 0) for entry in history_data]
    
    # Define chart dimensions
    height = 10
    width = len(iterations) * 3 + 5  # 5 for y-axis, 3 chars per data point
    
    # Create a simpler character-based chart
    result = []
    
    # Add title
    result.append(f"{Fore.YELLOW}Accuracy Over Iterations{Style.RESET_ALL}")
    result.append(f"{Fore.CYAN}{'-' * 25}{Style.RESET_ALL}")
    
    # Create rows from top (100%) to bottom (0%)
    for h in range(height + 1):
        row = []
        # Y-axis labels
        y_value = 100 - (h * 100 // height)
        
        # Color the y-axis labels based on value
        if y_value >= 90:
            y_color = Fore.GREEN
        elif y_value >= 70:
            y_color = Fore.YELLOW
        else:
            y_color = Fore.WHITE
            
        row.append(f"{y_color}{y_value:3d}%{Style.RESET_ALL} {Fore.CYAN}|{Style.RESET_ALL}")
        
        # Plot points and lines
        for i, acc in enumerate(accuracies):
            x_pos = i * 3 + 5
            # Determine if this accuracy level should be marked at this height
            acc_height = int((100 - acc) * height / 100)
            
            if h == acc_height:
                # Choose color based on accuracy
                if acc >= 90:
                    point_color = Fore.GREEN
                elif acc >= 70:
                    point_color = Fore.YELLOW
                else:
                    point_color = Fore.WHITE
                    
                row.append(f"{point_color}o{Style.RESET_ALL}  ")  # Data point
            elif h < acc_height:
                # Check if there should be a connecting line
                if i > 0:
                    prev_acc = accuracies[i-1]
                    prev_height = int((100 - prev_acc) * height / 100)
                    if prev_height < h < acc_height or acc_height < h < prev_height:
                        row.append(f"{Fore.BLUE}|{Style.RESET_ALL}  ")
                    else:
                        row.append("   ")
                else:
                    row.append("   ")
            else:
                row.append("   ")
        
        result.append("".join(row))
    
    # Add x-axis
    x_axis = f"     {Fore.CYAN}" + "".join(f"{it:3d}" for it in iterations) + f"{Style.RESET_ALL}"
    result.append(f"     {Fore.CYAN}" + "-" * (len(iterations) * 3) + f"{Style.RESET_ALL}")
    result.append(x_axis)
    result.append(f"     {Fore.YELLOW}Iteration{Style.RESET_ALL}")
    
    return "\n".join(result)

def generate_colored_simple_ascii_chart(history_data):
    """Generate a very simple colored ASCII bar chart showing accuracy progress"""
    if not history_data:
        return f"{Fore.RED}No data available for chart{Style.RESET_ALL}"
    
    # Extract data
    iterations = [entry.get("iteration", i+1) for i, entry in enumerate(history_data)]
    accuracies = [entry.get("accuracy", 0) for entry in history_data]
    
    result = [f"{Fore.YELLOW}Accuracy Over Iterations:{Style.RESET_ALL}", f"{Fore.CYAN}{'-' * 25}{Style.RESET_ALL}"]
    
    # Generate a simple bar chart
    for i, acc in enumerate(accuracies):
        iter_num = iterations[i]
        # Scale accuracy to 0-20 character width
        bar_width = int(acc / 5)  # 100% = 20 characters
        
        # Choose color based on accuracy
        if acc >= 90:
            bar_color = Fore.GREEN
        elif acc >= 70:
            bar_color = Fore.YELLOW
        else:
            bar_color = Fore.WHITE
            
        bar = bar_color + "█" * bar_width + Style.RESET_ALL
        
        result.append(f"{Fore.CYAN}Iter {iter_num:2d} |{Style.RESET_ALL} {bar} {Fore.WHITE}{acc:.1f}%{Style.RESET_ALL}")
    
    return "\n".join(result)
