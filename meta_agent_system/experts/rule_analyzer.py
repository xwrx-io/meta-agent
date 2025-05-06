from typing import Dict, Any
import json
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, RESULTS_DIR

logger = get_logger(__name__)

def create_rule_analyzer(llm_client: OpenAIClient) -> ExpertAgent:
    """Create a rule analysis expert agent."""
    system_prompt = """
You are a Credit Card Data Analyst specializing in pattern detection. Your job is to 
analyze credit card applications and identify distinguishing patterns between approved and declined
applications.

Focus on identifying:
1. Key thresholds that separate approved from declined applications
2. Feature combinations that strongly predict approval or decline
3. Unusual or non-obvious patterns in the data
4. Potential edge cases that might be difficult to classify
"""
    
    def rule_analysis_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze application data to find patterns"""
        logger.info("Starting pattern analysis")
        
        # Load required data
        validation_diagnostics = load_file(os.path.join(RESULTS_DIR, "validation_diagnostics.json"))
        applications = load_applications()
        hidden_approvals = load_hidden_approvals()
        
        # Prepare data for analysis
        approved_apps = []
        declined_apps = []
        
        for idx, app in enumerate(applications):
            app_id = idx + 1
            key = str(app_id)
            if hidden_approvals.get(key, False):
                approved_apps.append(app)
            else:
                declined_apps.append(app)
        
        # Create analysis text for LLM
        analysis_prompt = create_analysis_prompt(approved_apps, declined_apps, validation_diagnostics)
        
        # Get LLM analysis
        try:
            logger.info("Requesting pattern analysis from LLM")
            llm_response = llm_client.generate(
                prompt=analysis_prompt,
                system_message=system_prompt,
                temperature=0.3
            )
            
            # Save analysis results
            save_analysis_results(llm_response)
            
            return {
                "status": "success",
                "message": "Analyzed application patterns successfully"
            }
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return {
                "status": "error",
                "message": f"Error analyzing patterns: {str(e)}"
            }
    
    def load_file(file_path):
        """Load a JSON file with error handling"""
        if not os.path.exists(file_path):
            return {}
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return {}
    
    def load_applications():
        """Load applications from files"""
        applications = []
        
        if os.path.exists(APPLICATIONS_DIR):
            app_files = [f for f in os.listdir(APPLICATIONS_DIR) 
                        if f.startswith("application_") and f.endswith(".json")]
            
            for f in app_files:
                file_path = os.path.join(APPLICATIONS_DIR, f)
                try:
                    with open(file_path, 'r') as file:
                        applications.append(json.load(file))
                except Exception as e:
                    logger.error(f"Error loading application: {str(e)}")
        
        return applications
    
    def load_hidden_approvals():
        """Load hidden approvals"""
        file_path = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading hidden approvals: {str(e)}")
        
        return {}
    
    def create_analysis_prompt(approved_apps, declined_apps, validation_diagnostics):
        """Create a prompt for pattern analysis"""
        # Extract misclassified applications
        misclassified = []
        for eval in validation_diagnostics.get("rule_evaluations", []):
            if not eval.get("correct", True):
                misclassified.append({
                    "id": eval.get("application_id"),
                    "expected": "Approve" if eval.get("expected") else "Decline",
                    "actual": "Approve" if eval.get("actual") else "Decline",
                    "data": eval.get("application_data", {})
                })
        
        # Calculate approved application statistics
        approved_stats = calculate_stats(approved_apps)
        declined_stats = calculate_stats(declined_apps)
        
        return f"""
# Credit Card Application Pattern Analysis

## Dataset Overview
- Total Applications: {len(approved_apps) + len(declined_apps)}
- Approved: {len(approved_apps)} applications
- Declined: {len(declined_apps)} applications

## Approved Applications Statistics
{approved_stats}

## Declined Applications Statistics
{declined_stats}

## Misclassified Applications ({len(misclassified)})
{format_misclassified(misclassified[:5])}

## Your Tasks
1. Identify the key patterns that distinguish approved from declined applications
2. Find the most important thresholds for numeric values
3. Identify combinations of features that strongly predict approval
4. Analyze edge cases that are difficult to classify
5. Suggest specific rules that could better classify the applications

Provide your analysis in clear sections. Focus on actionable insights that can be used to create credit approval rules.
"""
    
    def calculate_stats(applications):
        """Calculate statistics for a set of applications"""
        if not applications:
            return "No applications available"
        
        credit_scores = [app.get("creditHistory", {}).get("creditScore", 0) for app in applications]
        incomes = [app.get("financialInformation", {}).get("annualIncome", 0) for app in applications]
        debts = [app.get("financialInformation", {}).get("existingDebt", 0) for app in applications]
        
        # Filter out missing values
        credit_scores = [s for s in credit_scores if s]
        incomes = [i for i in incomes if i]
        debts = [d for d in debts if d]
        
        # Calculate averages
        avg_credit = sum(credit_scores) / len(credit_scores) if credit_scores else 0
        avg_income = sum(incomes) / len(incomes) if incomes else 0
        avg_debt = sum(debts) / len(debts) if debts else 0
        
        return f"""
- Average Credit Score: {avg_credit:.1f}
- Average Annual Income: ${avg_income:.2f}
- Average Existing Debt: ${avg_debt:.2f}
- Debt-to-Income Ratio: {(avg_debt / avg_income if avg_income else 0):.2f}
"""
    
    def format_misclassified(misclassified):
        """Format misclassified applications for analysis"""
        if not misclassified:
            return "No misclassified applications"
        
        formatted = []
        for app in misclassified:
            data = app.get("data", {})
            formatted.append(f"""
Application #{app.get('id')}: Should be {app.get('expected')} but was {app.get('actual')}
- Credit Score: {data.get('credit_score')}
- Credit Tier: {data.get('credit_tier')}
- Annual Income: {data.get('annual_income')}
- Existing Debt: {data.get('existing_debt')}
- Payment History: {data.get('payment_history')}
- Employment Status: {data.get('employment_status')}
""")
        
        return "\n".join(formatted)
    
    def save_analysis_results(analysis):
        """Save analysis results to files"""
        # Save full analysis
        analysis_file = os.path.join(RESULTS_DIR, "credit_card_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump({"analysis": analysis}, f, indent=2)
        
        # Save readable text version
        text_file = os.path.join(RESULTS_DIR, "credit_card_analysis.txt")
        with open(text_file, 'w') as f:
            f.write(analysis)
        
        logger.info(f"Saved analysis results to {analysis_file} and {text_file}")
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Rule Analyzer",
        capabilities=["pattern_analysis", "data_analysis"],
        behavior=rule_analysis_behavior,
        description="Analyzes application patterns to identify distinguishing features"
    ) 