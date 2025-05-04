from typing import Dict, Any, List
import json
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR, RESULTS_DIR
import time

logger = get_logger(__name__)

def create_data_analyzer(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a data analyzer expert agent.
    
    This expert analyzes data to identify patterns and insights.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for data analysis
    """
    system_prompt = """
You are a Data Analysis Expert. Your job is to analyze data to identify patterns, correlations, and potential insights.

For credit card applications, you need to:
1. Analyze the applications to identify what factors might influence approval/rejection
2. Look for correlations and patterns across different fields
3. Identify potential thresholds or rules that might be in use
4. Organize your findings in a clear, structured way

Focus on finding meaningful patterns rather than superficial correlations.
Consider combinations of factors that might work together (e.g., income relative to requested credit limit).

Your analysis should be thorough, insightful, and provide a foundation for rule extraction.
"""
    
    def data_analysis_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior function for data analysis"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        context = task.get("context", {})
        
        # Get applications from context if available
        applications = []
        application_files = []
        
        for key, value in context.items():
            if isinstance(value, dict) and value.get("agent_name") == "Data Generator":
                if "applications" in value.get("result", {}):
                    applications = value["result"]["applications"]
                    application_files = value["result"].get("application_files", [])
                    logger.info(f"Found {len(applications)} applications in context")
                    break
        
        if not applications:
            # Try to load applications from files if they exist
            if os.path.exists(APPLICATIONS_DIR):
                application_files = [os.path.join(APPLICATIONS_DIR, f) for f in os.listdir(APPLICATIONS_DIR) 
                                   if f.startswith("application_") and f.endswith(".json")]
                
                if application_files:
                    logger.info(f"Found {len(application_files)} application files on disk")
                    applications = []
                    for file_path in application_files:
                        try:
                            with open(file_path, 'r') as f:
                                application = json.load(f)
                                applications.append(application)
                            logger.info(f"Loaded application from {file_path}")
                        except Exception as e:
                            logger.error(f"Error loading application from {file_path}: {str(e)}")
        
        if not applications:
            logger.error("No applications found in context or files")
            return {
                "status": "error",
                "error": "No applications found in context or files. Data Generator must run first."
            }
        
        # Try to load hidden approvals if available
        approvals = {}
        approvals_file = os.path.join(APPLICATIONS_DIR, "hidden_approvals.json")
        if os.path.exists(approvals_file):
            try:
                with open(approvals_file, 'r') as f:
                    approvals = json.load(f)
                logger.info(f"Loaded hidden approvals from {approvals_file}")
            except Exception as e:
                logger.error(f"Error loading hidden approvals: {str(e)}")
        
        # Construct a prompt for data analysis
        prompt = f"""
Task Description: {task_description}

You are analyzing {len(applications)} credit card applications to discover patterns that differentiate approved from declined applications.
Based only on the data, identify what factors seem to determine approval or rejection.

Consider analyzing:
- Income levels and how they relate to requested credit limits
- Credit scores and their impact
- Debt-to-income ratios
- Employment stability
- Credit history factors
- Any other relevant factors in the data

The first 10 applications (indices 0-9) are APPROVED.
The next 10 applications (indices 10-19) are DECLINED.

Please provide a detailed analysis with clear, specific observations about what differentiates approved applications from declined ones.
Your analysis should be thorough enough to allow extraction of specific rules with numeric thresholds.
"""
        
        # Include sample applications in the prompt
        prompt += "\n\nHere are the first few applications for reference:\n"
        sample_apps = applications[:3] + applications[10:13]  # 3 approved, 3 declined
        prompt += json.dumps(sample_apps, indent=2)
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.4,  # Lower temperature for more consistent analysis
            max_tokens=4000
        )
        
        # Save the analysis to file
        analysis_file = os.path.join(RESULTS_DIR, "credit_card_analysis.json")
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
        
        # Create a structured analysis result
        analysis_result = {
            "analysis": llm_response,
            "num_applications_analyzed": len(applications),
            "timestamp": time.time(),
            "approved_indices": list(range(10)),  # First 10 are approved
            "declined_indices": list(range(10, 20))  # Last 10 are declined
        }
        
        # Save as JSON
        with open(analysis_file, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        logger.info(f"Saved analysis to {analysis_file}")
        
        # Also save a text version for easier reading
        text_file = os.path.join(RESULTS_DIR, "credit_card_analysis.txt")
        with open(text_file, 'w') as f:
            f.write(f"Credit Card Application Analysis\n")
            f.write(f"================================\n\n")
            f.write(llm_response)
        logger.info(f"Saved analysis text to {text_file}")
        
        return {
            "status": "success",
            "result": {
                "analysis": llm_response,
                "analysis_file": analysis_file,
                "num_applications_analyzed": len(applications),
                "message": "Completed analysis of credit card applications"
            }
        }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Data Analyzer",
        capabilities=["data_analysis"],
        behavior=data_analysis_behavior,
        description="Analyzes data to identify patterns and insights"
    )
