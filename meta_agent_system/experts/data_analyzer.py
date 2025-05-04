from typing import Dict, Any, List
import json
import os
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import APPLICATIONS_DIR

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
                    break
        
        if not applications:
            # Try to load applications from files if they exist
            if os.path.exists(APPLICATIONS_DIR):
                application_files = [os.path.join(APPLICATIONS_DIR, f) for f in os.listdir(APPLICATIONS_DIR) 
                                   if f.startswith("application_") and f.endswith(".json")]
                
                applications = []
                for file_path in application_files:
                    try:
                        with open(file_path, 'r') as f:
                            applications.append(json.load(f))
                    except Exception as e:
                        logger.error(f"Error loading application from {file_path}: {str(e)}")
        
        if not applications:
            return {
                "status": "error",
                "error": "No applications found in context or files. Data Generator must run first."
            }
        
        # Construct a prompt for data analysis
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

Applications Data:
{json.dumps(applications, indent=2)}

Please analyze these credit card applications to identify patterns and factors that might 
influence approval or rejection decisions. Look for correlations, potential thresholds, 
and combinations of factors that might work together.

Consider analyzing:
- Income levels and how they relate to requested credit limits
- Credit scores and their impact
- Debt-to-income ratios
- Employment stability
- Credit history factors
- Age, education, or demographic patterns
- Any other relevant factors in the data

Provide a detailed analysis of what you find, including specific observations, potential patterns, 
and hypotheses about what rules might be in use for approving or declining applications.

Your analysis will be used to extract specific rules for credit card approval in the next step.
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=4000
        )
        
        # Return the analysis
        return {
            "status": "success",
            "analysis": llm_response,
            "num_applications_analyzed": len(applications),
            "message": "Completed analysis of credit card applications"
        }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Data Analyzer",
        capabilities=["data_analysis"],
        behavior=data_analysis_behavior,
        description="Analyzes data to identify patterns and insights"
    )
