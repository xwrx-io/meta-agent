from typing import Dict, Any, List
import json
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
import os
from meta_agent_system.config.settings import RESULTS_DIR
import time
import re
from datetime import datetime

logger = get_logger(__name__)

def create_expertise_recommender(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create an expertise recommender agent.
    
    This expert analyzes the current state and recommends new expertise that might be needed.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for expertise recommendation
    """
    system_prompt = """
You are an AI Agent Architect specializing in creating optimized expert agents for credit card approval systems.

Your task is to design specialized AI expert agents for specific aspects of credit assessment.
For each expert agent, you will:

1. Define a specialized domain of expertise relevant to credit approval decisions
2. Create a precise system prompt that configures an AI to operate in that domain
3. Specify the exact response format (preferably JSON) that the expert should use
4. List concrete capabilities the expert agent should demonstrate

IMPORTANT GUIDELINES:
- Each system prompt must be designed for an AI language model, not a human
- Include clear instructions for structured JSON outputs in each system prompt
- Focus on pattern recognition and rule refinement in different domains
- Avoid vague instructions; specify exact evaluation criteria 
- Design prompts that encourage analytical reasoning and concrete recommendations
- Target expertise areas that would help refine credit approval rules

CRITICAL: Your output must be VALID JSON. Do not use trailing commas, ensure all property names are in quotes,
and verify that all brackets and braces are properly matched.
"""
    
    def expertise_recommendation_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate expertise recommendations based on validation results."""
        logger.info("Expertise Recommender processing task")
        
        # Define clean_json_string inside the function where it's used
        def clean_json_string(json_str):
            """Clean up common JSON syntax issues"""
            # Replace JavaScript comments with empty strings
            cleaned = re.sub(r'//.*', '', json_str)
            
            # Remove trailing commas (a common error in JSON)
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            # Ensure property names are quoted
            cleaned = re.sub(r'(\s*)(\w+)(\s*):([^"])', r'\1"\2"\3:\4', cleaned)
            
            # Replace single quotes with double quotes (another common error)
            cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
            
            return cleaned
        
        # Get validation result from task data or context
        validation_result = task.get("data", {}).get("validation_result")
        if not validation_result:
            # Try to find in context
            for context_item in task.get("context", {}).values():
                if isinstance(context_item, dict) and context_item.get("agent_name") == "Validator":
                    validation_result = context_item.get("result")
                    break
        
        if not validation_result:
            logger.error("No validation result found")
            return {
                "status": "error",
                "message": "No validation data available"
            }
        
        # Extract useful information for prompting
        accuracy = task.get("data", {}).get("current_accuracy", validation_result.get("accuracy", 0))
        misclassified = validation_result.get("misclassified_applications", [])
        
        # Get current ruleset
        ruleset_file = os.path.join(RESULTS_DIR, "credit_card_approval_rules.json")
        current_ruleset = {}
        try:
            with open(ruleset_file, 'r') as f:
                current_ruleset = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ruleset: {str(e)}")
        
        # Construct prompt for the LLM
        prompt = f"""
Design specialized AI expert agents that can improve our credit card approval system.
Our current accuracy is {accuracy}%, and we need to reach 100%.

Current ruleset:
```json
{json.dumps(current_ruleset, indent=2)}
```

We currently have {len(misclassified) if isinstance(misclassified, list) else 0} misclassified applications.

For each recommended expert, provide:

1. **name**: Descriptive of their specialized domain (e.g., "Credit Tier Evaluator")
2. **capabilities**: List of specific analytical abilities the expert has
3. **system_prompt**: Detailed instructions for how the AI agent should operate, including:
   - Exact expertise domain focus
   - Specific analytical approach
   - REQUIRED JSON response format with examples
   - How to evaluate patterns from input data
   - How to generate concrete rule recommendations

Create 3-4 diverse, specialized experts whose combined expertise would help achieve 100% accuracy.
Each expert should have a different specialization that would contribute to better rule discovery.

IMPORTANT: Ensure your response is valid JSON with the structure shown below. 
- Do not use trailing commas
- Ensure all property names have quotes
- Make sure all brackets and braces are matched

Your response must be a valid JSON object with this exact structure:
{{
  "recommended_experts": [
    {{
      "name": "Expert Name",
      "capabilities": ["capability1", "capability2"],
      "system_prompt": "Detailed AI system prompt with JSON response format",
      "description": "Brief description of what this expert contributes"
    }}
  ]
}}
"""
        
        # Get response from LLM
        logger.info("Generating expertise recommendations with LLM")
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.5,  # Reduced temperature for more predictable output
            expert_name="Expertise Recommender"
        )
        
        # Extract and fix JSON response
        try:
            # Try to find JSON object in response
            match = re.search(r'\{[\s\S]*\}', llm_response)
            json_str = match.group(0) if match else llm_response
            
            # Try to clean up common JSON issues before parsing
            cleaned_json = clean_json_string(json_str)  # Now using the local function
            
            # Parse the JSON
            try:
                recommendations = json.loads(cleaned_json)
            except json.JSONDecodeError:
                # Try with original if cleaning failed
                recommendations = json.loads(json_str)
            
            # Save to file for tracking
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recommendations_file = os.path.join(RESULTS_DIR, f"expertise_recommendations_{timestamp}.json")
            os.makedirs(os.path.dirname(recommendations_file), exist_ok=True)
            
            with open(recommendations_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            logger.info(f"Saved expertise recommendations to {recommendations_file}")
            
            return {
                "status": "success",
                "recommendations": recommendations.get("recommended_experts", []),
                "recommendations_file": recommendations_file,
                "message": f"Generated {len(recommendations.get('recommended_experts', []))} expert recommendations"
            }
        except Exception as e:
            logger.error(f"Error parsing expertise recommendations: {str(e)}")
            
            # Create a fallback response with a generic expert recommendation
            fallback_recommendations = {
                "recommended_experts": [
                    {
                        "name": "Credit Risk Pattern Analyzer",
                        "capabilities": [
                            "Credit tier evaluation",
                            "Rule pattern discovery",
                            "Decision boundary analysis",
                            "Edge case identification"
                        ],
                        "system_prompt": "You are a Credit Risk Pattern Analyzer specializing in discovering optimal decision boundaries for credit card approvals. Analyze application data to identify precise combinations of credit tier, income tier, and debt tier factors that determine approval outcomes. When examining applications, determine the exact thresholds that separate approvals from declines. Your responses should be structured as JSON with specific rule recommendations and associated confidence scores.",
                        "description": "Identifies precise approval boundaries between credit factors"
                    }
                ]
            }
            
            # Save fallback recommendations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_file = os.path.join(RESULTS_DIR, f"expertise_recommendations_fallback_{timestamp}.json")
            with open(fallback_file, 'w') as f:
                json.dump(fallback_recommendations, f, indent=2)
            
            # Also save the raw response for debugging
            raw_file = os.path.join(RESULTS_DIR, f"expertise_recommendations_raw_{timestamp}.txt")
            with open(raw_file, 'w') as f:
                f.write(llm_response)
            
            logger.info(f"Using fallback recommendations due to parsing error. Raw response saved to {raw_file}")
            
            return {
                "status": "partial_success",
                "recommendations": fallback_recommendations.get("recommended_experts", []),
                "recommendations_file": fallback_file,
                "message": f"Generated fallback expert recommendations due to parsing error: {str(e)}"
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Expertise Recommender",
        capabilities=["expertise_recommendation"],
        behavior=expertise_recommendation_behavior,
        description="Analyzes current state and recommends new AI expert agents"
    )

def process(self, task):
    """Recommend new expertise based on task needs."""
    self.logger.info(f"Processing task: {task.description}")
    
    # Find validation results if available
    validation_results = task.data.get('validation_results', {})
    
    # If no validation results, look for the most recent validation task
    if not validation_results:
        # This would need to access tasks from meta_agent, for now use static recommendations
        pass
    
    # Extract insights from validation results
    inconsistencies = validation_results.get('inconsistencies', [])
    suggestions = validation_results.get('suggestions', [])
    
    # Identify needed expertise areas
    areas_needing_expertise = []
    
    for inconsistency in inconsistencies:
        issue = inconsistency.get('issue', '').lower()
        if 'employment status' in issue:
            areas_needing_expertise.append('employment status evaluation')
        if 'payment history' in issue:
            areas_needing_expertise.append('credit history evaluation')
    
    for suggestion in suggestions:
        suggestion_text = suggestion.get('suggestion', '').lower()
        if 'employment' in suggestion_text:
            areas_needing_expertise.append('employment status evaluation')
        if 'payment history' in suggestion_text or 'credit history' in suggestion_text:
            areas_needing_expertise.append('credit history evaluation')
    
    # If no specific areas found, add general ones
    if not areas_needing_expertise:
        areas_needing_expertise = ['credit risk assessment', 'rule optimization']
    
    # Remove duplicates
    areas_needing_expertise = list(set(areas_needing_expertise))
    
    # Generate recommendations
    recommendations = []
    for area in areas_needing_expertise:
        agent_name = f"{area.replace(' ', '_').title()}_Expert"
        capabilities = [area, 'rule_refinement']
        
        system_prompt = f"""You are an expert in {area} for credit card applications.
        Your role is to evaluate applications specifically focusing on {area} aspects and provide
        specialized insights that can improve approval rules and decision-making.
        
        When asked to refine rules, suggest specific adjustments that improve accuracy
        without compromising risk management principles.
        """
        
        recommendations.append({
            "agent_name": agent_name,
            "capabilities": capabilities,
            "system_prompt": system_prompt,
            "rationale": f"Created to address inconsistencies in {area}"
        })
    
    # Save recommendations
    timestamp = int(time.time())
    feedback_iteration = task.data.get('feedback_iteration', 1)
    recommendation_file = os.path.join(RESULTS_DIR, f'expertise_recommendations_iteration_{feedback_iteration}.json')
    with open(recommendation_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    return {
        "status": "success",
        "recommendations": recommendations,
        "recommendation_file": recommendation_file,
        "message": f"Generated {len(recommendations)} expertise recommendations"
    }