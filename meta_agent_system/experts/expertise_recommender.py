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
You are an Expertise Recommender. Your job is to analyze the current state of problem-solving
and identify what additional specialized expertise might be needed.

When analyzing the situation:
1. Identify gaps in the current expertise roster
2. Spot areas where specialized knowledge would improve results
3. Consider both technical and domain-specific expertise
4. Be specific about what expertise is needed and why
5. Explain how the new expertise would improve outcomes

Your recommendations should be specific, actionable, and justified by the current context.
"""
    
    def expertise_recommendation_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate expertise recommendations based on validation results."""
        logger.info("Expertise Recommender processing task")
        
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
        inconsistencies = validation_result.get("validation_results", {}).get("inconsistencies", [])
        suggestions = validation_result.get("validation_results", {}).get("suggestions", [])
        
        logger.info(f"Found {len(inconsistencies)} inconsistencies and {len(suggestions)} suggestions")
        
        # Construct prompt for the LLM
        prompt = f"""
Based on the validation results of our credit card approval rules system, we need specialized experts to improve the ruleset.

Current accuracy: {validation_result.get('accuracy', 0)}%

Inconsistencies detected:
{json.dumps(inconsistencies, indent=2)}

Improvement suggestions:
{json.dumps(suggestions, indent=2)}

Please recommend specialized experts that would help achieve 100% accuracy. For each expert, provide:
1. Name (descriptive of their expertise)
2. List of capabilities (skills they have)
3. System prompt (detailed instructions for the expert)

Format your response as a valid JSON object with a 'recommended_experts' array containing expert objects.
"""
        
        # Get response from LLM
        logger.info("Generating expertise recommendations with LLM")
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7
        )
        
        # Extract JSON response
        try:
            # Try to find JSON object in response
            match = re.search(r'\{[\s\S]*\}', llm_response)
            if match:
                recommendations = json.loads(match.group(0))
            else:
                recommendations = json.loads(llm_response)
            
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
            return {
                "status": "error",
                "error": f"Failed to parse recommendations: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Expertise Recommender",
        capabilities=["expertise_recommendation"],
        behavior=expertise_recommendation_behavior,
        description="Analyzes current state and recommends new expertise needed"
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