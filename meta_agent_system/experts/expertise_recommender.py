from typing import Dict, Any, List
import json
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger

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
        """Behavior function for expertise recommendation"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        context = task.get("context", {})
        
        # Analyze validation results and current performance
        validation_results = None
        ruleset = None
        
        # Find the most recent validation result
        for key, value in reversed(list(context.items())):
            if isinstance(value, dict) and value.get("agent_name") == "Validator":
                if "result" in value and "validation_results" in value["result"]:
                    validation_results = value["result"]["validation_results"]
                    break
        
        # Find the most recent ruleset
        for key, value in reversed(list(context.items())):
            if isinstance(value, dict) and (value.get("agent_name") == "Rule Extractor" or 
                                          value.get("agent_name") == "Rule Refiner"):
                if "result" in value and "ruleset" in value["result"]:
                    ruleset = value["result"]["ruleset"]
                    break
        
        # Get list of current agents/expertise
        current_expertise = []
        for key, value in context.items():
            if isinstance(value, dict) and "agent_name" in value:
                current_expertise.append(value["agent_name"])
        
        # Construct a prompt for expertise recommendation
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

Current Experts Available:
{json.dumps(list(set(current_expertise)), indent=2)}

Validation Results:
{json.dumps(validation_results, indent=2) if validation_results else "No validation results yet"}

Current Ruleset:
{json.dumps(ruleset, indent=2) if ruleset else "No ruleset yet"}

Based on the current state of the problem and the performance so far, please identify what 
additional specialized expertise might be needed to improve the results.

Consider:
1. Are there patterns or aspects of the data that current experts might be missing?
2. Would specific domain knowledge help in formulating better rules?
3. Are there technical aspects that need specialized attention?
4. What expertise would help address the gaps in the current solution?

For each recommended expertise, explain:
- What specific expertise is needed
- Why it would be valuable
- How it would improve the current solution
- What tasks this expertise would perform
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract recommendations
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            
            if json_match:
                recommendations = json.loads(json_match.group(0))
            else:
                # If no JSON found, structure the response ourselves
                recommendations = {
                    "recommended_expertise": [],
                    "explanation": llm_response
                }
                
                # Try to extract recommendations from text
                expertise_pattern = r'(?:Expertise|Expert):\s*([^\n]+)'
                expertise_matches = re.findall(expertise_pattern, llm_response)
                
                why_pattern = r'(?:Why|Reason|Rationale|Justification):\s*([^\n]+)'
                why_matches = re.findall(why_pattern, llm_response)
                
                for i, expertise in enumerate(expertise_matches):
                    why = why_matches[i] if i < len(why_matches) else "To improve the solution"
                    
                    recommendations["recommended_expertise"].append({
                        "name": expertise.strip(),
                        "reason": why.strip(),
                        "task_type": expertise.strip().lower().replace(" ", "_")
                    })
            
            # Generate tasks for new experts
            new_expertise_tasks = []
            for expertise in recommendations.get("recommended_expertise", []):
                task_type = expertise.get("task_type", "custom")
                expertise_name = expertise.get("name", "Specialized Expert")
                reason = expertise.get("reason", "To improve the solution")
                
                new_expertise_tasks.append({
                    "description": f"Apply {expertise_name} to improve the solution",
                    "task_type": task_type,
                    "data": {
                        "context": reason,
                        "required_expertise": expertise_name,
                        "new_expertise_needed": True
                    },
                    "priority": 3
                })
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "new_expertise_tasks": new_expertise_tasks,
                "message": f"Identified {len(new_expertise_tasks)} potential new expertise areas"
            }
        except Exception as e:
            logger.error(f"Error parsing expertise recommendations: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to parse expertise recommendations: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Expertise Recommender",
        capabilities=["expertise_recommendation"],
        behavior=expertise_recommendation_behavior,
        description="Analyzes current state and recommends new expertise needed"
    )
