from typing import Dict, Any, List
import json
import re
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.config.settings import TASK_TYPES
from meta_agent_system.utils.logger import get_logger

logger = get_logger(__name__)

def create_task_decomposer(llm_client: OpenAIClient) -> ExpertAgent:
    """
    Create a task decomposition expert agent.
    
    This expert breaks down complex problems into manageable subtasks
    and identifies the expertise required for each subtask.
    
    Args:
        llm_client: OpenAI client for generating responses
        
    Returns:
        Expert agent for task decomposition
    """
    system_prompt = """
You are a Task Decomposition Specialist. Your job is to break down complex problems into smaller, 
manageable subtasks that can be solved independently by specialized expert agents.

For each subtask you create, provide:
1. A clear description of the subtask
2. The specific expertise required to solve it (be precise about the exact expertise needed)
3. The task type that best matches this subtask
4. Any input data required
5. Dependencies on other subtasks (if any)
6. Priority level (1-10, where 1 is highest priority)

Your output should be a JSON object with a "subtasks" array containing the subtask objects.

Available task types:
- task_decomposition: Breaking down a complex task into smaller, manageable subtasks
- schema_design: Creating JSON schemas for data structures
- data_generation: Generating sample data based on schemas
- data_analysis: Analyzing data to identify patterns and insights
- rule_extraction: Extracting rules from data patterns
- validation: Validating results against criteria
- programming: Writing code to implement solutions

If none of these task types fit the subtask, you should specify what new expertise would be required.
"""
    
    def decompose_behavior(task: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior function for task decomposition"""
        task_description = task.get("description", "")
        task_data = task.get("data", {})
        
        # Construct a prompt for decomposition
        prompt = f"""
Task Description: {task_description}

Context: {task_data.get('context', '')}

Requirements:
{json.dumps(task_data.get('requirements', []), indent=2)}

Constraints:
{json.dumps(task_data.get('constraints', []), indent=2)}

Please decompose this task into subtasks. For each subtask, identify:
1. What needs to be done
2. What specific expertise is required
3. The appropriate task type
4. Any dependencies on other subtasks
5. Priority level (1-10)

If a subtask requires expertise not covered by the standard task types, please specify what new expertise would be needed.
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract JSON from the response
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                result_data = json.loads(json_match.group(0))
            else:
                # If no JSON object found, try parsing the entire output
                result_data = json.loads(llm_response)
            
            # Extract subtasks from result
            subtasks = result_data.get("subtasks", [])
            if not isinstance(subtasks, list):
                raise ValueError("Expected a list of subtasks")
            
            # Process and format subtasks
            spawn_tasks = []
            for i, subtask in enumerate(subtasks):
                # Ensure each subtask has required fields
                if "description" not in subtask:
                    subtask["description"] = f"Subtask {i+1}"
                
                # Extract task_type and required_expertise
                task_type = subtask.get("task_type", "")
                required_expertise = subtask.get("required_expertise", "")
                
                # If task_type is not in standard types, it's a new expertise
                new_expertise_needed = task_type not in TASK_TYPES.keys() and task_type != ""
                
                # Prepare the spawn task
                spawn_task = {
                    "description": subtask["description"],
                    "task_type": task_type if not new_expertise_needed else "custom",
                    "data": subtask.get("data", {}),
                    "priority": subtask.get("priority", 10 - i),
                    "dependencies": subtask.get("dependencies", []),
                    "required_expertise": required_expertise or task_type,
                    "new_expertise_needed": new_expertise_needed
                }
                
                spawn_tasks.append(spawn_task)
            
            return {
                "spawn_tasks": spawn_tasks,
                "message": f"Decomposed into {len(spawn_tasks)} subtasks",
                "subtasks_details": subtasks  # Include original details for context
            }
        except Exception as e:
            logger.error(f"Error parsing decomposition output: {str(e)}")
            return {
                "error": f"Failed to parse decomposition output: {str(e)}",
                "raw_output": llm_response
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Task Decomposer",
        capabilities=["task_decomposition", "planning"],
        behavior=decompose_behavior,
        description="Breaks down complex problems into manageable subtasks and identifies required expertise"
    )
