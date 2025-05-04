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
You are a Task Decomposition Expert. Your job is to break down complex problems into well-defined, manageable subtasks.

For each subtask, you must specify:
1. A clear, concise description of what needs to be done
2. The type of expertise required (task_type)
3. IMPORTANT: Explicit dependencies between tasks - which tasks MUST be completed before this task can start
4. Appropriate priority level (1-10, with 1 being highest priority)
5. Any additional data or context needed

When specifying dependencies:
- Use EXACT and FULL task descriptions for dependencies
- Only add dependencies that are truly required (direct prerequisites)
- If a task has no dependencies, leave the dependencies list empty

Format your response as a structured set of tasks in the requested format.
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
3. The appropriate task type from this list: schema_design, data_generation, data_analysis, rule_extraction, validation, rule_refinement
4. Any dependencies on other subtasks
5. Priority level (1-10, with 1 being highest priority)

Your response MUST be in valid JSON format with the following structure:
{{
  "subtasks": [
    {{
      "description": "Clear description of the subtask",
      "task_type": "One of the task types listed above",
      "data": {{}},
      "priority": 1,
      "dependencies": ["Full description of prerequisite task"],
      "required_expertise": "Description of required expertise"
    }},
    // more subtasks...
  ]
}}

Make sure each dependency listed is the EXACT COMPLETE description of another task.
Ensure your JSON is properly formatted without any markdown formatting or extra text.
"""
        
        # Get response from LLM
        llm_response = llm_client.generate(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.3,  # Lower temperature for more deterministic output
            max_tokens=2000
        )
        
        # Extract JSON from the response
        try:
            # First, try to clean up the response to handle common formatting issues
            cleaned_response = llm_response.strip()
            
            # Remove any markdown code block markers
            cleaned_response = re.sub(r'^```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'^```\s*', '', cleaned_response)
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
            
            # Try to extract JSON using regex first
            json_match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if json_match:
                result_data = json.loads(json_match.group(0))
            else:
                # Fall back to manual subtask creation if JSON parsing fails
                logger.warning("Failed to parse JSON from LLM response, using fallback method")
                
                # Create default subtasks based on requirements in the task data
                subtasks = []
                requirements = task_data.get('requirements', [])
                
                if "schema" in task_description.lower() or any("schema" in req.lower() for req in requirements):
                    subtasks.append({
                        "description": "Create a JSON schema for credit card applications",
                        "task_type": "schema_design",
                        "data": {},
                        "priority": 1,
                        "dependencies": [],
                        "required_expertise": "schema_design",
                    })
                
                if "generate" in task_description.lower() or any("generate" in req.lower() for req in requirements):
                    subtasks.append({
                        "description": "Generate 20 diverse credit card applications (10 approved, 10 declined)",
                        "task_type": "data_generation",
                        "data": {},
                        "priority": 2,
                        "dependencies": ["Create a JSON schema for credit card applications"],
                        "required_expertise": "data_generation",
                    })
                
                if "analyze" in task_description.lower() or any("analyze" in req.lower() for req in requirements):
                    subtasks.append({
                        "description": "Analyze the credit card applications to discover patterns",
                        "task_type": "data_analysis",
                        "data": {},
                        "priority": 3,
                        "dependencies": ["Generate 20 diverse credit card applications (10 approved, 10 declined)"],
                        "required_expertise": "data_analysis",
                    })
                
                if "extract" in task_description.lower() or any("extract" in req.lower() or "rule" in req.lower() for req in requirements):
                    subtasks.append({
                        "description": "Extract rules that determine approval or rejection from discovered patterns",
                        "task_type": "rule_extraction",
                        "data": {},
                        "priority": 4,
                        "dependencies": ["Analyze the credit card applications to discover patterns"],
                        "required_expertise": "rule_extraction",
                    })
                
                if "validate" in task_description.lower() or any("validate" in req.lower() for req in requirements):
                    subtasks.append({
                        "description": "Validate the extracted rules by applying them to all 20 applications",
                        "task_type": "validation",
                        "data": {},
                        "priority": 5,
                        "dependencies": ["Extract rules that determine approval or rejection from discovered patterns"],
                        "required_expertise": "validation",
                    })
                
                result_data = {"subtasks": subtasks}
            
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
            logger.error(f"Raw response: {llm_response}")
            
            # Fallback to hardcoded tasks for credit card problem
            fallback_tasks = [
                {
                    "description": "Create a JSON schema for credit card applications",
                    "task_type": "schema_design",
                    "data": {},
                    "priority": 1,
                    "dependencies": [],
                    "required_expertise": "schema_design",
                    "new_expertise_needed": False
                },
                {
                    "description": "Generate 20 diverse credit card applications (10 approved, 10 declined)",
                    "task_type": "data_generation",
                    "data": {},
                    "priority": 2, 
                    "dependencies": ["Create a JSON schema for credit card applications"],
                    "required_expertise": "data_generation",
                    "new_expertise_needed": False
                },
                {
                    "description": "Analyze the credit card applications to discover patterns",
                    "task_type": "data_analysis",
                    "data": {},
                    "priority": 3,
                    "dependencies": ["Generate 20 diverse credit card applications (10 approved, 10 declined)"],
                    "required_expertise": "data_analysis", 
                    "new_expertise_needed": False
                },
                {
                    "description": "Extract rules that determine approval or rejection",
                    "task_type": "rule_extraction",
                    "data": {},
                    "priority": 4,
                    "dependencies": ["Analyze the credit card applications to discover patterns"],
                    "required_expertise": "rule_extraction",
                    "new_expertise_needed": False
                },
                {
                    "description": "Validate the extracted rules by applying them to all 20 applications",
                    "task_type": "validation", 
                    "data": {},
                    "priority": 5,
                    "dependencies": ["Extract rules that determine approval or rejection"],
                    "required_expertise": "validation",
                    "new_expertise_needed": False
                }
            ]
            
            return {
                "spawn_tasks": fallback_tasks,
                "message": "Using fallback task decomposition due to error",
                "error": f"Original error: {str(e)}"
            }
    
    # Create and return the expert agent
    return ExpertAgent(
        name="Task Decomposer",
        capabilities=["task_decomposition", "planning"],
        behavior=decompose_behavior,
        description="Breaks down complex problems into manageable subtasks and identifies required expertise"
    )
