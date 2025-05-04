from typing import Dict, Any, List, Callable, Optional
from meta_agent_system.utils.logger import get_logger

logger = get_logger(__name__)

class ExpertAgent:
    """
    Base class for expert agents that can handle specific types of tasks.
    
    Attributes:
        name: Name of the agent
        capabilities: List of task types this agent can handle
        behavior: Function that processes tasks and returns results
        description: Description of what this agent does
    """
    def __init__(self, 
                 name: str, 
                 capabilities: List[str], 
                 behavior: Callable[[Dict[str, Any]], Dict[str, Any]],
                 description: str = ""):
        """
        Initialize an expert agent.
        
        Args:
            name: Name of the agent
            capabilities: List of task types this agent can handle
            behavior: Function that processes tasks and returns results
            description: Description of what this agent does
        """
        self.name = name
        self.capabilities = capabilities
        self.behavior = behavior
        self.description = description
        logger.info(f"Initialized expert agent '{name}' with capabilities: {capabilities}")
    
    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle a given task type"""
        return task_type in self.capabilities
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task and return the result.
        
        Args:
            task: Task data dictionary
            
        Returns:
            Dictionary with processing result
        """
        logger.info(f"Agent '{self.name}' processing task: {task.get('description', 'Unknown')}")
        try:
            result = self.behavior(task)
            return {
                "status": "completed",
                "agent_name": self.name,
                "task_id": task.get("id"),
                "result": result
            }
        except Exception as e:
            logger.error(f"Error in agent '{self.name}': {str(e)}", exc_info=True)
            return {
                "status": "error",
                "agent_name": self.name,
                "task_id": task.get("id"),
                "error": str(e)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation"""
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "description": self.description
        }
