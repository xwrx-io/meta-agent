from typing import Dict, Any, Callable

class ExpertAgent:
    """A simplified expert agent that performs a specialized task"""
    
    def __init__(self, name: str, capabilities: list, behavior: Callable, description: str = ""):
        """
        Initialize an expert agent
        
        Args:
            name: Name of the expert
            capabilities: List of capabilities/skills this expert has
            behavior: Function that implements the expert's behavior
            description: Description of what this expert does
        """
        self.name = name
        self.capabilities = capabilities
        self.behavior = behavior
        self.description = description
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the expert's behavior on a task
        
        Args:
            task: Task dictionary with description and data
            
        Returns:
            Result dictionary with status and output
        """
        try:
            result = self.behavior(task)
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing {self.name}: {str(e)}"
            }
