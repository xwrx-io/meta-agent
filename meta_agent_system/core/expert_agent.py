from typing import Dict, Any, Callable, List

class ExpertAgent:
    """Simple expert agent that executes a specific behavior function."""
    
    def __init__(self, name: str, behavior: Callable, capabilities: List[str] = None, description: str = ""):
        """Initialize the expert agent."""
        self.name = name
        self.behavior = behavior
        self.capabilities = capabilities or []
        self.description = description
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the expert's behavior on a task."""
        return self.behavior(task)
    
    def has_capability(self, capability: str) -> bool:
        """Check if the expert has a specific capability."""
        return capability in self.capabilities
    
    def __str__(self) -> str:
        """String representation of the expert."""
        return f"{self.name}: {self.description}"
