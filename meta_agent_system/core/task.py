import uuid
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class Task(BaseModel):
    """
    Represents a task to be executed by an expert agent.
    
    Attributes:
        id: Unique identifier for the task
        description: Description of the task
        task_type: Type of task (used to route to appropriate expert)
        data: Additional data required for the task
        priority: Priority level (lower number = higher priority)
        parent_id: ID of parent task if this is a subtask
        status: Current status of the task
        result: Result of task execution
        agent_assigned: Name of agent assigned to this task
        created_at: Timestamp when task was created
        completed_at: Timestamp when task was completed
        dependencies: List of task IDs this task depends on
        spawn_tasks: List of task IDs spawned by this task
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    task_type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 5
    parent_id: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    agent_assigned: Optional[str] = None
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependencies: List[str] = Field(default_factory=list)
    spawn_tasks: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = time.time()
    
    def mark_completed(self, result: Dict[str, Any]):
        """Mark task as completed with result"""
        self.status = "completed"
        self.result = result
        self.completed_at = time.time()
    
    def mark_failed(self, error: str):
        """Mark task as failed with error message"""
        self.status = "failed"
        self.result = {"error": error}
        self.completed_at = time.time()
    
    def mark_in_progress(self, agent_name: str):
        """Mark task as in progress with assigned agent"""
        self.status = "in_progress"
        self.agent_assigned = agent_name
    
    def add_dependency(self, task_id: str):
        """Add a dependency to this task"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
    
    def add_spawn_task(self, task_id: str):
        """Add a spawned task ID to this task"""
        if task_id not in self.spawn_tasks:
            self.spawn_tasks.append(task_id)
    
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds"""
        if self.completed_at and self.created_at:
            return self.completed_at - self.created_at
        return None
    
    def __lt__(self, other):
        """Comparison for priority queue"""
        if not isinstance(other, Task):
            return NotImplemented
        return self.priority < other.priority

    def dict(self):
        if hasattr(self, 'model_dump'):
            return self.model_dump()
        # Fall back to original implementation
