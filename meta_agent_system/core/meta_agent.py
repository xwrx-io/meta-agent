import time
from typing import Dict, Any, List, Optional
from queue import PriorityQueue
from meta_agent_system.core.task import Task
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import MAX_ITERATIONS
from meta_agent_system.core.expert_factory import ExpertFactory

logger = get_logger(__name__)

class MetaAgent:
    """
    Meta agent that coordinates multiple expert agents to solve complex problems.
    
    Attributes:
        name: Name of the meta agent
        agents: List of registered expert agents
        task_queue: Priority queue for pending tasks
        completed_tasks: List of completed tasks
        results: List of task results
        context: Shared context dictionary for all tasks
    """
    def __init__(self, name: str):
        """
        Initialize a meta agent.
        
        Args:
            name: Name of the meta agent
        """
        self.name = name
        self.agents = []
        self.task_queue = PriorityQueue()
        self.completed_tasks = []
        self.task_store = {}  # Dictionary to store all tasks by ID
        self.results = []
        self.context = {}
        logger.info(f"Initialized meta agent '{name}'")
    
    def register_agent(self, agent: ExpertAgent):
        """
        Register an expert agent with the meta agent.
        
        Args:
            agent: Expert agent to register
        """
        self.agents.append(agent)
        logger.info(f"Registered agent '{agent.name}' with meta agent '{self.name}'")
    
    def add_task(self, task: Task):
        """
        Add a task to the queue.
        
        Args:
            task: Task to add to the queue
        """
        # Store task in the task store
        self.task_store[task.id] = task
        
        # Add to priority queue
        self.task_queue.put((task.priority, task.id))
        logger.info(f"Added task to queue: {task.description} (ID: {task.id})")
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Task object or None if not found
        """
        return self.task_store.get(task_id)
    
    def find_agent_for_task(self, task_type: str) -> Optional[ExpertAgent]:
        """
        Find an agent that can handle a given task type.
        
        Args:
            task_type: Type of task to find an agent for
            
        Returns:
            Expert agent that can handle the task or None if none found
        """
        for agent in self.agents:
            if agent.can_handle(task_type):
                return agent
        return None
    
    def create_expert_for_task(self, task_type: str, required_expertise: str, task_description: str) -> Optional[ExpertAgent]:
        """
        Create a new expert agent for a task type that doesn't have an existing handler.
        
        Args:
            task_type: Type of task needing an expert
            required_expertise: Description of expertise required
            task_description: Description of the task
            
        Returns:
            Newly created expert agent or None if creation failed
        """
        # Skip if we don't have a factory
        if not hasattr(self, 'expert_factory'):
            logger.warning("No expert factory available to create new experts")
            return None
        
        # Generate a name for the expert
        expert_name = f"{required_expertise.replace(' ', '_').title()}_Expert"
        
        # Create capabilities list
        capabilities = [task_type] if task_type != "custom" else []
        
        # Add a custom capability if needed
        if task_type == "custom":
            custom_capability = required_expertise.lower().replace(' ', '_')
            capabilities.append(custom_capability)
        
        # Create the expert
        try:
            new_expert = self.expert_factory.create_expert(
                name=expert_name,
                expertise=required_expertise,
                capabilities=capabilities,
                task_description=task_description
            )
            
            if new_expert:
                # Register the new expert
                self.register_agent(new_expert)
                logger.info(f"Created and registered new expert: {new_expert.name}")
                return new_expert
            else:
                logger.error(f"Failed to create expert for {required_expertise}")
                return None
        except Exception as e:
            logger.error(f"Error creating expert: {str(e)}")
            return None
    
    def solve(self, problem: Dict[str, Any], max_iterations: int = MAX_ITERATIONS) -> Dict[str, Any]:
        """
        Solve a complex problem by coordinating expert agents.
        
        Args:
            problem: Dictionary describing the problem
            max_iterations: Maximum number of iterations to run
            
        Returns:
            Dictionary with results and execution statistics
        """
        logger.info(f"Starting to solve problem: {problem.get('description', 'Unknown')}")
        
        # Create initial task from problem
        initial_task = Task(
            description=problem.get("description", "Solve this problem"),
            task_type=problem.get("type", "task_decomposition"),
            data=problem.get("data", {})
        )
        self.add_task(initial_task)
        
        # Execute task loop
        iterations = 0
        tasks_completed = 0
        
        while not self.task_queue.empty() and iterations < max_iterations:
            iterations += 1
            logger.info(f"Iteration {iterations}/{max_iterations}")
            
            # Get the highest priority task ID from the queue
            _, task_id = self.task_queue.get()
            
            # Get the task from the store
            task = self.get_task_by_id(task_id)
            
            if not task:
                logger.error(f"Task with ID {task_id} not found in task store")
                continue
                
            # Check if all dependencies are satisfied
            dependencies_satisfied = True
            for dep_id in task.dependencies:
                dep_task = self.get_task_by_id(dep_id)
                if not dep_task or dep_task.status != "completed":
                    dependencies_satisfied = False
                    # Put task back in queue with slightly lower priority
                    self.task_queue.put((task.priority + 1, task.id))
                    logger.info(f"Task {task.id} waiting for dependency {dep_id}")
                    break
            
            if not dependencies_satisfied:
                continue
            
            # Find an agent that can handle this task
            agent = self.find_agent_for_task(task.task_type)

            # If no agent found and we have expertise requirements, try to create one
            if not agent and hasattr(self, 'expert_factory'):
                required_expertise = task.data.get("required_expertise", "")
                new_expertise_needed = task.data.get("new_expertise_needed", False)
                
                if new_expertise_needed and required_expertise:
                    logger.info(f"No existing agent for task type '{task.task_type}'. Creating new expert for '{required_expertise}'")
                    agent = self.create_expert_for_task(
                        task_type=task.task_type,
                        required_expertise=required_expertise,
                        task_description=task.description
                    )

            if agent:
                # Process the task
                logger.info(f"Assigning task '{task.description}' to agent '{agent.name}'")
                task.mark_in_progress(agent.name)
                
                # Prepare task dictionary with context
                task_dict = task.dict()
                task_dict["context"] = self.context  # Add context to task
                
                # Process the task
                result = agent.process_task(task_dict)
                
                # Update task status
                if result["status"] == "completed":
                    task.mark_completed(result.get("result", {}))
                else:
                    task.mark_failed(result.get("error", "Unknown error"))
                
                # Add to completed tasks
                self.completed_tasks.append(task)
                tasks_completed += 1
                
                # Store the result
                self.results.append(result)
                
                # Update context with result
                self.context[f"task_{task.id}"] = task.dict()
                
                # Process any new tasks from the result
                spawn_tasks = result.get("result", {}).get("spawn_tasks", [])
                if spawn_tasks:
                    logger.info(f"Spawning {len(spawn_tasks)} new tasks from result")
                    for spawn_task_data in spawn_tasks:
                        spawn_task = Task(
                            description=spawn_task_data.get("description", "Subtask"),
                            task_type=spawn_task_data.get("task_type", "general"),
                            data=spawn_task_data.get("data", {}),
                            priority=spawn_task_data.get("priority", 5),
                            parent_id=task.id
                        )
                        # Add dependency relationship
                        spawn_task.add_dependency(task.id)
                        
                        # Add spawned task to parent's spawn list
                        task.add_spawn_task(spawn_task.id)
                        
                        # Add the new task to the queue
                        self.add_task(spawn_task)
            else:
                # No agent can handle this task
                logger.warning(f"No agent found for task type: {task.task_type}")
                task.mark_failed(f"No agent available for task type: {task.task_type}")
                self.completed_tasks.append(task)
        
        # After the main task loop completes, check if we should continue with feedback loop
        if self.task_queue.empty() and hasattr(self, 'feedback_loop'):
            should_continue = self.process_feedback_loop()
            if should_continue:
                # Continue the task loop with the new tasks
                return self.solve(problem, max_iterations)
        
        # Compile the final results
        return {
            "status": "completed" if self.task_queue.empty() else "incomplete",
            "iterations": iterations,
            "tasks_completed": tasks_completed,
            "tasks_remaining": self.task_queue.qsize(),
            "results": [result for result in self.results if result["status"] == "completed"],
            "errors": [result for result in self.results if result["status"] == "error"],
            "context": self.context
        }

    def process_feedback_loop(self) -> bool:
        """
        Process the feedback loop to determine if we should continue iterating.
        
        Returns:
            True if the loop should continue, False if it should stop
        """
        if not hasattr(self, 'feedback_loop'):
            return False
        
        # Process validation results through the feedback loop
        feedback_result = self.feedback_loop.process_validation_results(self.context)
        
        if feedback_result.get("status") == "success":
            logger.info(f"Feedback loop completed successfully: {feedback_result.get('message', '')}")
            return False
        
        # Create new tasks for the next iteration
        next_tasks = feedback_result.get("next_tasks", [])
        if not next_tasks:
            return False
        
        # Add the next tasks to the queue
        created_task_ids = []
        for task_data in next_tasks:
            task = Task(
                description=task_data.get("description", "Feedback loop task"),
                task_type=task_data.get("task_type", "general"),
                data=task_data.get("data", {}),
                priority=task_data.get("priority", 5)
            )
            created_task_ids.append(task.id)
            self.add_task(task)
        
        # Update dependencies between tasks
        for i, task_data in enumerate(next_tasks):
            dependencies = task_data.get("dependencies", [])
            if dependencies:
                task_id = created_task_ids[i]
                task = self.get_task_by_id(task_id)
                
                for dep in dependencies:
                    # Replace placeholder with actual task ID
                    if dep == "<REFINEMENT_TASK_ID>" and i > 0:
                        dep = created_task_ids[0]  # First task is refinement
                    
                    if dep in created_task_ids:
                        task.add_dependency(dep)
        
        # Look for any expertise recommendations
        for key, value in reversed(list(self.context.items())):
            if isinstance(value, dict) and value.get("agent_name") == "Expertise Recommender":
                if "result" in value and "new_expertise_tasks" in value["result"]:
                    expertise_tasks = value["result"]["new_expertise_tasks"]
                    if expertise_tasks:
                        logger.info(f"Adding {len(expertise_tasks)} tasks for new expertise areas")
                        for task_data in expertise_tasks:
                            task = Task(
                                description=task_data.get("description", "Apply new expertise"),
                                task_type=task_data.get("task_type", "custom"),
                                data=task_data.get("data", {}),
                                priority=task_data.get("priority", 4)
                            )
                            self.add_task(task)
                    break
        
        logger.info(f"Feedback loop continuing: {feedback_result.get('message', '')}")
        return True
