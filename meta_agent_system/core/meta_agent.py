import time
from typing import Dict, Any, List, Optional
from queue import PriorityQueue
from meta_agent_system.core.task import Task
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import MAX_ITERATIONS
from meta_agent_system.core.expert_factory import ExpertFactory
import logging
import uuid

logger = get_logger(__name__)

class MetaAgent:
    """
    Meta agent that coordinates multiple expert agents to solve complex problems.
    
    Attributes:
        name: Name of the meta agent
        agents: List of registered expert agents
        task_store: Dictionary to store all tasks by ID
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
        self.logger = get_logger(__name__)
        self.name = name
        self.agents = []
        self.task_store = {}  # Dictionary to store all tasks by ID
        self.completed_tasks = []
        self.results = []
        self.context = {}
        self.feedback_iterations = 0
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
        Add a task to the task store.
        
        Args:
            task: Task to add
        """
        # Store task in the task store
        self.task_store[task.id] = task
        logger.info(f"Added task to store: {task.description} (ID: {task.id}, Priority: {task.priority})")
    
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
    
    def get_runnable_tasks(self):
        """
        Get tasks that are ready to run (all dependencies satisfied).
        
        Returns:
            List of Task objects with all dependencies satisfied, sorted by priority
        """
        runnable_tasks = []
        pending_count = 0
        blocked_count = 0
        
        # Check all tasks in the store for pending tasks with satisfied dependencies
        for task_id, task in self.task_store.items():
            if task.status == "pending":
                pending_count += 1
                if not task.dependencies:
                    # Task has no dependencies, it's runnable
                    runnable_tasks.append(task)
                    continue
                
                # Check if all dependencies are satisfied
                dependencies_satisfied = True
                unsatisfied_deps = []
                for dep_id in task.dependencies:
                    dep_task = self.get_task_by_id(dep_id)
                    if not dep_task:
                        logger.warning(f"Task {task.id} depends on missing task {dep_id}")
                        dependencies_satisfied = False
                        unsatisfied_deps.append(f"missing:{dep_id}")
                    elif dep_task.status != "completed":
                        dependencies_satisfied = False
                        unsatisfied_deps.append(f"{dep_id}:{dep_task.status}")
                
                if dependencies_satisfied:
                    runnable_tasks.append(task)
                else:
                    blocked_count += 1
                    logger.debug(f"Task '{task.description}' blocked by dependencies: {', '.join(unsatisfied_deps)}")
        
        logger.info(f"Task status: {len(runnable_tasks)} runnable, {blocked_count} blocked, {pending_count} total pending")
        
        # Sort by priority (lower number = higher priority)
        return sorted(runnable_tasks, key=lambda t: t.priority)

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
        feedback_loops = 0
        
        # Main execution loop - continue until max iterations or no more pending tasks
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Iteration {iterations}/{max_iterations}")
            
            # Get all runnable tasks (topological sort approach)
            runnable_tasks = self.get_runnable_tasks()
            
            if not runnable_tasks:
                # If no tasks are runnable, check if all tasks are complete
                pending_tasks = [t for _, t in self.task_store.items() if t.status == "pending"]
                if not pending_tasks:
                    logger.info("No more tasks to process - all tasks completed")
                    break
                else:
                    # Check for circular dependencies or blocked tasks
                    logger.warning(f"No runnable tasks but {len(pending_tasks)} tasks are pending - possible dependency deadlock")
                    for task in pending_tasks:
                        logger.warning(f"Pending task: {task.description} (ID: {task.id}) - Dependencies: {task.dependencies}")
                    
                    # If we have the feedback loop, try to use it to break deadlocks
                    if hasattr(self, 'feedback_loop') and feedback_loops < 5:
                        logger.info(f"Attempting feedback loop to resolve deadlock (iteration: {feedback_loops})")
                        should_continue = self.process_feedback_loop()
                        
                        if should_continue:
                            feedback_loops += 1
                            continue
                        else:
                            logger.info("Feedback loop unable to resolve deadlock")
                            break
                    else:
                        # Unable to make progress
                        break
            
            # Process the highest priority runnable task
            task = runnable_tasks[0]
            
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
                    # Create a map of descriptions to task IDs for dependency mapping
                    description_to_id = {}
                    for existing_id, existing_task in self.task_store.items():
                        description_to_id[existing_task.description] = existing_id
                    
                    # First pass: create all tasks to build the ID mapping
                    new_tasks = []
                    for spawn_task_data in spawn_tasks:
                        spawn_task = Task(
                            description=spawn_task_data.get("description", "Subtask"),
                            task_type=spawn_task_data.get("task_type", "general"),
                            data=spawn_task_data.get("data", {}),
                            priority=spawn_task_data.get("priority", 5),
                            parent_id=task.id
                        )
                        new_tasks.append((spawn_task, spawn_task_data))
                        # Add to task store and update mapping
                        self.add_task(spawn_task)
                        description_to_id[spawn_task.description] = spawn_task.id
                    
                    # Second pass: set up all dependencies now that all IDs exist
                    for spawn_task, spawn_task_data in new_tasks:
                        # Add dependency relationships - map text descriptions to task IDs
                        text_dependencies = spawn_task_data.get("dependencies", [])
                        if text_dependencies:
                            logger.info(f"Task '{spawn_task.description}' has {len(text_dependencies)} dependencies")
                            
                        for dep_desc in text_dependencies:
                            # Look for exact matches first
                            if dep_desc in description_to_id:
                                dep_id = description_to_id[dep_desc]
                                spawn_task.add_dependency(dep_id)
                                dep_task = self.get_task_by_id(dep_id)
                                dep_status = dep_task.status if dep_task else "unknown"
                                logger.info(f"Added dependency: '{spawn_task.description}' → '{dep_desc}' (ID: {dep_id}, Status: {dep_status})")
                            else:
                                # If no exact match, look for partial matches
                                found_match = False
                                for existing_desc, existing_id in description_to_id.items():
                                    # Check if one is a substring of the other
                                    if dep_desc in existing_desc or existing_desc in dep_desc:
                                        spawn_task.add_dependency(existing_id)
                                        dep_task = self.get_task_by_id(existing_id)
                                        dep_status = dep_task.status if dep_task else "unknown"
                                        logger.info(f"Added partial dependency match: '{spawn_task.description}' → '{existing_desc}' (ID: {existing_id}, Status: {dep_status})")
                                        found_match = True
                                        break
                                
                                if not found_match:
                                    logger.warning(f"Could not find dependency '{dep_desc}' for task '{spawn_task.description}'")
                        
                        # Add spawned task to parent's spawn list
                        task.add_spawn_task(spawn_task.id)
            else:
                # No agent can handle this task
                logger.warning(f"No agent found for task type: {task.task_type}")
                task.mark_failed(f"No agent available for task type: {task.task_type}")
                self.completed_tasks.append(task)
            
            # After processing a batch of tasks, check if we need to run the feedback loop
            if not self.get_runnable_tasks() and hasattr(self, 'feedback_loop') and feedback_loops < 5:
                logger.info(f"No more runnable tasks, checking feedback loop (feedback iterations: {feedback_loops})")
                should_continue = self.process_feedback_loop()
                
                if should_continue:
                    feedback_loops += 1
                    logger.info(f"Feedback loop {feedback_loops} added new tasks, continuing execution")
                else:
                    logger.info("Feedback loop indicated completion or max feedback loops reached")
                    break
        
        # Count remaining pending tasks
        pending_tasks = [t for _, t in self.task_store.items() if t.status == "pending"]
        
        # Compile the final results
        return {
            "status": "completed" if not pending_tasks else "incomplete",
            "iterations": iterations,
            "tasks_completed": tasks_completed,
            "tasks_remaining": len(pending_tasks),
            "feedback_loops": feedback_loops,
            "results": [result for result in self.results if result["status"] == "completed"],
            "errors": [result for result in self.results if result["status"] == "error"],
            "context": self.context
        }

    def process_feedback_loop(self):
        """Process the feedback loop to get new tasks or determine if we're done."""
        self.logger.info(f"Processing feedback loop with {len(self.completed_tasks)} completed tasks")
        
        # Call the feedback loop processor
        new_tasks, complete = self.feedback_loop.process(self.completed_tasks)
        
        # Check for expertise recommendations from completed tasks
        for task in self.completed_tasks:
            agent_name = getattr(task, 'agent_assigned', None)
            result = getattr(task, 'result', {})
            
            if agent_name == "Expertise Recommender" and result and result.get("status") == "success":
                # Found expertise recommendations - try to create new experts
                self.logger.info("Found expertise recommendations in completed tasks")
                
                if "recommendations_file" in result:
                    recommendations_file = result["recommendations_file"]
                    self.logger.info(f"Processing recommendations from file: {recommendations_file}")
                    
                    if hasattr(self, 'expert_factory'):
                        # Create the experts - this needs to be properly implemented
                        new_experts = self.expert_factory.create_experts_from_recommendations(recommendations_file)
                        if new_experts:
                            for expert in new_experts:
                                self.register_agent(expert)
                            self.logger.info(f"Created and registered {len(new_experts)} new expert agents")
        
        # If the success criteria is met, we're done
        if complete:
            self.logger.info("Success criteria met! Problem solved.")
            return False
        
        # If we have new tasks, add them to the queue
        if new_tasks:
            for task_data in new_tasks:
                # Create a new Task object and add it to the queue
                task = Task(
                    id=str(uuid.uuid4()),
                    description=task_data.get("description", ""),
                    task_type=task_data.get("task_type", "expertise_recommendation") if "expertise" in task_data.get("description", "").lower() else "general",
                    priority=task_data.get("priority", 1),
                    dependencies=task_data.get("dependencies", []),
                    data=task_data.get("data", {}),
                    context=task_data.get("context", {})
                )
                self.add_task(task)
            
            self.logger.info(f"Feedback loop {self.feedback_iterations + 1} added new tasks, continuing execution")
            self.feedback_iterations += 1
            return True
        
        self.logger.info("Feedback loop didn't add any new tasks, execution complete")
        return False

    def process_feedback(self):
        # Check for recommendations of new expertise
        logging.info(f"Completed tasks structure: {self.completed_tasks}")
        logging.info(f"First item type: {type(self.completed_tasks[0]) if self.completed_tasks else 'No items'}")
        for task_item in self.completed_tasks:
            # Access attributes directly for Pydantic Task objects
            task_id = task_item.id if hasattr(task_item, 'id') else None
            agent_name = task_item.agent_name if hasattr(task_item, 'agent_name') else None
            result = task_item.result if hasattr(task_item, 'result') else None
            
            # Then use these variables in your logic
            if agent_name == "Expertise Recommender" and result:
                recommendations = result.get("recommendations", {})
                new_expertise_tasks = result.get("new_expertise_tasks", [])
                
                if new_expertise_tasks:
                    for expertise_task in new_expertise_tasks:
                        # Create a new expert agent for the recommended expertise
                        expertise_name = expertise_task.get("data", {}).get("required_expertise", "Specialized Expert")
                        task_type = expertise_task.get("task_type", "custom")
                        
                        # Create the expert using the factory
                        factory = ExpertFactory(self.llm_client)
                        new_expert = factory.create_expert(
                            name=expertise_name,
                            expertise=expertise_task.get("data", {}).get("context", ""),
                            capabilities=[task_type],
                            task_description=expertise_task.get("description", "")
                        )
                        
                        if new_expert:
                            # Register the new expert
                            self.register_agent(new_expert)
                            logger.info(f"Created and registered new expert agent '{expertise_name}' with capabilities: {[task_type]}")
