from typing import Dict, Any, List, Optional
from meta_agent_system.core.task import Task
from meta_agent_system.core.success_criteria import SuccessCriteriaChecker
from meta_agent_system.utils.logger import get_logger

logger = get_logger(__name__)

class FeedbackLoop:
    """
    Manages the feedback loop for iterative improvement.
    
    This class coordinates the cycle of rule extraction, validation, and refinement
    to progressively improve the ruleset until success criteria are met.
    """
    def __init__(self, success_criteria_checker=None, expertise_recommender=None, expert_factory=None):
        """
        Initialize the feedback loop.
        
        Args:
            success_criteria_checker: Checker for determining when to stop the loop
            expertise_recommender: Expert recommender for recommending new experts
            expert_factory: Factory for creating new experts from recommendations
        """
        self.logger = get_logger(__name__)
        
        # Store the parameters as class attributes
        self.success_criteria_checker = success_criteria_checker
        self.expertise_recommender = expertise_recommender
        self.expert_factory = expert_factory
        self.iteration = 0
        logger.info("Initialized feedback loop")
    
    def process_validation_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process validation results and determine next steps.
        
        Args:
            context: Current context from the meta agent
            
        Returns:
            Dictionary with feedback loop status and next tasks
        """
        # Check if we've met the success criteria
        success_check = self.success_criteria_checker.check_success(context)
        
        # Always force at least one feedback iteration to ensure expertise recommendation runs
        if self.iteration == 0 or not success_check.get("success", False):
            # If not successful, increment iteration and create refinement task
            self.iteration += 1
            logger.info(f"Feedback loop iteration {self.iteration}: {success_check.get('reason', '')}")
            
            # Create tasks for the next iteration
            next_tasks = []
            
            # Add rule refinement task
            refinement_task = {
                "description": f"Refine credit card approval rules (iteration {self.iteration})",
                "task_type": "rule_refinement",
                "data": {
                    "context": "Based on validation results, refine the rules to improve accuracy",
                    "iteration": self.iteration,
                    "accuracy": success_check.get("accuracy", 0),
                    "reason": success_check.get("reason", "")
                },
                "priority": 3
            }
            next_tasks.append(refinement_task)
            
            # Add validation task (to be executed after refinement)
            validation_task = {
                "description": f"Validate refined credit card approval rules (iteration {self.iteration})",
                "task_type": "validation",
                "data": {
                    "context": "Validate the refined ruleset against all applications",
                    "iteration": self.iteration
                },
                "priority": 4,
                "dependencies": ["<REFINEMENT_TASK_ID>"]  # This will be replaced with the actual ID
            }
            next_tasks.append(validation_task)
            
            # Add expertise recommendation task
            recommendation_task = {
                "description": f"Recommend additional expertise needed (iteration {self.iteration})",
                "task_type": "expertise_recommendation",
                "data": {
                    "context": "Analyze the current state and recommend additional expertise that might improve results",
                    "iteration": self.iteration
                },
                "priority": 3
            }
            next_tasks.append(recommendation_task)
            
            return {
                "status": "continue",
                "message": f"Continuing feedback loop (iteration {self.iteration}): {success_check.get('reason', '')}",
                "next_tasks": next_tasks,
                "iteration": self.iteration
            }
        else:
            # Success criteria met
            logger.info(f"Success criteria met: {success_check.get('reason', '')}")
            return {
                "status": "success",
                "message": success_check.get("reason", "Success criteria met"),
                "next_tasks": []
            }

    def process(self, completed_tasks):
        """Process completed tasks and recommend next actions."""
        self.logger.info("Processing feedback loop")
        
        # Check if we've hit success criteria
        problem_solved = False
        if self.success_criteria_checker:
            problem_solved = self.success_criteria_checker.check_success(completed_tasks)
            if problem_solved:
                self.logger.info("Success criteria met! Problem solved.")
                return [], True
        
        # If problem not solved, we need to generate new tasks
        new_tasks = []
        
        # Find validator result
        validator_result = None
        for task in completed_tasks:
            agent_name = None
            if hasattr(task, 'agent_assigned'):
                agent_name = task.agent_assigned
            
            if agent_name == "Validator" and hasattr(task, 'result'):
                validator_result = task.result
                break
        
        if not problem_solved and validator_result:
            self.logger.info("Problem not solved yet. Creating expertise recommendation task...")
            
            # Create context from completed tasks for new expertise task
            task_context = {}
            for task in completed_tasks:
                if hasattr(task, 'id') and hasattr(task, 'result'):
                    agent_name = getattr(task, 'agent_assigned', 'Unknown')
                    task_context[task.id] = {"agent_name": agent_name, "result": task.result}
            
            # Add expertise recommender task
            expertise_task = {
                "description": "Recommend expertise for improving rule accuracy",
                "priority": 4,
                "data": {"validation_result": validator_result},
                "context": task_context
            }
            new_tasks.append(expertise_task)
            
            # Add rule refiner task
            refine_task = {
                "description": "Refine rules based on validation feedback",
                "priority": 5,
                "data": {"validation_result": validator_result},
                "context": task_context
            }
            new_tasks.append(refine_task)
            
            self.logger.info(f"Created {len(new_tasks)} new tasks to improve ruleset")
            return new_tasks, False
        
        # If we get here, no new tasks were added
        self.logger.info("No new tasks created in feedback loop")
        return [], problem_solved
