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
    def __init__(self, success_criteria_checker: SuccessCriteriaChecker):
        """
        Initialize the feedback loop.
        
        Args:
            success_criteria_checker: Checker for determining when to stop the loop
        """
        self.success_criteria_checker = success_criteria_checker
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
        
        if success_check.get("success", False):
            logger.info(f"Success criteria met: {success_check.get('reason', '')}")
            return {
                "status": "success",
                "message": success_check.get("reason", "Success criteria met"),
                "next_tasks": []
            }
        
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
