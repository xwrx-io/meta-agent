from typing import Dict, Any, Optional
from meta_agent_system.utils.logger import get_logger
import os
import json
from meta_agent_system.config.settings import RESULTS_DIR
from datetime import datetime

logger = get_logger(__name__)

class SuccessCriteriaChecker:
    """
    Checks if the current state meets the success criteria for stopping the loop.
    
    This class determines when the meta agent has successfully solved the problem
    and can stop the execution loop.
    """
    def __init__(self, criteria_type="credit_card_rules"):
        """
        Initialize the success criteria checker.
        
        Args:
            criteria_type: Type of criteria to check (default: credit_card_rules)
        """
        self.criteria_type = criteria_type
        self.logger = get_logger(__name__)
        logger.info(f"Initialized success criteria checker for: {criteria_type}")
    
    def check_success(self, completed_tasks):
        """Check if the success criteria are met."""
        # Make sure we examine each task to see which field contains the agent name
        meta_agent_context = {}
        
        for task in completed_tasks:
            if hasattr(task, 'result') and task.result:
                # Try different ways to get the agent name
                agent_name = None
                if hasattr(task, 'agent'):
                    agent_name = task.agent
                elif hasattr(task, 'agent_name'):
                    agent_name = task.agent_name
                # Add more fallbacks if needed
                
                meta_agent_context[task.id] = {
                    "agent_name": agent_name,
                    "result": task.result
                }
        
        if self.criteria_type == "credit_card_rules":
            return self._check_credit_card_rules_success(meta_agent_context)
        # Add more criteria types as needed
        return False
    
    def _check_credit_card_rules_success(self, context):
        """
        Check if credit card rules success criteria are met.
        
        Success is defined as:
        1. Validator has validated the rules with 100% accuracy
        2. All applications are correctly classified (approved/declined)
        """
        self.logger.info("Checking credit card rules success criteria")
        
        for key, value in context.items():
            if value.get("agent_name") == "Validator":
                result = value.get("result", {})
                
                # Check if validation was successful and if accuracy is 100%
                accuracy = result.get("accuracy", 0)
                success_flag = result.get("success", False)
                
                self.logger.info(f"Validator accuracy: {accuracy}%, Success flag: {success_flag}")
                
                if success_flag is True and accuracy >= 100:
                    self.logger.info("Credit card rules success criteria met! 100% accuracy achieved.")
                    
                    # Save the final results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
                    
                    try:
                        os.makedirs(os.path.dirname(results_file), exist_ok=True)
                        with open(results_file, 'w') as f:
                            json.dump(context, f, indent=2)
                        self.logger.info(f"Saved final results to {results_file}")
                    except Exception as e:
                        self.logger.error(f"Error saving results: {str(e)}")
                    
                    return True
        
        self.logger.info("Credit card rules success criteria not yet met")
        return False

    def check(self, task_results):
        """Check if success criteria are met."""
        # For credit card rules, we need to check validation results
        validation_result = None
        for task in task_results:
            if task.get('agent_name') == 'Validator':
                validation_result = task.get('result', {})
                break
        
        if not validation_result:
            self.logger.warning("No validation result found, cannot determine success")
            return False
        
        # Check both accuracy and success status
        validation_status = validation_result.get('status', '')
        validation_success = validation_result.get('success', False)
        
        if validation_status == 'success' and validation_success is True:
            self.logger.info("Success criteria met: Validation was fully successful")
            return True
        else:
            self.logger.info(f"Success criteria not met: Validation status={validation_status}, success={validation_success}")
            return False
