from typing import Dict, Any, Optional
from meta_agent_system.utils.logger import get_logger

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
        logger.info(f"Initialized success criteria checker for: {criteria_type}")
    
    def check_success(self, meta_agent_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the current state meets the success criteria.
        
        Args:
            meta_agent_context: Current context from the meta agent
            
        Returns:
            Dictionary with success status and details
        """
        if self.criteria_type == "credit_card_rules":
            return self._check_credit_card_rules_success(meta_agent_context)
        else:
            logger.warning(f"Unknown criteria type: {self.criteria_type}")
            return {"success": False, "reason": f"Unknown criteria type: {self.criteria_type}"}
    
    def _check_credit_card_rules_success(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check success criteria for credit card rules discovery.
        
        The criteria are:
        1. Rules have been extracted
        2. Rules have been validated
        3. Validation accuracy is 100%
        
        Args:
            context: Current context from the meta agent
            
        Returns:
            Dictionary with success status and details
        """
        # Check if rules have been extracted
        ruleset = None
        validation_results = None
        
        # Find the most recent rule extraction result
        for key, value in reversed(list(context.items())):
            if isinstance(value, dict) and value.get("agent_name") == "Rule Extractor":
                if "result" in value and "ruleset" in value["result"]:
                    ruleset = value["result"]["ruleset"]
                    break
        
        # Find the most recent validation result
        for key, value in reversed(list(context.items())):
            if isinstance(value, dict) and value.get("agent_name") == "Validator":
                if "result" in value and "validation_results" in value["result"]:
                    validation_results = value["result"]["validation_results"]
                    break
        
        # Check success criteria
        if not ruleset:
            return {"success": False, "reason": "No rules have been extracted yet"}
        
        if not validation_results:
            return {"success": False, "reason": "Rules have not been validated yet"}
        
        # Check accuracy
        accuracy = validation_results.get("calculated_accuracy")
        if accuracy is None:
            # Try to get from validation_results directly
            accuracy_str = validation_results.get("accuracy", "0%")
            try:
                accuracy = float(accuracy_str.strip('%')) / 100
            except:
                accuracy = 0
        
        # Success criteria: 100% accuracy
        if accuracy >= 1.0:
            return {
                "success": True, 
                "reason": "Rules have been validated with 100% accuracy",
                "ruleset": ruleset,
                "validation_results": validation_results
            }
        else:
            return {
                "success": False, 
                "reason": f"Rules accuracy is {accuracy*100:.2f}%, below the required 100%",
                "ruleset": ruleset,
                "validation_results": validation_results,
                "accuracy": accuracy
            }
