import os
import sys
import json
import time
import unittest
from dotenv import load_dotenv

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.core.meta_agent import MetaAgent
from meta_agent_system.core.expert_factory import ExpertFactory
from meta_agent_system.core.success_criteria import SuccessCriteriaChecker
from meta_agent_system.core.feedback_loop import FeedbackLoop
from meta_agent_system.experts.task_decomposer import create_task_decomposer
from meta_agent_system.experts.schema_designer import create_schema_designer
from meta_agent_system.experts.data_generator import create_data_generator
from meta_agent_system.experts.data_analyzer import create_data_analyzer
from meta_agent_system.experts.rule_extractor import create_rule_extractor
from meta_agent_system.experts.rule_refiner import create_rule_refiner
from meta_agent_system.experts.validator import create_validator
from meta_agent_system.experts.expertise_recommender import create_expertise_recommender
from meta_agent_system.utils.helpers import ensure_directory_exists
from meta_agent_system.config.settings import RESULTS_DIR, APPLICATIONS_DIR, SCHEMA_DIR

class TestEndToEnd(unittest.TestCase):
    """Test the entire meta agent system end-to-end"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Load environment variables
        load_dotenv()
        
        # Ensure API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Create necessary directories
        ensure_directory_exists(RESULTS_DIR)
        ensure_directory_exists(APPLICATIONS_DIR)
        ensure_directory_exists(SCHEMA_DIR)
        
        # Initialize OpenAI client
        cls.openai_client = OpenAIClient()
        
    def test_credit_card_rules_discovery(self):
        """Test the credit card rules discovery problem end-to-end"""
        # Start timing
        start_time = time.time()
        
        # Create meta agent
        meta = MetaAgent("Test Meta Agent")
        
        # Create expert factory
        expert_factory = ExpertFactory(llm_client=self.openai_client)
        meta.expert_factory = expert_factory
        
        # Register core experts
        self._register_experts(meta)
        
        # Set up feedback loop
        success_checker = SuccessCriteriaChecker(criteria_type="credit_card_rules")
        feedback_loop = FeedbackLoop(success_checker)
        meta.feedback_loop = feedback_loop
        
        # Define the problem
        problem = self._get_credit_card_problem()
        
        print("\nStarting credit card rules discovery test...")
        
        # Solve the problem
        max_iterations = 25  # Allow more iterations for testing
        results = meta.solve(problem, max_iterations=max_iterations)
        
        # End timing
        elapsed_time = time.time() - start_time
        
        # Print results
        print(f"\nTest completed in {elapsed_time:.2f} seconds")
        print(f"Status: {results['status']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Tasks completed: {results['tasks_completed']}")
        
        # Verify results
        self.assertEqual(results["status"], "completed", "Problem should be solved completely")
        
        # Check for ruleset and validation in results
        ruleset = None
        validation = None
        
        for key, value in results["context"].items():
            if isinstance(value, dict):
                if value.get("agent_name") == "Rule Extractor" or value.get("agent_name") == "Rule Refiner":
                    if "result" in value and "ruleset" in value["result"]:
                        ruleset = value["result"]["ruleset"]
                
                if value.get("agent_name") == "Validator":
                    if "result" in value and "validation_results" in value["result"]:
                        validation = value["result"]["validation_results"]
        
        # Verify ruleset exists
        self.assertIsNotNone(ruleset, "Should have extracted rules")
        
        # Verify validation exists
        self.assertIsNotNone(validation, "Should have validation results")
        
        # Check if a high accuracy was achieved
        if "calculated_accuracy" in validation:
            accuracy = validation["calculated_accuracy"]
            print(f"Achieved accuracy: {accuracy * 100:.2f}%")
            self.assertGreaterEqual(accuracy, 0.9, "Should achieve at least 90% accuracy")
    
    def _register_experts(self, meta):
        """Register all experts with the meta agent"""
        # Create and register experts
        experts = [
            create_task_decomposer(self.openai_client),
            create_schema_designer(self.openai_client),
            create_data_generator(self.openai_client),
            create_data_analyzer(self.openai_client),
            create_rule_extractor(self.openai_client),
            create_rule_refiner(self.openai_client),
            create_validator(self.openai_client),
            create_expertise_recommender(self.openai_client)
        ]
        
        for expert in experts:
            meta.register_agent(expert)
    
    def _get_credit_card_problem(self):
        """Get the credit card ruleset problem"""
        return {
            "description": "Discover credit card application approval rules from sample applications",
            "type": "task_decomposition",
            "data": {
                "context": """
We need to analyze credit card applications to discover the ruleset used for approving or declining applications.
We have 20 credit card applications, 10 of which were approved and 10 of which were declined.
The meta agent system must discover the rules used for approval or rejection without being told which applications were approved or declined.
                """,
                "requirements": [
                    "Create a JSON schema for credit card applications",
                    "Generate 20 diverse credit card applications (10 approved, 10 declined)",
                    "Analyze the applications to discover patterns",
                    "Extract rules that determine approval or rejection",
                    "Validate the rules by applying them to all 20 applications"
                ],
                "constraints": [
                    "The meta agent must not know in advance which applications were approved or declined",
                    "The rules must be clear and understandable",
                    "The rules must achieve 100% accuracy on the 20 applications"
                ]
            }
        }

if __name__ == "__main__":
    unittest.main()
