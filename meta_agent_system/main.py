import os
import sys
import json
from dotenv import load_dotenv
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.utils.helpers import save_json, get_timestamp, ensure_directory_exists
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.core.meta_agent import MetaAgent
from meta_agent_system.core.expert_factory import ExpertFactory
from meta_agent_system.experts.task_decomposer import create_task_decomposer
from meta_agent_system.config.settings import RESULTS_DIR
from meta_agent_system.experts.schema_designer import create_schema_designer
from meta_agent_system.experts.data_generator import create_data_generator
from meta_agent_system.experts.data_analyzer import create_data_analyzer
from meta_agent_system.experts.rule_extractor import create_rule_extractor
from meta_agent_system.experts.validator import create_validator
from meta_agent_system.core.success_criteria import SuccessCriteriaChecker
from meta_agent_system.core.feedback_loop import FeedbackLoop
from meta_agent_system.experts.rule_refiner import create_rule_refiner
from meta_agent_system.experts.expertise_recommender import create_expertise_recommender

# Configure logging
logger = get_logger(__name__)

def main():
    """Main entry point for meta agent system"""
    # Load environment variables
    load_dotenv()
    
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    print("\n=== Meta Agent System ===\n")
    print("Initializing system...")
    
    # Initialize OpenAI client
    openai_client = OpenAIClient()
    
    # Create expert factory
    expert_factory = ExpertFactory(llm_client=openai_client)
    
    # Create meta agent
    meta = MetaAgent("Dynamic Meta Agent")
    
    # Attach expert factory to meta agent
    meta.expert_factory = expert_factory
    
    # Register task decomposer expert
    task_decomposer = create_task_decomposer(openai_client)
    meta.register_agent(task_decomposer)
    
    # Register core experts in the main function
    schema_designer = create_schema_designer(openai_client)
    data_generator = create_data_generator(openai_client)
    data_analyzer = create_data_analyzer(openai_client)
    rule_extractor = create_rule_extractor(openai_client)
    validator = create_validator(openai_client)
    
    meta.register_agent(schema_designer)
    meta.register_agent(data_generator)
    meta.register_agent(data_analyzer)
    meta.register_agent(rule_extractor)
    meta.register_agent(validator)
    
    # Create and register the rule refiner
    rule_refiner = create_rule_refiner(openai_client)
    meta.register_agent(rule_refiner)
    
    # Create success criteria checker and feedback loop
    success_checker = SuccessCriteriaChecker(criteria_type="credit_card_rules")
    feedback_loop = FeedbackLoop(success_checker)
    
    # Attach feedback loop to meta agent
    meta.feedback_loop = feedback_loop
    
    # Define problem
    problem = get_credit_card_ruleset_problem()
    
    # Display problem info
    print(f"\nProblem: {problem['description']}")
    print("\nStarting execution...\n")
    
    # Solve the problem
    results = meta.solve(problem, max_iterations=30)
    
    # Display summary
    print("\n=== Execution Summary ===")
    print(f"Status: {results['status']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Tasks completed: {results['tasks_completed']}")
    print(f"Tasks remaining: {results['tasks_remaining']}")
    
    # Save results
    timestamp = get_timestamp()
    results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    ensure_directory_exists(RESULTS_DIR)
    save_json(results, results_file)
    
    print(f"\nResults saved to {results_file}")
    print("\nMeta Agent execution complete!")

    # Add this after registering other experts:
    expertise_recommender = create_expertise_recommender(openai_client)
    meta.register_agent(expertise_recommender)

def get_credit_card_ruleset_problem():
    """Define the credit card ruleset problem"""
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
    main()
