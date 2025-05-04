import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Meta agent settings
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
DEFAULT_TASK_PRIORITY = 5

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Task types
TASK_TYPES = {
    "task_decomposition": "Breaking down a complex task into smaller, manageable subtasks",
    "schema_design": "Creating JSON schemas for data structures",
    "data_generation": "Generating sample data based on schemas",
    "data_analysis": "Analyzing data to identify patterns and insights",
    "rule_extraction": "Extracting rules from data patterns",
    "validation": "Validating results against criteria",
    "programming": "Writing code to implement solutions",
}

# Expert capabilities
EXPERT_CAPABILITIES = {
    "task_decomposer": ["task_decomposition"],
    "schema_designer": ["schema_design"],
    "data_generator": ["data_generation"],
    "data_analyzer": ["data_analysis"],
    "rule_extractor": ["rule_extraction", "data_analysis"],
    "validator": ["validation"],
    "programmer": ["programming"],
}

# Data paths
DATA_DIR = "data"
SCHEMA_DIR = os.path.join(DATA_DIR, "schema")
APPLICATIONS_DIR = os.path.join(DATA_DIR, "applications")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
