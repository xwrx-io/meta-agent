from setuptools import setup, find_packages

setup(
    name="meta_agent_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pydantic>=2.0.0",
        "psutil>=5.9.0"
    ],
    author="Meta Agent Team",
    description="A framework for solving complex problems using a meta-agent approach",
    python_requires=">=3.8",
)
