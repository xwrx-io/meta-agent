from typing import List, Dict, Any, Optional
from meta_agent_system.core.expert_agent import ExpertAgent
from meta_agent_system.core.expert_factory import ExpertFactory
from meta_agent_system.llm.openai_client import OpenAIClient
from meta_agent_system.utils.logger import get_logger
from meta_agent_system.config.settings import RESULTS_DIR
import os
import json
import time
from colorama import Fore, Style

logger = get_logger(__name__)

class ExpertManager:
    """
    Manages the creation and coordination of dynamic expert agents.
    
    This class handles:
    1. Creating experts from recommendations
    2. Collecting insights from experts
    3. Tracking expert contributions to accuracy improvements
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """Initialize the expert manager with an OpenAI client"""
        self.openai_client = openai_client
        self.factory = ExpertFactory(openai_client)
        self.dynamic_experts = []
        self.expert_contributions = {}
        
    def create_experts_from_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[ExpertAgent]:
        """
        Create dynamic expert agents from expertise recommendations
        
        Args:
            recommendations: List of expert recommendation objects
            
        Returns:
            List of created expert agents
        """
        # Basic info at INFO level
        logger.info(f"Creating dynamic experts from {len(recommendations)} recommendations")
        # Detailed info only at DEBUG level
        logger.debug(f"Recommendation details: {recommendations}")
        
        # Create a recommendation format expected by factory
        formatted_recs = {"recommended_experts": recommendations}
        
        # Create the experts
        self.dynamic_experts = self.factory.create_experts_from_recommendations(formatted_recs)
        
        # Log created experts - keep it simple at INFO level
        expert_names = [expert.name for expert in self.dynamic_experts]
        if expert_names:
            logger.info(f"Created experts: {', '.join(expert_names)}")
        else:
            logger.info("No experts were created")
            
        return self.dynamic_experts
    
    def gather_expert_insights(self, 
                              iteration: int, 
                              current_ruleset: Dict[str, Any],
                              validation_result: Dict[str, Any],
                              applications: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Gather insights from all dynamic experts
        
        Args:
            iteration: Current iteration number
            current_ruleset: Current ruleset being evaluated
            validation_result: Results from the validation step
            applications: Application data (optional)
            
        Returns:
            List of expert insights
        """
        if not self.dynamic_experts:
            logger.info("No dynamic experts available to provide insights")
            return []
            
        # Just log the count at INFO level
        logger.info(f"Gathering insights from {len(self.dynamic_experts)} dynamic experts")
        
        insights = []
        for expert in self.dynamic_experts:
            # Basic info at INFO level
            logger.info(f"Requesting insight from {expert.name}")
            
            # Prepare task data for this expert
            task_data = {
                "description": f"Analyze credit card applications as {expert.name}",
                "data": {
                    "current_ruleset": current_ruleset,
                    "validation_result": validation_result,
                    "iteration": iteration
                }
            }
            
            # More detailed info at DEBUG level
            logger.debug(f"Task data for {expert.name}: {task_data}")
            
            # Add applications data if available
            if applications:
                task_data["data"]["applications"] = applications
                
            # Execute the expert to get insight
            try:
                result = expert.execute(task_data)
                
                if result.get("status") == "success":
                    insight = {
                        "expert": expert.name,
                        "timestamp": int(time.time()),
                        "insight": result.get("result", {})
                    }
                    insights.append(insight)
                    logger.info(f"Received insight from {expert.name}")
                    logger.debug(f"Insight content from {expert.name}: {result.get('result', {})}")
                else:
                    logger.warning(f"Expert {expert.name} failed to provide insight: {result.get('message', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error getting insight from {expert.name}: {str(e)}")
        
        # Save all insights to file
        if insights:
            self._save_expert_insights(insights, iteration)
            
        return insights
    
    def _save_expert_insights(self, insights: List[Dict[str, Any]], iteration: int):
        """Save expert insights to a file"""
        insights_file = os.path.join(RESULTS_DIR, f"expert_insights_iteration_{iteration}.json")
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)
        logger.info(f"Saved {len(insights)} expert insights to {insights_file}")
    
    def record_expert_contribution(self, 
                                 iteration: int, 
                                 current_accuracy: float, 
                                 previous_accuracy: float,
                                 expert_insights: List[Dict[str, Any]]):
        """
        Record which experts contributed to accuracy improvement
        
        Args:
            iteration: Current iteration
            current_accuracy: Current accuracy level
            previous_accuracy: Previous accuracy level
            expert_insights: Expert insights used for this iteration
        """
        if current_accuracy <= previous_accuracy or not expert_insights:
            return
            
        improvement = current_accuracy - previous_accuracy
        experts_used = [insight['expert'] for insight in expert_insights]
        
        contribution = {
            "iteration": iteration,
            "improvement": improvement,
            "previous_accuracy": previous_accuracy,
            "new_accuracy": current_accuracy,
            "contributing_experts": experts_used
        }
        
        # Save to file
        contribution_file = os.path.join(RESULTS_DIR, f"expert_contributions_iteration_{iteration}.json")
        with open(contribution_file, 'w') as f:
            json.dump(contribution, f, indent=2)
            
        logger.info(f"Recorded expert contributions for iteration {iteration} with {improvement:.2f}% improvement")
        
        # Update internal tracking
        self.expert_contributions[iteration] = contribution
    
    def get_experts_summary(self) -> str:
        """
        Get a summary of dynamic experts and their contributions
        
        Returns:
            Formatted summary string
        """
        if not self.dynamic_experts:
            return f"{Fore.YELLOW}No dynamic experts were created{Style.RESET_ALL}"
            
        summary = [
            f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}",
            f"{Fore.YELLOW}DYNAMIC EXPERTS SUMMARY{Style.RESET_ALL}",
            f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}\n"
        ]
        
        # Summarize experts
        summary.append(f"{Fore.GREEN}Created {len(self.dynamic_experts)} dynamic experts:{Style.RESET_ALL}\n")
        
        for i, expert in enumerate(self.dynamic_experts, 1):
            summary.append(f"{Fore.YELLOW}{i}. {expert.name}{Style.RESET_ALL}")
            summary.append(f"   {Fore.WHITE}Capabilities: {', '.join(expert.capabilities)}{Style.RESET_ALL}")
            
        # Summarize contributions if any
        if self.expert_contributions:
            summary.append(f"\n{Fore.GREEN}Expert Contributions:{Style.RESET_ALL}\n")
            
            # Sort contributions by improvement amount
            sorted_contribs = sorted(
                self.expert_contributions.values(), 
                key=lambda x: x.get("improvement", 0), 
                reverse=True
            )
            
            for contrib in sorted_contribs:
                iter_num = contrib.get("iteration", "?")
                improvement = contrib.get("improvement", 0)
                experts = contrib.get("contributing_experts", [])
                
                summary.append(f"{Fore.CYAN}Iteration {iter_num}:{Style.RESET_ALL} "
                              f"{Fore.GREEN}+{improvement:.2f}%{Style.RESET_ALL} "
                              f"{Fore.WHITE}improvement with input from: "
                              f"{Fore.YELLOW}{', '.join(experts)}{Style.RESET_ALL}")
        
        summary.append(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        
        return "\n".join(summary)
