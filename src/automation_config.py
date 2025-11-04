# automation_config.py
# Configuration for automating user interactions in INTELLISEARCH

from typing import Dict, Any

class AutomationConfig:
    """
    Configuration class for automating INTELLISEARCH workflow.
    """
    
    def __init__(
        self,
        auto_approve_queries: bool = True,
        non_interactive: bool = True,
        approval_choice: str = "yes"  # "yes" or "no"
    ):
        """
        Initialize automation configuration.
        
        Args:
            auto_approve_queries: Automatically approve generated search queries
            non_interactive: Run in non-interactive mode (no user input prompts)
            approval_choice: Simulated user response for query approval ("yes" or "no")
        """
        self.auto_approve_queries = auto_approve_queries
        self.non_interactive = non_interactive
        self.approval_choice = approval_choice
    
    def to_state_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to state dictionary for AgentState.
        
        Returns:
            Dictionary with automation flags for AgentState
        """
        return {
            "auto_approve": self.auto_approve_queries,
            "non_interactive": self.non_interactive,
            "approval_choice": self.approval_choice
        }
    
    @classmethod
    def full_automation(cls) -> 'AutomationConfig':
        """
        Create configuration for full automation (no user input required).
        
        Returns:
            AutomationConfig with all automation features enabled
        """
        return cls(
            auto_approve_queries=True,
            non_interactive=True,
            approval_choice="yes"
        )
    
    @classmethod
    def query_automation_only(cls) -> 'AutomationConfig':
        """
        Create configuration that only automates query approval.
        
        Returns:
            AutomationConfig with only query approval automated
        """
        return cls(
            auto_approve_queries=True,
            non_interactive=False,
            approval_choice="yes"
        )
    
    @classmethod
    def no_automation(cls) -> 'AutomationConfig':
        """
        Create configuration with no automation (full user interaction).
        
        Returns:
            AutomationConfig with all automation disabled
        """
        return cls(
            auto_approve_queries=False,
            non_interactive=False,
            approval_choice=None
        )

# Pre-defined automation profiles
AUTOMATION_PROFILES = {
    "full": AutomationConfig.full_automation(),
    "query_only": AutomationConfig.query_automation_only(), 
    "none": AutomationConfig.no_automation()
}


def get_automation_config(profile: str = "full") -> AutomationConfig:
    """
    Get automation configuration by profile name.
    
    Args:
        profile: Profile name ("full", "query_only", "none")
        
    Returns:
        AutomationConfig instance
        
    Raises:
        ValueError: If profile name is not recognized
    """
    if profile not in AUTOMATION_PROFILES:
        available = ", ".join(AUTOMATION_PROFILES.keys())
        raise ValueError(f"Unknown automation profile '{profile}'. Available: {available}")
    
    return AUTOMATION_PROFILES[profile]


# Example usage:
if __name__ == "__main__":
    # Full automation example
    full_auto = AutomationConfig.full_automation()
    print("Full automation state:", full_auto.to_state_dict())
    
    # Query-only automation example  
    query_auto = AutomationConfig.query_automation_only()
    print("Query automation state:", query_auto.to_state_dict())
    
    # No automation example
    no_auto = AutomationConfig.no_automation()
    print("No automation state:", no_auto.to_state_dict())