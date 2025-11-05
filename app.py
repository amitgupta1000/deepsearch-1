# app.py - INTELLISEARCH Main Application
# Unified interface supporting both interactive and automated modes

import asyncio
import logging
import argparse
from typing import Dict, Any, List

# Basic logging configuration in case setup.py didn't configure it
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Startup validation
def validate_startup():
    """Quick startup validation to ensure critical imports work."""
    try:
        # Test critical imports
        from src.config import validate_config
        from src.graph import app
        
        if not validate_config():
            logging.warning("Configuration validation failed - check .env file")
            
        if app is None:
            logging.error("LangGraph app failed to compile - check LangChain imports")
            return False
            
        return True
        
    except ImportError as e:
        logging.error(f"Critical import failed: {e}")
        logging.error("Run 'python startup_validation.py' for detailed diagnostics")
        return False
    except Exception as e:
        logging.error(f"Startup validation error: {e}")
        return False

# RunnableConfig is optional (depends on langchain_core availability)
try:
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(recursion_limit=100)
except Exception:
    logging.debug("langchain_core.runnables not available; continuing without RunnableConfig.")
    config = None

# Import necessary components from your modules
try:
    import setup
    logging.info("setup.py imported and likely executed initial setup.")
except ImportError as e:
    logging.error(f"Could not import setup.py: {e}. Initial setup may be incomplete.")
    # Basic logging config if setup failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Import the compiled LangGraph application
    from src.graph import app 
except ImportError as e:
    logging.error(f"Could not import the LangGraph app from graph.py: {e}. Cannot run the workflow.")
    app = None # Set app to None if import fails

try:
    # Import AgentState for type hinting initial state
    from src.nodes import AgentState
except ImportError as e:
    logging.error(f"Could not import AgentState from nodes.py: {e}. Type hinting for initial state may be missing.")
    # Define a dummy AgentState if import fails, or handle gracefully
    AgentState = Dict[str, Any] # Fallback to a generic dictionary type hint

try:
    # Import automation configuration
    from src.automation_config import AutomationConfig, get_automation_config
except ImportError as e:
    logging.error(f"Could not import automation configuration: {e}. Automation features disabled.")
    AutomationConfig = None
    get_automation_config = None


async def run_workflow(
    initial_query: str, 
    prompt_type: str, 
    enable_automation: bool = False,
    automation_profile: str = "full"
):
    """
    Runs the LangGraph workflow with an initial query.
    Supports both interactive and automated modes.

    Args:
        initial_query: The user's initial research query.
        prompt_type: Type of prompt to use.
        enable_automation: If True, run with full automation (no user input).
        automation_profile: Automation profile ('full', 'query_only', 'none').
        report_type: Type of report to generate ('concise' or 'detailed').
    """
    if app is None:
        logging.error("LangGraph app is not compiled or imported. Cannot run workflow.")
        print("Workflow cannot be run due to errors in graph compilation or imports.")
        return None

    mode_text = "AUTOMATED" if enable_automation else "INTERACTIVE"
    logging.info(f"Starting {mode_text} workflow for query: '{initial_query}'")
    
    if enable_automation:
        logging.info(f"Configuration: prompt_type={prompt_type}, automation={automation_profile}")

    # Get automation configuration if enabled
    automation_config = None
    if enable_automation and get_automation_config:
        try:
            automation_config = get_automation_config(automation_profile)
        except Exception as e:
            logging.warning(f"Could not load automation profile '{automation_profile}': {e}")

    # Define the initial state for the graph
    initial_state = {
        "new_query": initial_query,
        "search_queries": [],
        "rationale": None,
        "data": [],
        "relevant_contexts": {},
        "relevant_chunks": [],
        "proceed": True, # Start by proceeding to query generation
        "visited_urls": [],
        "failed_urls": [],
        "iteration_count": 0,
        "report": None,
        "report_filename": "IntelliSearchReport",
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "prompt_type": prompt_type, # Added prompt type to the state
        "approval_iteration_count": 0,  # Counts loops between user_approval ↔ create_queries
        "search_iteration_count": 0,  # Counts loops from AI_evaluate ↔ evaluate_search_results    
        "report_type": None,  # Will be set by choose_report_type node
        
        # Automation flags (set based on enable_automation parameter)
        "non_interactive": enable_automation,
        "auto_approve": enable_automation,
        "approval_choice": "yes" if enable_automation else None,
        "auto_report_type": "unified" if enable_automation else None,
        "report_type_choice": "unified" if enable_automation else None,
        "new_query_override": None
    }

    # Apply automation configuration if available
    if automation_config:
        automation_state = automation_config.to_state_dict()
        initial_state.update(automation_state)
        logging.info(f"Applied automation configuration: {automation_state}")

    # Run the compiled workflow
    try:
        if config is not None:
            astream = app.astream(initial_state, config=config)
        else:
            astream = app.astream(initial_state)

        # Track execution progress
        executed_nodes = []
        async for step in astream:
             for key, value in step.items():
                 logging.info("Node executed: %s", key)
                 executed_nodes.append(key)

        logging.info(f"{mode_text} workflow finished successfully.")
        if enable_automation:
            logging.info(f"Executed nodes: {' → '.join(executed_nodes)}")
            
        final_report_filename = initial_state.get("report_filename", "No report file generated.")
        logging.info("Check for report files: %s and %s", 
                    f"{final_report_filename}.txt", 
                    (setup.REPORT_FILENAME_PDF if hasattr(setup, 'REPORT_FILENAME_PDF') else 'CrystalSearchReport.pdf')) 

        # Check for any errors in the final state
        final_error_state = initial_state.get('error')
        if final_error_state:
             logging.warning("Workflow completed with errors: %s", final_error_state)
        else:
             logging.info("Workflow completed successfully without errors.")

        return initial_state

    except Exception as e:
        logging.exception(f"An error occurred during workflow execution: {e}")
        return None


async def batch_research(queries: List[str], **kwargs):
    """
    Run multiple research queries in batch.
    
    Args:
        queries: List of research queries to process
        **kwargs: Additional arguments to pass to run_workflow
    """
    results = []
    
    for i, query in enumerate(queries, 1):
        logging.info(f"Processing batch query {i}/{len(queries)}: {query}")
        
        try:
            result = await run_workflow(query, **kwargs)
            results.append({
                "query": query,
                "success": result is not None,
                "result": result
            })
            
            if result:
                logging.info(f"Batch query {i} completed successfully")
            else:
                logging.warning(f"Batch query {i} failed")
                
        except Exception as e:
            logging.error(f"Error processing batch query {i}: {e}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    logging.info(f"Batch processing completed: {successful}/{len(queries)} queries successful")
    
    return results


def interactive_mode():
    """
    Run the application in interactive mode with user prompts.
    """
    print("INTELLISEARCH Research Tool")
    print("=" * 50)
    
    # Ask about automation mode
    automation_choice = input("Enable automation (skip all user prompts)? (y/n): ").lower().strip()
    enable_automation = automation_choice in ['y', 'yes', '1', 'true']
    
    if enable_automation:
        print("🤖 AUTOMATION ENABLED - No user prompts will be shown during workflow")
    else:
        print("👤 INTERACTIVE MODE - You will be prompted for approvals during workflow")
    
    # Get research query
    user_research_query = input("\nEnter your Research Query: ")

    # Get prompt type from user
    print("\nSelect prompt type:")
    print("1: Legal")
    print("2: General") 
    print("3: Macro")
    print("4: DeepSearch")
    print("5: Person Search")
    print("6: Investment Research")
    prompt_type_choice = input("Enter the number for your desired prompt type: ")

    # Map user choice to prompt type string
    prompt_type_mapping = {
        "1": "legal",
        "2": "general",
        "3": "macro",
        "4": "deepsearch",
        "5": "person_search",
        "6": "investment"
    }
    selected_prompt_type = prompt_type_mapping.get(prompt_type_choice, "general")
    print(f"Selected prompt type: {selected_prompt_type}")

    if enable_automation:
        print(f"\n🚀 Starting automated workflow...")
        print(f"   Query: {user_research_query}")
        print(f"   Prompt Type: {selected_prompt_type}")
        print(f"   Report Format: Unified (500-2000 words)")
        print(f"   All queries will be auto-approved")

    # Run the async workflow
    return asyncio.run(run_workflow(
        user_research_query, 
        selected_prompt_type, 
        enable_automation
    ))


def command_line_mode():
    """
    Run the application in command-line mode with argument parsing.
    """
    parser = argparse.ArgumentParser(description="INTELLISEARCH Research Tool")
    parser.add_argument("query", nargs='?', help="Research query to process")
    parser.add_argument("--prompt-type", 
                       choices=["legal", "general", "macro", "deepsearch", "person_search", "investment"],
                       default="general", help="Prompt type (default: general)")
    parser.add_argument("--automation", choices=["full", "query_only", "none"],
                       default="full", help="Automation profile (default: full)")
    parser.add_argument("--batch-file", help="Text file with one query per line for batch processing")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # If no query and no batch file, or interactive flag is set, run interactive mode
    if (not args.query and not args.batch_file) or args.interactive:
        return interactive_mode()
    
    if args.batch_file:
        # Batch processing mode
        try:
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            logging.info(f"Loaded {len(queries)} queries from {args.batch_file}")
            
            return asyncio.run(batch_research(
                queries,
                prompt_type=args.prompt_type,
                enable_automation=True,
                automation_profile=args.automation
            ))
            
        except FileNotFoundError:
            logging.error(f"Batch file not found: {args.batch_file}")
            return None
        except Exception as e:
            logging.error(f"Error processing batch file: {e}")
            return None
    else:
        # Single query mode
        return asyncio.run(run_workflow(
            args.query,
            prompt_type=args.prompt_type,
            enable_automation=True,
            automation_profile=args.automation
        ))


# Quick research function for programmatic use
async def quick_research(query: str, **kwargs):
    """
    Quick research function for programmatic use.
    
    Args:
        query: Research question
        **kwargs: Additional configuration options
        
    Returns:
        Final state dictionary or None if failed
    """
    kwargs.setdefault('enable_automation', True)
    return await run_workflow(query, **kwargs)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Perform startup validation
    if not validate_startup():
        print("\n💥 Startup validation failed!")
        print("   Run 'python startup_validation.py' for detailed diagnostics")
        print("   Or run 'run_setup_and_interactive.bat' to fix environment")
        exit(1)
    
    command_line_mode()


