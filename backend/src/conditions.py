# conditions.py
import logging

# --- Search Loop Logic ---
def should_continue_search(state: dict, max_loops: int = 3) -> bool:
    """
    Determines if search refinement should continue.
    """
    return state.get("proceed", False) and state.get("search_iteration_count", 0) < max_loops

def should_terminate_search(state: dict, max_loops: int = 3) -> bool:
    """
    Determines if search refinement should end and write report.
    """
    return not state.get("proceed", False) or state.get("search_iteration_count", 0) >= max_loops

def route_ai_evaluate(state: dict) -> str:
    """
    Routes AI evaluation result to the appropriate next node.
    Uses MAX_AI_ITERATIONS from config for iteration cap.
    """
    # Use dynamic config from state, fallback to global config
    # This makes the routing respect the search_mode
    max_loops = state.get("max_ai_iterations")
    if max_loops is None:
        from .config import MAX_AI_ITERATIONS
        max_loops = MAX_AI_ITERATIONS

    if state.get("search_iteration_count", 0) >= max_loops:
        return "write_report"
    if state.get("proceed", True):
        return "write_report"
    else:
        return "create_queries"


def route_retrieval_method(state: dict) -> str:
    """
    Determines the next node based on the chosen retrieval method.

    Args:
        state (dict): The current graph state.

    Returns:
        str: The name of the next node.
    """
    retrieval_method = state.get("retrieval_method", "hybrid").lower() # Ensure case-insensitivity
    logging.info(f"Routing after extract_content, method: '{retrieval_method}'")
    if retrieval_method == "file_search":
        route = "fss_retrieve"
        logging.info(f"Decision: Routing to '{route}' for file-based retrieval.")
        return route
    else: # Default to hybrid path
        route = "embed_and_retrieve"
        logging.info(f"Decision: Routing to '{route}' for hybrid retrieval.")
        return route