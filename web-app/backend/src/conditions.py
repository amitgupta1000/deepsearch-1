# conditions.py

# --- Search Loop Logic ---
def should_continue_search(state: dict, max_loops: int = 5) -> bool:
    """
    Determines if search refinement should continue.
    """
    return state.get("proceed", False) and state.get("search_iteration_count", 0) < max_loops

def should_terminate_search(state: dict, max_loops: int = 5) -> bool:
    """
    Determines if search refinement should end and write report.
    """
    return not state.get("proceed", False) or state.get("search_iteration_count", 0) >= max_loops

# --- Approval Loop Logic ---
def should_continue_approval(state: dict, max_loops: int = 3) -> bool:
    """
    Determines if approval loop should continue.
    """
    return state.get("proceed", False) and state.get("approval_iteration_count", 0) < max_loops

def should_terminate_approval(state: dict, max_loops: int = 3) -> bool:
    """
    Determines if approval loop should terminate and skip to report.
    """
    return not state.get("proceed", False) or state.get("approval_iteration_count", 0) >= max_loops


def route_user_approval(state: dict) -> str:
    """
    Routes user approval state to the appropriate next node.
    """
    max_loops = 3  # You can make this configurable

    if state.get("approval_iteration_count", 0) >= max_loops:
        return "choose_report_type"
    elif state.get("proceed", False):
        return "evaluate_search_results"
    else:
        return "create_queries"

def route_ai_evaluate(state: dict) -> str:
    """
    Routes AI evaluation result to the appropriate next node.
    """
    max_loops = 5  # You can make this configurable

    if state.get("search_iteration_count", 0) >= max_loops:
        return "choose_report_type"
    elif state.get("proceed", False):
        return "choose_report_type"
    else:
        return "evaluate_search_results"

