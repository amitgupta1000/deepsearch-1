# graph.py
# This file defines the LangGraph workflow.
from .logging_setup import logger

# Try to import LangGraph; provide a minimal fallback for static linting/runtime without the package
try:
    from langgraph.graph import StateGraph, START, END, add_messages
except (ImportError, Exception):
    logging.warning("langgraph.graph not available; using fallback StateGraph for static checks.")
    # Minimal fallback implementation so other modules can import graph during lint/static analysis
    START = "__START__"
    END = "__END__"
    add_messages = None
    class StateGraph:
        def __init__(self, state_type=None):
            self.nodes = {}
        def add_node(self, name, func):
            self.nodes[name] = func
        def add_edge(self, a, b):
            pass
        def add_conditional_edges(self, node, route_fn, mapping):
            pass
        def compile(self):
            logger.info("Fallback StateGraph.compile() called — no-op.")
            return None

from typing import TypedDict, Optional, List, Dict, Any # Import necessary types
from .conditions import (
    route_ai_evaluate,
    route_retrieval_method # Import the new router
)
# Import nodes and AgentState from nodes.py
try:
    from .nodes import (
        AgentState,
        create_queries,
        fast_search_results_to_final_urls,
        extract_content,
        embed_and_retrieve,
        fss_retrieve,
        create_qa_pairs,
        AI_evaluate,
        write_report,
    )
except ImportError as e:
    import logging
    logger.exception("Error importing nodes: %s. Cannot define graph.", e)
    
# Initialize StateGraph
workflow = StateGraph(AgentState)

# Add nodes
# Assuming all imported node functions are async
if 'create_queries' in locals():
    workflow.add_node("create_queries", create_queries)
if 'fast_search_results_to_final_urls' in locals():
    workflow.add_node("fast_search_results_to_final_urls", fast_search_results_to_final_urls)
if 'extract_content' in locals():
    workflow.add_node("extract_content", extract_content)
if 'embed_and_retrieve' in locals():
    workflow.add_node("embed_and_retrieve", embed_and_retrieve)
if 'fss_retrieve' in locals():
    workflow.add_node("fss_retrieve", fss_retrieve)
if 'create_qa_pairs' in locals():
    workflow.add_node("create_qa_pairs", create_qa_pairs)
if 'AI_evaluate' in locals():
    workflow.add_node("AI_evaluate", AI_evaluate)
if 'write_report' in locals():
    workflow.add_node("write_report", write_report)

# Add edges - check if nodes were successfully added before adding edges

# FSS-only workflow: create_queries -> fast_search_results_to_final_urls -> extract_content -> fss_retrieve -> write_report -> END
if all(node_name in workflow.nodes for node_name in ["create_queries", "fast_search_results_to_final_urls", "extract_content", "fss_retrieve", "write_report"]):
    workflow.add_edge(START, "create_queries")
    workflow.add_edge("create_queries", "fast_search_results_to_final_urls")
    workflow.add_edge("fast_search_results_to_final_urls", "extract_content")
    workflow.add_edge("extract_content", "fss_retrieve")
    workflow.add_edge("fss_retrieve", "write_report")
    workflow.add_edge("write_report", END)


# Compile the workflow if nodes were successfully added
try:
    app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
except Exception as e:
    logger.exception("Error compiling LangGraph workflow: %s", e)
    app = None # Set app to None if compilation fails