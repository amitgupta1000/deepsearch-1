# graph.py
# This file defines the LangGraph workflow.

import logging

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
            logging.info("Fallback StateGraph.compile() called — no-op.")
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
        evaluate_search_results,
        extract_content,
        embed_and_retrieve,
        fss_retrieve,
        create_qa_pairs,
        AI_evaluate,
        write_report,
    )
except ImportError as e:
    import logging
    logging.exception("Error importing nodes: %s. Cannot define graph.", e)
    
# Initialize StateGraph
workflow = StateGraph(AgentState)

# Add nodes
# Assuming all imported node functions are async
if 'create_queries' in locals():
    workflow.add_node("create_queries", create_queries)
if 'evaluate_search_results' in locals():
    workflow.add_node("evaluate_search_results", evaluate_search_results)
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
if all(node_name in workflow.nodes for node_name in ["create_queries", "evaluate_search_results"]):
    workflow.add_edge(START, "create_queries")
    workflow.add_edge("create_queries", "evaluate_search_results")

# Add sequential edges
if "evaluate_search_results" in workflow.nodes and "extract_content" in workflow.nodes:
    workflow.add_edge("evaluate_search_results", "extract_content")


workflow.add_conditional_edges(
    "extract_content",
    route_retrieval_method,
    {
        "embed_and_retrieve": "embed_and_retrieve",
        "fss_retrieve": "fss_retrieve"
    }
)

# After embed_and_retrieve, go to create_qa_pairs (hybrid path)
if "embed_and_retrieve" in workflow.nodes and "create_qa_pairs" in workflow.nodes:
    workflow.add_edge("embed_and_retrieve", "create_qa_pairs")

# After create_qa_pairs, go to AI_evaluate
if "create_qa_pairs" in workflow.nodes and "AI_evaluate" in workflow.nodes:
    workflow.add_edge("create_qa_pairs", "AI_evaluate")


# After fss_retrieve, go directly to write_report
if "fss_retrieve" in workflow.nodes and "write_report" in workflow.nodes:
    workflow.add_edge("fss_retrieve", "write_report")


# After AI_evaluate, route based on proceed and retrieval_method
workflow.add_conditional_edges(
    "AI_evaluate",
    lambda state: "write_report" if state.get("proceed", True) else ("fss_retrieve" if state.get("retrieval_method") == "fss_retriever" else "evaluate_search_results"),
    {
        "write_report": "write_report",
        "fss_retrieve": "fss_retrieve",
        "evaluate_search_results": "evaluate_search_results"
    }
)

# Edge to END from write_report
if "write_report" in workflow.nodes:
    workflow.add_edge("write_report", END)


# Compile the workflow if nodes were successfully added
try:
    app = workflow.compile()
    logging.info("LangGraph workflow compiled successfully.")
except Exception as e:
    logging.exception("Error compiling LangGraph workflow: %s", e)
    app = None # Set app to None if compilation fails