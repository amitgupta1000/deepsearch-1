# nodes.py
# This file contains the individual nodes (functions) for the LangGraph workflow.
# Using llm call from Langchain-ChatGoogleGenerativeAI for generating search strings, evaluations etc., and Google GenAI for report writing for verbosity

import logging
import json
import re
import asyncio
from typing import Dict, Any, List, Optional

# Try to import optional dependencies with fallbacks
try:
    from pydantic import BaseModel, Field, ValidationError, conlist
    PYDANTIC_AVAILABLE = True
except ImportError:
    logging.warning("pydantic not available. Using basic classes instead.")
    # Create fallback classes
    class BaseModel:
        pass
    Field = lambda **kwargs: None
    ValidationError = Exception
    conlist = lambda type_, **kwargs: List[type_]
    PYDANTIC_AVAILABLE = False

try:
    from langchain_core.documents import Document
    LANGCHAIN_DOCS_AVAILABLE = True
except ImportError:
    logging.warning("langchain_core.documents not available. Using basic Document class.")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    LANGCHAIN_DOCS_AVAILABLE = False

try:
    from langchain_community.document_loaders import PyMuPDFLoader, RecursiveUrlLoader, CSVLoader
    LANGCHAIN_LOADERS_AVAILABLE = True
except ImportError:
    logging.warning("langchain_community.document_loaders not available. Some features may be limited.")
    PyMuPDFLoader = RecursiveUrlLoader = CSVLoader = None
    LANGCHAIN_LOADERS_AVAILABLE = False

try:
    from langchain_core.messages import (
        AnyMessage,
        AIMessage,
        SystemMessage,
        HumanMessage,
        ToolMessage,
    )
    LANGCHAIN_MESSAGES_AVAILABLE = True
except ImportError:
    logging.warning("langchain_core.messages not available. Using basic message classes.")
    class AnyMessage: pass
    class AIMessage: pass
    class SystemMessage: pass
    class HumanMessage: pass
    class ToolMessage: pass
    LANGCHAIN_MESSAGES_AVAILABLE = False


# Import necessary classes and functions from other modules
try:
    from .llm_calling import llm, llm_call_async, embeddings # Assuming these are initialized in llm_calling.py
except ImportError:
    logging.error("Could not import LLM/Embeddings from llm_calling. Some nodes may not function.")
    llm, llm_call_async, embeddings = None, None, None

try:
    from .search import UnifiedSearcher, SearchResult # Assuming SearchResult and UnifiedSearcher are in search.py
except ImportError:
    logging.error("Could not import search components from search.py. Search node will not function.")
    UnifiedSearcher, SearchResult = None, None

# Fallback to lightweight types if SearchResult or Document not available
try:
    from .data_types import SearchResult as TypesSearchResult, Document as TypesDocument
except Exception:
    TypesSearchResult, TypesDocument = None, None

if SearchResult is None and TypesSearchResult is not None:
    SearchResult = TypesSearchResult

# Ensure Document is available for typing (may be imported earlier)
try:
    Document
except NameError:
    Document = TypesDocument

try:
    from .scraper import Scraper, ScrapedContent # Assuming Scraper and ScrapedContent are in scraper.py
except ImportError:
    logging.error("Could not import scraper components from scraper.py. Extraction node will not function.")
    Scraper, ScrapedContent = None, None

try:
    from .hybrid_retriever import HybridRetriever, HybridRetrieverConfig, create_hybrid_retriever
except ImportError:
    logging.warning("Could not import hybrid retriever. Using fallback retrieval.")
    HybridRetriever, HybridRetrieverConfig, create_hybrid_retriever = None, None, None

try:
    from .utils import (
        safe_format, get_current_date, clean_extracted_text,
        fetch_pdf_content, rank_urls, save_report_to_text,
        generate_pdf_from_md # Import utility functions
    )
except ImportError:
    logging.error("Could not import utility functions from utils.py. Some nodes may be limited.")
    # Define dummy functions or handle missing utilities within nodes if necessary

# Note: question_analyzer module is not needed as LLM-based query generation 
# in create_queries() provides superior question analysis and query generation

try:
    from .config import(
            USE_PERSISTENCE,
            MAX_RESULTS,
            CACHE_TTL,
            CACHE_ENABLED,
            EMBEDDING_MODEL,
            REPORT_FORMAT,
            REPORT_FILENAME_PDF,
            REPORT_FILENAME_TEXT,
            MAX_SEARCH_QUERIES,
            MAX_CONCURRENT_SCRAPES,
            MAX_SEARCH_RETRIES,
            MAX_AI_ITERATIONS,
            MAX_USER_QUERY_LOOPS,
            DEFAULT_USER_AGENT,
            DEFAULT_REFERER,
            URL_TIMEOUT,
            SKIP_EXTENSIONS,
            BLOCKED_DOMAINS,
            YELLOW,
            ENDC,
            RED,
            GREEN,
            BLUE,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            # Hybrid retrieval configuration
            USE_HYBRID_RETRIEVAL,
            RETRIEVAL_TOP_K,
            HYBRID_VECTOR_WEIGHT,
            HYBRID_BM25_WEIGHT,
            HYBRID_FUSION_METHOD,
            HYBRID_RRF_K,
            VECTOR_SCORE_THRESHOLD,
            MIN_CHUNK_LENGTH,
            MIN_WORD_COUNT,
            USE_RERANKING,
            RERANKER_CANDIDATES_MULTIPLIER,
            # Enhanced embedding configuration
            USE_ENHANCED_EMBEDDINGS,
            EMBEDDING_TASK_TYPE,
            EMBEDDING_DIMENSIONALITY,
            EMBEDDING_NORMALIZE,
            EMBEDDING_BATCH_SIZE,
            # Multi-query retrieval settings
            USE_MULTI_QUERY_RETRIEVAL,
            MAX_RETRIEVAL_QUERIES,
            QUERY_CHUNK_DISTRIBUTION,
            # Enhanced deduplication configuration
            USE_LLM_DEDUPLICATION,
            LLM_DEDUP_DETAILED_ONLY
            )
    from .enhanced_deduplication import enhanced_deduplicate_content
except ImportError:
    logging.warning("Could not import config settings. Using defaults.")
    # Fallback defaults to ensure nodes run in degraded mode
    USE_PERSISTENCE = False
    MAX_RESULTS = 10
    CACHE_TTL = 3600
    CACHE_ENABLED = False
    EMBEDDING_MODEL = "text-embedding-3-small"
    REPORT_FORMAT = "md"
    REPORT_FILENAME_PDF = "CrystalSearchReport.pdf"
    REPORT_FILENAME_TEXT = "CrystalSearchReport.txt"
    MAX_SEARCH_QUERIES = 5
    MAX_CONCURRENT_SCRAPES = 4
    MAX_SEARCH_RETRIES = 2
    MAX_AI_ITERATIONS = 3
    MAX_USER_QUERY_LOOPS = 3
    DEFAULT_USER_AGENT = "intellISearch-bot/1.0"
    DEFAULT_REFERER = "https://example.com"
    URL_TIMEOUT = 10
    SKIP_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.zip', '.exe']
    BLOCKED_DOMAINS = []
    # Hybrid retrieval fallback defaults
    USE_HYBRID_RETRIEVAL = True
    RETRIEVAL_TOP_K = 20
    HYBRID_VECTOR_WEIGHT = 0.6
    HYBRID_BM25_WEIGHT = 0.4
    HYBRID_FUSION_METHOD = "rrf"
    HYBRID_RRF_K = 60
    VECTOR_SCORE_THRESHOLD = 0.1
    MIN_CHUNK_LENGTH = 50
    MIN_WORD_COUNT = 10
    USE_RERANKING = False
    RERANKER_CANDIDATES_MULTIPLIER = 3
    # Enhanced embedding fallback defaults
    USE_ENHANCED_EMBEDDINGS = True
    EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
    EMBEDDING_DIMENSIONALITY = 768
    EMBEDDING_NORMALIZE = True
    EMBEDDING_BATCH_SIZE = 50
    # Multi-query retrieval fallback defaults
    USE_MULTI_QUERY_RETRIEVAL = True
    MAX_RETRIEVAL_QUERIES = 5
    QUERY_CHUNK_DISTRIBUTION = True
    # Color constants and chunking fallbacks
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    # Simple ANSI color fallbacks
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    # Enhanced deduplication fallback defaults
    USE_LLM_DEDUPLICATION = False
    LLM_DEDUP_DETAILED_ONLY = True
    # Import fallback deduplication function
    def enhanced_deduplicate_content(text: str, report_type: str = "detailed") -> str:
        return deduplicate_content(text)  # Fallback to basic deduplication


# Import prompt instructions
try:
    from .prompt import (
        query_writer_instructions_legal,
        query_writer_instructions_general,
        query_writer_instructions_macro,
        query_writer_instructions_deepsearch,
        query_writer_instructions_person_search,
        query_writer_instructions_investment,
        web_search_validation_instructions,
        reflection_instructions_modified,
        report_writer_instructions_legal,
        report_writer_instructions_general,
        report_writer_instructions_macro,
        report_writer_instructions_deepsearch,
        report_writer_instructions_person_search,
        report_writer_instructions_investment,
    ) 
except ImportError:
    logging.error("Could not import prompt instructions from prompt.py. LLM nodes will not function.")
    # Define dummy variables to prevent NameError
    query_writer_instructions_legal = ""
    query_writer_instructions_general = ""
    query_writer_instructions_macro = ""
    query_writer_instructions_deepsearch = ""
    query_writer_instructions_person_search = ""
    query_writer_instructions_investment = ""
    web_search_validation_instructions = ""
    reflection_instructions_modified = ""
    report_writer_instructions_legal = ""
    report_writer_instructions_general = ""
    report_writer_instructions_macro = ""
    report_writer_instructions_deepsearch = ""
    report_writer_instructions_person_search = ""
    report_writer_instructions_investment = ""
    

# Import LangChain components used in nodes
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS # Assuming using FAISS

except ImportError:
    logging.error("Could not import necessary LangChain components. Embedding and indexing may fail.")
    RecursiveCharacterTextSplitter, FAISS = None, None

from typing import TypedDict, Union
class AgentState(TypedDict):
    new_query: Optional[str]
    reasoning_mode: bool
    new_queries: Optional[List[str]] # Not used in current flow, consider removal
    search_queries: Optional[List[str]] # The primary list of queries
    rationale: Optional[str]
    data: Optional[List[SearchResult]] # List of SearchResult objects after evaluation
    relevant_contexts: Optional[Dict[str, Dict[str, str]]] # Extracted content and metadata from relevant URLs
    relevant_chunks: Optional[List[Document]] # Relevant chunks extracted from contexts per query
    proceed: Optional[bool] # Flag for conditional transitions (e.g., proceed to next stage or loop)
    visited_urls: Optional[List[str]] # List of URLs that have been visited/processed
    failed_urls: Optional[List[str]] # List of URLs that have failed and should not be revisited
    iteration_count: Optional[int] # Counter for search refinement iterations
    report: Optional[str] # Final report content
    report_filename: Optional[str] # Filename of the saved text report
    error: Optional[str] # Field to store error messages
    evaluation_response: Optional[str]
    suggested_follow_up_queries: Optional[List[str]] # New queries suggested by AI evaluation
    prompt_type: Optional[str]
    approval_iteration_count: Optional[int]  # Counts loops between user_approval ↔ create_queries
    search_iteration_count: Optional[int]    # Counts loops from AI_evaluate ↔ evaluate_search_results
    snippet_state: Optional[Dict[str, str]]
    report_type: Optional[str] # "concise" (600-1200 words) or "detailed" (800-3000 words)
    
    # Automation flags
    non_interactive: Optional[bool] # Run without blocking user input
    auto_approve: Optional[bool] # Automatically approve generated queries
    approval_choice: Optional[str] # Simulated user response for approval ("yes" or "no")
    auto_report_type: Optional[str] # Automatically select report type ("concise" or "detailed")
    report_type_choice: Optional[str] # Simulated user choice for report type
    new_query_override: Optional[str] # Override query if approval is rejected in non-interactive mode

# Pydantic models for LLM output validation (moved from initial cells)
class SearchQueryResponse(BaseModel):
    """Represents the expected JSON structure from the create_queries LLM call."""
    rationale: Optional[str] = Field(default=None, description="The rationale for the generated search queries.")
    query: conlist(str, min_length=1) = Field(description="A list of search queries.")

class EvaluationResponse(BaseModel):
    """Represents the expected JSON structure from the AI_evaluate LLM call."""
    is_sufficient: bool = Field(description="Whether the extracted information is sufficient to answer the query.")
    knowledge_gap: str = Field(description="Description of the knowledge gap if the information is not sufficient.")
    follow_up_queries: List[str] = Field(description="A list of follow-up queries if the information is not sufficient.")


# --- Node Functions ---

async def create_queries(state: AgentState) -> AgentState:
    """
    Uses the user input from the initial state to generate rationale and a list of queries using LLM.
    Uses Pydantic for robust parsing of LLM output and includes error handling.
    Also checks for and uses suggested_follow_up_queries if available.
    """
    # Use RED color constant from setup if available
    try:
        from setup import RED, ENDC
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'

    # Get prompt type from state
    prompt_type = state.get("prompt_type", "general") # Default to general
    logging.info(f"Using prompt type '%s' for query generation.", prompt_type)

    # Select the correct prompt template based on prompt_type
    query_writer_instructions = query_writer_instructions_general # Default
    if prompt_type == "legal":
        query_writer_instructions = query_writer_instructions_legal
    elif prompt_type == "macro":
         query_writer_instructions = query_writer_instructions_macro
    elif prompt_type == "general":
         query_writer_instructions = query_writer_instructions_general
    elif prompt_type == "deepsearch":
         query_writer_instructions = query_writer_instructions_deepsearch
    elif prompt_type == "person_search":
         query_writer_instructions = query_writer_instructions_person_search
    elif prompt_type == "investment":
         query_writer_instructions = query_writer_instructions_investment

    # Prioritize suggested_follow_up_queries if they exist and we are in a refinement loop
    suggested_queries = state.get("suggested_follow_up_queries")
    new_query = state.get("new_query")
    current_iteration = state.get("iteration_count", 0)

    generated_search_queries = set()
    rationale = ""
    error = None

    # If there are suggested follow-up queries from a previous AI evaluation and we haven't exceeded max iterations,
    # use them directly instead of asking the LLM to generate new ones based on the original query.
    if suggested_queries and current_iteration < MAX_AI_ITERATIONS and state.get("relevant_chunks"):
         logging.info("Using %d suggested follow-up queries from previous iteration.", len(suggested_queries))
         generated_search_queries.update(suggested_queries)
         rationale = f"Refining search based on the previous evaluation's suggested queries ({len(suggested_queries)} queries)."
         # Clear suggested_follow_up_queries after using them
         state["suggested_follow_up_queries"] = []
         state["search_queries"] = list(generated_search_queries)
         state["rationale"] = rationale
         state["error"] = None # Clear previous error if using suggested queries
         return state


    # Proceed with initial query generation if no suggested queries or max iterations reached
    logging.info("Generating initial search queries based on user query: %s", new_query)

    number_queries = MAX_SEARCH_QUERIES # Use config constant if available

    if new_query and llm: # Ensure llm is available
        # Import SystemMessage and HumanMessage locally if needed
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=query_writer_instructions.format(
                number_queries=number_queries,
                current_date=get_current_date(),
                topic=new_query
            )),
            HumanMessage(content=f"User Query: {new_query}\n\nPlease provide a JSON object with the key 'query', where 'query' is a list of {number_queries} search queries.")
        ]

        try:
            # Use the general llm_call_async utility
            response = await llm.ainvoke(messages) # Use the primary llm

            if response and isinstance(response.content, str):
                # Use a more robust regex to find the JSON block
                json_match = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', response.content, re.DOTALL)

                if json_match:
                    json_string = json_match.group(1) if json_match.group(1) else json_match.group(2)
                    # Clean the JSON string: remove trailing commas before brackets/braces and control characters
                    cleaned_json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
                    cleaned_json_string = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_json_string)


                    # Using model_validate_json for Pydantic V2 compatibility
                    try:
                        parsed_response = SearchQueryResponse.model_validate_json(cleaned_json_string)
                        rationale = parsed_response.rationale or "No rationale provided."
                        search_queries = parsed_response.query

                        if isinstance(search_queries, list) and all(isinstance(q, str) for q in search_queries):
                            generated_search_queries.update(search_queries)
                            logging.info("Generated %d search queries.", len(generated_search_queries))
                        else:
                            # This case should ideally be caught by Pydantic validation, but keeping as a safeguard
                            error = "LLM response 'query' key is not a valid list of strings after Pydantic parsing."
                            logging.error(f"{error} Response: {response.content}")

                    except ValidationError as e:
                        error = f"Pydantic validation error parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logging.error(error)
                    except json.JSONDecodeError as e:
                        error = f"JSON decoding error parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logging.error(error)
                    except Exception as e:
                        error = f"An unexpected error occurred parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logging.error(error)

                else:
                    error = "Could not find JSON block in LLM response for query generation."
                    logging.error(f"{error} Response: {response.content}")


            else:
                error = "No or invalid response received from LLM for query generation."
                logging.error(error)

        except Exception as e:
            error = f"An unexpected error occurred during LLM call for query generation: {e}"
            logging.error(error)


    else:
        if not new_query:
            error = "No initial query provided in state."
            logging.warning(error)
        elif not llm_call_async:
             error = "Primary LLM is not initialized. Cannot generate queries."
             logging.error(error)
        elif not query_writer_instructions:
             error = f"Prompt instructions for type '{prompt_type}' not loaded. Cannot generate queries."
             logging.error(error)


    state['rationale'] = rationale if rationale else "No rationale generated."
    state['search_queries'] = list(generated_search_queries) if generated_search_queries else []

    # Append new error to existing error state
    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + error).strip() if error else current_error.strip()
    state['error'] = None if state['error'] == "" else state['error']

    # Ensure suggested_follow_up_queries is cleared if we generated new queries
    state["suggested_follow_up_queries"] = []

    return state

async def user_approval_for_queries(state: AgentState) -> AgentState:
    """
    Displays the generated rationale and search queries to the user and asks for approval to proceed.
    Supports non-interactive operation via state flags:
      - state['non_interactive'] = True : run without blocking input()
      - state['auto_approve'] = True : automatically approve generated queries
      - state['approval_choice'] = 'yes'|'no' : simulated user response
    """

    try:
        from setup import RED, ENDC
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'

    rationale = state.get('rationale', 'No rationale provided.')
    search_queries = state.get('search_queries', [])
    current_error = state.get('error')

    # Non-interactive shortcuts
    non_interactive = state.get('non_interactive', False)
    auto_approve = state.get('auto_approve', False)
    approval_choice = state.get('approval_choice', None)  # 'yes' or 'no'

    if current_error:
        logging.error("Error from previous step: %s", current_error)

    logging.info("Generated Search Queries for Approval")
    logging.debug("Rationale: %s", rationale)
    logging.debug("Search Queries: %s", search_queries)

    # Increment approval iteration count
    state['approval_iteration_count'] = state.get('approval_iteration_count', 0) + 1

    # Handle non-interactive / automated approval
    if non_interactive:
        if approval_choice is not None:
            choice = approval_choice.strip().lower()
            logging.info("Non-interactive approval choice provided: %s", choice)
            if choice in ['yes', 'y']:
                state['proceed'] = True
                state['error'] = None
                return state
            else:
                # Simulate user entering a new query via state['new_query_override'] if provided
                new_query = state.get('new_query_override') or state.get('new_query')
                logging.info("Non-interactive approval: choice is no; new_query set to %s", new_query)
                state['proceed'] = False
                state['new_query'] = new_query
                state['search_queries'] = []
                state['data'] = []
                state['relevant_contexts'] = {}
                state['relevant_chunks'] = []
                state['visited_urls'] = []
                state['iteration_count'] = 0
                state['approval_iteration_count'] = 0
                state['error'] = None
                state['suggested_follow_up_queries'] = []
                state['prompt_type'] = state.get('prompt_type', 'general')
                return state
        elif auto_approve:
            logging.info("Non-interactive auto-approve enabled. Proceeding.")
            state['proceed'] = True
            state['error'] = None
            return state
        else:
            # Default for non_interactive with no explicit choice: do not proceed
            logging.warning("Non-interactive mode with no approval flags set. Defaulting to no.")
            state['proceed'] = False
            return state

    # Interactive mode: preserve existing behavior but with logging
    if search_queries:
        for i, query in enumerate(search_queries, 1):
            logging.info("%d. %s", i, query)
        logging.debug("--- end queries ---")

        for _ in range(5):  # Limit invalid attempts
            user_input = input("Approve queries and proceed? (yes/no): ").lower().strip()
            if user_input in ['yes', 'y']:
                state['proceed'] = True
                state['error'] = None
                return state
            elif user_input in ['no', 'n']:
                new_query = input("Please enter your initial query: ")
                logging.info("New query received: %s", new_query)

                # Reset state
                state['proceed'] = False
                state['new_query'] = new_query
                state['search_queries'] = []
                state['data'] = []
                state['relevant_contexts'] = {}
                state['relevant_chunks'] = []
                state['visited_urls'] = []
                state['iteration_count'] = 0
                state['approval_iteration_count'] = 0
                state['error'] = None
                state['suggested_follow_up_queries'] = []
                state['prompt_type'] = state.get('prompt_type', 'general')
                return state
            else:
                logging.warning("Invalid input received for approval prompt.")

        logging.warning("Too many invalid attempts at approval prompt. Defaulting to no.")
        state['proceed'] = False
        return state

    else:
        logging.warning("No search queries were generated.")
        new_query = input("No queries were generated. Please enter a new initial query: ")
        logging.info("New query received: %s", new_query)

        # Reset state
        state['proceed'] = False
        state['new_query'] = new_query
        state['search_queries'] = []
        state['data'] = []
        state['relevant_contexts'] = {}
        state['relevant_chunks'] = []
        state['visited_urls'] = []
        state['iteration_count'] = 0
        state['approval_iteration_count'] = 0
        state['error'] = None
        state['suggested_follow_up_queries'] = []
        state['prompt_type'] = state.get('prompt_type', 'general')

        return state


async def choose_report_type(state: AgentState) -> AgentState:
    """
    Sets unified report type. No user selection needed since we now use a single
    500-2000 word report format with query-answer structure.
    Maintains compatibility with existing automation flags.
    """
    
    # Set unified report type - no user interaction needed
    state['report_type'] = 'unified'
    logging.info("Using unified report format (500-2000 words)")
    return state


# Consolidated helper and evaluation implementation
import hashlib
from typing import List

def hash_snippet(url: str, snippet: str) -> str:
    return hashlib.sha256(f"{url}|{snippet}".encode()).hexdigest()

async def evaluate_search_results(state: AgentState) -> AgentState:
    """
    Consolidated and robust evaluate_search_results implementation.
    - Uses UnifiedSearcher to run concurrent searches
    - Uses an LLM to validate snippets with timeout handling
    - Caches snippet verdicts in state['snippet_cache']
    - Deduplicates results and preserves previous data
    """
    try:
        from setup import RED, ENDC, YELLOW
    except ImportError:
        RED, ENDC, YELLOW = '\033[91m', '\033[0m', '\033[93m'

    search_queries = state.get("search_queries", []) or []
    existing_data = state.get("data", []) or []
    visited_urls = set(state.get("visited_urls", []) or [])
    failed_urls = set(state.get("failed_urls", []) or [])  # Get failed URLs
    snippet_cache = state.get("snippet_cache", {}) or {}
    errors = []

    if not search_queries:
        logging.warning("No search queries found to evaluate.")
        state.update({
            "data": existing_data,
            "visited_urls": list(visited_urls),
            "error": state.get('error'),
            "snippet_cache": snippet_cache
        })
        return state

    if not UnifiedSearcher:
        error_msg = "UnifiedSearcher class not available. Cannot perform search."
        logging.error(error_msg)
        state.update({
            "data": [],
            "visited_urls": list(visited_urls),
            "error": error_msg,
            "snippet_cache": snippet_cache
        })
        return state

    search_engine = UnifiedSearcher()
    search_tasks = [search_engine.search(q) for q in search_queries]
    search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

    async def evaluate_snippet(result, query: str):
        url, snippet = getattr(result, 'url', None), getattr(result, 'snippet', None)
        if not url or url in visited_urls or url in failed_urls or not snippet:
            if url in failed_urls:
                logging.debug(f"Skipping previously failed URL: {url}")
            return None

        # Check for blocked domains
        if any(domain in url.lower() for domain in BLOCKED_DOMAINS):
            blocked_domain = next(domain for domain in BLOCKED_DOMAINS if domain in url.lower())
            logging.debug(f"Skipping blocked domain URL (%s): %s", blocked_domain, url)
            return None

        snippet_hash = hash_snippet(url, snippet)
        cached = snippet_cache.get(snippet_hash)
        if cached:
            logging.debug(f"Using cached verdict for {url}: {cached}")
            return result if cached == "yes" else None

        messages = [
            SystemMessage(content=web_search_validation_instructions.format(
                query=query,
                current_date=get_current_date()
            )),
            HumanMessage(content=f"Snippet: {snippet}")
        ]

        try:
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=15)
            answer = getattr(response, 'content', '') or ''
            answer_l = answer.strip().lower()
            verdict = "yes" if "yes" in answer_l else "no"
            snippet_cache[snippet_hash] = verdict
            logging.info(f"LLM verdict for {url}: {verdict}")
            return result if verdict == "yes" else None
        except asyncio.TimeoutError:
            logging.warning(f"Timeout evaluating {url}")
            return None
        except Exception as e:
            error_msg = f"Error evaluating {url}: {type(e).__name__} - {e}"
            logging.exception(error_msg)
            errors.append(error_msg)
            return None

    evaluated_results = []
    for i, result_set in enumerate(search_results_list):
        query = search_queries[i] if i < len(search_queries) else f"<unknown_{i}>"

        if isinstance(result_set, Exception):
            errors.append(f"Search failed for query '{query}': {result_set}")
            continue

        if not result_set:
            logging.info(f"No results for query: {query}")
            errors.append(f"No results returned for query: {query}")
            continue

        tasks = [evaluate_snippet(r, query) for r in result_set]
        results = await asyncio.gather(*tasks)
        for r in results:
            if r:
                evaluated_results.append(r)
                visited_urls.add(r.url)

    # Merge and deduplicate with previous data
    deduplicated = {item.url: item for item in existing_data + evaluated_results}
    final_data = list(deduplicated.values())

    state.update({
        "data": final_data,
        "visited_urls": list(visited_urls),
        "snippet_cache": snippet_cache,
        "error": "\n".join(errors) if errors else None
    })

    logging.info(f"evaluate_search_results: Final data size: {len(final_data)}")
    return state


async def extract_content(state: AgentState) -> AgentState:
    """
    Uses the Scraper to extract content from relevant URLs, handling PDFs separately.
    Ranks URLs based on relevance and extracts content from the top N.
    Includes error handling for fetching and content cleaning.
    Skips common non-HTML file types (except PDF now) and YouTube URLs based on config.
    Adds a timeout per URL based on config.
    """
    # Use color constants from setup if available
    try:
        from .config import RED, ENDC, YELLOW
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'
        YELLOW = '\033[93m'

    from .config import URL_TIMEOUT
    data = state.get('data', []) # Original search results including snippets
    relevant_contexts = {}
    errors = []
    url_timeout = URL_TIMEOUT # Use timeout from config
    # Modify SKIP_EXTENSIONS to potentially exclude .pdf if we are handling it
    skip_extensions = [ext for ext in SKIP_EXTENSIONS if ext.lower() != '.pdf'] # Exclude .pdf from general skip list


    if not Scraper:
         errors.append("Scraper class not available. Cannot extract content.")
         logging.error(errors[-1])
         state["relevant_contexts"] = {} # Ensure relevant_contexts is initialized
         # Append new errors to existing ones in state
         current_error = state.get('error', '') or ''
         state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else current_error.strip()
         state['error'] = None if state['error'] == "" else state['error']
         return state

    if not PyMuPDFLoader:
        errors.append("PyMuPDFLoader class not available. Cannot extract content from PDFs.")
        logging.warning(errors[-1])
        # continue; no PDF loader available - use snippets as fallback where possible
        logging.debug("PyMuPDFLoader not available; PDF extraction disabled.")
    # Check if data is empty
    if not data:
        logging.info("No data found to extract content from.")
        state["relevant_contexts"] = {} # Ensure relevant_contexts is initialized
        # Append new errors to existing ones in state
        current_error = state.get('error', '') or ''
        state['error'] = (current_error + "\nNo data found to extract content from.").strip()
        state['error'] = None if state['error'] == "" else state['error']
        return state

    # Rank URLs based on collected data (Assuming data contains SearchResult objects with url and snippet)
    valid_data = [item for item in data if isinstance(item, SearchResult) and item.url and item.snippet]

    if not valid_data:
         logging.info("No valid data found for ranking.")
         state["relevant_contexts"] = {}
         # Append new errors to existing ones in state
         current_error = state.get('error', '') or ''
         state['error'] = (current_error + "\nNo valid data found for ranking.").strip()
         state['error'] = None if state['error'] == "" else state['error']
         return state

    context_for_ranking = {item.url: item.snippet for item in valid_data}

    # Use the original new_query for ranking relevance if available, otherwise use the first search query
    ranking_query = state.get("new_query", state.get("search_queries", [None])[0])
    if ranking_query:
        # Use the imported rank_urls
        ranked_urls = rank_urls(ranking_query, [item.url for item in valid_data], context_for_ranking)
    else:
        logging.info("No query available for ranking URLs. Proceeding without ranking.")
        ranked_urls = [item.url for item in valid_data] # Use original order if no query

    urls_to_process = ranked_urls[:30] # limit to top 30 urls
    logging.info("Relevant and ranked URLs for extraction: %s", urls_to_process)

    # Get the list of previously failed URLs
    failed_urls = state.get('failed_urls', []) or []
    
    # Filter out failed URLs from processing
    urls_to_process = [url for url in urls_to_process if url not in failed_urls]
    if len(ranked_urls[:30]) > len(urls_to_process):
        skipped_count = len(ranked_urls[:30]) - len(urls_to_process)
        logging.info("Skipped %d previously failed URLs", skipped_count)

    # Initialize the Scraper
    scraper = Scraper()

    # Prepare tasks for each URL, including timeout and fallback logic
    processing_tasks = []

    # Create a mapping from URL to its original SearchResult for easy snippet access
    url_to_search_result = {item.url: item for item in valid_data}

    for url in urls_to_process:
         # Check for blocked domains
         if any(domain in url.lower() for domain in BLOCKED_DOMAINS):
             blocked_domain = next(domain for domain in BLOCKED_DOMAINS if domain in url.lower())
             logging.info("Skipping blocked domain URL (%s): %s", blocked_domain, url)
             continue # Skip blocked domain URLs

         # Check for other file extensions to skip (excluding .pdf now)
         skipped = False
         for ext in skip_extensions:
              if url.lower().endswith(ext):
                  logging.info("Skipping URL with unsupported extension %s: %s", ext, url)
                  skipped = True
                  break
         if skipped:
              continue # Skip if the extension is in the skip list

         # Create an async task for processing this URL with a timeout
         async def process_single_url_with_timeout(target_url):
             try:
                 extracted_content = None # Initialize extracted content

                 # Handle PDF URLs separately
                 if target_url.lower().endswith('.pdf') and PyMuPDFLoader:
                     logging.info("Attempting to load PDF from: %s using PyMuPDFLoader", target_url)
                     try:
                         # PyMuPDFLoader can be synchronous, run in thread pool
                         loader = PyMuPDFLoader(target_url)
                         # Use asyncio.wait_for to apply the timeout to the synchronous load operation
                         documents = await asyncio.wait_for(
                             asyncio.to_thread(loader.load),
                             timeout=url_timeout
                         )

                         if documents:
                             # Concatenate page content from all documents (pages)
                             pdf_text = "\n".join([doc.page_content for doc in documents])
                             extracted_content = clean_extracted_text(pdf_text) # Clean the extracted text
                             
                             # Try to extract title from metadata or use filename
                             pdf_title = "Untitled"
                             if documents[0].metadata and 'title' in documents[0].metadata:
                                 pdf_title = documents[0].metadata['title']
                             elif documents[0].metadata and 'source' in documents[0].metadata:
                                 # Extract filename from source as fallback
                                 import os
                                 pdf_title = os.path.basename(documents[0].metadata['source']).replace('.pdf', '').replace('_', ' ').title()
                             else:
                                 # Use URL as last resort
                                 import os
                                 pdf_title = os.path.basename(target_url).replace('.pdf', '').replace('_', ' ').title()
                             
                             logging.info("Successfully extracted content from PDF: %s", target_url)
                             return target_url, {"content": extracted_content[:10000], "title": pdf_title} # Return URL and content/title dict
                         else:
                             # If loader returned no documents or empty content
                             error_msg = f"PyMuPDFLoader returned no documents or empty content for {target_url}."
                             logging.warning(error_msg)
                             # Fallback to snippet for PDFs if available, though snippets for PDFs are rare/less useful
                             original_result = url_to_search_result.get(target_url)
                             if original_result and original_result.snippet:
                                  logging.info("Using snippet as fallback for PDF %s after loading failure.", target_url)
                                  fallback_title = original_result.title if hasattr(original_result, 'title') and original_result.title else "Untitled"
                                  return target_url, {"content": original_result.snippet[:10000], "title": fallback_title} # Use truncated snippet with title
                             else:
                                  logging.warning("No snippet available for fallback for PDF %s", target_url)
                                  return target_url, None # No content if PDF loading fails and no snippet fallback

                     except asyncio.TimeoutError:
                         error_msg = f"PyMuPDFLoader timed out after {url_timeout}s for {target_url}."
                         logging.warning(error_msg)
                         # No snippet fallback for PDFs after timeout (as per previous logic, snippets for PDFs are less reliable)
                         return target_url, None
                     except Exception as e:
                         error_msg = f"Error using PyMuPDFLoader for {target_url}: {e}"
                         logging.error(error_msg)
                         # No snippet fallback for PDFs after general error
                         return target_url, None

                 else: # Handle non-PDF URLs using the general Scraper
                     # Use the scraper for this URL
                     scraped_content = await asyncio.wait_for(
                         scraper.scrape_url(target_url, dynamic=False), # Use dynamic=False explicitly
                         timeout=url_timeout
                     )

                     if scraped_content.is_successful():
                         # Truncate content if it's excessively long
                         extracted_content = scraped_content.text[:10000] if scraped_content.text else "" # Limit to 10k characters
                         extracted_title = scraped_content.title if scraped_content.title else "Untitled"
                         logging.info("Successfully extracted content from %s", target_url)
                         return target_url, {"content": extracted_content, "title": extracted_title} # Return URL and content/title dict
                     else:
                          # If scraping failed but didn't timeout, log the error and use snippet if available
                          error_msg = f"Scraping failed for {target_url}: {scraped_content.error}"
                          logging.error(error_msg)
                          # Fallback to snippet if scraping failed for a non-timeout reason
                          original_result = url_to_search_result.get(target_url)
                          if original_result and original_result.snippet:
                               logging.info("Using snippet as fallback for %s after scraping failure.", target_url)
                               fallback_title = original_result.title if hasattr(original_result, 'title') and original_result.title else "Untitled"
                               return target_url, {"content": original_result.snippet[:10000], "title": fallback_title} # Use truncated snippet with title
                          else:
                               # If no snippet, return None for content
                               logging.warning("No snippet available for fallback for %s", target_url)
                               return target_url, None


             except asyncio.TimeoutError:
                 # Handle timeout for non-PDF URLs: Use snippet as fallback
                 logging.warning(f"Processing timed out for URL: {target_url} after {url_timeout}s.")

                 # Fallback to snippet if the URL was originally from search results
                 if target_url in url_to_search_result:
                      original_result = url_to_search_result[target_url]
                      if original_result.snippet:
                           logging.info("Using snippet as fallback for %s after timeout.", target_url)
                           fallback_title = original_result.title if hasattr(original_result, 'title') and original_result.title else "Untitled"
                           return target_url, {"content": original_result.snippet[:10000], "title": fallback_title} # Use truncated snippet with title
                      else:
                          # If no snippet, return None for content
                          logging.warning("No snippet available for fallback after timeout for %s", target_url)
                          return target_url, None
                 else:
                     # No snippet fallback for URLs not in original search results
                     logging.warning("No snippet fallback available or applicable for %s after timeout.", target_url)
                     return target_url, None # Return None for content on timeout without snippet fallback


             except Exception as e:
                 # Handle any other unexpected exceptions during processing
                 error_msg = f"An unexpected error occurred processing {target_url}: {e}"
                 logging.error(error_msg, exc_info=True) # Log traceback for unexpected errors
                 # Fallback to snippet if unexpected error occurs for a non-PDF URL
                 if target_url in url_to_search_result and not target_url.lower().endswith('.pdf'):
                      original_result = url_to_search_result[target_url]
                      if original_result.snippet:
                           logging.info("Using snippet as fallback for %s after unexpected error.", target_url)
                           return target_url, original_result.snippet[:10000] # Use truncated snippet
                      else:
                           logging.warning("No snippet available for fallback after unexpected error for %s", target_url)
                           return target_url, None
                 else:
                     logging.warning("No snippet fallback available or applicable for %s after unexpected error.", target_url)
                     return target_url, None


         # Add the processing task for this URL
         processing_tasks.append(process_single_url_with_timeout(url))

    # Run all processing tasks concurrently
    processed_results = await asyncio.gather(*processing_tasks)

    logging.info(f"extract_content: Finished all URL processing tasks. Processing {len(processed_results)} results.")

    # Track failed URLs to avoid revisiting them
    new_failed_urls = []

    # Process the results from the gathered tasks
    for url, content_data in processed_results:
        if content_data:
            relevant_contexts[url] = content_data # Add successfully extracted content and title (or snippet fallback)
        else:
             # If content_data is None, it means extraction/fallback failed or was skipped
             # Add this URL to the failed URLs list to avoid revisiting it
             new_failed_urls.append(url)
             logging.info(f"Adding URL to failed list: {url}")

    # Update the failed URLs list in state
    current_failed_urls = state.get('failed_urls', []) or []
    updated_failed_urls = list(set(current_failed_urls + new_failed_urls))  # Remove duplicates
    state['failed_urls'] = updated_failed_urls
    
    if new_failed_urls:
        logging.info(f"Added {len(new_failed_urls)} URLs to failed list. Total failed URLs: {len(updated_failed_urls)}")

    state["relevant_contexts"] = relevant_contexts

    # Append new errors to existing ones in state
    # Errors from process_single_url_with_timeout are already logged, no need to add to state['error'] unless critical
    # Let's just add a summary error if relevant_contexts is empty despite having valid_data initially
    if not relevant_contexts and valid_data:
         summary_error = "Failed to extract usable content from any relevant URLs."
         errors.append(summary_error)
         logging.error(summary_error)


    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else current_error.strip()

    # Add logging to inspect the final state["relevant_contexts"]
    logging.info("extract_content: Final state['relevant_contexts'] contains %d items.", len(state.get('relevant_contexts', {})))


    return state


async def embed_index_and_extract(state: AgentState) -> AgentState:
    """
    Enhanced embedding, indexing, and retrieval using hybrid approach.
    Combines BM25 (sparse) and vector search (dense) for improved relevance.
    """
    # Use color constants from setup if available
    try:
        from setup import RED, ENDC
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'

    relevant_contexts = state.get("relevant_contexts", {})
    relevant_chunks = [] # Initialize relevant_chunks as a list of Documents
    errors = []
    N_CHUNKS = 20 # Increased from 10 to 20 for better context coverage

    # Retrieve current error state safely
    current_error_state = state.get('error')

    # Check if we have contexts to process
    if not relevant_contexts:
        logging.warning("No relevant contexts found for embedding and retrieval.")
        state["relevant_chunks"] = []
        new_error = (str(current_error_state or '') + "\nNo relevant contexts found for embedding and retrieval.").strip()
        state['error'] = None if new_error == "" else new_error
        return state

    # Try hybrid retriever first
    if create_hybrid_retriever and embeddings and USE_HYBRID_RETRIEVAL:
        try:
            logging.info("Using hybrid retriever (BM25 + Vector Search)")
            
            # Create hybrid retriever with configuration from config.py
            hybrid_retriever = create_hybrid_retriever(
                embeddings=embeddings,
                top_k=RETRIEVAL_TOP_K,
                vector_weight=HYBRID_VECTOR_WEIGHT,
                bm25_weight=HYBRID_BM25_WEIGHT,
                fusion_method=HYBRID_FUSION_METHOD,
                rrf_k=HYBRID_RRF_K,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                score_threshold=VECTOR_SCORE_THRESHOLD,
                min_chunk_length=MIN_CHUNK_LENGTH,
                min_word_count=MIN_WORD_COUNT
            )
            
            # Build indices
            if hybrid_retriever.build_index(relevant_contexts):
                # Get retrieval queries - use the specific search queries that were used to find documents
                search_queries = state.get("search_queries", [])
                original_query = state.get("new_query")
                
                if search_queries:
                    # Use the specific search queries for better relevance
                    logging.info(f"Using {len(search_queries)} specific search queries for retrieval")
                    relevant_chunks = hybrid_retriever.retrieve_multi_query(search_queries)
                    retrieval_method = "multi-query hybrid retrieval"
                elif original_query:
                    # Fallback to original query if no search queries available
                    logging.info("Using original query for retrieval (fallback)")
                    relevant_chunks = hybrid_retriever.retrieve(original_query)
                    retrieval_method = "single-query hybrid retrieval"
                else:
                    logging.warning("No retrieval queries found for hybrid retrieval.")
                    relevant_chunks = []
                    retrieval_method = "no queries"
                
                if relevant_chunks:
                    # Log retrieval stats
                    stats = hybrid_retriever.get_stats()
                    logging.info(f"Hybrid retrieval stats: {stats}")
                    logging.info(f"Retrieved {len(relevant_chunks)} chunks using {retrieval_method}")
                    
                    state["relevant_chunks"] = relevant_chunks
                    return state
                else:
                    logging.warning("No chunks retrieved from hybrid retrieval, falling back to standard approach")
            else:
                logging.warning("Failed to build hybrid retriever indices, falling back to standard approach")
                
        except Exception as e:
            logging.error(f"Hybrid retriever failed: {e}, falling back to standard approach")
            errors.append(f"Hybrid retriever error: {str(e)}")

    # Fallback to existing FAISS-only approach
    logging.info("Using fallback vector search approach")
    
    # Ensure necessary components are available
    if not embeddings or not Document or not RecursiveCharacterTextSplitter:
         errors.append("Required components for embedding/indexing (embeddings, Document, RecursiveCharacterTextSplitter) are not available.")
         logging.error(errors[-1])
         state["relevant_chunks"] = [] # Ensure relevant_chunks is initialized
         new_error = (str(current_error_state or '') + "\n" + "\n".join(errors)).strip() if errors else (current_error_state.strip() if current_error_state is not None else None)
         state['error'] = None if new_error is None or new_error == "" else new_error
         return state

    # Check if FAISS is available
    faiss_available = FAISS is not None
    if not faiss_available:
        logging.warning("FAISS not available. Using fallback text-based similarity search.")

    # --- Embedding and Indexing Logic (with FAISS fallback) ---
    logging.info("Processing %d relevant contexts for embedding and indexing%s.", 
                 len(relevant_contexts), " using FAISS" if faiss_available else " using fallback method")

    documents_content = []
    document_metadatas = []

    try:
        # Initialize embedding model (already imported as embeddings)
        if not embeddings:
            error_msg = "Embeddings model not available - check Google API key configuration"
            errors.append(error_msg)
            logging.error(error_msg)
            state["relevant_chunks"] = []
            new_error = (str(current_error_state or '') + "\n" + error_msg).strip()
            state['error'] = None if new_error == "" else new_error
            return state
            
        logging.info("Using embeddings model: %s", type(embeddings).__name__)
        logging.info("Processing %d document chunks for embedding", len(documents_content))

        # Process and chunk content using config constants
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better separators for semantic chunks
        )

        for url, content_data in relevant_contexts.items():
            if content_data and isinstance(content_data, dict) and content_data.get('content'):
                content = content_data['content']
                title = content_data.get('title', 'Untitled')
                
                # Filter out very short or low-quality content
                if len(content.strip()) < 100:  # Skip very short content
                    continue
                    
                chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    # Additional quality filters for chunks
                    chunk_text = chunk.strip()
                    if len(chunk_text) < 50:  # Skip very short chunks
                        continue
                    if len(chunk_text.split()) < 10:  # Skip chunks with less than 10 words
                        continue
                        
                    documents_content.append(chunk_text)
                    document_metadatas.append({
                        "source": url, 
                        "title": title,  # Add the title to metadata
                        "chunk_index": i,
                        "chunk_length": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    })


        if not documents_content:
             logging.warning("No documents created from relevant contexts.")
             # Safely update error state
             new_error = (str(current_error_state or '') + "\nNo documents created from relevant contexts for embedding.").strip()
             state['error'] = None if new_error == "" else new_error
             state["relevant_chunks"] = [] # Ensure relevant_chunks is initialized even on error
             return state

        logging.info("Created %d document chunks for embedding.", len(documents_content))

        # Create vector database (FAISS or fallback)
        documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents_content, document_metadatas)]
        
        if faiss_available:
            # Use FAISS for vector similarity search
            vector_db = FAISS.from_documents(documents, embeddings)
            logging.info("FAISS VectorStore created and documents added.")
        else:
            # Use fallback: store documents for text-based similarity search
            vector_db = {
                'documents': documents,
                'content': documents_content,
                'embeddings': None,  # Could compute embeddings here if needed
                'type': 'fallback'
            }
            logging.info("Fallback document store created (no FAISS available).")

    except Exception as e:
        error_msg = f"An error occurred during embedding and indexing: {e}"
        errors.append(error_msg)
        logging.exception("Error in embed_and_index_content part: %s", e)
        # If embedding/indexing fails, there's nothing to extract from
        state["relevant_chunks"] = [] # Ensure relevant_chunks is initialized even on error
        # Safely update error state
        new_error = (str(current_error_state or '') + "\n" + "\n".join(errors)).strip() if errors else (current_error_state.strip() if current_error_state is not None else None)
        state['error'] = None if new_error is None or new_error == "" else new_error
        return state

    # --- Relevant Chunk Retrieval Logic ---
    # Use the vector_db (FAISS VectorStore) or fallback document store

    # Define the queries for retrieval - prioritize specific search queries over original query
    search_queries = state.get("search_queries", [])
    original_query = state.get("new_query")
    
    # Determine retrieval strategy
    if search_queries:
        retrieval_queries = search_queries[:5]  # Limit to top 5 queries for performance
        retrieval_method = f"multi-query retrieval ({len(retrieval_queries)} queries)"
        logging.info(f"Using {len(retrieval_queries)} specific search queries for chunk retrieval")
    elif original_query:
        retrieval_queries = [original_query]
        retrieval_method = "single-query retrieval (fallback)"
        logging.info("Using original query for chunk retrieval (fallback)")
    else:
        logging.warning("No retrieval queries found for chunk retrieval.")
        state["relevant_chunks"] = []
         # Safely update error state
        new_error = (str(current_error_state or '') + "\nNo retrieval queries found for chunk retrieval.").strip()
        state['error'] = None if new_error == "" else new_error
        return state

    try:
        all_relevant_chunks = []
        
        if vector_db and hasattr(vector_db, 'as_retriever'):
            # FAISS mode - perform vector similarity search for each query
            retriever = vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": max(5, N_CHUNKS // len(retrieval_queries)),  # Distribute chunks across queries
                    "score_threshold": 0.1,  # Only retrieve reasonably relevant chunks
                    "fetch_k": N_CHUNKS * 2  # Search more candidates for better selection
                }
            )
            
            # Retrieve chunks for each query
            seen_chunks = set()
            for query in retrieval_queries:
                query_chunks = retriever.invoke(query)
                
                # Add unique chunks
                for chunk in query_chunks:
                    chunk_key = f"{chunk.metadata.get('source', '')}_{chunk.metadata.get('chunk_index', 0)}"
                    if chunk_key not in seen_chunks:
                        all_relevant_chunks.append(chunk)
                        seen_chunks.add(chunk_key)
                
                if len(all_relevant_chunks) >= N_CHUNKS:
                    break
            
            relevant_chunks = all_relevant_chunks[:N_CHUNKS]
            logging.info("Retrieved %d relevant chunks using %s with FAISS and score threshold.", 
                        len(relevant_chunks), retrieval_method)
            
        elif vector_db and isinstance(vector_db, dict) and vector_db.get('type') == 'fallback':
            # Fallback mode - enhanced text matching for multiple queries
            docs = vector_db.get('documents', [])
            all_scored_docs = []
            
            # Score documents for each query
            for query in retrieval_queries:
                query_words = query.lower().split()
                
                for doc in docs:
                    content = getattr(doc, 'page_content', '') or str(doc)
                    content_lower = content.lower()
                    score = sum(1 for word in query_words if word in content_lower)
                    if score > 0:
                        all_scored_docs.append((score, doc, query))
            
            # Sort by score and deduplicate
            all_scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            seen_chunks = set()
            candidate_chunks = []
            for score, doc, query in all_scored_docs:
                doc_key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('chunk_index', 0)}"
                if doc_key not in seen_chunks:
                    candidate_chunks.append(doc)
                    seen_chunks.add(doc_key)
                    
                if len(candidate_chunks) >= N_CHUNKS * RERANKER_CANDIDATES_MULTIPLIER if USE_RERANKING else N_CHUNKS:
                    break
            
            # Apply reranking if enabled and we have enough chunks
            if USE_RERANKING and len(candidate_chunks) > N_CHUNKS:
                # TODO: Implement rerank_chunks function for advanced reranking
                # relevant_chunks = rerank_chunks(retrieval_queries[0], candidate_chunks, target_count=N_CHUNKS)
                relevant_chunks = candidate_chunks[:N_CHUNKS]  # Fallback to simple truncation
                logging.info("Retrieved %d fallback candidates, truncated to %d relevant chunks using %s (reranking disabled).", 
                           len(candidate_chunks), len(relevant_chunks), retrieval_method)
            else:
                relevant_chunks = candidate_chunks[:N_CHUNKS]
                logging.info("Retrieved %d relevant chunks using %s with fallback text search%s.", 
                           len(relevant_chunks), retrieval_method,
                           " (reranking disabled)" if not USE_RERANKING else "")
        else:
            # No vector_db available at all
            logging.warning("No vector database or fallback store available for chunk retrieval.")
            relevant_chunks = []

    except Exception as e:
        error_msg = f"Error during similarity search using {retrieval_method}: {e}"
        errors.append(error_msg)
        logging.exception("Error during chunk retrieval: %s", e)
        relevant_chunks = [] # Ensure relevant_chunks is empty on error

    state["relevant_chunks"] = relevant_chunks # Store the retrieved chunks

    # Safely update error in state after processing
    current_error_state = state.get('error')
    if errors:
        new_error_message = (str(current_error_state or '') + "\n" + "\n".join(errors)).strip()
        state['error'] = None if new_error_message == "" else new_error_message
    else:
        if current_error_state is not None:
            stripped_current_error = current_error_state.strip()
            state['error'] = stripped_current_error if stripped_current_error != "" else None
        else:
            state['error'] = None


    # Add logging to inspect the final state["relevant_chunks"]
    logging.info("embed_index_and_extract: Final state['relevant_chunks'] contains %d items.", len(state.get('relevant_chunks', [])))

    return state

async def AI_evaluate(state: AgentState) -> AgentState:
    """
    Evaluates extracted relevant_chunks using an LLM to determine if the info is sufficient.
    Updates 'proceed' based on AI assessment. Suggests follow-up queries if needed.
    Tracks search_iteration_count to prevent infinite loops.
    """

    try:
        from setup import RED, ENDC, YELLOW
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'
        YELLOW = '\033[93m'

    relevant_chunks = state.get("relevant_chunks", [])
    logging.info("AI Evaluation started: evaluating %d relevant chunks.", len(relevant_chunks))

    state["search_iteration_count"] = state.get("search_iteration_count", 0) + 1
    max_iterations = MAX_AI_ITERATIONS
    errors = []
    state["proceed"] = True

    if not llm:
        msg = "LLM not initialized. Skipping AI evaluation."
        errors.append(msg)
        logging.error(msg)
        state['error'] = msg
        return state

    if not relevant_chunks:
        msg = f"No relevant chunks found ({state['search_iteration_count']}/{max_iterations})."
        logging.warning(msg)
        if state["search_iteration_count"] < max_iterations:
            state["proceed"] = False
            state['suggested_follow_up_queries'] = []
            state['knowledge_gap'] = "No relevant info extracted this round."
        else:
            state["proceed"] = True
            state['knowledge_gap'] = "Max iterations reached. Proceeding with current info."
        state['error'] = msg
        return state

    chunks_text = "\n---\n".join(
        [f"Source: {doc.metadata.get('source', 'Unknown')}\nContent:\n{doc.page_content}" for doc in relevant_chunks]
    )

    from .prompt import reflection_instructions_modified

    messages = [
        SystemMessage(content=reflection_instructions_modified.format(
            research_topic=state.get("new_query", ""),
            extracted_info_json=chunks_text
        )),
        HumanMessage(content=f"User Question: {state.get('new_query', '')}\n\nExtracted Information:\n{chunks_text}")
    ]

    try:
        response = await llm.ainvoke(messages)
        response_text = getattr(response, "content", None)

        if not response_text:
            raise ValueError("No response received from LLM.")

        # Extract and clean JSON
        match = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', response_text, re.DOTALL)
        if match:
            json_block = match.group(1) if match.group(1) else match.group(2)
            json_block = re.sub(r',\s*([\]}])', r'\1', json_block)
            json_block = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_block)

            eval_result = EvaluationResponse.model_validate_json(json_block)

            if eval_result.is_sufficient:
                state["proceed"] = True
                state["suggested_follow_up_queries"] = []
                state["knowledge_gap"] = ""
                logging.debug("AI_evaluate: info sufficient, proceeding to report.")
            else:
                state["proceed"] = False
                state["suggested_follow_up_queries"] = eval_result.follow_up_queries
                state["knowledge_gap"] = eval_result.knowledge_gap
                logging.debug("AI_evaluate: not sufficient, looping back with follow-ups.")

        else:
            raise ValueError("JSON block not found in response.")

    except Exception as e:
        logging.exception("AI_evaluate error: %s", e)
        state["proceed"] = True
        state["suggested_follow_up_queries"] = []
        state["knowledge_gap"] = f"Fallback triggered due to error: {e}"
        errors.append(str(e))

    # Final check for iteration cap
    if not state["proceed"] and state["search_iteration_count"] >= max_iterations:
        logging.warning("Max iterations reached. Forcing report.")
        state["proceed"] = True
        state["suggested_follow_up_queries"] = []
        state["knowledge_gap"] = "Max iterations hit. Report generated with partial info."

    # Error handling
    if errors:
        prev_error = state.get("error", "") or ""
        state["error"] = (prev_error + "\n" + "\n".join(errors)).strip()

    return state

#=============================================================================================
unified_report_instruction = (
    "Generate a comprehensive research report with the following structure: "
    "1) Main Research Query (the original question), "
    "2) Research Results section where you address each sub-query individually with detailed answers, "
    "3) Conclusion that synthesizes all findings, and "
    "4) Citations and Sources. "
    "For each sub-query in the Research Results section, provide: the query itself, a comprehensive answer based on sources, "
    "supporting evidence, and relevant citations [1], [2], etc. "
    "Ensure every sub-query from the search generation phase is explicitly addressed. "
    "Aim for 500-2000 words with balanced coverage. Target 100-200 words per sub-query answer. "
    "Use clear markdown formatting with proper headings and structure."
)
#=============================================================================================

def deduplicate_content(text: str) -> str:
    """
    Remove duplicate sentences and similar content from the report to reduce repetition.
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        # Clean and normalize the sentence for comparison
        cleaned = re.sub(r'\s+', ' ', sentence.strip().lower())
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove punctuation for comparison
        
        # Skip very short sentences or headers
        if len(cleaned.split()) < 3:
            unique_sentences.append(sentence)
            continue
            
        # Check for similarity with existing content
        is_duplicate = False
        for seen in seen_content:
            # Calculate simple similarity (common words)
            words1 = set(cleaned.split())
            words2 = set(seen.split())
            if len(words1 & words2) / max(len(words1), len(words2), 1) > 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
            seen_content.add(cleaned)
    
    return ' '.join(unique_sentences)


def generate_citations_section(relevant_chunks) -> tuple[str, dict]:
    """
    Generate a citations section from the relevant chunks, excluding it from word counts.
    Returns (citations_text, source_mapping) where source_mapping maps URLs to citation numbers.
    """
    if not relevant_chunks:
        return "", {}
    
    # Extract unique sources
    sources = {}
    source_mapping = {}
    
    for chunk in relevant_chunks:
        source_url = chunk.metadata.get('source', 'Unknown URL')
        
        # Avoid duplicate sources
        if source_url not in sources:
            citation_number = len(sources) + 1
            sources[source_url] = {
                'index': citation_number,
                'url': source_url
            }
            source_mapping[source_url] = citation_number
    
    # Format citations section
    citations_text = "\n\n---\n\n# 📚 Sources and References\n\n"
        
    for source in sources.values():
        citations_text += f"[{source['index']}] {source['url']}\n\n"
    
    citations_text += f"*Total sources referenced: {len(sources)}*\n"
    citations_text += f"*Research conducted on: {get_current_date()}*"
    
    return citations_text, source_mapping


async def write_report(state: AgentState):
    """
    Generates the final report and saves to text and PDF.
    Supports concise (600-1200 words) and detailed (800-3000 words) reports.
    """
    # Use color constants from setup if available
    try:
        from setup import RED, ENDC, GREEN
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'
        GREEN = '\033[92m'

    errors: List[str] = []

    # Config and state
    prompt_type = state.get("prompt_type", "general")
    # Remove reasoning_mode and report_type dependencies - use unified approach
    
    logging.info("Using prompt type '%s' with unified report generation (500-2000 words).", prompt_type)

    # Set unified word limits (500-2000 words)
    max_words = 2000
    min_sections = 3
    max_sections = 6
    default_section_words = 400
    logging.info("Generating unified report (500-2000 words)")

    report_writer_instructions = report_writer_instructions_general
    if prompt_type == "legal":
        report_writer_instructions = report_writer_instructions_legal
    elif prompt_type == "macro":
        report_writer_instructions = report_writer_instructions_macro
    elif prompt_type == "deepsearch":
        report_writer_instructions = report_writer_instructions_deepsearch
    elif prompt_type == "person_search":
        report_writer_instructions = report_writer_instructions_person_search
    elif prompt_type == "investment":
        report_writer_instructions = report_writer_instructions_investment

    relevant_chunks = state.get('relevant_chunks', [])
    research_topic = state.get('new_query', 'the topic')
    search_queries = state.get('search_queries', [])

    if not relevant_chunks:
        final_report_content = f"Could not generate a report. No relevant information was found for the topic: '{research_topic}'."
        errors.append(final_report_content)
        logging.warning(final_report_content)

        # Save and update state as before
        text_filename = save_report_to_text(final_report_content, REPORT_FILENAME_TEXT)
        if not text_filename:
            errors.append(f"Failed to save report to text file: {REPORT_FILENAME_TEXT}.")
            logging.error(errors[-1])

        pdf_result_message = generate_pdf_from_md(final_report_content, REPORT_FILENAME_PDF)
        if "Error generating PDF" in pdf_result_message:
            errors.append(pdf_result_message)
            logging.error(pdf_result_message)
        else:
            logging.info(pdf_result_message)

        state["report"] = final_report_content
        state["report_filename"] = text_filename
        current_error = state.get('error', '') or ''
        state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else current_error.strip()
        state['error'] = None if state['error'] == "" else state['error']
        return state

    # Prepare search queries context for report instructions
    search_queries_context = ""
    if search_queries:
        search_queries_context = f"""

**REPORT STRUCTURE REQUIRED:**

## Main Research Query
**{research_topic}**

## Research Results
For each sub-query below, create a dedicated subsection with the query as the heading and provide a comprehensive answer:

{chr(10).join(f"### Sub-Query {i+1}: {query}" + chr(10) + "- [Provide detailed answer based on sources]" + chr(10) + "- [Include supporting evidence and citations [1], [2], etc.]" + chr(10) for i, query in enumerate(search_queries))}

## Conclusion
- Synthesize findings from all sub-queries
- Address the main research question comprehensively
- Highlight key insights and implications

## Citations and Sources
- List all sources used with [1], [2], etc. format

CRITICAL: Each sub-query listed above MUST be explicitly addressed in the Research Results section with its own subsection.
"""
    else:
        search_queries_context = """

**REPORT STRUCTURE REQUIRED:**
Follow the template: Main Research Query → Research Results (with sub-queries and answers) → Conclusion → Citations and Sources
"""

    # Prepare the combined chunk context (keep it reasonably sized to avoid extremely long prompts)
    formatted_chunks = "\n---\n".join([
        f"Source: {chunk.metadata.get('source', 'Unknown URL')}\nContent:\n{chunk.page_content}"
        for chunk in relevant_chunks
    ])

    selected_instruction = unified_report_instruction
    
    # Simplified instruction focusing on key requirements
    enhanced_instruction = f"""
    {selected_instruction}
    
    Answer: "{research_topic}"
    Focus on specific data from content chunks. Include relevant citations [1], [2], etc.
    {search_queries_context}
    """

    # Helper to call the LLM with flexible wrappers
    async def _call_llm(messages: List[Any]):
        # Try llm_call_async first (may accept kwargs), then llm.ainvoke, then fallback to previously used llm_call_async(signature)
        try:
            return await llm_call_async(messages)
        except Exception as e:
            logging.debug("llm_call_async failed: %s", e2)
            
            try:
                return await llm.ainvoke(messages)
            except Exception as e2:
                logging.debug("llm.ainvoke failed: %s", e)
        return None

    # 1) Request an outline (JSON) specifying section titles and target word counts
    outline_prompt = f"""
    Create an outline for a research report answering: "{research_topic}"
    
    {report_writer_instructions.format(research_topic=research_topic, summaries=formatted_chunks, current_date=get_current_date())}
    {search_queries_context}

    Provide JSON outline with {min_sections}-{max_sections} sections, maximum {max_words} words total.
    
    JSON format:
    {{"sections": [{{"title": "Section Name", "target_words": {default_section_words}}}]}}
    """

    messages = [SystemMessage(content=outline_prompt), HumanMessage(content=f"Provide the outline for: {research_topic}")]
    outline_resp = await _call_llm(messages)
    outline_text = getattr(outline_resp, 'content', None) if outline_resp is not None else None

    sections = None
    if outline_text:
        # Try to extract JSON block
        m = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', str(outline_text), re.DOTALL)
        json_string = m.group(1) if m and m.group(1) else (m.group(2) if m else None)
        if json_string:
            try:
                parsed = json.loads(json_string)
                if isinstance(parsed, dict) and 'sections' in parsed and isinstance(parsed['sections'], list):
                    sections = parsed['sections']
            except Exception as e:
                logging.debug("Outline JSON parse error: %s", e)

    # Fallback simple outline if parsing failed
    if not sections:
        logging.warning("Could not parse outline from LLM. Falling back to unified report structure.")
        # Calculate words per query based on number of search queries
        num_queries = len(search_queries) if search_queries else 3
        words_per_query = min(200, max(100, 1200 // num_queries))  # 100-200 words per query
        
        sections = [
            {"title": "Main Research Query", "target_words": 100},
            {"title": "Research Results", "target_words": words_per_query * num_queries},
            {"title": "Conclusion", "target_words": 400},
            {"title": "Citations and Sources", "target_words": 100}
        ]

    # Clamp and sanitize sections based on unified report type
    sanitized_sections = []
    total_target = 0
    max_section_words = max_words // 3  # For unified reports
    
    for s in sections[:max_sections]:
        title = s.get('title') if isinstance(s, dict) else str(s)
        try:
            tw = int(s.get('target_words', default_section_words)) if isinstance(s, dict) else default_section_words
        except Exception:
            tw = default_section_words
        
        # Adjust word count for unified reports
        tw = max(100, min(tw, max_section_words))  # Standard sections for unified reports
            
        sanitized_sections.append({"title": title.strip(), "target_words": tw})
        total_target += tw
        
        # Stop if we're approaching the limit
        if total_target >= max_words * 0.9:
            break

    # Adjust if total exceeds limit
    if total_target > max_words:
        scale_factor = max_words / total_target
        for section in sanitized_sections:
            section["target_words"] = max(50, int(section["target_words"] * scale_factor))

    # Generate citation mapping for use in sections
    _, source_mapping = generate_citations_section(relevant_chunks)

    # 2) Expand each section individually with content distribution
    section_texts = []
    total_chunks = len(relevant_chunks)
    chunks_per_section = max(1, total_chunks // len(sanitized_sections)) if sanitized_sections else 1
    
    for i, sec in enumerate(sanitized_sections):
        sec_title = sec['title']
        target_words = sec['target_words']

        # Distribute content chunks to avoid repetition across sections
        start_idx = i * chunks_per_section
        end_idx = min(start_idx + chunks_per_section + 1, total_chunks)  # +1 for overlap
        section_chunks = relevant_chunks[start_idx:end_idx]
        
        # If last section, include any remaining chunks
        if i == len(sanitized_sections) - 1:
            section_chunks = relevant_chunks[start_idx:]
        
        # Format section-specific chunks with citation numbers
        section_formatted_chunks = "\n---\n".join([
            f"[Citation {source_mapping.get(chunk.metadata.get('source', ''), 'N/A')}] Source: {chunk.metadata.get('source', 'Unknown URL')}\nContent:\n{chunk.page_content}"
            for chunk in section_chunks
        ])

        expand_prompt = f"""
        Section: {sec_title}
        Target: ~{target_words} words for unified research report
        Research Question: "{research_topic}"
        
        Extract specific information from content chunks to answer the research question.
        Use citations [1], [2], etc. from the content below.
        Focus on unique data for this section.
        {search_queries_context}
        
        Content chunks:
        {section_formatted_chunks}
        """

        messages = [SystemMessage(content=report_writer_instructions.format(research_topic=research_topic, summaries=section_formatted_chunks, current_date=get_current_date()) + "\n\n" + enhanced_instruction), HumanMessage(content=expand_prompt)]
        sec_resp = await _call_llm(messages)
        sec_content = None
        if sec_resp is not None:
            sec_content = getattr(sec_resp, 'content', None)
            # Some wrappers return a string directly
            if isinstance(sec_content, str):
                section_texts.append(f"# {sec_title}\n\n" + sec_content.strip())
            else:
                # If wrapper returns raw string
                section_texts.append(f"# {sec_title}\n\n" + str(sec_resp).strip())
        else:
            logging.warning("LLM did not return content for section %s. Inserting placeholder.", sec_title)
            section_texts.append(f"# {sec_title}\n\n" + "[No content generated for this section due to LLM failure.]")

    final_report_content = "\n\n".join(section_texts)

    # Post-generation checks: if the result is shorter than intended, ask for expansion
    def _word_count(text: str) -> int:
        return len(text.split())

    expected_words = sum(s['target_words'] for s in sanitized_sections)
    actual_words = _word_count(final_report_content)
    logging.info("Initial generated report: target_words=%d actual_words=%d max_words=%d", expected_words, actual_words, max_words)

    # For unified reports, use moderate expansion threshold 
    min_threshold = 0.6  # Moderate threshold for expansion
    min_expected_words = max(int(expected_words * min_threshold), 500)  # Minimum 500 words

    expansion_attempts = 0
    max_expansions = 1  # Allow one expansion attempt
    
    # If actual words are significantly less than expected and under the max limit, request an expansion pass
    while actual_words < min_expected_words and actual_words < max_words * 0.8 and expansion_attempts < max_expansions:
        deficit = max(0, min(expected_words - actual_words, max_words - actual_words))
        logging.warning("Report shorter than expected (have=%d want=%d max=%d). Requesting focused expansion #%d.", actual_words, expected_words, max_words, expansion_attempts+1)

        # Focus on adding NEW information rather than rewriting existing content
        expand_all_prompt = f"""
        The current research report below needs approximately {deficit} additional words but must not exceed {max_words} words total.
        
        CRITICAL: Add NEW specific information that answers this research question: "{research_topic}"
        
        EXPANSION INSTRUCTIONS - ADD NEW CONTENT ONLY:
        1. Review what information is MISSING from the current report that would better answer the research question
        2. Look through the original content chunks for ADDITIONAL unused data, values, or facts
        3. ADD approximately {deficit} words of NEW information while preserving the existing structure
        4. Focus on COMPLEMENTARY information that wasn't already covered
        5. Add more specific data points, quotes, or other relevant details
        6. DO NOT restate or rephrase information already in the report
        {search_queries_context}
        
        APPEND NEW SECTIONS OR EXPAND EXISTING ONES - Do not rewrite the entire report.
        
        Original content chunks for reference:
        {formatted_chunks}

        Current report to ADD TO (do not replace):
        {final_report_content}
        
        ADD new specific information that complements the existing content and better answers "{research_topic}":
        """
        messages = [SystemMessage(content="You are an expert report writer. Your task is to ADD new relevant information to an existing report without duplicating content."), HumanMessage(content=expand_all_prompt)]
        expand_resp = await _call_llm(messages)
        addition = getattr(expand_resp, 'content', None) if expand_resp is not None else None
        if addition:
            # APPEND new content instead of replacing (to avoid duplication)
            addition_clean = str(addition).strip()
            if addition_clean and not addition_clean.lower().startswith(final_report_content[:100].lower()):
                final_report_content = final_report_content + "\n\n" + addition_clean
                actual_words = _word_count(final_report_content)
                logging.info("After expansion #%d actual_words=%d (added content)", expansion_attempts+1, actual_words)
            else:
                logging.warning("Expansion attempt returned duplicate content, skipping.")
                break
        else:
            logging.warning("Expansion attempt returned no content.")
            break
        expansion_attempts += 1

    # Apply enhanced deduplication to reduce repetitive content
    logging.info("Applying enhanced content deduplication...")
    final_report_content = await enhanced_deduplicate_content(final_report_content, "unified")
    
    # Add citations section (excluded from word count)
    citations_section, source_mapping = generate_citations_section(relevant_chunks)
    final_report_with_citations = final_report_content + citations_section
    logging.info("Added citations section with %d unique sources", len(source_mapping))
    
    # Final word count check and truncation if necessary (excluding citations)
    final_words = _word_count(final_report_content)  # Count only main content, not citations
    if final_words > max_words:
        logging.warning("Report exceeds %d word limit (%d words). Truncating.", max_words, final_words)
        words = final_report_content.split()
        final_report_content = ' '.join(words[:max_words])
        final_words = max_words
        # Regenerate citations after truncation
        citations_section, source_mapping = generate_citations_section(relevant_chunks)
        final_report_with_citations = final_report_content + citations_section

    # Final logging and saving (including citations in saved files)
    total_chars = len(final_report_content or "")
    total_chars_with_citations = len(final_report_with_citations or "")
    logging.info("Final unified report size: %d chars, %d words (limit: %d words) + %d chars for citations", 
                 total_chars, final_words, max_words, total_chars_with_citations - total_chars)

    # Save to files (with citations)
    text_filename = save_report_to_text(final_report_with_citations, REPORT_FILENAME_TEXT)
    if not text_filename:
         errors.append(f"Failed to save report to text file: {REPORT_FILENAME_TEXT}.")
         logging.error(errors[-1])

    pdf_result_message = generate_pdf_from_md(final_report_with_citations, REPORT_FILENAME_PDF)
    if "Error generating PDF" in pdf_result_message:
         errors.append(pdf_result_message)
         logging.error(pdf_result_message)
    else:
         logging.info(pdf_result_message)

    # Update state and return (with citations)
    state["report"] = final_report_with_citations
    state["report_filename"] = text_filename

    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else current_error.strip()
    state['error'] = None if state['error'] == "" else state['error']

    # Clear intermediate data
    state['data'] = []
    state['relevant_contexts'] = {}
    state['relevant_chunks'] = []
    state['search_queries'] = []
    state['suggested_follow_up_queries'] = []
    state['knowledge_gap'] = ""
    state['rationale'] = ""
    state['iteration_count'] = 0

    return state # Return the updated state

logging.info("nodes.py loaded with LangGraph node functions.")