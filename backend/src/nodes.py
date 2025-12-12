import json, re, asyncio
from typing import Dict, Any, List, Optional
from .logging_setup import logger
from .api_keys import GOOGLE_API_KEY, SERPER_API_KEY

# Try to import optional dependencies with fallbacks
try:
    from pydantic import BaseModel, Field, ValidationError, conlist
    PYDANTIC_AVAILABLE = True
except ImportError:
    logger.warning("pydantic not available. Using basic classes instead.")
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
    logger.warning("langchain_core.documents not available. Using basic Document class.")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    LANGCHAIN_DOCS_AVAILABLE = False

try:
    from langchain_community.document_loaders import RecursiveUrlLoader, CSVLoader
    LANGCHAIN_LOADERS_AVAILABLE = True
except ImportError:
    logger.warning("langchain_community.document_loaders not available. Some features may be limited.")
    RecursiveUrlLoader = CSVLoader = None
    LANGCHAIN_LOADERS_AVAILABLE = False

try:
    from langchain_core.messages import (
        SystemMessage,
        HumanMessage,
    )
    LANGCHAIN_MESSAGES_AVAILABLE = True
except ImportError:
    logger.warning("langchain_core.messages not available. Using basic message classes.")
    
    # Create proper fallback message classes with content attribute
    class BaseMessage:
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def __str__(self):
            return f"{self.__class__.__name__}(content='{self.content}')"
    
    class SystemMessage(BaseMessage): pass
    class HumanMessage(BaseMessage): pass
    
    LANGCHAIN_MESSAGES_AVAILABLE = False

# Import necessary classes and functions from other modules
try:
    from .llm_utils import llm_call_async, embeddings
except ImportError:
    logger.error("Could not import LLM/Embeddings from llm_utils. Some nodes may not function.")
    llm_call_async, embeddings = None, None
try:
    from .search import UnifiedSearcher, SearchResult # Assuming SearchResult and UnifiedSearcher are in search.py
except ImportError:
    logger.error("Could not import search components from search.py. Search node will not function.")
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
    logger.error("Could not import scraper components from scraper.py. Extraction node will not function.")
    Scraper, ScrapedContent = None, None

try:
    from backend.src.hybrid_retriever import HybridRetriever, create_hybrid_retriever
except ImportError:
    logger.warning("Could not import hybrid retriever. Using fallback retrieval.")
    create_hybrid_retriever = None


# FSS retriever import with fallback
from typing import Any
try:
    from backend.src.fss_retriever import GeminiFileSearchRetriever, delete_gemini_file_search_store
except ImportError as e:
    logger.error(f"Failed to import GeminiFileSearchRetriever: {e}")
    GeminiFileSearchRetriever = None
try:
    from .utils import (
        safe_format, 
        get_current_date, 
        clean_extracted_text,
        rank_urls, 
        save_report_to_text,
        format_research_report, 
        enhance_report_readability  # Import utility functions
    )
except ImportError:
    logger.error("Could not import utility functions from utils.py. Some nodes may be limited.")
    
    # Fallback formatting functions
    def format_research_report(content: str) -> str:
        """Fallback formatting function"""
        return content
    
    def enhance_report_readability(content: str) -> str:
        """Fallback readability enhancement function"""
        return content
    
    def get_current_date() -> str:
        """Fallback current date function"""
        from datetime import datetime
        return datetime.now().strftime("%B %d, %Y")
    
    def save_report_to_text(content: str, filename: str) -> str:
        """Fallback save function"""
        return filename

try:
    from .config import(
            USE_PERSISTENCE,
            MAX_RESULTS,
            CACHE_TTL,
            CACHE_ENABLED,
            EMBEDDING_MODEL,
            REPORT_FORMAT,
            REPORT_FILENAME_TEXT,
            MAX_SEARCH_QUERIES,
            MAX_SEARCH_RESULTS,
            MAX_CONCURRENT_SCRAPES,
            MAX_SEARCH_RETRIES,
            MAX_AI_ITERATIONS,
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
            # Cross-encoder reranking configuration
            USE_CROSS_ENCODER_RERANKING,
            CROSS_ENCODER_MODEL,
            CROSS_ENCODER_TOP_K,
            RERANK_TOP_K,
            CROSS_ENCODER_BATCH_SIZE,
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
    )
except ImportError:
    logger.warning("Could not import config settings. Using defaults.")
    
    print("Imports from config failed. Using defaults.")
    USE_PERSISTENCE = False
    MAX_RESULTS = 5
    CACHE_TTL = 3600
    CACHE_ENABLED = False
    EMBEDDING_MODEL = "text-embedding-3-small"
    REPORT_FORMAT = "md"
    REPORT_FILENAME_TEXT = "CrystalSearchReport.txt"
    MAX_SEARCH_QUERIES = 5
    MAX_SEARCH_RESULTS = 10
    MAX_CONCURRENT_SCRAPES = 4
    MAX_SEARCH_RETRIES = 2
    MAX_AI_ITERATIONS = 1
    MAX_USER_QUERY_LOOPS = 1
    DEFAULT_USER_AGENT = "intellISearch-bot/1.0"
    DEFAULT_REFERER = "https://www.google.com"
    URL_TIMEOUT = 45
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
    ) 
except ImportError:
    logger.error("Could not import prompt instructions from prompt.py. LLM nodes will not function.")
    # Define dummy variables to prevent NameError
    query_writer_instructions_legal = ""
    query_writer_instructions_general = ""
    query_writer_instructions_macro = ""
    query_writer_instructions_deepsearch = ""
    query_writer_instructions_person_search = ""
    query_writer_instructions_investment = ""
    web_search_validation_instructions = ""
    

# Import LangChain components used in nodes
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS # Assuming using FAISS

except ImportError:
    logger.error("Could not import necessary LangChain components. Embedding and indexing may fail.")
    
    # Create fallback text splitter
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text: str) -> List[str]:
            """Simple text splitting fallback."""
            if not text:
                return []
            
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks
        
        def split_documents(self, documents):
            """Split documents into chunks."""
            result = []
            for doc in documents:
                text_chunks = self.split_text(doc.page_content)
                for chunk in text_chunks:
                    new_doc = Document(page_content=chunk, metadata=doc.metadata.copy())
                    result.append(new_doc)
            return result
    
    FAISS = None

from typing import TypedDict, Union
class AgentState(TypedDict):
    session_id: Optional[str]
    new_query: Optional[str]
    search_queries: Optional[List[str]]
    rationale: Optional[str]
    data: Optional[List[Any]]
    relevant_contexts: Optional[Dict[str, Dict[str, str]]]
    relevant_chunks: Optional[List[Document]]
    retriever_responses: Optional[Dict[str, str]]
    qa_pairs: Optional[List[Dict]]
    all_citations: Optional[List[Dict]]
    proceed: Optional[bool]
    visited_urls: Optional[List[str]]
    failed_urls: Optional[List[str]]
    iteration_count: Optional[int]
    error: Optional[str]
    evaluation_response: Optional[str]
    suggested_follow_up_queries: Optional[List[str]]
    prompt_type: Optional[str]
    search_iteration_count: Optional[int]
    snippet_state: Optional[Dict[str, str]]
    analysis_content: Optional[str]
    appendix_content: Optional[str]
    analysis_filename: Optional[str]
    appendix_filename: Optional[str]
    max_search_queries: Optional[int]
    max_search_results: Optional[int]
    max_ai_iterations: Optional[int]


# Pydantic models for LLM output validation (moved from initial cells)
class SearchQueryResponse(BaseModel):
    """Represents the expected JSON structure from the create_queries LLM call."""
    rationale: Optional[str] = Field(default=None, description="The rationale for the generated search queries.")
    query: List[str] = Field(default_factory=list, min_length=1, description="A list of search queries.")

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
    # Color constants not needed; removed for simplicity

    # Get prompt type from state
    prompt_type = state.get("prompt_type", "general") # Default to general
    logger.info(f"Using prompt type '%s' for query generation.", prompt_type)

    # Select the correct prompt template based on prompt_type
    query_writer_instructions = query_writer_instructions_general # Default
    if prompt_type == "legal":
        query_writer_instructions = query_writer_instructions_legal
    elif prompt_type == "macro":
         query_writer_instructions = query_writer_instructions_macro
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
    if suggested_queries and current_iteration < MAX_AI_ITERATIONS:
         logger.info("Using %d suggested follow-up queries from previous iteration.", len(suggested_queries))
         generated_search_queries.update(suggested_queries)
         rationale = f"Refining search based on the previous evaluation's suggested queries ({len(suggested_queries)} queries)."
         # Clear suggested_follow_up_queries after using them
         state["suggested_follow_up_queries"] = []
         state["search_queries"] = list(generated_search_queries)
         state["rationale"] = rationale
         state["error"] = None # Clear previous error if using suggested queries
         return state


    # Proceed with initial query generation if no suggested queries or max iterations reached
    logger.info("Generating initial search queries based on user query: %s", new_query)

    # Use dynamic config from state, fallback to global config
    number_queries = state.get("max_search_queries", MAX_SEARCH_QUERIES)

    if new_query and llm_call_async: # Ensure llm_call_async is available
        # Use the message classes (either LangChain or our fallback implementation)
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
            response = await llm_call_async(messages) # Use llm_call_async instead of llm.ainvoke

            if response and isinstance(response, str):
                # Use a more robust regex to find the JSON block
                json_match = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', response, re.DOTALL)

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
                            logger.info("Generated %d search queries.", len(generated_search_queries))
                        else:
                            # This case should ideally be caught by Pydantic validation, but keeping as a safeguard
                            error = "LLM response 'query' key is not a valid list of strings after Pydantic parsing."
                            logger.error(f"{error} Response: {response.content}")

                    except ValidationError as e:
                        error = f"Pydantic validation error parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logger.error(error)
                    except json.JSONDecodeError as e:
                        error = f"JSON decoding error parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logger.error(error)
                    except Exception as e:
                        error = f"An unexpected error occurred parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logger.error(error)

                else:
                    error = "Could not find JSON block in LLM response for query generation."
                    logger.error(f"{error} Response: {response}")


            else:
                error = "No or invalid response received from LLM for query generation."
                logger.error(error)

        except Exception as e:
            error = f"An unexpected error occurred during LLM call for query generation: {e}"
            logger.error(error)


    else:
        if not new_query:
            error = "No initial query provided in state."
            logger.warning(error)
        elif not llm_call_async:
             error = "Primary LLM is not initialized. Cannot generate queries."
             logger.error(error)
        elif not query_writer_instructions:
             error = f"Prompt instructions for type '{prompt_type}' not loaded. Cannot generate queries."
             logger.error(error)


    state['rationale'] = rationale if rationale else "No rationale generated."
    state['search_queries'] = list(generated_search_queries) if generated_search_queries else []

    # Append new error to existing error state
    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + error).strip() if error else current_error.strip()
    state['error'] = None if state['error'] == "" else state['error']

    # Ensure suggested_follow_up_queries is cleared if we generated new queries
    state["suggested_follow_up_queries"] = []

    return state

    # user_approval_for_queries and choose_report_type functions removed as requested
#=============================================================================================
# Consolidated helper and evaluation implementation
import hashlib
from typing import List

def hash_snippet(url: str, snippet: str) -> str:
    return hashlib.sha256(f"{url}|{snippet}".encode()).hexdigest()


# --- LLM Evaluation Function (Preserved for Future Use) ---
async def llm_evaluate_snippets(state: AgentState, search_results: list, visited_urls: set, failed_urls: set, snippet_cache: dict) -> list:
    """
    Evaluate search result snippets using LLM. Returns a filtered list of results deemed relevant by the LLM.
    """
    async def evaluate_snippet(result, query: str):
        url, snippet = getattr(result, 'url', None), getattr(result, 'snippet', None)
        if not url or url in visited_urls or url in failed_urls or not snippet:
            return None
        if any(domain in url.lower() for domain in BLOCKED_DOMAINS):
            return None
        snippet_hash = hash_snippet(url, snippet)
        cached = snippet_cache.get(snippet_hash)
        if cached:
            return result if cached == "yes" else None
        messages = [
            SystemMessage(content=web_search_validation_instructions.format(
                query=query,
                current_date=get_current_date()
            )),
            HumanMessage(content=f"Snippet: {snippet}")
        ]
        try:
            response = await asyncio.wait_for(llm_call_async(messages), timeout=15)
            answer = response if isinstance(response, str) else getattr(response, 'content', '') or ''
            answer_l = answer.strip().lower()
            verdict = "yes" if "yes" in answer_l else "no"
            snippet_cache[snippet_hash] = verdict
            return result if verdict == "yes" else None
        except Exception:
            return None
    # This function is not used in the main workflow for now.
    return []

# --- Fast Search Results to Deduplication Workflow ---
async def fast_search_results_to_final_urls(state: AgentState) -> AgentState:
    """
    Go straight from search results to deduplication and save results in final_urls (no LLM evaluation).
    """
    search_queries = state.get("search_queries", []) or []
    existing_data = state.get("data", []) or []
    visited_urls = set(state.get("visited_urls", []) or [])
    failed_urls = set(state.get("failed_urls", []) or [])
    errors = []

    if not search_queries:
        logger.warning("No search queries found to evaluate.")
        state.update({
            "data": existing_data,
            "visited_urls": list(visited_urls),
            "error": state.get('error'),
        })
        return state

    if not UnifiedSearcher:
        error_msg = "UnifiedSearcher class not available. Cannot perform search."
        logger.error(error_msg)
        state.update({
            "data": [],
            "visited_urls": list(visited_urls),
            "error": error_msg,
        })
        return state

    # Use dynamic config from state for max_results, fallback to global config
    max_results_per_query = state.get("max_search_results", MAX_SEARCH_RESULTS)
    search_engine = UnifiedSearcher(max_results=max_results_per_query)

    search_tasks = [search_engine.search(q) for q in search_queries]
    search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

    all_results = []
    for i, result_set in enumerate(search_results_list):
        query = search_queries[i] if i < len(search_queries) else f"<unknown_{i}>"
        if isinstance(result_set, Exception):
            errors.append(f"Search failed for query '{query}': {result_set}")
            continue
        if not result_set:
            logger.info(f"No results for query: {query}")
            errors.append(f"No results returned for query: {query}")
            continue
        for r in result_set:
            url = getattr(r, 'url', None)
            # Filter out blocked domains
            if url and url not in visited_urls and url not in failed_urls:
                if any(domain in url.lower() for domain in BLOCKED_DOMAINS):
                    logger.info(f"Skipping blocked domain URL: {url}")
                    continue
                all_results.append(r)
                visited_urls.add(url)

    # Deduplicate by URL
    deduplicated = {item.url: item for item in existing_data + all_results if hasattr(item, 'url') and not any(domain in item.url.lower() for domain in BLOCKED_DOMAINS)}
    final_data = list(deduplicated.values())
    final_urls = [item.url for item in final_data if hasattr(item, 'url')]
    state.update({
        "data": final_data,
        "visited_urls": list(visited_urls),
        "final_urls": final_urls,
        "error": "\n".join(errors) if errors else None
    })
    logger.info(f"fast_search_results_to_final_urls: Final URLs count: {len(final_urls)}")
    
    
    return state
#=============================================================================================
async def extract_content(state: AgentState) -> AgentState:
    """
    Uses the Scraper to extract content from relevant URLs.
    Ranks URLs based on relevance and extracts content from the top N.
    Includes error handling for fetching and content cleaning.
    Skips common non-HTML file types and YouTube URLs based on config.
    Adds a timeout per URL based on config.
    """
    # Use color constants from setup if available
    try:
           from .config import RED, ENDC
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'
        YELLOW = '\033[93m'

    from .config import URL_TIMEOUT
    data = state.get('data', []) # Original search results including snippets
    relevant_contexts = {}
    errors = []
    url_timeout = URL_TIMEOUT # Use timeout from config
    skip_extensions = SKIP_EXTENSIONS


    if not Scraper:
         errors.append("Scraper class not available. Cannot extract content.")
         logger.error(errors[-1])
         state["relevant_contexts"] = {} # Ensure relevant_contexts is initialized
         # Append new errors to existing ones in state
         current_error = state.get('error', '') or ''
         state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else current_error.strip()
         state['error'] = None if state['error'] == "" else state['error']
         return state

    # Check if data is empty
    if not data:
        logger.info("No data found to extract content from.")
        state["relevant_contexts"] = {} # Ensure relevant_contexts is initialized
        # Append new errors to existing ones in state
        current_error = state.get('error', '') or ''
        state['error'] = (current_error + "\nNo data found to extract content from.").strip()
        state['error'] = None if state['error'] == "" else state['error']
        return state

    # Rank URLs based on collected data (Assuming data contains SearchResult objects with url and snippet)
    valid_data = [item for item in data if isinstance(item, SearchResult) and item.url and item.snippet]

    if not valid_data:
         logger.info("No valid data found for ranking.")
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
        logger.info("No query available for ranking URLs. Proceeding without ranking.")
        ranked_urls = [item.url for item in valid_data] # Use original order if no query

    urls_to_process = ranked_urls[:30] # limit to top 30 urls
    logger.info("Relevant and ranked URLs for extraction: %s", urls_to_process)

    # Get the list of previously failed URLs
    failed_urls = state.get('failed_urls', []) or []
    
    # Filter out failed URLs from processing
    urls_to_process = [url for url in urls_to_process if url not in failed_urls]
    if len(ranked_urls[:30]) > len(urls_to_process):
        skipped_count = len(ranked_urls[:30]) - len(urls_to_process)
        logger.info("Skipped %d previously failed URLs", skipped_count)

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
            logger.info("Skipping blocked domain URL (%s): %s", blocked_domain, url)
            continue # Skip blocked domain URLs

        # Check for other file extensions to skip
        skipped = False
        for ext in skip_extensions:
            if url.lower().endswith(ext):
                logger.info("Skipping URL with unsupported extension %s: %s", ext, url)
                skipped = True
                break
        if skipped:
            continue # Skip if the extension is in the skip list

        # Create an async task for processing this URL with a timeout
        async def process_single_url_with_timeout(target_url):
            try:
                extracted_content = None # Initialize extracted content
                # Use the scraper for this URL
                scraped_content = await asyncio.wait_for(
                    scraper.scrape_url(target_url, dynamic=False), # Use dynamic=False explicitly
                    timeout=url_timeout
                )

                if scraped_content.is_successful():
                    # Truncate content if it's excessively long
                    extracted_content = scraped_content.text[:10000] if scraped_content.text else "" # Limit to 10k characters
                    extracted_title = scraped_content.title if scraped_content.title else "Untitled"
                    print(f"[EXTRACT SUCCESS] {target_url}")
                    logger.info("Successfully extracted content from %s", target_url)
                    return target_url, {"content": extracted_content, "title": extracted_title} # Return URL and content/title dict
                else:
                    print(f"[EXTRACT FAIL] {target_url} | Error: {scraped_content.error}")
                    # If scraping failed but didn't timeout, log the error and use snippet if available
                    error_msg = f"Scraping failed for {target_url}: {scraped_content.error}"
                    logger.error(error_msg)
                    # Fallback to snippet if scraping failed for a non-timeout reason
                    original_result = url_to_search_result.get(target_url)
                    if original_result and original_result.snippet:
                        logger.info("Using snippet as fallback for %s after scraping failure.", target_url)
                        fallback_title = original_result.title if hasattr(original_result, 'title') and original_result.title else "Untitled"
                        return target_url, {"content": original_result.snippet[:10000], "title": fallback_title} # Use truncated snippet with title
                    else:
                        # If no snippet, return None for content
                        logger.warning("No snippet available for fallback for %s", target_url)
                        return target_url, None

            except asyncio.TimeoutError:
                # Handle timeout: Use snippet as fallback
                logger.warning(f"Processing timed out for URL: {target_url} after {url_timeout}s.")
                # Fallback to snippet if the URL was originally from search results
                if target_url in url_to_search_result:
                    original_result = url_to_search_result[target_url]
                    if original_result.snippet:
                        logger.info("Using snippet as fallback for %s after timeout.", target_url)
                        fallback_title = original_result.title if hasattr(original_result, 'title') and original_result.title else "Untitled"
                        return target_url, {"content": original_result.snippet[:10000], "title": fallback_title}
                    else:
                        logger.warning("No snippet available for fallback after timeout for %s", target_url)
                        return target_url, None
                else:
                    logger.warning("No snippet fallback available or applicable for %s after timeout.", target_url)
                    return target_url, None

            except Exception as e:
                # Handle any other unexpected exceptions during processing
                error_msg = f"An unexpected error occurred processing {target_url}: {e}"
                logger.error(error_msg, exc_info=True)
                if target_url in url_to_search_result:
                    original_result = url_to_search_result[target_url]
                    if original_result.snippet:
                        logger.info("Using snippet as fallback for %s after unexpected error.", target_url)
                        fallback_title = original_result.title if hasattr(original_result, 'title') and original_result.title else "Untitled"
                        return target_url, {"content": original_result.snippet[:10000], "title": fallback_title}
                    else:
                        logger.warning("No snippet available for fallback after unexpected error for %s", target_url)
                        return target_url, None
                else:
                    logger.warning("No snippet fallback available or applicable for %s after unexpected error.", target_url)
                    return target_url, None

        # Add the processing task for this URL
        processing_tasks.append(process_single_url_with_timeout(url))

    # Run all processing tasks concurrently
    processed_results = await asyncio.gather(*processing_tasks)

    logger.info(f"extract_content: Finished all URL processing tasks. Processing {len(processed_results)} results.")

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
             logger.info(f"Adding URL to failed list: {url}")

    # Update the failed URLs list in state
    current_failed_urls = state.get('failed_urls', []) or []
    updated_failed_urls = list(set(current_failed_urls + new_failed_urls))  # Remove duplicates
    state['failed_urls'] = updated_failed_urls
    
    if new_failed_urls:
        logger.info(f"Added {len(new_failed_urls)} URLs to failed list. Total failed URLs: {len(updated_failed_urls)}")

    state["relevant_contexts"] = relevant_contexts

    # Append new errors to existing ones in state
    # Errors from process_single_url_with_timeout are already logged, no need to add to state['error'] unless critical
    # Let's just add a summary error if relevant_contexts is empty despite having valid_data initially
    if not relevant_contexts and valid_data:
         summary_error = "Failed to extract usable content from any relevant URLs."
         errors.append(summary_error)
         logger.error(summary_error)


    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else current_error.strip()

    # Add logger to inspect the final state["relevant_contexts"]
    logger.info("extract_content: Final state['relevant_contexts'] contains %d items.", len(state.get('relevant_contexts', {})))
    return state
#=============================================================================================
async def embed_and_retrieve(state: AgentState) -> AgentState:
    """
    Enhanced embedding, indexing, and retrieval using hybrid approach.
    Combines BM25 (sparse) and vector search (dense) for improved relevance.
    Focuses only on creating relevant_chunks from relevant_contexts.
    """
    # Use color constants from setup if available
    try:
           from .config import RED, ENDC
    except ImportError:
        RED = '\033[91m'
        ENDC = '\033[0m'

    relevant_contexts = state.get("relevant_contexts", {})
    relevant_chunks = [] # Initialize relevant_chunks as a list of Documents
    errors = []
    N_CHUNKS = 30 # Increased from 10 to 30 for better context coverage

    # Retrieve current error state safely
    current_error_state = state.get('error')

    # Check if we have contexts to process
    if not relevant_contexts:
        logger.warning("No relevant contexts found for embedding and retrieval.")
        new_error = (str(current_error_state or '') + "\nNo relevant contexts found for embedding and retrieval.").strip()
        state['error'] = None if new_error == "" else new_error
        return state

    # Try hybrid retriever first
    if create_hybrid_retriever and embeddings and USE_HYBRID_RETRIEVAL:
        try:
            logger.info("Using hybrid retriever (BM25 + Vector Search)")
            # Create hybrid retriever with configuration from config.py
            hybrid_retriever = create_hybrid_retriever(embeddings=embeddings)
            
            # Build indices
            if hybrid_retriever.build_index(relevant_contexts):
                # Get retrieval queries - use the specific search queries that were used to find documents
                search_queries = state.get("search_queries", [])
                original_query = state.get("new_query")
                retriever_responses = {}
                
                if search_queries:
                    # Use the specific search queries for better relevance and capture individual responses
                    logger.info(f"Using {len(search_queries)} specific search queries for retrieval")
                    relevant_chunks, retriever_responses = hybrid_retriever.retrieve_with_query_responses(search_queries)
                    retrieval_method = "multi-query hybrid retrieval with responses"
                elif original_query:
                    # Fallback to original query if no search queries available
                    logger.info("Using original query for retrieval (fallback)")
                    relevant_chunks = hybrid_retriever.retrieve(original_query)
                    retriever_responses = {original_query: f"Retrieved {len(relevant_chunks)} documents for original query."}
                    retrieval_method = "single-query hybrid retrieval"
                else:
                    logger.warning("No retrieval queries found for hybrid retrieval.")
                    relevant_chunks = []
                    retriever_responses = {}
                    retrieval_method = "no queries"
                
                if relevant_chunks:
                    # Log retrieval stats
                    stats = hybrid_retriever.get_stats()
                    logger.info(f"Hybrid retrieval stats: {stats}")
                    logger.info(f"Retrieved {len(relevant_chunks)} chunks using {retrieval_method}")
                    logger.info(f"Captured {len(retriever_responses)} query responses")
                    
                    state["relevant_chunks"] = relevant_chunks
                    state["retriever_responses"] = retriever_responses
                    return state
                else:
                    logger.warning("No chunks retrieved from hybrid retrieval, falling back to standard approach")
            else:
                logger.warning("Failed to build hybrid retriever indices, falling back to standard approach")
                
        except Exception as e:
            logger.error(f"Hybrid retriever failed: {e}, falling back to standard approach")
            errors.append(f"Hybrid retriever error: {str(e)}")

    logger.error("Hybrid retriever failed and no fallback available")
    errors.append("Hybrid retriever failed to build indices")
    new_error = (str(current_error_state or '') + "\n" + "\n".join(errors)).strip() if errors else (current_error_state.strip() if current_error_state is not None else None)
    state['error'] = None if new_error is None or new_error == "" else new_error
    state["relevant_chunks"] = []

#=============================================================================================
async def fss_retrieve(state: dict) -> dict:
    """
    Creates a Gemini File Search Store, uploads the relevant contexts,
    generates an answer using the store, and then cleans up the store.
    The generated answer is stored in 'analysis_content'.
    """
    contexts_to_use = state.get("relevant_contexts", {})
    search_queries = state.get("search_queries", [])
    session_id = state.get("session_id", "default-session")

    logger.info(f"[FSS Node] fss_retrieve received retrieval_method: '{state.get('retrieval_method')}'")
    logger.debug(f"[FSS Node] Entering fss_retrieve for {len(search_queries)} queries with {len(contexts_to_use)} contexts.")
    print(f"[DEBUG][fss_retrieve] queries: {search_queries[:3]}... (total {len(search_queries)}) | contexts: {list(contexts_to_use.keys())[:3]}... (total {len(contexts_to_use)})")

    try:
        if not contexts_to_use or not search_queries:
            logger.warning("[FSS Node] No relevant contexts or queries available. Skipping FSS batch QA.")
            state["qa_pairs"] = []
            return state

        if not GeminiFileSearchRetriever:
            logger.error("[FSS Node] GeminiFileSearchRetriever class not available.")
            raise RuntimeError("GeminiFileSearchRetriever not available.")

        retriever = GeminiFileSearchRetriever(display_name_prefix=f"crystal-{session_id}")
        qa_dict = await retriever.answer_batch_questions(search_queries, contexts_to_use)
        logger.debug(f"[FSS Node] Batch QA complete. Got {len(qa_dict)} QA pairs.")

        # Convert to list of dicts for downstream compatibility
        qa_pairs = []
        for q in search_queries:
            answer = qa_dict.get(q, "No answer generated.")
            qa_pairs.append({"question": q, "answer": answer, "citations": []})

        state["qa_pairs"] = qa_pairs
        state["file_store_name"] = None
        logger.info(f"[FSS Node] Stored {len(qa_pairs)} QA pairs in state.")

    except Exception as e:
        error_msg = f"[FSS Node] failed: {e}"
        logger.error(error_msg, exc_info=True)
        state["error"] = error_msg
        state["file_store_name"] = None
    return state

#=======================================================================================
async def create_qa_pairs(state: AgentState) -> AgentState:
    """
    Creates Q&A pairs with citations from relevant chunks for each search query.
    Takes relevant_chunks from previous embedding/retrieval step and generates
    structured Q&A pairs that will be used in the report appendix.
    """
    # Get relevant chunks and search queries from state
    relevant_chunks = state.get("relevant_chunks", [])
    search_queries = state.get("search_queries", [])
    
    # Retrieve current error state safely
    current_error_state = state.get('error')
    logger.info(f"Creating Q&A pairs from {len(relevant_chunks)} chunks for {len(search_queries)} queries.")
    
    if not relevant_chunks:
        logger.warning("No relevant chunks available for Q&A creation.")
        state["qa_pairs"] = []
        state["all_citations"] = []
        new_error = (str(current_error_state or '') + "\nNo relevant chunks available for Q&A creation.").strip()
        state['error'] = None if new_error == "" else new_error
        logger.info(f"Created {len(state['qa_pairs'])} Q&A pairs.")
        return state
    
    if not search_queries:
        logger.warning("No search queries available for Q&A creation.")
        state["qa_pairs"] = []
        state["all_citations"] = []
        new_error = (str(current_error_state or '') + "\nNo search queries available for Q&A creation.").strip()
        state['error'] = None if new_error == "" else new_error
        return state
    
    try:
        qa_pairs = []
        all_citations = []
        citation_counter = 1
        
        for i, query in enumerate(search_queries):
            # Find relevant chunks for this specific query
            query_words = set(word.lower() for word in query.split() if len(word) > 3)
            query_chunks = []
            
            # Score chunks based on relevance to this query
            chunk_scores = []
            for chunk in relevant_chunks:
                content_words = set(word.lower() for word in chunk.page_content.split())
                overlap = len(query_words.intersection(content_words))
                chunk_scores.append((overlap, chunk))
            
            # Sort by relevance and take top chunks for this query
            chunk_scores.sort(key=lambda x: x[0], reverse=True)
            query_chunks = [chunk for score, chunk in chunk_scores[:3] if score > 0]  # Top 3 relevant chunks
            
            if not query_chunks:
                # Fallback: use some chunks even if no direct word overlap
                query_chunks = relevant_chunks[i:i+2] if i < len(relevant_chunks) else relevant_chunks[:2]
            
            # Create answer with citations for this query
            answer_parts = []
            query_citations = []
            
            for chunk in query_chunks:
                # Create citation
                citation = {
                    "number": citation_counter,
                    "source": chunk.metadata.get('source', 'Unknown'),
                    "title": chunk.metadata.get('title', 'Untitled'),
                    "url": chunk.metadata.get('source', ''),
                    "content_preview": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                }
                all_citations.append(citation)
                query_citations.append(citation)
                
                # Add chunk content with citation reference
                answer_parts.append(f"{chunk.page_content[:400]}{'...' if len(chunk.page_content) > 400 else ''} [{citation_counter}]")
                citation_counter += 1
            
            # Create the Q&A pair
            qa_pair = {
                "question": query,
                "answer": " ".join(answer_parts) if answer_parts else "No specific information found for this query.",
                "citations": query_citations
            }
            qa_pairs.append(qa_pair)
        
        # Store Q&A pairs and all citations in state
        state["qa_pairs"] = qa_pairs
        state["all_citations"] = all_citations
        logger.info(f"Created {len(qa_pairs)} Q&A pairs with {len(all_citations)} total citations.")
        

        # Print a summary of the appendix state instead of full content
        qa_pairs = state.get("qa_pairs", [])
        all_citations = state.get("all_citations", [])
        if qa_pairs:
            print(f"[APPENDIX SUMMARY] Q&A pairs: {len(qa_pairs)}, Citations: {len(all_citations)}")

    except Exception as e:
        error_msg = f"Error creating Q&A pairs: {e}"
        logger.error(error_msg, exc_info=True)
        state["qa_pairs"] = []
        state["all_citations"] = []
        new_error = (str(current_error_state or '') + "\n" + error_msg).strip()
        state['error'] = None if new_error == "" else new_error
    
    return state

#=============================================================================================
async def AI_evaluate(state: AgentState) -> AgentState:
    """
    Evaluates the Q&A pairs to determine if they provide sufficient depth to answer the original user query.
    Updates 'proceed' based on AI assessment of Q&A pair coverage.
    Suggests follow-up queries if the Q&A pairs don't adequately address the original question.
    Tracks search_iteration_count to prevent infinite loops.
    """

    print(f"[DEBUG][AI_evaluate] state['qa_pairs'] length: {len(state.get('qa_pairs', []))}")
    if state.get('qa_pairs'):
        print(f"[DEBUG][AI_evaluate] First QA: {state['qa_pairs'][0]}")

    qa_pairs = state.get("qa_pairs", [])
    original_query = state.get("new_query", "")
    search_queries = state.get("search_queries", [])
    
    logger.info("AI Evaluation started: evaluating %d Q&A pairs for original query coverage.", len(qa_pairs))

    state["search_iteration_count"] = state.get("search_iteration_count", 0) + 1
    # Use dynamic config from state, fallback to global config
    max_iterations = state.get("max_ai_iterations", MAX_AI_ITERATIONS)
    errors = []
    state["proceed"] = True

    if not llm_call_async:
        msg = "LLM not initialized. Skipping AI evaluation."
        errors.append(msg)
        logger.error(msg)
        state['error'] = msg
        return state

    if not qa_pairs:
        msg = f"No Q&A pairs available for evaluation ({state['search_iteration_count']}/{max_iterations})."
        logger.warning(msg)
        if state["search_iteration_count"] < max_iterations:
            state["proceed"] = False
            state['suggested_follow_up_queries'] = []
            state['knowledge_gap'] = "No Q&A pairs generated this round."
        else:
            state["proceed"] = True
            state['knowledge_gap'] = "Max iterations reached. Proceeding with current Q&A pairs."
        state['error'] = msg
        return state

    # Format Q&A pairs for evaluation
    qa_text = "\n\n".join([
        f"**Q{i+1}: {pair['question']}**\n**A{i+1}:** {pair['answer']}"
        for i, pair in enumerate(qa_pairs)
    ])

    # Create evaluation prompt focused on Q&A coverage
    evaluation_prompt = f"""
    Analyze whether the following Q&A pairs provide sufficient depth and coverage to comprehensively answer the original user query.

    **Original User Query:** {original_query}

    **Available Q&A Pairs:**
    {qa_text}

    **Adversarial Evaluation Task:**
    1. **Play Devil's Advocate**: Act as a skeptical user. Do the answers truly and comprehensively address the original query, or are they superficial?
    2. **Identify Weaknesses**: Pinpoint the weakest part of the collected information. What crucial aspect is missing or poorly explained?
    3. **Check for Contradictions**: Do any of the answers appear to contradict each other? (Note this in the knowledge_gap).
    4. **Assess Sufficiency**: Based on this adversarial review, is the information sufficient to create a high-quality, trustworthy report?

    
    Respond with a JSON object containing:
    {{
        "is_sufficient": boolean,
        "knowledge_gap": "string describing any gaps if not sufficient",
        "follow_up_queries": ["list of specific queries needed to fill gaps"],
        "coverage_assessment": "string explaining how well the Q&A pairs address the original query"
    }}
    """

    messages = [
        SystemMessage(content="You are an expert research analyst evaluating whether Q&A pairs provide sufficient coverage for answering a user's original query."),
        HumanMessage(content=evaluation_prompt)
    ]

    try:
        response = await llm_call_async(messages)
        response_text = response

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
                logger.info("AI_evaluate: Q&A pairs provide sufficient coverage, proceeding to report.")
            else:
                state["proceed"] = False
                state["suggested_follow_up_queries"] = eval_result.follow_up_queries
                state["knowledge_gap"] = eval_result.knowledge_gap
                logger.info("AI_evaluate: Q&A pairs insufficient, generating follow-up queries.")

        else:
            raise ValueError("JSON block not found in response.")

    except Exception as e:
        logger.exception("AI_evaluate error: %s", e)
        state["proceed"] = True
        state["suggested_follow_up_queries"] = []
        state["knowledge_gap"] = f"Fallback triggered due to error: {e}"
        errors.append(str(e))

    # Final check for iteration cap
    if not state["proceed"] and state["search_iteration_count"] >= max_iterations:
        logger.warning("Max iterations reached. Forcing report generation.")
        state["proceed"] = True
        state["suggested_follow_up_queries"] = []
        state["knowledge_gap"] = "Max iterations hit. Report generated with available Q&A pairs."

    # Error handling
    if errors:
        prev_error = state.get("error", "") or ""
        state["error"] = (prev_error + "\n" + "\n".join(errors)).strip()

    return state

#=============================================================================================
unified_report_instruction = (
    "Generate a comprehensive research report with the following THREE-PART structure: "
    "\n\n**PART 1: Original User Query**\n"
    "Present the original user question exactly as submitted."
    "\n\n**PART 2: IntelliSearch Response**\n"
    "Provide an LLM-driven analysis that directly answers the user's query by synthesizing information from the Q&A pairs. "
    "This should be a cohesive, analytical response that covers all search queries without duplication. "
    "Focus on answering the user's original question comprehensively using insights from the collected data. "
    "Target 400-1000 words for this main analytical section."
    "\n\n**PART 3: Appendix - Q&A Pairs with Citations**\n"
    "Present each search query as a question-answer pair with enclosed citations and sources. "
    "Format each pair as: **Q: [search query]** followed by **A: [detailed answer with citations [1], [2], etc.]**. "
    "Include all sources and citations at the end. "
    "This appendix serves as supporting evidence for the main IntelliSearch response."
    "\n\n Use clear markdown formatting with proper headings."
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

#=============================================================================================
async def write_report(state: AgentState) -> AgentState:
    """
    Generates the final report using the new three-part structure:
    1) Original User Query
    2) IntelliSearch Response (LLM analysis based on qa_pairs or file_search)
    3) Appendix with Q&A pairs and citations (only for hybrid method)
    """    
    errors: List[str] = []
    research_topic = state.get('new_query', 'the topic')
    full_session_id = state.get('session_id', 'unknown_session')
    short_session_id = full_session_id.split('-')[0]
    analysis_content = ""
    analysis_filename = None
    logger.info("[write_report] Only file_search retrieval method is supported.")
    print("[DEBUG] Entering write_report")
    logger.info(f"--- Entering write_report node for session '{short_session_id}' ---")


    # PART 1: Original User Query
    part1_query = f"# Research Report\n\n## 1. Original User Query\n\n**{research_topic}**\n\n---\n"

    # PART 2: Synthesize Final Response from QA_pairs
    qa_pairs = state.get("qa_pairs", [])
    if not qa_pairs:
        intellisearch_response = "No Q&A pairs were generated."
    else:
        # Synthesize final response using LLM
        if not llm_call_async:
            intellisearch_response = "LLM not available to synthesize final response."
        else:
            # Prepare a markdown summary of all Q&A pairs
            qa_md = "\n".join([f"### Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])
            synthesis_prompt = f"""
                You are an expert research analyst. Given the following Q&A pairs from a research process, synthesize a comprehensive, analytical response to the original user query below. Use the Q&A pairs as your main evidence base, but write a unified, well-structured report in markdown, with a clear conclusion section.

                Original User Query: {research_topic}

                Q&A Pairs:
                {qa_md}
                """
            messages = [
                SystemMessage(content="You are a helpful research analyst."),
                HumanMessage(content=synthesis_prompt)
            ]
            try:
                intellisearch_response = await llm_call_async(messages)
            except Exception as e:
                logger.error(f"Error during LLM synthesis in write_report: {e}")
                intellisearch_response = "Error synthesizing final response."

    part2_response = f"## 2. IntelliSearch Response\n\n{intellisearch_response}\n\n---\n"
    analysis_content = part1_query + part2_response

    # Appendix: List all QA pairs
    appendix_md = "\n".join([f"### Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])
    appendix_content = f"## Appendix: Q&A Pairs\n\n{appendix_md if appendix_md else 'No Q&A pairs available.'}"
    appendix_filename = None

    # Ensure file store is deleted after session closes
    try:
        if "file_store_name" in state and state["file_store_name"]:
            if 'delete_gemini_file_search_store' in globals():
                delete_gemini_file_search_store(state["file_store_name"])
            state["file_store_name"] = None
    except Exception as e:
        logger.error(f"Error deleting file store: {e} it might have been already deleted.")

    # Save analysis to Firestore
    db = None
    try:
        from google.cloud import firestore
        db = firestore.Client()
    except (ImportError, Exception) as e:
        errors.append(f"Firestore client not available: {e}")

    if db and analysis_content:
        analysis_filename = f"{short_session_id}_analysis.txt"
        try:
            db.collection("report_files").document(analysis_filename).set({"content": analysis_content})
            logger.info(f"Successfully saved analysis report to Firestore: {analysis_filename}")
        except Exception as e:
            errors.append(f"Failed to save analysis to Firestore: {e}")

    # Final state update
    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else (current_error.strip() if current_error else None)
    if state['error'] == '': state['error'] = None

    state.update({
        "analysis_content": analysis_content,
        "appendix_content": appendix_content,
        "analysis_filename": analysis_filename,
        "appendix_filename": appendix_filename,
    })

    # Clean up intermediate data
    state.update({
        'data': [],
        'relevant_contexts': {},
        'suggested_follow_up_queries': [],
        'knowledge_gap': "",
        'rationale': "",
        'iteration_count': 0,
    })
    logger.info("Report generation completed. State updated.")
    return state

logger.info("nodes.py loaded with LangGraph node functions.")
#=============================================================================================