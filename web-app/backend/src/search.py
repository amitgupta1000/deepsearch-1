import os, sys
import asyncio
import time
import random
import json
import re # Added re import here as it's used in cache key sanitization
import requests  # Add requests for direct Serper API calls
from typing import List, Dict, Any, Optional, Union, Set
from functools import lru_cache
from dataclasses import dataclass
import logging
from urllib.parse import quote_plus, urlparse

# Serper API endpoint
SERPER_ENDPOINT = "https://google.serper.dev/search"

# Try to import optional dependencies with fallbacks
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    logging.warning("aiohttp not available. Some search features may be limited.")
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    logging.warning("BeautifulSoup not available. Some search features may be limited.")
    BeautifulSoup = None
    BS4_AVAILABLE = False

try:
    from fake_useragent import UserAgent
    FAKE_UA_AVAILABLE = True
except ImportError:
    logging.warning("fake_useragent not available. Using default user agent.")
    UserAgent = None
    FAKE_UA_AVAILABLE = False

try:
    from googlesearch import search as google_search
    GOOGLESEARCH_AVAILABLE = True
except ImportError:
    logging.warning("googlesearch not available. Google search functionality disabled.")
    google_search = None
    GOOGLESEARCH_AVAILABLE = False


try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    logging.warning("pydantic not available. Using basic class instead of BaseModel.")
    BaseModel = object
    PYDANTIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(__name__)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


try:
    from .api_keys import SERPER_API_KEY
    logging.info("Successfully imported API keys from api_keys.py")
except ImportError:
    logging.error("Could not import API keys from api_keys.py. API keys will not be available.")
    # Define dummy variables to prevent NameError later, but warn the user
    SERPER_API_KEY = None

# Only set environment variable if SERPER_API_KEY is not None
# Import unified configuration
from .config import (
    SERPER_API_KEY, USER_AGENT, CACHE_ENABLED, CACHE_TTL,
    MAX_SEARCH_RESULTS as DEFAULT_MAX_RESULTS, BLOCKED_DOMAINS
)

if SERPER_API_KEY is not None:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
serper_api_key = os.environ.get("SERPER_API_KEY")

# Cache settings from unified config
CACHE_DIR = os.path.expanduser("~/.deepsearcher_amit/cache/search") # Directory for search cache

# Ensure cache directory exists when module is imported
# This runs once when the search.py module is loaded.
if CACHE_ENABLED and not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Global cache directory created at: {CACHE_DIR}")
    except Exception as e:
        logger.warning(f"Could not create global cache directory {CACHE_DIR}: {e}. Disabling cache globally.")
        CACHE_ENABLED = False # Disable globally if creation fails

# URL classification function for search results
def classify_url(url: str) -> str:
    """
    Classify URL by domain type for search result categorization.
    
    Args:
        url: The URL to classify
        
    Returns:
        A classification tag (e.g., 'news', 'academic', 'government', 'general')
    """
    try:
        domain = urlparse(url).netloc.lower()
        
        # News sites
        if any(news_domain in domain for news_domain in [
            'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com',
            'bbc.com', 'cnn.com', 'theguardian.com', 'nytimes.com', 'washingtonpost.com'
        ]):
            return 'news'
        
        # Academic and research
        if any(academic_indicator in domain for academic_indicator in [
            '.edu', '.ac.', 'scholar.google', 'arxiv.org', 'researchgate.net',
            'jstor.org', 'pubmed.ncbi', 'sciencedirect.com'
        ]):
            return 'academic'
        
        # Government sites
        if any(gov_indicator in domain for gov_indicator in [
            '.gov', '.mil', 'europa.eu', 'un.org', 'who.int'
        ]):
            return 'government'
        
        # Financial sites
        if any(finance_domain in domain for finance_domain in [
            'sec.gov', 'finra.org', 'nasdaq.com', 'nyse.com', 'morningstar.com',
            'yahoo.com/finance', 'google.com/finance'
        ]):
            return 'financial'
        
        # Default classification
        return 'general'
        
    except Exception:
        return 'general'

# Moved SearchResult definition here to be used by UnifiedSearcher and Pydantic models

@dataclass # Using dataclass as it's lighter and sufficient here, removed Pydantic BaseModel inheritance to avoid conflict
class SearchResult:
    """
    Dataclass to store individual search results.

    Attributes:
        url: The URL of the search result.
        title: The title of the search result.
        snippet: A brief description or snippet of the content.
        source: The search engine that provided the result (e.g., "Google", "DuckDuckGo").
    """
    url: str
    title: str
    snippet: str
    source: str

    def __str__(self) -> str:
        """String representation of search result for easy printing."""
        return f"Title: {self.title}\nURL: {self.url}\nSnippet: {self.snippet}\nSource: {self.source}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult object to a dictionary format, useful for serialization (e.g., caching)."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source
        }

class UnifiedSearcher:
    """
    Unified search engine that can query multiple search engines concurrently
    with features like caching, retries, and basic result deduplication.
    """

    def __init__(self, max_results: int = DEFAULT_MAX_RESULTS, cache_enabled: bool = CACHE_ENABLED, cache_ttl: int = CACHE_TTL, **kwargs):
        """
        Initialize the unified searcher instance.

        Args:
            max_results: The maximum number of *unique* results to return in total
                         from a unified search (after combining and deduplicating results
                         from all engines). Note that each engine might be asked
                         for `max_results` or more internally.
            cache_enabled: Boolean flag to enable or disable caching for this instance.
                           Defaults to the global CACHE_ENABLED setting.
            cache_ttl: Time-to-live in seconds for cached search results for this instance.
                       Defaults to the global CACHE_TTL setting.
        """
        # Clamp max_results to a reasonable range (e.g., 1 to 50) to prevent excessively large requests.
        self.max_results = max(1, min(max_results, 50))
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.default_engine = "serper"
        self.max_retries = kwargs.get("max_retries", 2)
        self.retry_timeout = kwargs.get("retry_timeout", 15)
        logger.info(f"Initializing UnifiedSearcher with max_results={self.max_results}, cache_enabled={cache_enabled}")

        # Consolidate user_agent handling: Prefer environment variable if set, then try fake_useragent, else use default.
        if 'USER_AGENT' in os.environ:
            self.user_agent = os.environ.get('USER_AGENT')
            logger.info(f"Using USER_AGENT from environment: {self.user_agent[:50]}...")
        else:
            try:
                # Attempt to use fake_useragent for a more realistic user agent string.
                ua = UserAgent()
                self.user_agent = ua.random
                logger.info(f"Using random user agent: {self.user_agent[:50]}...")
            except Exception as e:
                logger.warning(f"Could not generate random user agent: {e}. Using default.")
                # Fallback to a standard default user agent if fake_useragent fails.
                self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'


        self.default_engine = "serper"  # Default engine to use if none are specified in search calls.
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        # Set to track queries currently being processed by this instance to avoid redundant work.
        self.in_progress_queries: Set[str] = set()
        # Dictionary to store asyncio Semaphores. A separate semaphore is needed for each event loop
        # if the instance is used across different threads/loops (though typically async runs on one loop).
        self._semaphores = {}
        # Lock to ensure thread-safe access when creating/retrieving semaphores from the _semaphores dictionary.
        self._semaphore_lock = asyncio.Lock()

        # Check for SERPER_API_KEY availability (now using direct API calls)
        if not SERPER_API_KEY:
            logger.warning("SERPER_API_KEY environment variable is not set. Serper search engine will not work.")
        else:
            logger.info("Serper API key found. Serper search engine is enabled (using direct API).")

    async def _check_cache(self, query: str, engine: str) -> Optional[List[SearchResult]]:
        """
        Check if search results for a given query and engine are available in cache and not expired.

        Args:
            query: The search query.
            engine: The name of the search engine (used in the cache key).

        Returns:
            A list of SearchResult objects from the cache if found and valid, otherwise None.
        """
        if not self.cache_enabled:
            logger.debug("Cache check skipped: cache is disabled.")
            return None

        try:
            # Generate a unique and filesystem-safe cache key for the query and engine.
            cache_key = self._get_formatted_query(query, engine)
            # Sanitize the key to be safe for filenames: replace invalid characters with underscores.
            sanitized_key = "".join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in cache_key)
            # Replace multiple consecutive underscores/dashes/dots with a single underscore.
            sanitized_key = re.sub(r'[_\-\.]+', '_', sanitized_key)
            # Trim leading/trailing underscores.
            sanitized_key = sanitized_key.strip('_')

            # Add a hash component for very long keys to ensure uniqueness and stay within filesystem path limits.
            import hashlib # hashlib is needed here for the hash component of the cache key.
            if len(sanitized_key) > 150: # Arbitrary threshold for applying hashing.
                 key_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
                 # Combine a truncated sanitized key with the hash.
                 sanitized_key = sanitized_key[:150] + "_" + key_hash
                 logger.debug(f"Long cache key for '{query}' on '{engine}', using hash. Final key length: {len(sanitized_key)}")

            # Final limit on the key length to avoid filesystem issues and add the .json extension.
            cache_filename = sanitized_key[:250] + ".json"
            # Construct the full cache file path.
            cache_path = os.path.join(CACHE_DIR, cache_filename)
            logger.debug(f"Checking cache at {cache_path} for query '{query}' on engine '{engine}'")

        except Exception as e:
            logger.error(f"Error generating cache path for query '{query}' on engine '{engine}': {e}")
            return None

        # Use asyncio.to_thread to perform blocking file operations without blocking the event loop.
        # Check file existence asynchronously.
        cache_exists = await asyncio.to_thread(os.path.exists, cache_path)
        if not cache_exists:
            logger.debug(f"Cache file not found for '{query}' at {cache_path}")
            return None

        try:
            # Check file modification time for expiration asynchronously.
            mod_time = await asyncio.to_thread(os.path.getmtime, cache_path)
            if time.time() - mod_time > self.cache_ttl:
                logger.info(f"Cache for '{query}' on '{engine}' expired (TTL: {self.cache_ttl}s). Deleting {cache_path}")
                # Delete the expired cache file asynchronously.
                try:
                    await asyncio.to_thread(os.remove, cache_path)
                except OSError as e:
                     logger.warning(f"Could not delete expired cache file {cache_path}: {e}")
                return None

            # Load cached content asynchronously by running the blocking file read in a thread.
            logger.info(f"Loading cache from {cache_path} for '{query}' on '{engine}'")

            def _load_cache_sync():
                 # Synchronous file reading function.
                 with open(cache_path, 'r', encoding='utf-8') as f:
                     return json.load(f)

            data = await asyncio.to_thread(_load_cache_sync)

            # Basic validation of cache data structure: ensure it's a list of dictionaries with required keys.
            if not isinstance(data, list) or not data:
                logger.warning(f"Invalid cache file format for '{query}' on '{engine}': Expected list of results. Deleting {cache_path}")
                # Delete the invalid cache file asynchronously.
                try:
                    await asyncio.to_thread(os.remove, cache_path)
                except OSError as e:
                     logger.warning(f"Could not delete invalid cache file {cache_path}: {e}")
                return None # Return None to indicate cache is invalid

            required_keys = ["url", "title", "snippet", "source"]
            cached_results = []
            # Iterate through the loaded data and convert dictionaries back to SearchResult objects.
            for item in data:
                 # Validate each item in the list.
                 if not isinstance(item, dict) or not all(k in item for k in required_keys):
                    logger.warning(f"Invalid item format in cache file for '{query}' on '{engine}'. Skipping item or deleting file.")
                    # If any item is invalid, consider the whole cache file corrupted and delete it.
                    try:
                        await asyncio.to_thread(os.remove, cache_path)
                        logger.warning(f"Deleted invalid cache file: {cache_path}")
                        return None # Return None to indicate cache is invalid
                    except OSError as e:
                        logger.warning(f"Could not delete invalid cache file {cache_path}: {e}")
                        return None # Cannot recover, return None

                 # Create SearchResult object from the dictionary item.
                 cached_results.append(SearchResult(
                    url=item.get("url", ""), # Use .get() with default to handle potential missing keys gracefully.
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    source=item.get("source", "cached") # Default source for cached items if not present.
                ))

            logger.info(f"Cache hit for '{query}' on '{engine}'")
            return cached_results # Return the list of cached SearchResult objects.

        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.error(f"Error reading or parsing cache file for '{query}' on '{engine}' at {cache_path}: {e}. Attempting to remove corrupted file.")
            # Attempt to remove potentially corrupted cache file asynchronously.
            try:
                if await asyncio.to_thread(os.path.exists, cache_path):
                    await asyncio.to_thread(os.remove, cache_path)
                    logger.info(f"Removed corrupted cache file: {cache_path}")
            except OSError as cleanup_e:
                logger.warning(f"Could not remove corrupted cache file {cache_path}: {cleanup_e}")
            return None # Return None on error.
        except Exception as e:
            logger.error(f"Unexpected error during cache check for '{query}' on '{engine}': {e}")
            return None # Return None on unexpected error.


    async def _save_to_cache(self, query: str, engine: str, results: List[SearchResult]) -> bool:
        """
        Save search results to the cache.

        Args:
            query: The search query.
            engine: The name of the search engine.
            results: The list of SearchResult objects to cache.

        Returns:
            True if caching was successful, False otherwise.
        """
        if not self.cache_enabled:
            logger.debug("Cache save skipped: cache is disabled.")
            return False

        # Only cache non-empty lists of results.
        if not results:
            logger.debug(f"Cache save skipped for query '{query}' on engine '{engine}': No results to save.")
            return False

        # Initialize temp_cache_path to None
        temp_cache_path = None

        try:
            # Generate and sanitize the cache filename using the same logic as _check_cache.
            cache_key = self._get_formatted_query(query, engine)
            sanitized_key = "".join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in cache_key)
            sanitized_key = re.sub(r'[_\-\.]+', '_', sanitized_key).strip('_')
            # hashlib is needed here for the hash component of the cache key.
            import hashlib
            if len(sanitized_key) > 150:
                 key_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
                 sanitized_key = sanitized_key[:150] + "_" + key_hash
            cache_filename = sanitized_key[:250] + ".json"
            cache_path = os.path.join(CACHE_DIR, cache_filename)
            logger.debug(f"Saving cache to {cache_path} for query '{query}' on engine '{engine}'")

        except Exception as e:
            logger.error(f"Error generating cache path for query '{query}' on engine '{engine}' during save: {e}")
            return False # Return False on error.

        # Ensure the cache directory exists before attempting to write the file.
        # Use asyncio.to_thread for the blocking directory creation call.
        try:
            await asyncio.to_thread(os.makedirs, os.path.dirname(cache_path), exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating cache directory {os.path.dirname(cache_path)}: {e}")
            return False # Return False on error.

        try:
            # Prepare the list of results for JSON serialization.
            # Explicitly convert each item to a dictionary, handling potential type issues.
            data_to_cache = []
            for item in results:
                if hasattr(item, 'to_dict') and callable(item.to_dict):
                    # If the object has a to_dict method (like our SearchResult dataclass), use it.
                    data_to_cache.append(item.to_dict())
                elif isinstance(item, dict):
                    # If the item is already a dictionary, use it directly.
                    data_to_cache.append(item)
                else:
                    # If it's neither, try to construct a dictionary from expected attributes.
                    # This is a fallback for unexpected object types.
                    try:
                         data_to_cache.append({
                             "url": getattr(item, 'url', ''),
                             "title": getattr(item, 'title', ''),
                             "snippet": getattr(item, 'snippet', ''),
                             "source": getattr(item, 'source', 'unknown') # Default source if attribute missing
                         })
                         logger.warning(f"Caching a non-standard object type by attributes for query '{query}': {type(item)}")
                    except Exception as attr_e:
                        logger.error(f"Could not convert item to dictionary by attributes for caching for query '{query}': {item}. Error: {attr_e}")
                        continue # Skip this item if conversion fails


            logger.info(f"Saving cache for '{query}' on '{engine}' to {cache_path}")
            # Use a temporary file for atomic write: write to a temp file first, then rename it.
            # This prevents corrupted cache files if the writing process is interrupted.
            temp_cache_path = cache_path + ".tmp"

            def _save_cache_sync():
                # Synchronous file writing function.
                with open(temp_cache_path, 'w', encoding='utf-8') as f:
                    # Dump the data to the temporary file. Use compact format (indent=None) to save space.
                    json.dump(data_to_cache, f, ensure_ascii=False, indent=None)
                # Atomically rename the temporary file to the final cache path.
                # os.replace is generally atomic on POSIX systems for files within the same filesystem.
                os.replace(temp_cache_path, cache_path)

            # Run the synchronous save operation in a thread.
            await asyncio.to_thread(_save_cache_sync)

            logger.info(f"Cache saved successfully for '{query}' on '{engine}'")
            return True # Return True on success.

        except (IOError, OSError) as e:
            logger.error(f"Error writing cache file for '{query}' on '{engine}' at {cache_path}: {e}")
            # Attempt to clean up the temporary file if it exists asynchronously.
            if temp_cache_path and await asyncio.to_thread(os.path.exists, temp_cache_path):
                try:
                    await asyncio.to_thread(os.remove, temp_cache_path)
                except OSError as cleanup_e:
                    logger.warning(f"Could not remove temporary cache file {temp_cache_path}: {cleanup_e}")
            return False # Return False on error.
        except Exception as e:
            logger.error(f"Unexpected error during cache save for '{query}' on '{engine}': {e}")
             # Attempt to clean up the temporary file if it exists asynchronously on unexpected error.
            if temp_cache_path and await asyncio.to_thread(os.path.exists, temp_cache_path):
                try:
                    await asyncio.to_thread(os.remove, temp_cache_path)
                except OSError as cleanup_e:
                    logger.warning(f"Could not remove temporary cache file {temp_cache_path}: {cleanup_e}")
            return False # Return False on unexpected error.


    async def search(self, query: str, engines: Optional[List[str]] = None, force_refresh: bool = False) -> List[SearchResult]:
        """
        Search for a query using multiple search engines concurrently.

        Combines results from specified engines, deduplicates them, and limits the final list.

        Args:
            query: The search query string.
            engines: A list of search engine names to use (e.g., ["google"]).
                     If None, the default engine ("serper") is used.
                     Supported engines: "google", "serper".
            force_refresh: If True, ignore cached results and force fresh searches.

        Returns:
            A list of unique SearchResult objects, limited by self.max_results.
            Returns an empty list if no valid engines are specified or if all searches fail.
        """
        logger.info(f"Starting unified search for query: '{query}' with engines: {engines}, force_refresh: {force_refresh}")
        
        # Enhance query with current date context for better recency
        from datetime import datetime
        current_year = datetime.now().year
        
        # Add current date context to prioritize recent information (unless it's already specific)
        if str(current_year) not in query and str(current_year-1) not in query and "recent" not in query.lower():
            enhanced_query = f"{query} recent latest {current_year}"
            logger.debug(f"Enhanced query with recency context: '{enhanced_query}'")
        else:
            enhanced_query = query
            logger.debug(f"Query already contains date/recency context: '{query}'")
        
        # Use default engine if none are provided.
        if engines is None:
            engines = [self.default_engine]

        # Ensure engines is a list, even if a single string was passed.
        if isinstance(engines, str):
            engines = [engines]

        # Use a set to get unique engine names and convert to lowercase for consistent lookup.
        unique_engines = set(engine.lower() for engine in engines)
        logger.debug(f"Unique engines to search: {unique_engines}")

        tasks = [] # List to hold async tasks for each search engine.
        # Create a task for each unique engine.
        for engine in unique_engines:
            # Skip unsupported engines with a warning.
            if engine not in ["google", "serper"]:
                logger.warning(f"Skipping unsupported search engine: {engine}")
                continue

            # Check cache first unless force_refresh is True.
            if not force_refresh:
                cached_results = await self._check_cache(enhanced_query, engine)
                if cached_results:
                    logger.info(f"Using cached results for '{enhanced_query}' on {engine}")
                    # If cached results are found, create a completed task that immediately returns them.
                    # This allows cached results to be included in the asyncio.gather call alongside live searches.
                    # asyncio.sleep(0, result=...) is a simple way to create an already-completed awaitable.
                    tasks.append(asyncio.create_task(asyncio.sleep(0, result=cached_results)))
                    continue # Skip live search if cache hit.

            # If not using cache, add a task to perform a live search for the engine, wrapped in retry logic.
            logger.debug(f"Adding live search task for engine: {engine}")
            # Map engine name to the corresponding internal search method.
            if engine == "google":
                tasks.append(asyncio.create_task(self._search_with_retry(self._search_google, enhanced_query)))
            elif engine == "serper":
                 # Check if Serper API key is available before creating the task.
                 if not SERPER_API_KEY:
                      logger.warning(f"Serper API key missing. Skipping Serper search.")
                 else:
                    tasks.append(asyncio.create_task(self._search_with_retry(self._search_serper, enhanced_query)))

        # If no valid tasks were created (e.g., all engines unsupported or Serper key missing), return an empty list.
        if not tasks:
            logger.warning(f"No valid search engine tasks created for query: '{enhanced_query}'.")
            return []

        logger.info(f"Running {len(tasks)} search tasks concurrently for query: '{enhanced_query}'")
        results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f"Finished asyncio.gather for query: '{query}'. Received {len(results_from_gather)} results/exceptions.")

        # Process the results from asyncio.gather.
        all_results = [] # List to collect all successful SearchResult objects from all engines.
        # Iterate through the results returned by gather.
        for result in results_from_gather:
            # Check if the result is an exception.
            if isinstance(result, Exception):
                # Exceptions are logged by the _search_with_retry method.
                # We just note here that a task failed.
                logger.error(f"A search task resulted in an exception: {result}")
                pass # Skip adding results from this failed task.
            # Check if the result is a list (expected format for successful search results).
            elif isinstance(result, list):
                # Extend the all_results list with the results from this engine.
                all_results.extend(result)
                logger.debug(f"Task returned {len(result)} search results.")
            else:
                 # Log a warning for any unexpected return types from a task.
                 logger.warning(f"Search task returned unexpected type: {type(result)}. Skipping result.")

        logger.debug(f"Combined raw results before deduplication: {len(all_results)}")

        # Deduplicate results based on URL to ensure uniqueness.
        unique_urls = set() # Set to keep track of URLs already seen.
        unique_results = [] # List to store unique SearchResult objects.
        for result in all_results:
            # Ensure the item is a SearchResult object and has a valid URL before attempting to deduplicate.
            if isinstance(result, SearchResult) and result.url and urlparse(result.url).scheme in ['http', 'https']:
                # If the URL hasn't been seen before, add it to the set and add the result to the unique list.
                if result.url not in unique_urls:
                    unique_urls.add(result.url)
                    unique_results.append(result)
            else:
                # Log if a result is skipped during deduplication due to being invalid or malformed.
                logger.debug(f"Skipping invalid or malformed result during deduplication: {result}")

        logger.debug(f"Results after deduplication: {len(unique_results)}")

        # Shuffle the unique results to mix results from different engines before limiting.
        # Avoid shuffling an empty list.
        if unique_results:
            random.shuffle(unique_results)
            logger.debug("Shuffled unique search results.")

        # Limit the final list of unique results to the specified maximum number of results.
        final_limited_results = unique_results[:self.max_results]
        logger.info(f"Final search results count (after limit): {len(final_limited_results)}")

        return final_limited_results # Return the final list of unique and limited results.


    async def _search_with_retry(
        self,
        search_function: callable,
        query: str,
        max_retries: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Wrapper function that adds retry logic to an asynchronous search function.

        Attempts to execute the provided `search_function` up to `max_retries + 1` times
        in case of exceptions, with increasing random delays between retries.

        Args:
            search_function: The asynchronous search function to wrap (e.g., self._search_google).
                             Expected to return a List[SearchResult] on success or raise an Exception on failure.
            query: The search query string to pass to the search function.
            max_retries: Optional override for maximum retry attempts. Falls back to `self.max_retries` if not set.

        Returns:
            A list of SearchResult objects if the search function succeeds within the allowed retries.
            Returns an empty list if all attempts fail.
        """
        retries = 0
        effective_max_retries = max_retries if max_retries is not None else self.max_retries
        engine_name = search_function.__name__.replace("_search_", "")

        while retries <= effective_max_retries:
            try:
                if retries > 0:
                    delay = random.uniform(1.0, 3.0) * retries
                    logger.info(f"Retry {retries + 1}/{effective_max_retries + 1} for engine '{engine_name}' after {delay:.2f}s delay.")
                    await asyncio.sleep(delay)

                semaphore = await self._get_semaphore()
                async with semaphore:
                    results = await asyncio.wait_for(
                        search_function(query),
                        timeout=self.retry_timeout
                    )

                    if results:
                        await self._save_to_cache(query, engine_name, results)
                        logger.debug(f"Successfully cached results for engine '{engine_name}' and query '{query}'.")
                    else:
                        logger.debug(f"No results returned by '{engine_name}' for query '{query}'. Not caching.")

                    logger.info(f"Search successful for engine '{engine_name}' on attempt {retries + 1}.")
                    return results

            except asyncio.TimeoutError:
                logger.warning(f"Search engine '{engine_name}' timed out after {self.retry_timeout}s on attempt {retries + 1} for query '{query}'.")
            except Exception as e:
                logger.warning(f"Search attempt {retries + 1}/{effective_max_retries + 1} failed for engine '{engine_name}' and query '{query}': {e}")

            retries += 1

        logger.error(f"All {effective_max_retries + 1} attempts failed for engine '{engine_name}' search for query '{query}'.")
        return []


    async def _search_google(self, query: str) -> List[SearchResult]:
        """
        Search Google for a query using the googlesearch-python library and subsequent HTML scraping for details.

        Note: The googlesearch library is blocking, and scraping Google's HTML is fragile as
        their structure changes frequently. This method is a best effort.

        Args:
            query: The search query string.

        Returns:
            A list of SearchResult objects obtained from Google.
        """
        logger.info(f"Starting Google search for query: '{query}'")
        try:
            # Check if google_search is available
            if google_search is None:
                logger.warning("googlesearch library not available. Google search functionality disabled.")
                return []
                
            # Use the googlesearch-python library which is blocking. Run it in a thread pool.
            loop = asyncio.get_event_loop()
            # Request more results than max_results from the initial googlesearch call
            # to have a higher chance of getting `self.max_results` unique results
            # after enrichment and potential filtering.
            google_results_urls = await loop.run_in_executor(
                None, # Use the default thread pool executor.
                lambda: list(google_search(query, num=self.max_results * 2)) # num is results per page, request a larger batch.
            )

            results = []
            # googlesearch only returns URLs. Create initial SearchResult objects.
            for url in google_results_urls:
                 # Basic URL validation for results from external libraries.
                if url and urlparse(url).scheme in ['http', 'https']:
                    results.append(SearchResult(
                        url=url,
                        title="",  # Title will be enriched later by scraping Google's page.
                        snippet="",  # Snippet will be enriched later by scraping Google's page.
                        source="Google" # Explicitly set the source.
                    ))
                else:
                    # Log a warning if an invalid URL is received from googlesearch.
                    logger.warning(f"Skipping invalid URL from Google search: {url}")


            # Attempt to enrich the results with titles and snippets by scraping the Google search results page.
            if results:
                logger.debug(f"Attempting to enrich {len(results)} Google results for query: '{query}'")
                # Call the asynchronous enrichment method.
                await self._enrich_google_results(results, query)

            # Limit the final list of results to self.max_results after enrichment.
            final_results = results[:self.max_results]

            logger.info(f"Finished Google search for query: '{query}'. Found {len(final_results)} results.")
            return final_results # Return the list of SearchResult objects.
        except Exception as e:
            # Catch any exception during the process and log it.
            logger.error(f"Error during Google search for query '{query}': {e}", exc_info=True)
            # Re-raise the exception to be caught by the retry mechanism in _search_with_retry.
            raise

    async def _enrich_google_results(self, results: List[SearchResult], query: str) -> None:
        """
        Asynchronously enrich a list of Google SearchResult objects with titles and snippets
        by scraping the current Google search results page for the given query.

        Modifies the SearchResult objects in the input list in place.

        Args:
            results: A list of SearchResult objects to enrich (must contain URLs).
            query: The original search query string.
        """
        if not results:
            logger.debug("No results to enrich for Google.")
            return # Nothing to do if the input list is empty.

        logger.debug(f"Starting Google results enrichment for query: '{query}' with {len(results)} initial results.")
        try:
            # Use aiohttp for asynchronous HTTP request to fetch the Google search results page HTML.
            # Set a timeout for the request.
            timeout = aiohttp.ClientTimeout(total=20)  # Increased timeout as loading/parsing can take time.
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Construct the URL for the Google search results page.
                # Request 'num' results equal to self.max_results for parsing.
                enrichment_url = f"https://www.google.com/search?q={quote_plus(query)}&num={self.max_results}"
                # Include the user agent in the request headers.
                headers = {"User-Agent": self.user_agent}

                logger.debug(f"Fetching Google search page for enrichment: {enrichment_url}")
                # Make the asynchronous GET request.
                async with session.get(enrichment_url, headers=headers) as response:
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx status codes).
                    html = await response.text() # Get the HTML content of the page.

            # Parse the HTML content using BeautifulSoup. This is a blocking operation,
            # so run it in a thread pool to avoid blocking the event loop.
            def parse_google_html_sync(html_content):
                 return BeautifulSoup(html_content, features="lxml") # Use lxml for potentially faster parsing.

            soup = await asyncio.to_thread(parse_google_html_sync, html)

            # Find the main search result elements in the parsed HTML.
            # Google's HTML structure is prone to change, so the CSS selector 'div.g' might need updates.
            search_divs = soup.find_all("div", class_="g")
            logger.debug(f"Found {len(search_divs)} potential result divs on Google search page.")

            enriched_count = 0 # Counter for the number of results successfully enriched.
            # Iterate through the original list of SearchResult objects.
            for initial_result in results:
                # Try to find the HTML element corresponding to the current SearchResult's URL.
                # This assumes the URL from googlesearch matches an href attribute in the scraped HTML.
                link_elem = soup.find("a", href=initial_result.url)

                # If a link element with the matching URL is found.
                if link_elem:
                    # Try to find the parent element that represents the entire search result block.
                    # This heuristic depends on Google's HTML structure.
                    result_div = link_elem.find_parent("div", class_="g")

                    # If the parent result div is found.
                    if result_div:
                        # Extract the title from the result div (usually within an h3 tag).
                        title_elem = result_div.find("h3")
                        if title_elem:
                            initial_result.title = title_elem.text.strip()

                        # Extract the snippet from the result div. Look for common snippet container classes.
                        # The class names ('VwiC3b', 's3gt2') are based on current Google HTML structure and may need updates.
                        snippet_elem = result_div.find("div", class_=lambda c: c and ("VwiC3b" in c or "s3gt2" in c))
                        if snippet_elem:
                            # Get the text content of the snippet element.
                            initial_result.snippet = snippet_elem.get_text(separator=" ", strip=True)
                            # Clean up potential extra whitespace/newlines in the snippet.
                            initial_result.snippet = re.sub(r'\s+', ' ', initial_result.snippet).strip()

                        enriched_count += 1 # Increment the counter.
                        logger.debug(f"Enriched result for URL: {initial_result.url}")
                    else:
                         # Log if the parent result div could not be found for a URL.
                         logger.debug(f"Could not find parent result div for URL: {initial_result.url}")
                else:
                    # Log if the link element for a URL could not be found in the scraped HTML.
                    logger.debug(f"Could not find link element for URL: {initial_result.url} in enrichment HTML.")

            logger.debug(f"Finished Google results enrichment. Enriched {enriched_count} results.")

        except aiohttp.ClientResponseError as e:
            # Log HTTP errors during the request (e.g., 404, 500).
            logger.warning(f"HTTP error during Google enrichment for query '{query}': Status {e.status}, Message: {e.message}")
        except aiohttp.ClientTimeout as e:
             # Log timeout errors during the request.
             logger.warning(f"Timeout during Google enrichment for query '{query}': {e}")
        except Exception as e:
            # Catch and log any other unexpected errors during enrichment.
            logger.error(f"Error during Google results enrichment for query '{query}': {e}", exc_info=True)
            # Do NOT re-raise here. Enrichment is supplementary, and its failure should not fail the main search result retrieval.


    async def _search_bing(self, query: str) -> List[SearchResult]:
        """
        Stub for Bing search. Support removed to simplify the unified searcher.
        Returns an empty list and logs a warning if called.
        """
        logger.warning("Bing search support has been removed. Returning empty result list.")
        return []


    async def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """
        Stub for Wikipedia search. Support removed to simplify the unified searcher.
        Returns an empty list and logs a warning if called.
        """
        logger.warning("Wikipedia search support has been removed. Returning empty result list.")
        return []


    async def _search_serper(self, query: str) -> List[SearchResult]:
        """
        Search Serper using direct API calls instead of LangChain wrapper.
        
        This approach provides better control over the API requests and error handling.

        Args:
            query: The search query string.

        Returns:
            A list of SearchResult objects obtained from Serper.
            Returns an empty list if the API key is not set or the API call fails.
        """
        logger.info(f"Starting Serper search for query: '{query}'")
        
        # Check if API key is available
        if not SERPER_API_KEY:
            logger.warning(f"Serper API key is not set. Skipping Serper search for query: '{query}'.")
            return []

        try:
            # Prepare headers and payload for direct API call
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            
            # Use max_results for this query, with a reasonable limit
            max_results_per_query = min(self.max_results, 10)
            
            payload = {
                "q": query,
                "num": max_results_per_query,
                "sort": "date"  # Sort by date when possible
            }
            
            logger.debug(f"Calling Serper API with payload: {payload}")
            
            # Make the API request
            response = requests.post(SERPER_ENDPOINT, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            organic_results = data.get("organic", [])
            
            logger.debug(f"Received {len(organic_results)} organic results from Serper API")
            
            results = []
            for result in organic_results:
                url = result.get("link")
                if not url:
                    continue
                    
                # Check if the URL's domain is in the blocked list
                domain = urlparse(url).netloc.lower()
                if any(blocked_domain in domain for blocked_domain in BLOCKED_DOMAINS):
                    logger.info(f"Excluded URL: {url} based on domain filter.")
                    continue
                
                # Basic URL validation
                if not (url and urlparse(url).scheme in ['http', 'https']):
                    logger.warning(f"Skipping invalid URL from Serper search: {url}")
                    continue
                
                # Create SearchResult with classification
                search_result = SearchResult(
                    url=url,
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    source="Serper"
                )
                
                # Add classification tag (if needed for future use)
                classification = classify_url(url)
                # Note: SearchResult dataclass doesn't have a tag field, 
                # but we could extend it or use metadata if needed
                
                results.append(search_result)
                
                # Stop if we've reached our limit
                if len(results) >= self.max_results:
                    break
            
            logger.info(f"Finished Serper search for query: '{query}'. Found {len(results)} results after filtering.")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during Serper search for query '{query}': {e}")
            raise
        except requests.exceptions.Timeout:
            logger.error(f"Timeout during Serper search for query '{query}'")
            raise
        except ValueError as e:
            logger.error(f"JSON parsing error during Serper search for query '{query}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Serper search for query '{query}': {e}", exc_info=True)
            raise


    def search_sync(self, query: str, engines: Optional[List[str]] = None, force_refresh: bool = False) -> List[SearchResult]:
        """
        Synchronous wrapper for the asynchronous search method.

        Runs the asynchronous `search` method using `asyncio.run`.
        Note: `asyncio.run` creates a new event loop and closes it, which might
        have implications if used repeatedly in some environments. Consider using
        the async `search` method directly within an existing event loop if possible.

        Args:
            query: The search query string.
            engines: A list of search engine names to use.
            force_refresh: If True, ignore cached results.

        Returns:
            A list of SearchResult objects.
        """
        logger.info(f"Starting synchronous search for query: '{query}'")
        # Run the asynchronous search method using asyncio.run.
        # nest_asyncio.apply() should be called once at the start of the application
        # if asyncio.run needs to be called from an already running loop (e.g., in notebooks).
        try:
            results = asyncio.run(self.search(query, engines, force_refresh))
            logger.info(f"Synchronous search finished for query: '{query}'. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during synchronous search for query '{query}': {e}", exc_info=True)
            return [] # Return empty list on error.


    async def _get_semaphore(self) -> asyncio.Semaphore:
        """
        Get or create an asyncio Semaphore instance bound to the current event loop.

        This method ensures that the concurrency limit (`self.max_concurrent`, currently hardcoded to 5)
        is applied correctly within the specific event loop where the async task is running.
        It is thread-safe for scenarios where the UnifiedSearcher instance might be accessed
        from different threads/loops (though typically all async operations run on one loop).

        Returns:
            An asyncio.Semaphore instance for the current event loop.

        Raises:
            RuntimeError: If called when no asyncio event loop is currently running.
                          The caller is responsible for ensuring a loop is running (e.g., using asyncio.run).
        """
        try:
            # Get the currently running event loop. This will raise RuntimeError if no loop is running.
            loop = asyncio.get_running_loop()
            # Use the loop's ID as the key in the semaphores dictionary.
            loop_id = id(loop)

            # Optimistic check: If a semaphore for this loop already exists, return it immediately
            # without acquiring the lock, which is faster.
            if loop_id in self._semaphores:
                return self._semaphores[loop_id]

            # If the semaphore was not found, acquire the lock to safely check and potentially create it.
            async with self._semaphore_lock:
                # Double-check after acquiring the lock in case another task created the semaphore
                # while this task was waiting for the lock.
                if loop_id in self._semaphores:
                    return self._semaphores[loop_id]

                # If still not found, create a new semaphore for this loop.
                # The concurrency limit is hardcoded to 5 here. Consider making this configurable
                # in __init__ if different concurrency levels are needed.
                semaphore = asyncio.Semaphore(5)
                # Store the new semaphore in the dictionary.
                self._semaphores[loop_id] = semaphore
                logger.debug(f"Created new semaphore for loop {loop_id} with max_concurrent 5")
                return semaphore

        except RuntimeError:
            # This exception occurs if asyncio.get_running_loop() is called when no loop is active.
            # Re-raise the error, indicating that the caller must ensure an async loop is running.
            logger.error("RuntimeError: No running asyncio loop found. Ensure search or search_sync is called within a running loop (e.g., using asyncio.run()).")
            raise # Propagate the error.


    @lru_cache(maxsize=128) # Cache the result of this function based on query and engine.
    def _get_formatted_query(self, query: str, engine: str) -> str:
        """
        Create a cache-friendly formatted query string.

        Combines the engine name and query string into a consistent format,
        converting them to lowercase for case-insensitive caching.
        Used to generate a base string for cache keys.

        Args:
            query: The search query.
            engine: The name of the search engine.

        Returns:
            A formatted string "engine:query" in lowercase.
        """
        return f"{engine.lower()}:{query.lower()}"

# Keep the global cache directory creation logic outside the class
# Ensure cache directory exists when module is imported (Global handling)
if CACHE_ENABLED and not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Global cache directory created at: {CACHE_DIR}")
    except Exception as e:
        logger.warning(f"Could not create global cache directory {CACHE_DIR}: {e}. Disabling cache globally.")
        CACHE_ENABLED = False # Disable globally if creation fails

