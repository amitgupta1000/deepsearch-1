# config.py - Unified Configuration System
# All configuration values loaded from environment variables (.env file)

import os
import logging
from typing import List, Any, Callable
import sys

# Dictionary to track config values and their sources
CONFIG_SOURCES = {}

def get_config_value(key: str, default: Any, cast_func: Callable = str) -> Any:
    value = os.getenv(key)
    if value is not None:
        try:
            casted_value = cast_func(value)
        except Exception:
            casted_value = default
        CONFIG_SOURCES[key] = {"value": casted_value, "source": "env"}
        return casted_value
    else:
        CONFIG_SOURCES[key] = {"value": default, "source": "default"}
        return default

# Load environment variables (for local dev only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_env_bool(key: str, default: bool = False) -> bool:
    def cast_bool(val):
        return val.lower() in ('true', '1', 'yes', 'on')
    return get_config_value(key, default, cast_bool)

def get_env_int(key: str, default: int) -> int:
    def cast_int(val):
        try:
            return int(val)
        except ValueError:
            logging.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    return get_config_value(key, default, cast_int)

def get_env_float(key: str, default: float) -> float:
    def cast_float(val):
        try:
            return float(val)
        except ValueError:
            logging.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    return get_config_value(key, default, cast_float)

def get_env_list(key: str, default: List[str] = None, separator: str = ',') -> List[str]:
    if default is None:
        default = []
    def cast_list(val):
        return [item.strip() for item in val.split(separator) if item.strip()]
    return get_config_value(key, default, cast_list)

# =============================================================================
# API KEYS  
# =============================================================================

GOOGLE_API_KEY = get_config_value("GOOGLE_API_KEY", "")
SERPER_API_KEY = get_config_value("SERPER_API_KEY", "")

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

PRIMARY_LLM_PROVIDER = get_config_value("PRIMARY_LLM_PROVIDER", "google")
GOOGLE_MODEL = get_config_value("GOOGLE_MODEL", "gemini-2.0-flash")

# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")  # Updated to latest model

# Enhanced embedding configuration
EMBEDDING_TASK_TYPE = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")  # Default task type
EMBEDDING_DIMENSIONALITY = get_env_int("EMBEDDING_DIMENSIONALITY", 768)  # Efficient default
EMBEDDING_NORMALIZE = get_env_bool("EMBEDDING_NORMALIZE", True)  # Auto-normalize for non-3072 dims
EMBEDDING_BATCH_SIZE = get_env_int("EMBEDDING_BATCH_SIZE", 100)  # Batch processing
USE_ENHANCED_EMBEDDINGS = get_env_bool("USE_ENHANCED_EMBEDDINGS", True)  # Use new implementation

# LLM Settings
LLM_TEMPERATURE = get_env_float("LLM_TEMPERATURE", 0.1)  # Low temperature for factual research
MAX_TOKENS = get_env_int("MAX_TOKENS", 30000)  # High token limit for comprehensive reports
DEFAULT_LLM_TIMEOUT = get_env_int("DEFAULT_LLM_TIMEOUT", 120)  # Extended timeout for complex queries

# Legacy support for old variable names
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
DEFAULT_GEMINI_MODEL = GOOGLE_MODEL
gemini_model = GOOGLE_MODEL
gemini1 = "gemini-2.0-flash-lite"
gemini2 = "gemini-2.0-flash"
MAX_TOKENS = get_env_int("MAX_TOKENS", 30000)
LLM_TEMPERATURE = get_env_float("LLM_TEMPERATURE", 0.1)
DEFAULT_LLM_MODEL_NAME = GOOGLE_MODEL
DEFAULT_MAX_TOKENS = MAX_TOKENS
DEFAULT_TEMPERATURE = LLM_TEMPERATURE

# =============================================================================
# SEARCH AND PROCESSING CONFIGURATION
# =============================================================================

# -- Fast Search (Default) Mode Settings --
MAX_SEARCH_QUERIES = get_env_int("MAX_SEARCH_QUERIES", 5)  # Default for "fast" search
MAX_SEARCH_RESULTS = get_env_int("MAX_SEARCH_RESULTS", 3)  # Balanced between quality and performance
MAX_AI_ITERATIONS = get_env_int("MAX_AI_ITERATIONS", 1)

# -- Ultra Search Mode Settings --
ULTRA_MAX_SEARCH_QUERIES = get_env_int("ULTRA_MAX_SEARCH_QUERIES", 15)
ULTRA_MAX_SEARCH_RESULTS = get_env_int("ULTRA_MAX_SEARCH_RESULTS", 10)
ULTRA_MAX_AI_ITERATIONS = get_env_int("ULTRA_MAX_AI_ITERATIONS", 3)

# Other Search limits
MAX_CONCURRENT_SCRAPES = get_env_int("MAX_CONCURRENT_SCRAPES", 6)  # Reasonable concurrency for stability
MAX_SEARCH_RETRIES = get_env_int("MAX_SEARCH_RETRIES", 2)  # Limited retries to prevent hanging
MAX_USER_QUERY_LOOPS = get_env_int("MAX_USER_QUERY_LOOPS", 2)

# Content processing
CHUNK_SIZE = get_env_int("CHUNK_SIZE", 1000)  # Optimized for embedding model context
CHUNK_OVERLAP = get_env_int("CHUNK_OVERLAP", 100)  # Minimal overlap for efficiency
MAX_CONTENT_LENGTH = get_env_int("MAX_CONTENT_LENGTH", 10000)  # Reasonable limit per source
URL_TIMEOUT = get_env_int("URL_TIMEOUT", 30)  # Quick timeout to prevent hanging


# =============================================================================
# HYBRID RETRIEVAL CONFIGURATION
# =============================================================================
RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "hybrid")
HYBRID_VECTOR_WEIGHT = get_env_float("HYBRID_VECTOR_WEIGHT", 0.6)
HYBRID_BM25_WEIGHT = get_env_float("HYBRID_BM25_WEIGHT", 0.4)
HYBRID_FUSION_METHOD = os.getenv("HYBRID_FUSION_METHOD", "rrf")
HYBRID_RRF_K = get_env_int("HYBRID_RRF_K", 60)
VECTOR_SCORE_THRESHOLD = get_env_float("VECTOR_SCORE_THRESHOLD", 0.1)
VECTOR_FETCH_K_MULTIPLIER = get_env_int("VECTOR_FETCH_K_MULTIPLIER", 2)
RETRIEVAL_TOP_K = get_env_int("RETRIEVAL_TOP_K", 20)
MIN_CHUNK_LENGTH = get_env_int("MIN_CHUNK_LENGTH", 50)
MIN_WORD_COUNT = get_env_int("MIN_WORD_COUNT", 10)
USE_MULTI_QUERY_RETRIEVAL = get_env_bool("USE_MULTI_QUERY_RETRIEVAL", True)

import logging
MAX_RETRIEVAL_QUERIES = get_env_int("MAX_RETRIEVAL_QUERIES", 5)
QUERY_CHUNK_DISTRIBUTION = get_env_bool("QUERY_CHUNK_DISTRIBUTION", True)
USE_HYBRID_RETRIEVAL = get_env_bool("USE_HYBRID_RETRIEVAL", True)
USE_RERANKING = get_env_bool("USE_RERANKING", True)
RERANKER_CANDIDATES_MULTIPLIER = get_env_int("RERANKER_CANDIDATES_MULTIPLIER", 3)
USE_CROSS_ENCODER_RERANKING = get_env_bool("USE_CROSS_ENCODER_RERANKING", True)

# Only the fast cross-encoder model is supported
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K = get_env_int("CROSS_ENCODER_TOP_K", 30)
RERANK_TOP_K = get_env_int("RERANK_TOP_K", 10)
CROSS_ENCODER_BATCH_SIZE = get_env_int("CROSS_ENCODER_BATCH_SIZE", 32)
MAX_RESULTS = MAX_SEARCH_RESULTS
MAX_RETRIES = MAX_SEARCH_RETRIES

# Log cross-encoder reranking config at startup
logging.info(f"USE_CROSS_ENCODER_RERANKING={USE_CROSS_ENCODER_RERANKING}, CROSS_ENCODER_MODEL={CROSS_ENCODER_MODEL}, CROSS_ENCODER_TOP_K={CROSS_ENCODER_TOP_K}, RERANK_TOP_K={RERANK_TOP_K}, CROSS_ENCODER_BATCH_SIZE={CROSS_ENCODER_BATCH_SIZE}")


# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

REPORT_FORMAT = os.getenv("REPORT_FORMAT", "md")
DEFAULT_REPORT_TYPE = os.getenv("DEFAULT_REPORT_TYPE", "detailed")

# Legacy support
REPORT_FILENAME_TEXT = os.getenv("REPORT_FILENAME_TEXT", "Crystal_DeepSearch.txt")

# =============================================================================
# WEB SCRAPING CONFIGURATION
# =============================================================================

USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
DEFAULT_USER_AGENT = USER_AGENT  # Legacy support
REQUEST_TIMEOUT = get_env_int("REQUEST_TIMEOUT", 30)
REQUEST_DELAY = get_env_int("REQUEST_DELAY", 1)
DEFAULT_REFERER = "https://www.google.com/"

# Blocked domains and extensions
BLOCKED_DOMAINS = get_env_list("BLOCKED_DOMAINS", [
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com/posts", 
    "reddit.com/r/", "youtube.com/watch", "youtu.be", "nsearchives.nseindia.com", 
    "bseindia.com", "sebi.gov.in"
])
SKIP_EXTENSIONS = get_env_list("SKIP_EXTENSIONS", [
    ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3", ".zip", 
    ".exe", ".dmg", ".rar", ".7z"
])


# =============================================================================
# CACHING AND PERFORMANCE
# =============================================================================

CACHE_ENABLED = get_env_bool("CACHE_ENABLED", True)  # Enabled for production performance
CACHE_TTL = get_env_int("CACHE_TTL", 86400)  # 24 hours for good balance

# =============================================================================
# ENHANCED DEDUPLICATION SETTINGS
# =============================================================================

# Enable LLM-powered intelligent deduplication
USE_LLM_DEDUPLICATION = get_env_bool("USE_LLM_DEDUPLICATION", True)
DEDUPLICATION_CACHE_ENABLED = get_env_bool("DEDUPLICATION_CACHE_ENABLED", True)
DEDUPLICATION_CACHE_TTL = get_env_int("DEDUPLICATION_CACHE_TTL", 7200)  # 2 hours cache

# Deduplication thresholds
SIMILARITY_THRESHOLD = get_env_float("SIMILARITY_THRESHOLD", 0.75)  # When to consider content similar
MIN_SENTENCE_LENGTH = get_env_int("MIN_SENTENCE_LENGTH", 3)  # Minimum words per sentence
DEDUPLICATION_BATCH_SIZE = get_env_int("DEDUPLICATION_BATCH_SIZE", 10)  # Process sentences in batches

# When to apply LLM deduplication (word thresholds)
LLM_DEDUP_MIN_WORDS = get_env_int("LLM_DEDUP_MIN_WORDS", 800)  # Apply to reports above this size
LLM_DEDUP_DETAILED_ONLY = get_env_bool("LLM_DEDUP_DETAILED_ONLY", True)  # Only for detailed reports

# Rate limiting
MAX_CONCURRENT_CALLS = get_env_int("MAX_CONCURRENT_CALLS", 10)  # Conservative for stability
MAX_CALLS_PER_SECOND = get_env_int("MAX_CALLS_PER_SECOND", 30)  # Reasonable rate limiting
BASE_DELAY = get_env_int("BASE_DELAY", 1)  # Small delay between requests
API_REQUESTS_PER_MINUTE = get_env_int("API_REQUESTS_PER_MINUTE", 30)  # API-friendly limits
SCRAPING_REQUESTS_PER_MINUTE = get_env_int("SCRAPING_REQUESTS_PER_MINUTE", 30)  # Respectful scraping

## All automation settings removed

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
DEBUG_MODE = get_env_bool("DEBUG_MODE", False)
VERBOSE = get_env_bool("VERBOSE", False)

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

SSL_VERIFY = get_env_bool("SSL_VERIFY", True)

# =============================================================================
# LEGACY SUPPORT AND COLORS
# =============================================================================

# Color constants for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

# Legacy persistence setting
USE_PERSISTENCE = get_env_bool("USE_PERSISTENCE", False)

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate critical configuration values."""
    errors = []
    warnings = []
    
    # Check API keys
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is required")
    if not SERPER_API_KEY:
        warnings.append("SERPER_API_KEY not set - search functionality may be limited")
    
    # Check numeric limits
    if MAX_SEARCH_QUERIES <= 0:
        errors.append("MAX_SEARCH_QUERIES must be positive")
    if CACHE_TTL <= 0:
        warnings.append("CACHE_TTL should be positive for effective caching")
    
    # Log results
    if errors:
        for error in errors:
            logging.error(f"Configuration error: {error}")
    if warnings:
        for warning in warnings:
            logging.warning(f"Configuration warning: {warning}")
    
    return len(errors) == 0

# Validate configuration on import
config_valid = validate_config()

logging.info("config.py loaded successfully with unified environment-based configuration")

# Export commonly used values for backward compatibility
__all__ = [
    # API Keys (Google-only)
    'GOOGLE_API_KEY', 'SERPER_API_KEY',
    
    # LLM Configuration  
    'PRIMARY_LLM_PROVIDER', 'GOOGLE_MODEL', 'EMBEDDING_MODEL',
    'LLM_TEMPERATURE', 'MAX_TOKENS',
    
    # Enhanced Embedding Configuration
    'EMBEDDING_PROVIDER', 'EMBEDDING_TASK_TYPE', 'EMBEDDING_DIMENSIONALITY',
    'EMBEDDING_NORMALIZE', 'EMBEDDING_BATCH_SIZE', 'USE_ENHANCED_EMBEDDINGS',
    
    # Search and Processing
    'MAX_SEARCH_QUERIES', 'MAX_SEARCH_RESULTS', 'MAX_AI_ITERATIONS',
    'ULTRA_MAX_SEARCH_QUERIES', 'ULTRA_MAX_SEARCH_RESULTS', 'ULTRA_MAX_AI_ITERATIONS',
    'MAX_CONCURRENT_SCRAPES', 'CHUNK_SIZE', 'CHUNK_OVERLAP',
    
    # Hybrid Retrieval
    'RETRIEVAL_METHOD', 'HYBRID_VECTOR_WEIGHT', 'HYBRID_BM25_WEIGHT',
    'HYBRID_FUSION_METHOD', 'HYBRID_RRF_K', 'VECTOR_SCORE_THRESHOLD',
    'RETRIEVAL_TOP_K', 'USE_HYBRID_RETRIEVAL', 'USE_RERANKING',
    'USE_MULTI_QUERY_RETRIEVAL', 'MAX_RETRIEVAL_QUERIES', 'QUERY_CHUNK_DISTRIBUTION',
    'USE_CROSS_ENCODER_RERANKING', 'CROSS_ENCODER_MODEL', 'CROSS_ENCODER_TOP_K', 'RERANK_TOP_K', 'CROSS_ENCODER_BATCH_SIZE',
    
    # Reports
    'REPORT_FORMAT', 'REPORT_FILENAME_TEXT',
    
    # Web Scraping
    'USER_AGENT', 'BLOCKED_DOMAINS', 'SKIP_EXTENSIONS', 'REQUEST_TIMEOUT',
    
    # Caching
    'CACHE_ENABLED', 'CACHE_TTL',
    
    # Debug
    'DEBUG_MODE',
    
    # Colors
    'RED', 'GREEN', 'BLUE', 'YELLOW', 'ENDC',
    
    # Legacy support
    'MAX_RESULTS', 'DEFAULT_USER_AGENT', 'DEFAULT_GEMINI_MODEL'
]