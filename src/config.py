# config.py - Unified Configuration System
# All configuration values loaded from environment variables (.env file)

import os
import logging
from typing import List, Optional, Union

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Loaded environment variables from .env file")
except ImportError:
    logging.warning("dotenv not available, using system environment variables only")

# Detect if we're in a production environment (Render, Heroku, etc.)
PRODUCTION_MODE = bool(os.getenv('PORT') or os.getenv('RENDER') or os.getenv('DYNO'))

if PRODUCTION_MODE:
    logging.info("Production mode detected - using optimized defaults")
else:
    logging.info("Development mode - reading from environment variables")

def get_env_bool(key: str, default: bool = False) -> bool:
    """Convert environment variable to boolean."""
    if PRODUCTION_MODE:
        return default  # Use defaults in production
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')

def get_env_int(key: str, default: int) -> int:
    """Convert environment variable to integer."""
    if PRODUCTION_MODE:
        return default  # Use defaults in production
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logging.warning(f"Invalid integer value for {key}, using default: {default}")
        return default

def get_env_float(key: str, default: float) -> float:
    """Convert environment variable to float."""
    if PRODUCTION_MODE:
        return default  # Use defaults in production
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logging.warning(f"Invalid float value for {key}, using default: {default}")
        return default

def get_env_list(key: str, default: List[str] = None, separator: str = ',') -> List[str]:
    """Convert environment variable to list."""
    if PRODUCTION_MODE:
        return default or []  # Use defaults in production
    if default is None:
        default = []
    value = os.getenv(key, '')
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API Keys (Google-only configuration)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "google")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")


# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-005")

# Retrieval Configuration
USE_RERANKING = get_env_bool("USE_RERANKING", True)  # Enable reranking by default
RERANKER_CANDIDATES_MULTIPLIER = get_env_int("RERANKER_CANDIDATES_MULTIPLIER", 3)  # Retrieve 3x more candidates
RERANK_TOP_K = get_env_int("RERANK_TOP_K", 20)  # Final number of chunks after reranking

# LLM Settings
LLM_TEMPERATURE = get_env_float("LLM_TEMPERATURE", 0.1)  # Low temperature for factual research
MAX_TOKENS = get_env_int("MAX_TOKENS", 30000)  # High token limit for comprehensive reports
DEFAULT_LLM_TIMEOUT = get_env_int("DEFAULT_LLM_TIMEOUT", 120)  # Extended timeout for complex queries

# Legacy support for old variable names
DEFAULT_GEMINI_MODEL = GOOGLE_MODEL
gemini_model = GOOGLE_MODEL
gemini1 = "gemini-2.0-flash-lite"
gemini2 = "gemini-2.0-flash"
DEFAULT_LLM_MODEL_NAME = GOOGLE_MODEL
DEFAULT_MAX_TOKENS = MAX_TOKENS
DEFAULT_TEMPERATURE = LLM_TEMPERATURE

# =============================================================================
# SEARCH AND PROCESSING CONFIGURATION
# =============================================================================

# Search limits
MAX_SEARCH_QUERIES = get_env_int("MAX_SEARCH_QUERIES", 15)  # Multiple queries for comprehensive coverage
MAX_SEARCH_RESULTS = get_env_int("MAX_SEARCH_RESULTS", 10)  # Balanced between quality and performance
MAX_CONCURRENT_SCRAPES = get_env_int("MAX_CONCURRENT_SCRAPES", 4)  # Reasonable concurrency for stability
MAX_SEARCH_RETRIES = get_env_int("MAX_SEARCH_RETRIES", 2)  # Limited retries to prevent hanging

# AI iteration limits
MAX_AI_ITERATIONS = get_env_int("MAX_AI_ITERATIONS", 3)
MAX_USER_QUERY_LOOPS = get_env_int("MAX_USER_QUERY_LOOPS", 3)

# Content processing
CHUNK_SIZE = get_env_int("CHUNK_SIZE", 1000)  # Optimized for embedding model context
CHUNK_OVERLAP = get_env_int("CHUNK_OVERLAP", 100)  # Minimal overlap for efficiency
MAX_CONTENT_LENGTH = get_env_int("MAX_CONTENT_LENGTH", 10000)  # Reasonable limit per source
URL_TIMEOUT = get_env_int("URL_TIMEOUT", 30)  # Quick timeout to prevent hanging

# Legacy support
MAX_RESULTS = MAX_SEARCH_RESULTS  # Backward compatibility
MAX_RETRIES = MAX_SEARCH_RETRIES

# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

REPORT_FORMAT = os.getenv("REPORT_FORMAT", "md")
DEFAULT_REPORT_TYPE = os.getenv("DEFAULT_REPORT_TYPE", "detailed")
REPORT_FILENAME_TEXT = os.getenv("REPORT_FILENAME_TEXT", "IntelliSearchReport.txt")
REPORT_FILENAME_PDF = os.getenv("REPORT_FILENAME_PDF", "IntelliSearchReport.pdf")

# Legacy support
REPORT_FILENAME_PDF = os.getenv("REPORT_FILENAME_PDF", "Crystal_DeepSearch.pdf")
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

# Rate limiting
MAX_CONCURRENT_CALLS = get_env_int("MAX_CONCURRENT_CALLS", 10)  # Conservative for stability
MAX_CALLS_PER_SECOND = get_env_int("MAX_CALLS_PER_SECOND", 30)  # Reasonable rate limiting
BASE_DELAY = get_env_int("BASE_DELAY", 1)  # Small delay between requests
API_REQUESTS_PER_MINUTE = get_env_int("API_REQUESTS_PER_MINUTE", 30)  # API-friendly limits
SCRAPING_REQUESTS_PER_MINUTE = get_env_int("SCRAPING_REQUESTS_PER_MINUTE", 30)  # Respectful scraping

# =============================================================================
# AUTOMATION SETTINGS
# =============================================================================

DEFAULT_AUTOMATION_MODE = os.getenv("DEFAULT_AUTOMATION_MODE", "none")
AUTO_APPROVE_QUERIES = get_env_bool("AUTO_APPROVE_QUERIES", False)
AUTO_REPORT_TYPE = os.getenv("AUTO_REPORT_TYPE", "detailed")

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
    
    # Search and Processing
    'MAX_SEARCH_QUERIES', 'MAX_SEARCH_RESULTS', 'MAX_CONCURRENT_SCRAPES',
    'MAX_AI_ITERATIONS', 'CHUNK_SIZE', 'CHUNK_OVERLAP',
    
    # Reports
    'REPORT_FORMAT', 'REPORT_FILENAME_TEXT', 'REPORT_FILENAME_PDF',
    
    # Web Scraping
    'USER_AGENT', 'BLOCKED_DOMAINS', 'SKIP_EXTENSIONS', 'REQUEST_TIMEOUT',
    
    # Caching
    'CACHE_ENABLED', 'CACHE_TTL',
    
    # Debug and Production
    'DEBUG_MODE', 'PRODUCTION_MODE',
    
    # Colors
    'RED', 'GREEN', 'BLUE', 'YELLOW', 'ENDC',
    
    # Legacy support
    'MAX_RESULTS', 'DEFAULT_USER_AGENT', 'DEFAULT_GEMINI_MODEL'
]

def validate_production_config():
    """Validate critical configuration for production deployment."""
    critical_missing = []
    
    # Check required API keys
    if not GOOGLE_API_KEY:
        critical_missing.append("GOOGLE_API_KEY")
    if not SERPER_API_KEY:
        critical_missing.append("SERPER_API_KEY")
    
    if critical_missing:
        error_msg = f"Critical environment variables missing: {', '.join(critical_missing)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Log production mode status
    if PRODUCTION_MODE:
        logging.info("Running in PRODUCTION mode - using optimized defaults")
        logging.info(f"LLM Temperature: {LLM_TEMPERATURE}")
        logging.info(f"Max Search Results: {MAX_SEARCH_RESULTS}")
        logging.info(f"Cache Enabled: {CACHE_ENABLED}")
        logging.info(f"Debug Mode: {DEBUG_MODE}")
    else:
        logging.info("Running in DEVELOPMENT mode - reading from environment variables")
    
    return True

# Validate configuration on import
try:
    validate_production_config()
except Exception as e:
    logging.error(f"Configuration validation failed: {e}")
    if PRODUCTION_MODE:
        raise  # Fail fast in production

