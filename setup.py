# setup.py
# This file handles initial setup, including package imports and global initializations.
# It should be run once at the beginning of your application.

import logging
import os
import sys
import subprocess
import time # Import time for cache initialization
from typing import Any

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import Validation ---
def validate_langchain_imports():
    """Validate LangChain/LangGraph imports and provide installation guidance."""
    try:
        from src.import_validator import validate_imports
        validator = validate_imports()
        validator.print_status_report()
        
        missing_packages = validator.get_missing_packages()
        if missing_packages:
            logging.warning(f"Missing LangChain packages: {missing_packages}")
            install_cmd = validator.get_installation_command()
            if install_cmd:
                logging.info(f"Install with: {install_cmd}")
            return False
        else:
            logging.info("✅ All LangChain/LangGraph packages are available")
            return True
            
    except ImportError:
        logging.warning("Import validator not available - skipping validation")
        return True
    except Exception as e:
        logging.error(f"Error during import validation: {e}")
        return False

# --- Package Installation ---
def install_packages():
    """Installs required Python packages using pip."""
    
    # Core packages that should be installed first
    core_packages = [
        "requests>=2.31.0",
        "python-dotenv>=1.0.0", 
        "pydantic>=2.8.0",
        "nest_asyncio>=1.5.6",
    ]
    
    # LangChain packages
    langchain_packages = [
        "langchain>=0.1.0",
        "langgraph>=0.1.0", 
        "langchain-core>=0.1.0",
        "langchain-community>=0.1.0",
        "langchain-text-splitters>=0.1.0",
    ]
    
    # Provider-specific packages (Google GenAI only)
    provider_packages = [
        "langchain-google-genai>=1.0.0",
        "google-genai>=1.0.0",
    ]
    
    # Additional packages
    additional_packages = [
        "beautifulsoup4>=4.12.2",
        "aiohttp>=3.9.0",
        "requests-html>=0.10.0",
        "lxml>=5.0.0",
        "ratelimit>=2.2.1",
        "pymupdf>=1.23.0",
        "pypdf>=4.0.0",
        "fpdf2>=2.8.0",
        "trafilatura>=1.7.2",
        "rank_bm25>=0.2.2",
        "faiss-cpu>=1.7.4",
        "rich>=13.3.4",
    ]

    logging.info("Installing required packages...")
    
    # Install in order: core -> langchain -> providers -> additional
    package_groups = [
        ("Core packages", core_packages),
        ("LangChain packages", langchain_packages), 
        ("Provider packages", provider_packages),
        ("Additional packages", additional_packages)
    ]
    
    for group_name, packages in package_groups:
        logging.info(f"Installing {group_name}...")
        for package in packages:
            try:
                # Use sys.executable to ensure the package is installed in the current environment
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package, "--no-cache-dir", "--upgrade"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                logging.info(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                stderr_output = e.stderr.decode() if e.stderr else "Unknown error"
                logging.warning(f"⚠️ Error installing {package}: {stderr_output}")
                # Continue with other packages instead of failing completely

    logging.info("Package installation process completed.")
    
    # Validate imports after installation
    logging.info("Validating LangChain imports after installation...")
    validation_success = validate_langchain_imports()
    if not validation_success:
        logging.warning("Some LangChain packages may not be properly installed")

## Uncomment and run install_packages() if you need to install packages within the notebook
# install_packages()
# subprocess.run(["playwright", "install"], check=True)

# --- Import necessary modules after potential installation ---
# This assumes packages are installed in the environment where this script runs.
# nest_asyncio removed - not needed for web deployment with FastAPI/uvicorn
logging.info("Skipping nest_asyncio - not needed for web deployment")

# Import API keys from unified configuration
try:
    from src.config import (
        GOOGLE_API_KEY, SERPER_API_KEY, 
        CACHE_ENABLED, CACHE_TTL, USER_AGENT,
        RED, GREEN, BLUE, YELLOW, ENDC
    )
    logging.info("Successfully imported configuration from unified config")
except ImportError:
    logging.error("Could not import from unified config. Using fallback values.")
    # Fallback values
    GOOGLE_API_KEY = None
    SERPER_API_KEY = None
    CACHE_ENABLED = False
    CACHE_TTL = 3600
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    RED = GREEN = BLUE = YELLOW = ENDC = ''


# Placeholder for embeddings - initialized in llm_calling.py
embeddings = None # For embedding/indexing


# Initialize cache here if using a global cache instance
if CACHE_ENABLED:
    try:
        # Assuming SimpleCache is defined in scraper.py or utils.py
        # Import it here or define it globally if needed across modules
        class SimpleCache:
            def __init__(self, ttl: int = 3600):
                self._cache = {}
                self._ttl = ttl
                self._timestamps = {}

            def get(self, key: str):
                if key in self._cache:
                    if time.time() - self._timestamps.get(key, 0) < self._ttl:
                        return self._cache[key]
                    else:
                        self._cache.pop(key, None)
                        self._timestamps.pop(key, None)
                return None

            def set(self, key: str, value: Any):
                self._cache[key] = value
                self._timestamps[key] = time.time()

        global cache
        cache = SimpleCache(ttl=CACHE_TTL)
        logging.info("Global cache initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize global cache: {e}. Caching disabled.")
        CACHE_ENABLED = False
        cache = None
else:
    cache = None


logging.info("setup.py created with initial setup and global variables.")
