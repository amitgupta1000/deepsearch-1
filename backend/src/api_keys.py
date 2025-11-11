# api_keys.py - API Key Access
# Simplified to import from unified config

from .config import (
    GOOGLE_API_KEY,
    SERPER_API_KEY,
)

import logging
logging.info("API keys loaded from unified configuration (Google only)")