# llm_calling.py
# This file now serves to make LLM and embedding objects available to other modules
# by importing them from llm_utils.py
# importing llm, llm_call_async and embeddings

import logging

# Import LLM and embeddings from llm_utils.py
try:
    from .llm_utils import embeddings, llm_call_async
    logging.info("Successfully imported embeddings and llm_call_async from llm_utils.py")
except ImportError:
    logging.error("Could not import embeddings or llm_call_async from llm_utils.py. LLM functionality will be limited.")
    # Define dummy variables to prevent NameError later, but warn the user
    embeddings = None
    async def llm_call_async(messages):
        logging.error("llm_call_async is not available.")
        return None
        return None

logging.info("llm_calling.py loaded and attempted to import llm utilities.")
