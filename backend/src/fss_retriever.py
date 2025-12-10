from google import genai
from google.genai import types
import asyncio
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import io

from backend.src.fss_capacity_check import get_fss_storage_usage
from .logging_setup import logger
from backend.src.config import GOOGLE_MODEL

# Import API keys from api_keys.py
try:
    from .api_keys import GOOGLE_API_KEY
except ImportError:
    logger.error("Could not import API keys from api_keys.py. LLMs and embeddings may not initialize.")
    GOOGLE_API_KEY = None

try:
    from .config import GOOGLE_MODEL
except ImportError:
    logger.error("Could not import GOOGLE_MODEL from config.py. Using default model.")
    GOOGLE_MODEL = "gemini-2.0-flash"

# Configure the generative AI library with the API key

gemini_api_key = GOOGLE_API_KEY
gemini_model = GOOGLE_MODEL
FILE_INDEXING_TIMEOUT = 300  # 5 minutes

class GeminiFileSearchRetriever:
    """
    Manages a Gemini File Search Store for a single research session.
    It handles store creation, file uploads from in-memory contexts,
    and deletion of the store after use.
    """
    def __init__(self, display_name_prefix: str = "crystal-fss"):
        # Generate a unique name for the store for this session
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.display_name = f"{display_name_prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
        self.client = genai.Client(api_key=gemini_api_key)
        self.async_client = self.client.aio
        self.file_store_name: Optional[str] = None
        self._file_search_store_obj = None
        logger.info(f"[{self.display_name}] Initialized GeminiFileSearchRetriever instance.")

    async def create_and_upload_contexts(self, relevant_contexts: Dict[str, Dict[str, Any]]) -> Optional[str]:
        if not relevant_contexts:
            logger.warning(f"[{self.display_name}] No relevant contexts provided to upload.")
            return None

        try:
            # 1. Create the Store
            logger.info(f"[{self.display_name}] Creating File Search Store.")
            self._file_search_store_obj = await self.async_client.file_search_stores.create(
                config=types.CreateFileSearchStoreConfig(display_name=self.display_name)
            )
            self.file_store_name = self._file_search_store_obj.name
            logger.info(f"[{self.display_name}] Store created successfully: {self.file_store_name}")

            upload_ops = []

            # 2. Iterate through your memory dictionary and upload
            for url, data in relevant_contexts.items():
                content_text = data.get("content", "")
                if not content_text.strip():
                    continue
                
                file_stream = io.BytesIO(content_text.encode("utf-8"))
                clean_name = url[-128:] if len(url) > 128 else url # Truncate long URLs for display name

                logger.info(f"[{self.display_name}] Uploading in-memory stream for: {url}")
                op = await self.async_client.file_search_stores.upload_to_file_search_store(
                    file=file_stream,
                    file_search_store_name=self.file_store_name, # Corrected parameter name
                    config=types.UploadToFileSearchStoreConfig(
                        display_name=clean_name,
                        mime_type="text/plain"
                    )
                )
                upload_ops.append(op)

            if not upload_ops:
                logger.warning(f"[{self.display_name}] No valid content to upload. Deleting empty store.")
                await self.delete_store()
                return None

            # 3. Wait for indexing to finish
            logger.info(f"[{self.display_name}] Waiting for remote indexing of {len(upload_ops)} files...")
            start_time = asyncio.get_event_loop().time()
            for op in upload_ops:
                current_op = op
                while not current_op.done:
                    if asyncio.get_event_loop().time() - start_time > FILE_INDEXING_TIMEOUT:
                        raise asyncio.TimeoutError(f"File indexing timed out after {FILE_INDEXING_TIMEOUT} seconds.")
                    await asyncio.sleep(5) # Poll every 5 seconds
                    current_op = await self.async_client.operations.get(op)
                if current_op.error:
                    raise Exception(f"File upload and indexing failed for operation {op.name}: {op.error.message}")

            logger.info(f"[{self.display_name}] All {len(upload_ops)} files indexed in store {self.file_store_name}.")
            return self.file_store_name

        except asyncio.TimeoutError as e:
            logger.error(f"[{self.display_name}] {e}", exc_info=True)
            if self.file_store_name:
                await self.delete_store()
            return None
        except Exception as e:
            logger.error(f"[{self.display_name}] Error during store creation or upload: {e}", exc_info=True)
            if self.file_store_name:
                await self.delete_store()
            return None


    async def answer_question(self, query: str, relevant_contexts: Dict[str, Dict[str, Any]], system_instruction: Optional[str] = None) -> str:
        """
        Creates a temporary store, uploads contexts, asks a question, and cleans up.
        """
        try:
            # 1. Create store and upload files
            # Ensure this returns the full resource name: "fileSearchStores/..."
            file_store_name = await self.create_and_upload_contexts(relevant_contexts)
            
            if not file_store_name:
                raise Exception("Failed to create and populate the file search store.")

            # 2. Define the file search tool
            # This structure is correct for the google-genai SDK
            tools = [types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_store_name]
                )
            )]

            intellisearch_prompt = f"""
            You are an expert research analyst. Provide a comprehensive, analytical response to the user's query by synthesizing information from the collected research data.

            USER QUERY: {query}

            INSTRUCTIONS:
            - You have access to a set of files containing research data. Use the file_search tool to find relevant information to answer the user's query.
            - Your task is to analyze this data and produce a well-structured research report that directly answers the user's query with clarity and full depth.
            - You must provide clear quantitative answers if that is required.
            - The report should be well-structured, using markdown for headings, subheadings, and bullet points where appropriate.
            - Present your conclusion in a clearly separated section at the end marked "Conclusion".
            - Target 500-3000 words for this response.
            - Use clear, professional language with proper markdown formatting.
            - Provide a list of references at the end citing the data sources used.

            Generate the report keeping these instructions in mind:
            """
            
            final_instruction = system_instruction if system_instruction else intellisearch_prompt
            
            logger.info(f"[{self.display_name}] Answering question '{query}' using store '{file_store_name}'")

            # 3. Generate content
            # FIX: Move system_instruction INSIDE GenerateContentConfig
            response = await self.async_client.models.generate_content(
                model=GOOGLE_MODEL, 
                contents=[query], # Passing the query as a list of strings is fine
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1,
                    system_instruction=final_instruction # <--- MOVED HERE
                )
            )
            
            # 4. Better Error Handling & Validation
            # Validating response.text directly can fail if the model filtered the response.
            if not response.text:
                # Check why the response is empty (e.g., safety filters, no relevant content found)
                finish_reason = "UNKNOWN"
                if response.candidates and response.candidates[0]:
                    finish_reason = response.candidates[0].finish_reason
                
                logger.error(f"Empty response. Finish reason: {finish_reason}. Full response: {response}")
                raise ValueError(f"Model returned empty response. Reason: {finish_reason}")
            
            return response.text

        except Exception as e:
            logger.error(f"[{self.display_name}] Error in answer_question: {e}", exc_info=True)
            raise
        finally:
            await self.delete_store()


    async def delete_store(self):
        """Deletes the File Search Store associated with this instance."""
        if not self.file_store_name:
            return

        store_to_delete = self.file_store_name
        self.file_store_name = None  # Prevent re-entry
        
        try:
            logger.warning(f"[{self.display_name}] Deleting File Search Store: {store_to_delete}")
            # List all files in the store to delete them first.
            all_files_pager = self.client.files.list()
            delete_tasks = []
            for file in all_files_pager:
                if hasattr(file, 'file_search_store_name') and file.file_search_store_name == store_to_delete:
                    logger.info(f"[{self.display_name}] Scheduling deletion for file: {file.name}")
                    delete_tasks.append(self.async_client.files.delete(name=file.name))
            await asyncio.gather(*delete_tasks, return_exceptions=True)
            logger.info(f"[{self.display_name}] All files in store {store_to_delete} have been deleted.")
            
            # Now delete the store itself
            await self.async_client.file_search_stores.delete(name=store_to_delete)
            logger.info(f"[{self.display_name}] Successfully deleted store: {store_to_delete}")
        except Exception as e:
            # Log error but don't raise, as it's a cleanup step
            logger.error(
                f"[{self.display_name}] Failed to delete store {store_to_delete}. Manual cleanup may be required. Error: {e}"
            )
async def delete_gemini_file_search_store(file_store_name: str):
    """
    Deletes a Gemini FileSearchStore by name.
    """
    if not file_store_name:
        logger.info("Skipping manual deletion: file_store_name is empty.")
        return

    client = genai.Client()
    try:
        logger.info(f"Attempting to manually delete FileSearchStore: {file_store_name} (and its files)")
        # List and delete all files within the store first.
        all_files_pager = client.files.list()
        delete_tasks = []
        for file in all_files_pager:
            if hasattr(file, 'file_search_store_name') and file.file_search_store_name == file_store_name:
                logger.info(f"Scheduling deletion for file: {file.name}")
                delete_tasks.append(client.aio.files.delete(name=file.name))

        await asyncio.gather(*delete_tasks, return_exceptions=True)
        logger.info(f"All files in store {file_store_name} have been deleted.")
        
        # Now delete the store itself
        await client.aio.file_search_stores.delete(name=file_store_name)
        logger.info(f"Manually deleted FileSearchStore: {file_store_name}")
    except Exception as e:
        logger.error(f"Failed to manually delete FileSearchStore {file_store_name}: {e}", exc_info=True)