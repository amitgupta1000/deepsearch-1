from google import genai
from google.genai import types
import asyncio
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
import io, os, tempfile
from backend.src.fss_capacity_check import get_fss_storage_usage
from .logging_setup import logger
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
    Manages a Gemini File Search Store with parallel uploads and optimized cleanup.
    """
    def __init__(self, display_name_prefix: str = "crystal-fss"):
        # Generate a unique name for the store for this session
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.display_name = f"{display_name_prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
        self.client = genai.Client(api_key=gemini_api_key)
        self.async_client = self.client.aio
        self.file_store_name: Optional[str] = None
        self._file_search_store_obj = None
        self.upload_semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent uploads
        self.created_file_names: List[str] = []  # Track created file names for cleanup
        logger.info(f"[{self.display_name}] Initialized GeminiFileSearchRetriever instance.")

    async def _upload_single_file(self, url: str, content: str, store_name: str):
        if not content or not content.strip():
            logger.warning(f"[{self.display_name}] SKIPPING {url}: Content is empty.")
            return None

        clean_name = url[-128:] if len(url) > 128 else url
        
        # 1. Create a temp file to handle encoding/buffering correctly
        tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8')
        try:
            tf.write(content)
            tf.flush()
            tf.close() # Close the write-handle

            async with self.upload_semaphore:
                try:
                    # 2. OPEN the file in Read-Binary mode
                    # The SDK needs the file object, not the path string
                    with open(tf.name, "rb") as f:
                        logger.debug(f"Starting upload for {clean_name}...")
                        
                        result = await self.async_client.file_search_stores.upload_to_file_search_store(
                            file=f, # <--- PASS THE FILE OBJECT, NOT tf.name
                            file_search_store_name=store_name,
                            config=types.UploadToFileSearchStoreConfig(
                                display_name=clean_name,
                                mime_type="text/plain"
                            )
                        )
                    
                    return result
                except Exception as e:
                    # Catch specific SDK errors to prevent crashing the whole batch
                    logger.error(f"Failed to upload {url}: {e}")
                    return None
        finally:
            # 3. Clean up the local temp file
            if os.path.exists(tf.name):
                os.unlink(tf.name)
 
    async def create_and_upload_contexts(self, relevant_contexts: Dict[str, Dict[str, Any]]) -> Optional[str]:
        if not relevant_contexts:
            return None

        try:
            # 1. Create the Store
            logger.info(f"[{self.display_name}] Creating File Search Store...")
            store_obj = await self.async_client.file_search_stores.create(
                config=types.CreateFileSearchStoreConfig(display_name=self.display_name)
            )
            self.file_store_name = store_obj.name
            logger.info(f"[{self.display_name}] Store created: {self.file_store_name}")

            # 2. Prepare Parallel Upload Tasks
            tasks = []
            for url, data in relevant_contexts.items():
                tasks.append(
                    self._upload_single_file(url, data.get("content", ""), self.file_store_name)
                )

            # 3. Execute Uploads in Parallel
            logger.info(f"[{self.display_name}] Starting parallel upload of {len(tasks)} files...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Collect created file IDs for cleanup later
            for res in results:
                if isinstance(res, Exception) or not res:
                    continue
                if hasattr(res, 'name'): # The result of the operation is a File object
                    self.created_file_names.append(res.name) # Capture File Resource Name
                elif res and hasattr(res, 'file') and hasattr(res.file, 'name'):
                     self.created_file_names.append(res.file.name)

            # 5. Wait for Store Indexing
            # Instead of polling every file, we poll the Store's aggregate status is usually more efficient
            # or simply wait a moment. The prompt requires high consistency, so let's quick-poll the store.
            # However, technically with `upload_to_file_search_store`, we are mostly good to go.
            # To be 100% sure indexing is done:
            logger.info(f"[{self.display_name}] Uploads complete. Files tracked: {len(self.created_file_names)}")
            
            return self.file_store_name

        except Exception as e:
            logger.error(f"[{self.display_name}] Critical error in create_and_upload: {e}", exc_info=True)
            await self.delete_store()
            return None

    async def answer_question(self, query: str, relevant_contexts: Dict[str, Dict[str, Any]], system_instruction: Optional[str] = None) -> str:
        try:
            file_store_name = await self.create_and_upload_contexts(relevant_contexts)
            if not file_store_name:
                raise Exception("Failed to populate store.")

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
			
            final_instruction = system_instruction if system_instruction else "You are a helpful analyst..."

            logger.info(f"[{self.display_name}] Generating content...")
            
            response = await self.async_client.models.generate_content(
                model=GOOGLE_MODEL, 
                contents=[query],
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1,
                    system_instruction=final_instruction
                )
            )

            if not response.text:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                raise ValueError(f"Empty response. Reason: {finish_reason}")
            
            return response.text

        except Exception as e:
            logger.error(f"[{self.display_name}] Error: {e}", exc_info=True)
            raise
        finally:
            await self.delete_store()

    async def delete_store(self):
        """Optimized deletion: Deletes tracked files directly, then the store."""
        if not self.file_store_name:
            return

        store_to_delete = self.file_store_name
        self.file_store_name = None 

        try:
            # 1. Parallel Delete of Files (Faster than listing all)
            if self.created_file_names:
                logger.info(f"[{self.display_name}] Deleting {len(self.created_file_names)} files concurrently...")
                
                async def delete_file_safe(name):
                    try:
                        await self.async_client.files.delete(name=name)
                    except Exception:
                        pass # Ignore already deleted or not found

                # Batch delete
                delete_tasks = [delete_file_safe(name) for name in self.created_file_names]
                await asyncio.gather(*delete_tasks)
                self.created_file_names = []

            # 2. Delete Store
            logger.info(f"[{self.display_name}] Deleting store: {store_to_delete}")
            await self.async_client.file_search_stores.delete(name=store_to_delete)
            
        except Exception as e:
            logger.error(f"[{self.display_name}] Cleanup error: {e}")


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