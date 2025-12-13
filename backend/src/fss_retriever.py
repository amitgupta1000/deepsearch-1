from google import genai
from google.genai import types
import asyncio
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
import io, os, tempfile
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
    Manages a Gemini File Search Store with parallel uploads, batch querying, 
    and optimized O(1) cleanup.
    """
    def __init__(self, display_name_prefix: str = "crystal-fss", max_concurrent_uploads: int = 10, max_concurrent_generation: int = 10):
        # Generate a unique name for the store for this session
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.display_name = f"{display_name_prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
        
        # Initialize Client
        self.client = genai.Client(api_key=gemini_api_key)
        self.async_client = self.client.aio
        
        self.file_store_name: Optional[str] = None
        
        # Concurrency Controls
        self.upload_semaphore = asyncio.Semaphore(max_concurrent_uploads)
        self.generation_semaphore = asyncio.Semaphore(max_concurrent_generation)
        
        # Track created file names for fast O(1) cleanup
        self.created_file_names: List[str] = []
        
        logger.info(f"[{self.display_name}] Initialized. Upload Limit: {max_concurrent_uploads}, Gen Limit: {max_concurrent_generation}")

    async def _upload_single_file(self, url: str, content: str, store_name: str):
        """Helper: Handles temp file creation and upload with semaphore."""
        if not content or not content.strip():
            logger.warning(f"[{self.display_name}] SKIPPING {url}: Content is empty.")
            return None

        clean_name = url[-128:] if len(url) > 128 else url
        
        # 1. Create a temp file to handle encoding/buffering correctly
        # delete=False is required for cross-platform compatibility (Windows locks open files)
        tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8')
        try:
            tf.write(content)
            tf.flush()
            tf.close() # Close write handle so we can read it safely

            async with self.upload_semaphore:
                try:
                    # 2. OPEN the file in Read-Binary mode
                    with open(tf.name, "rb") as f:
                        logger.debug(f"[{self.display_name}] Uploading: {clean_name}")
                        
                        result = await self.async_client.file_search_stores.upload_to_file_search_store(
                            file=f,
                            file_search_store_name=store_name,
                            config=types.UploadToFileSearchStoreConfig(
                                display_name=clean_name,
                                mime_type="text/plain"
                            )
                        )
                    return result
                except Exception as e:
                    logger.error(f"[{self.display_name}] Failed to upload {url}: {e}")
                    return None
        finally:
            # 3. Clean up the local temp file immediately
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
            count_success = 0
            for res in results:
                if isinstance(res, Exception) or not res:
                    continue
                
                # Check different SDK return shapes just in case
                if hasattr(res, 'name'): 
                    self.created_file_names.append(res.name)
                    count_success += 1
                elif hasattr(res, 'file') and hasattr(res.file, 'name'):
                     self.created_file_names.append(res.file.name)
                     count_success += 1

            logger.info(f"[{self.display_name}] Uploads complete. Files tracked: {count_success}/{len(tasks)}")
            
            if count_success == 0 and len(tasks) > 0:
                logger.warning(f"[{self.display_name}] All uploads failed. Deleting store.")
                await self.delete_store()
                return None

            return self.file_store_name

        except Exception as e:
            logger.error(f"[{self.display_name}] Critical error in create_and_upload: {e}", exc_info=True)
            await self.delete_store()
            return None

    def _get_default_prompt(self, query: str) -> str:
        return f"""
        You are an expert research analyst. Provide a comprehensive, analytical response to the user's query by synthesizing information from the collected research data.

        USER QUERY: {query}

        INSTRUCTIONS:
        - You have access to a set of files containing research data. Use the file_search tool to find relevant information.
        - Your task is to analyze this data and produce a well-structured analytical response.
        - You must provide clear quantitative data if that is required and available.
        - You must accord the highest priority to recent information from the provided files.
        - The report should be well-structured, using markdown for headings, subheadings, and bullet points where appropriate.
        - Target 500-2000 words.
        - Provide a list of references at the end citing the data sources used.
        """

    async def answer_question(self, query: str, relevant_contexts: Dict[str, Dict[str, Any]], system_instruction: Optional[str] = None) -> str:
        """Single question entry point."""
        try:
            file_store_name = await self.create_and_upload_contexts(relevant_contexts)
            if not file_store_name:
                raise Exception("Failed to populate store.")

            
            fs_tool = types.FileSearch(
                file_search_store_names=[file_store_name]
            )
            
            # 2. Wrap it in a Tool object explicitly
            tool = types.Tool(
                file_search=fs_tool
            )
            
            # 3. Create the AFC Disable config explicitly
            # This ensures the SDK strictly respects the disable flag
            afc_config = types.AutomaticFunctionCallingConfig(
                disable=True
            )
            
            final_instruction = system_instruction if system_instruction else self._get_default_prompt(query)

            logger.info(f"[{self.display_name}] Generating content for single query...")
            
            response = await self.async_client.models.generate_content(
                model=GOOGLE_MODEL, 
                contents=[query],
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1,
                    system_instruction=final_instruction,
                    automatic_function_calling={"disable": True},
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

    async def answer_batch_questions(self, queries: List[str], context_data: Dict, system_instruction_template: Optional[str] = None) -> Dict[str, str]:
        # 1. Create the store ONCE
        store_name = await self.create_and_upload_contexts(context_data)
        if not store_name:
            logger.error(f"[{self.display_name}] Store creation failed or empty.")
            return {}
        
        # 2. Define helper for single generation
        async def ask_single(q):
            final_instruction = system_instruction_template if system_instruction_template else self._get_default_prompt(q)
            
            # --- FIX: STRICT TYPE CONSTRUCTION ---
            # 1. Create the FileSearch object explicitly
            fs_tool = types.FileSearch(
                file_search_store_names=[store_name]
            )
            
            # 2. Wrap it in a Tool object explicitly
            tool = types.Tool(
                file_search=fs_tool
            )
            
            # 3. Create the AFC Disable config explicitly
            # This ensures the SDK strictly respects the disable flag
            afc_config = types.AutomaticFunctionCallingConfig(
                disable=True
            )

            async with self.generation_semaphore:
                try:
                    response = await self.async_client.models.generate_content(
                        model=GOOGLE_MODEL,
                        contents=[q],
                        config=types.GenerateContentConfig(
                            tools=[tool], # Pass as a list of types.Tool
                            temperature=0.1,
                            system_instruction=final_instruction,
                            automatic_function_calling=afc_config # Pass the Typed Object
                        )
                    )
                    
                    # Handle cases where response is filtered/blocked
                    if not response.text:
                         return q, f"Error: Empty response (Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'})"
                         
                    return q, response.text
                    
                except Exception as e:
                    logger.error(f"Error answering '{q}': {e}")
                    return q, f"Error generating response: {str(e)}"

        try:
            logger.info(f"[{self.display_name}] Processing {len(queries)} queries in parallel...")
            tasks = [ask_single(q) for q in queries]
            results = await asyncio.gather(*tasks)
            return dict(results)
        
        except Exception as e:
            logger.error(f"[{self.display_name}] Batch Error: {e}", exc_info=True)
            raise
        finally:
            await self.delete_store()

class GeminiFileSearchRetriever_old:
    """
    Manages a Gemini File Search Store with parallel uploads, batch querying, 
    and optimized O(1) cleanup.
    """
    def __init__(self, display_name_prefix: str = "crystal-fss", max_concurrent_uploads: int = 10, max_concurrent_generation: int = 10):
        # Generate a unique name for the store for this session
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.display_name = f"{display_name_prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
        
        # Initialize Client
        self.client = genai.Client(api_key=gemini_api_key)
        self.async_client = self.client.aio
        
        self.file_store_name: Optional[str] = None
        
        # Concurrency Controls
        self.upload_semaphore = asyncio.Semaphore(max_concurrent_uploads)
        self.generation_semaphore = asyncio.Semaphore(max_concurrent_generation)
        
        # Track created file names for fast O(1) cleanup
        self.created_file_names: List[str] = []
        
        logger.info(f"[{self.display_name}] Initialized. Upload Limit: {max_concurrent_uploads}, Gen Limit: {max_concurrent_generation}")

    async def _upload_single_file(self, url: str, content: str, store_name: str):
        """Helper: Handles temp file creation and upload with semaphore."""
        if not content or not content.strip():
            logger.warning(f"[{self.display_name}] SKIPPING {url}: Content is empty.")
            return None

        clean_name = url[-128:] if len(url) > 128 else url
        
        # 1. Create a temp file to handle encoding/buffering correctly
        # delete=False is required for cross-platform compatibility (Windows locks open files)
        tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8')
        try:
            tf.write(content)
            tf.flush()
            tf.close() # Close write handle so we can read it safely

            async with self.upload_semaphore:
                try:
                    # 2. OPEN the file in Read-Binary mode
                    with open(tf.name, "rb") as f:
                        logger.debug(f"[{self.display_name}] Uploading: {clean_name}")
                        
                        result = await self.async_client.file_search_stores.upload_to_file_search_store(
                            file=f,
                            file_search_store_name=store_name,
                            config=types.UploadToFileSearchStoreConfig(
                                display_name=clean_name,
                                mime_type="text/plain"
                            )
                        )
                    return result
                except Exception as e:
                    logger.error(f"[{self.display_name}] Failed to upload {url}: {e}")
                    return None
        finally:
            # 3. Clean up the local temp file immediately
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
            count_success = 0
            for res in results:
                if isinstance(res, Exception) or not res:
                    continue
                
                # Check different SDK return shapes just in case
                if hasattr(res, 'name'): 
                    self.created_file_names.append(res.name)
                    count_success += 1
                elif hasattr(res, 'file') and hasattr(res.file, 'name'):
                     self.created_file_names.append(res.file.name)
                     count_success += 1

            logger.info(f"[{self.display_name}] Uploads complete. Files tracked: {count_success}/{len(tasks)}")
            
            if count_success == 0 and len(tasks) > 0:
                logger.warning(f"[{self.display_name}] All uploads failed. Deleting store.")
                await self.delete_store()
                return None

            return self.file_store_name

        except Exception as e:
            logger.error(f"[{self.display_name}] Critical error in create_and_upload: {e}", exc_info=True)
            await self.delete_store()
            return None

    def _get_default_prompt(self, query: str) -> str:
        return f"""
        You are an expert research analyst. Provide a comprehensive, analytical response to the user's query by synthesizing information from the collected research data.

        USER QUERY: {query}

        INSTRUCTIONS:
        - You have access to a set of files containing research data. Use the file_search tool to find relevant information.
        - Your task is to analyze this data and produce a well-structured analytical response.
        - You must provide clear quantitative data if that is required and available.
        - You must accord the highest priority to recent information from the provided files.
        - The report should be well-structured, using markdown for headings, subheadings, and bullet points where appropriate.
        - Target 500-2000 words.
        - Provide a list of references at the end citing the data sources used.
        """

    async def answer_question(self, query: str, relevant_contexts: Dict[str, Dict[str, Any]], system_instruction: Optional[str] = None) -> str:
        """Single question entry point."""
        try:
            file_store_name = await self.create_and_upload_contexts(relevant_contexts)
            if not file_store_name:
                raise Exception("Failed to populate store.")

            tools = [types.Tool(
                file_search=types.FileSearch(file_search_store_names=[file_store_name])
            )]
            
            final_instruction = system_instruction if system_instruction else self._get_default_prompt(query)

            logger.info(f"[{self.display_name}] Generating content for single query...")
            
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

    async def answer_batch_questions(self, queries: List[str], context_data: Dict, system_instruction_template: Optional[str] = None) -> Dict[str, str]:
        """
        Answers multiple questions in parallel using the same File Store.
        """
        # 1. Create the store ONCE
        store_name = await self.create_and_upload_contexts(context_data)
        if not store_name:
            return {}
        
        # 2. Define the tool configuration ONCE
        tools = [types.Tool(
            file_search=types.FileSearch(file_search_store_names=[store_name])
        )]
        
        # 3. Create a helper for a single generation op
        async def ask_single(q):
            # Allow custom system instruction or fallback to default
            final_instruction = system_instruction_template if system_instruction_template else self._get_default_prompt(q)

            # Use semaphore to prevent Rate Limiting (RPM)
            async with self.generation_semaphore:
                try:
                    response = await self.async_client.models.generate_content(
                        model=GOOGLE_MODEL,
                        contents=[q],
                        config=types.GenerateContentConfig(
                            tools=tools,
                            temperature=0.1,
                            system_instruction=final_instruction
                        )
                    )
                    return q, response.text if response.text else "Error: Empty response"
                except Exception as e:
                    logger.error(f"Error answering '{q}': {e}")
                    return q, f"Error generating response: {str(e)}"

        try:
            # 4. Run all queries in parallel
            logger.info(f"[{self.display_name}] Processing {len(queries)} queries in parallel...")
            tasks = [ask_single(q) for q in queries]
            results = await asyncio.gather(*tasks)
            return dict(results)
        
        except Exception as e:
            logger.error(f"[{self.display_name}] Batch Error: {e}", exc_info=True)
            raise
        finally:
            await self.delete_store()

    async def delete_store(self):
        """Optimized deletion: Deletes tracked files directly, then the store."""
        if not self.file_store_name:
            return

        store_to_delete = self.file_store_name
        self.file_store_name = None  # Prevent re-entry

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
                tasks = [delete_file_safe(name) for name in self.created_file_names]
                await asyncio.gather(*tasks)
                self.created_file_names = []

            # 2. Delete Store
            logger.info(f"[{self.display_name}] Deleting store: {store_to_delete}")
            await self.async_client.file_search_stores.delete(name=store_to_delete)
            
        except Exception as e:
            logger.error(f"[{self.display_name}] Cleanup error: {e}")


class GeminiFileSearchRetriever_old:
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
            - Target 500-1000 words for this response.
            - Use clear, professional language with proper markdown formatting.
            - Provide a list of references at the end citing the data sources used.

            Generate the report keeping these instructions in mind:
            """
			
            final_instruction = intellisearch_prompt if system_instruction is None else system_instruction  

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


    async def answer_batch_questions(self, queries: List[str], context_data: Dict) -> Dict[str, str]:
        """
        Answers multiple questions in parallel using the same File Store.
        """
        # 1. Create the store ONCE (expensive/slow part)
        store_name = await self.create_and_upload_contexts(context_data)
        if not store_name:
            return {}
        
        # 2. Define the tool configuration ONCE
        tools = [types.Tool(
            file_search=types.FileSearch(file_search_store_names=[store_name])
        )]
        
      
        # 3. Create a helper for a single generation op
        async def ask_single(q):
            try:
                intellisearch_prompt = f"""
                You are an expert research analyst. Provide a comprehensive, analytical response to the user's query by synthesizing information from the collected research data.

                USER QUERY: {q}

                INSTRUCTIONS:
                - You have access to a set of files containing research data. Use the file_search tool to find relevant information to answer the user's query.
                - Your task is to analyze this data and produce a well-structured analytical response that directly answers the user's query with clarity and full depth.
                - You must provide clear quantitative data if that is required and available.
                - You must accord the highest priority to recent information from the provided files.                The report should be well-structured, using markdown for headings, subheadings, and bullet points where appropriate.
                - Target 500-1000 words for this response.
                - Provide a list of references at the end citing the data sources used.

                Generate the report keeping these instructions in mind:
                """
			
                final_instruction = intellisearch_prompt  

                # We use a separate config for each request, but same store
                response = await self.async_client.models.generate_content(
                    model=GOOGLE_MODEL,
                    contents=[q],
                    config=types.GenerateContentConfig(
                        tools=tools,
                        temperature=0.1,
                        system_instruction=final_instruction
                    )
                )
                return q, response.text
            except Exception as e:
                logger.error(f"Error answering '{q}': {e}")
                return q, "Error generating response."

        try:
            # 4. Run all queries in parallel
            # Use a semaphore if you have >100 queries to avoid RateLimit errors
            logger.info(f"[{self.display_name}] Processing {len(queries)} queries in parallel...")
            tasks = [ask_single(q) for q in queries]
            results = await asyncio.gather(*tasks)
            return dict(results)
        
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