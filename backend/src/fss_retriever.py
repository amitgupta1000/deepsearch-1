from google import genai
from google.genai import types
import asyncio
import logging
from typing import Dict, Any, Optional
import uuid
import io

# Import API keys from api_keys.py
try:
    from .api_keys import GOOGLE_API_KEY
except ImportError:
    logging.error("Could not import API keys from api_keys.py. LLMs and embeddings may not initialize.")
    GOOGLE_API_KEY = None

# Configure the generative AI library with the API key
gemini_api_key = GOOGLE_API_KEY

class GeminiFileSearchRetriever:
    """
    Manages a Gemini File Search Store for a single research session.
    It handles store creation, file uploads from in-memory contexts,
    and deletion of the store after use.

    Usage:
        retriever = GeminiFileSearchRetriever(display_name_prefix="session-xyz")
        await retriever.create_and_upload_contexts(relevant_contexts)
        # ... use retriever.file_store_name in other operations ...
        await retriever.delete_store()
    """
    def __init__(self, display_name_prefix: str = "crystal-fss"):
        # Generate a unique name for the store for this session
        self.display_name = f"{display_name_prefix}-{uuid.uuid4().hex[:8]}"
        self.client = genai.Client(api_key=gemini_api_key)  # Sync client for setup/teardown
        self.async_client = self.client.aio
        self.file_store_name: Optional[str] = None
        self._file_search_store_obj = None
        logging.info(f"[FSS] Initialized retriever for store: {self.display_name}")

    async def create_and_upload_contexts(self, relevant_contexts: Dict[str, Dict[str, Any]]) -> Optional[str]:
        if not relevant_contexts:
            logging.warning("[FSS] No relevant contexts provided to upload.")
            return None

        try:
            # 1. Create the Store
            logging.info(f"[FSS] Creating File Search Store: '{self.display_name}'")
            self._file_search_store_obj = await self.async_client.file_search_stores.create(
                config=types.CreateFileSearchStoreConfig(display_name=self.display_name)
            )
            self.file_store_name = self._file_search_store_obj.name
            logging.info(f"[FSS] Store created successfully: {self.file_store_name}")

            upload_ops = []

            # 2. Iterate through your memory dictionary and upload
            for url, data in relevant_contexts.items():
                content_text = data.get("content", "")
                if not content_text.strip():
                    continue

                file_stream = io.BytesIO(content_text.encode("utf-8"))
                clean_name = url[-128:] if len(url) > 128 else url # Truncate long URLs for display name

                logging.info(f"[FSS] Uploading in-memory stream for: {url}")
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
                logging.warning("[FSS] No valid content to upload after filtering. Deleting empty store.")
                await self.delete_store()
                return None

            # 3. Wait for indexing to finish
            logging.info(f"[FSS] Waiting for remote indexing of {len(upload_ops)} files...")
            for op in upload_ops:
                op_name = op.name
                current_op = op
                while not current_op.done:
                    await asyncio.sleep(5) # Poll every 5 seconds
                    current_op = await self.async_client.operations.get(op_name)
                if current_op.error:
                    raise Exception(f"File upload and indexing failed for operation {op.name}: {op.error.message}")

            logging.info(f"[FSS] All {len(upload_ops)} memory contents indexed successfully in store {self.file_store_name}.")
            return self.file_store_name # Return the store name on success

        except Exception as e:
            logging.error(f"[FSS] Error during store creation or upload: {e}", exc_info=True)
            # Attempt to clean up if the store was created
            if self.file_store_name:
                await self.delete_store()
            return None

    async def answer_question(self, query: str, relevant_contexts: Dict[str, Dict[str, Any]], system_instruction: Optional[str] = None) -> str:
        """
        Creates a temporary store, uploads contexts, asks a question, and cleans up.
        """
        file_store_name = None
        try:
            # Create store and upload files
            file_store_name = await self.create_and_upload_contexts(relevant_contexts)
            if not file_store_name:
                raise Exception("Failed to create and populate the file search store.")

            # Define the file search tool for the LLM call
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
            
            # Use the provided system_instruction if available, otherwise use the default prompt.
            final_instruction = system_instruction if system_instruction else intellisearch_prompt

            # Generate content using the file search tool
            logging.info(f"[FSS] Answering question '{query}' using store '{file_store_name}'")
            response = await self.async_client.models.generate_content(
                model="gemini-1.5-flash-latest",  # Use a valid model that supports File Search
                contents=query,
                system_instruction=final_instruction,
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1
                )
            )

            return response.text

        except Exception as e:
            logging.error(f"[FSS] Error in answer_question: {e}", exc_info=True)
            raise
        finally:
            await self.delete_store()

    async def delete_store(self):
        """Deletes the File Search Store associated with this instance."""
        if self.file_store_name:
            try:
                logging.info(f"[FSS] Listing files in store {self.file_store_name} for deletion.")
                # List all files in the store to delete them first.
                # The API has changed: list all files and filter by store name.
                all_files_pager = self.client.files.list(page_size=1000)
                delete_tasks = []
                for file in all_files_pager:
                    if hasattr(file, 'file_search_store_name') and file.file_search_store_name == self.file_store_name:
                        logging.info(f"[FSS] Scheduling deletion for file: {file.name}")
                        delete_tasks.append(self.async_client.files.delete(name=file.name))
                await asyncio.gather(*delete_tasks, return_exceptions=True)
                logging.info(f"[FSS] All files in store {self.file_store_name} have been deleted.")
                logging.warning(f"[FSS] Deleting File Search Store: {self.file_store_name}")
                await self.async_client.file_search_stores.delete(name=self.file_store_name)
                logging.info(f"[FSS] Successfully deleted store: {self.file_store_name}")
                self.file_store_name = None
                self._file_search_store_obj = None
            except Exception as e:
                # Log error but don't raise, as it's a cleanup step
                logging.error(
                    f"[FSS] Failed to delete store {self.file_store_name}. Manual cleanup may be required. Error: {e}"
                )

async def delete_gemini_file_search_store(file_store_name: str):
    """
    Deletes a Gemini FileSearchStore by name.
    """
    client = genai.Client()
    try:
        logging.info(f"[FSS] Listing files in store {file_store_name} for manual deletion.")
        # List and delete all files within the store first.
        all_files_pager = client.files.list(page_size=1000)
        delete_tasks = []
        for file in all_files_pager:
            if hasattr(file, 'file_search_store_name') and file.file_search_store_name == file_store_name:
                logging.info(f"[FSS] Scheduling deletion for file: {file.name}")
                delete_tasks.append(client.aio.files.delete(name=file.name))

        await asyncio.gather(*delete_tasks, return_exceptions=True)
        logging.info(f"[FSS] All files in store {file_store_name} have been deleted.")

        await client.aio.file_search_stores.delete(name=file_store_name)
        logging.info(f"[FSS] Manually deleted FileSearchStore: {file_store_name}")
    except Exception as e:
        logging.error(f"[FSS] Failed to manually delete FileSearchStore {file_store_name}: {e}", exc_info=True)
