# hybrid_retriever_new.py
"""
Simplified Hybrid retriever combining BM25 (sparse) and vector search (dense).
This version uses standard LangChain components for ensemble retrieval and reranking,
making it more maintainable and easier to understand.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Type definitions and fallbacks
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
    from langchain.retrievers import ContextualCompressionRetriever
except ImportError as e:
    logging.warning(f"LangChain imports failed: {e}. Using fallback types.")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    class RecursiveCharacterTextSplitter:
        def __init__(self, **kwargs): pass
        def split_text(self, text): return [text]
    FAISS, BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever = None, None, None, None

# Cross-encoder imports for semantic reranking
try:
    from langchain_community.cross_encoders import SentenceTransformerCrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    logging.warning("SentenceTransformerCrossEncoder not available. Reranking will be disabled.")
    SentenceTransformerCrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

# Configuration
@dataclass
class HybridRetrieverConfig:
    """Configuration for the new hybrid retriever."""
    # Retrieval parameters
    top_k: int = 20
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    
    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Quality filters
    min_chunk_length: int = 50
    min_word_count: int = 10
    
    # Cross-encoder reranking parameters
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20


class HybridRetriever:
    """
    A simplified hybrid retriever using standard LangChain components.
    """
    
    def __init__(self, embeddings=None, config: Optional[HybridRetrieverConfig] = None):
        self.embeddings = embeddings
        self.config = config or HybridRetrieverConfig()
        self.vector_store = None
        self.bm25_retriever = None
        self.final_retriever = None
        self.documents = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def build_index(self, relevant_contexts: Dict[str, Dict[str, str]]) -> bool:
        """
        Builds the retrieval pipeline from documents.
        """
        try:
            documents = self._process_documents(relevant_contexts)
            if not documents:
                self.logger.warning("No documents to index.")
                return False
            self.documents = documents

            # 1. Build base retrievers
            if not self._build_vector_index(documents) or not self._build_bm25_index(documents):
                self.logger.error("Failed to build one or more base retrievers.")
                return False

            # 2. Create Ensemble Retriever
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.config.top_k})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[self.config.bm25_weight, self.config.vector_weight],
            )
            self.logger.info("LangChain EnsembleRetriever created successfully.")

            # 3. Optionally wrap with Reranker
            if self.config.use_cross_encoder and CROSS_ENCODER_AVAILABLE:
                self.logger.info(f"Initializing reranker with model: {self.config.cross_encoder_model}")
                reranker = SentenceTransformerCrossEncoder(model_name=self.config.cross_encoder_model)
                
                self.final_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=ensemble_retriever,
                )
                self.final_retriever.base_compressor.top_n = self.config.rerank_top_k
                self.logger.info(f"Wrapped ensemble retriever with ContextualCompressionRetriever. Will return top {self.config.rerank_top_k} docs.")
            else:
                self.final_retriever = ensemble_retriever

            return True
            
        except Exception as e:
            self.logger.error(f"Error building hybrid index: {e}", exc_info=True)
            return False
    
    def _process_documents(self, relevant_contexts: Dict[str, Dict[str, str]]) -> List[Document]:
        """Process and chunk documents from relevant contexts."""
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        
        for url, content_data in relevant_contexts.items():
            content = content_data.get('content', '')
            if len(content.strip()) < 100:
                continue
            
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                if (len(chunk) < self.config.min_chunk_length or len(chunk.split()) < self.config.min_word_count):
                    continue
                
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "title": content_data.get('title', 'Untitled'),
                        "chunk_index": i,
                    }
                ))
        
        self.logger.info(f"Processed {len(documents)} document chunks.")
        return documents
            
    def _build_vector_index(self, documents: List[Document]) -> bool:
        """Build FAISS vector index."""
        try:
            if not self.embeddings or not FAISS:
                self.logger.warning("Embeddings or FAISS not available.")
                return False
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.logger.info("Vector index built successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error building vector index: {e}", exc_info=True)
            return False
    
    def _build_bm25_index(self, documents: List[Document]) -> bool:
        """Build BM25 index."""
        try:
            if not BM25Retriever:
                self.logger.warning("BM25Retriever not available.")
                return False
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.config.top_k
            self.logger.info("BM25 index built successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {e}", exc_info=True)
            return False
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using the configured pipeline.
        """
        if not self.final_retriever:
            self.logger.warning("Retriever not built. Returning empty list.")
            return []
        try:
            return self.final_retriever.invoke(query)
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []
    
    def retrieve_with_query_responses(self, queries: List[str]) -> Tuple[List[Document], Dict[str, str]]:
        """
        Retrieve documents using multiple queries and generate simple responses.
        This version is simplified as batching is handled by underlying components.
        """
        if not self.final_retriever:
            return [], {}

        all_docs = []
        query_responses = {}
        seen_docs = set()

        primary_query = queries[0] if queries else ""
        if not primary_query:
            return [], {}

        # The main retrieval is done with the primary query
        all_docs = self.retrieve(primary_query)

        # Generate simple responses for all queries based on the retrieved docs
        for query in queries:
            if not query.strip():
                continue
            
            if all_docs:
                top_docs_preview = all_docs[:3]
                response_parts = [
                    f"From {doc.metadata.get('title', 'Untitled')}: {doc.page_content[:150]}..."
                    for doc in top_docs_preview
                ]
                query_responses[query] = "\n\n".join(response_parts)
            else:
                query_responses[query] = "No relevant information found for this query."

        self.logger.info(f"Multi-query retrieval returned {len(all_docs)} documents.")
        return all_docs, query_responses

    def _get_doc_key(self, doc: Document) -> str:
        """Generate unique key for document deduplication."""
        source = doc.metadata.get('source', '')
        chunk_idx = doc.metadata.get('chunk_index', 0)
        return f"{source}_{chunk_idx}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_documents": len(self.documents),
            "retriever_type": self.final_retriever.__class__.__name__ if self.final_retriever else "None",
            "config": self.config.__dict__
        }


def create_hybrid_retriever(embeddings=None, **config_kwargs) -> HybridRetriever:
    """
    Factory function to create a new hybrid retriever.
    """
    config = HybridRetrieverConfig(**config_kwargs)
    return HybridRetriever(embeddings=embeddings, config=config)


logging.info("hybrid_retriever_new.py loaded successfully.")