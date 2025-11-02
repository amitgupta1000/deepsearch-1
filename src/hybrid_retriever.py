# hybrid_retriever.py
"""
Hybrid retriever combining BM25 (sparse) and vector search (dense) for improved relevance.
Implements ensemble retrieval with configurable weights and fusion strategies.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Type definitions and fallbacks
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
except ImportError as e:
    logging.warning(f"LangChain imports failed: {e}. Using fallback types.")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    class RecursiveCharacterTextSplitter:
        def __init__(self, **kwargs): pass
        def split_text(self, text): return [text]
    FAISS, BM25Retriever, EnsembleRetriever = None, None, None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logging.warning("rank_bm25 not available. BM25 functionality will be limited.")
    BM25Okapi = None

# Configuration
@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval."""
    # Retrieval parameters
    top_k: int = 20
    vector_weight: float = 0.6  # Weight for vector search
    bm25_weight: float = 0.4    # Weight for BM25 search
    
    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Quality filters
    min_chunk_length: int = 50
    min_word_count: int = 10
    
    # FAISS parameters
    score_threshold: float = 0.1
    fetch_k_multiplier: int = 2
    
    # Fusion strategy
    fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    rrf_k: int = 60  # RRF parameter


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search for improved document retrieval.
    Uses ensemble methods to leverage both sparse and dense retrieval approaches.
    """
    
    def __init__(self, embeddings=None, config: Optional[HybridRetrieverConfig] = None):
        self.embeddings = embeddings
        self.config = config or HybridRetrieverConfig()
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = []
        self.logger = logging.getLogger(__name__)
        
    def build_index(self, relevant_contexts: Dict[str, Dict[str, str]]) -> bool:
        """
        Build both vector and BM25 indices from relevant contexts.
        
        Args:
            relevant_contexts: Dict mapping URLs to their content and metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Process and chunk documents
            documents = self._process_documents(relevant_contexts)
            if not documents:
                self.logger.warning("No documents to index")
                return False
                
            self.documents = documents
            
            # Build vector index with task-optimized embeddings
            vector_success = self._build_vector_index(documents)
            
            # Build BM25 index
            bm25_success = self._build_bm25_index(documents)
            
            # Create ensemble retriever if both are available
            if vector_success and bm25_success and EnsembleRetriever:
                try:
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[
                            self.vector_store.as_retriever(
                                search_kwargs={"k": self.config.top_k}
                            ),
                            self.bm25_retriever
                        ],
                        weights=[self.config.vector_weight, self.config.bm25_weight]
                    )
                    self.logger.info("Ensemble retriever created successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to create ensemble retriever: {e}")
            
            return vector_success or bm25_success
            
        except Exception as e:
            self.logger.error(f"Error building hybrid index: {e}")
            return False
    
    def _process_documents(self, relevant_contexts: Dict[str, Dict[str, str]]) -> List[Document]:
        """Process and chunk documents from relevant contexts."""
        documents = []
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            for url, content_data in relevant_contexts.items():
                if not content_data or not isinstance(content_data, dict):
                    continue
                    
                content = content_data.get('content', '')
                title = content_data.get('title', 'Untitled')
                
                # Quality filters
                if len(content.strip()) < 100:
                    continue
                    
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.strip()
                    
                    # Additional quality filters
                    if (len(chunk_text) < self.config.min_chunk_length or 
                        len(chunk_text.split()) < self.config.min_word_count):
                        continue
                    
                    documents.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source": url,
                            "title": title,
                            "chunk_index": i,
                            "chunk_length": len(chunk_text),
                            "word_count": len(chunk_text.split())
                        }
                    ))
            
            self.logger.info(f"Processed {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            return []
    
    def _build_vector_index(self, documents: List[Document]) -> bool:
        """Build FAISS vector index with task-optimized embeddings."""
        try:
            if not self.embeddings or not FAISS:
                self.logger.warning("Embeddings or FAISS not available for vector index")
                return False
            
            # Use task-optimized embeddings if available
            if hasattr(self.embeddings, 'embed_documents'):
                # Check if it's our enhanced embeddings class
                if hasattr(self.embeddings, 'embed_for_retrieval') or 'Enhanced' in str(type(self.embeddings)):
                    # Use document-specific task type for indexing
                    self.logger.info("Using task-optimized embeddings for document indexing")
                    if hasattr(self.embeddings, 'default_task_type'):
                        original_task = self.embeddings.default_task_type
                        self.embeddings.default_task_type = "RETRIEVAL_DOCUMENT"
                        self.vector_store = FAISS.from_documents(documents, self.embeddings)
                        self.embeddings.default_task_type = original_task
                    else:
                        self.vector_store = FAISS.from_documents(documents, self.embeddings)
                else:
                    # Standard LangChain embeddings
                    self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
            self.logger.info("Vector index built successfully with optimized embeddings")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building vector index: {e}")
            return False
    
    def _build_bm25_index(self, documents: List[Document]) -> bool:
        """Build BM25 index."""
        try:
            if not BM25Retriever:
                self.logger.warning("BM25Retriever not available")
                return False
                
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.config.top_k
            self.logger.info("BM25 index built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {e}")
            return False
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Try ensemble retriever first (best option)
            if self.ensemble_retriever:
                return self._retrieve_ensemble(query)
            
            # Fallback to manual fusion
            return self._retrieve_manual_fusion(query)
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return []
    
    def retrieve_multi_query(self, queries: List[str], deduplicate: bool = True) -> List[Document]:
        """
        Retrieve relevant documents using multiple queries for better coverage.
        
        Args:
            queries: List of search queries to use for retrieval
            deduplicate: Whether to remove duplicate documents
            
        Returns:
            List of relevant Document objects from all queries
        """
        try:
            all_results = []
            seen_docs = set()
            
            # Retrieve documents for each query
            for query in queries:
                if not query.strip():
                    continue
                    
                query_results = self.retrieve(query)
                
                for doc in query_results:
                    doc_key = self._get_doc_key(doc)
                    
                    if deduplicate and doc_key in seen_docs:
                        continue
                        
                    all_results.append(doc)
                    seen_docs.add(doc_key)
                    
                    # Stop if we have enough results
                    if len(all_results) >= self.config.top_k:
                        break
                
                if len(all_results) >= self.config.top_k:
                    break
            
            # Re-rank combined results if we have more than needed
            if len(all_results) > self.config.top_k:
                # Use the first query as primary for ranking
                primary_query = queries[0] if queries else ""
                ranked_results = self._rerank_multi_query_results(primary_query, all_results)
                return ranked_results[:self.config.top_k]
            
            self.logger.info(f"Multi-query retrieval returned {len(all_results)} documents from {len(queries)} queries")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error during multi-query retrieval: {e}")
            # Fallback to single query with first query
            return self.retrieve(queries[0] if queries else "")
    
    def _rerank_multi_query_results(self, primary_query: str, documents: List[Document]) -> List[Document]:
        """Re-rank results from multiple queries using the primary query."""
        try:
            # Simple scoring based on primary query keyword overlap
            query_words = set(primary_query.lower().split())
            
            scored_docs = []
            for doc in documents:
                content = doc.page_content.lower()
                # Score based on keyword overlap and metadata relevance
                keyword_score = sum(1 for word in query_words if word in content)
                
                # Boost score for documents with query terms in title/source
                title = doc.metadata.get('title', '').lower()
                source = doc.metadata.get('source', '').lower()
                title_boost = sum(2 for word in query_words if word in title)
                source_boost = sum(1 for word in query_words if word in source)
                
                total_score = keyword_score + title_boost + source_boost
                scored_docs.append((total_score, doc))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scored_docs]
            
        except Exception as e:
            self.logger.error(f"Error re-ranking multi-query results: {e}")
            return documents
    
    def _retrieve_ensemble(self, query: str) -> List[Document]:
        """Retrieve using LangChain ensemble retriever."""
        try:
            results = self.ensemble_retriever.invoke(query)
            self.logger.info(f"Ensemble retrieval returned {len(results)} documents")
            return results[:self.config.top_k]
            
        except Exception as e:
            self.logger.error(f"Ensemble retrieval failed: {e}")
            return self._retrieve_manual_fusion(query)
    
    def _retrieve_manual_fusion(self, query: str) -> List[Document]:
        """Manual fusion of vector and BM25 results with task-optimized query embeddings."""
        try:
            vector_results = []
            bm25_results = []
            
            # Get vector results with query-optimized embeddings
            if self.vector_store:
                try:
                    # Optimize query embedding for retrieval if using enhanced embeddings
                    if hasattr(self.embeddings, 'embed_query') and hasattr(self.embeddings, 'default_task_type'):
                        # Temporarily set task type for query
                        original_task = getattr(self.embeddings, 'default_task_type', None)
                        if hasattr(self.embeddings, 'default_task_type'):
                            self.embeddings.default_task_type = "RETRIEVAL_QUERY"
                        
                        vector_results = self.vector_store.similarity_search_with_score(
                            query, 
                            k=self.config.top_k * self.config.fetch_k_multiplier,
                            score_threshold=self.config.score_threshold
                        )
                        
                        # Restore original task type
                        if original_task and hasattr(self.embeddings, 'default_task_type'):
                            self.embeddings.default_task_type = original_task
                    else:
                        # Standard vector search
                        vector_results = self.vector_store.similarity_search_with_score(
                            query, 
                            k=self.config.top_k * self.config.fetch_k_multiplier,
                            score_threshold=self.config.score_threshold
                        )
                    
                    vector_results = [(doc, score) for doc, score in vector_results]
                    self.logger.info(f"Vector search returned {len(vector_results)} results")
                    
                except Exception as e:
                    self.logger.warning(f"Vector search failed: {e}")
            
            # Get BM25 results
            if self.bm25_retriever:
                try:
                    bm25_results = self.bm25_retriever.invoke(query)
                    bm25_results = [(doc, 1.0) for doc in bm25_results]  # Uniform scores
                    self.logger.info(f"BM25 search returned {len(bm25_results)} results")
                except Exception as e:
                    self.logger.warning(f"BM25 search failed: {e}")
            
            # Fuse results
            if self.config.fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
            else:
                fused_results = self._weighted_fusion(vector_results, bm25_results)
            
            self.logger.info(f"Manual fusion returned {len(fused_results)} documents")
            return fused_results[:self.config.top_k]
            
        except Exception as e:
            self.logger.error(f"Manual fusion failed: {e}")
            # Fallback to vector or BM25 only
            if vector_results:
                return [doc for doc, _ in vector_results[:self.config.top_k]]
            elif bm25_results:
                return [doc for doc, _ in bm25_results[:self.config.top_k]]
            return []
    
    def _reciprocal_rank_fusion(self, vector_results: List[Tuple[Document, float]], 
                               bm25_results: List[Tuple[Document, float]]) -> List[Document]:
        """Implement Reciprocal Rank Fusion (RRF)."""
        try:
            scores = {}
            
            # Process vector results
            for i, (doc, _) in enumerate(vector_results):
                doc_key = self._get_doc_key(doc)
                scores[doc_key] = scores.get(doc_key, 0) + self.config.vector_weight / (self.config.rrf_k + i + 1)
            
            # Process BM25 results
            for i, (doc, _) in enumerate(bm25_results):
                doc_key = self._get_doc_key(doc)
                scores[doc_key] = scores.get(doc_key, 0) + self.config.bm25_weight / (self.config.rrf_k + i + 1)
            
            # Create document mapping
            doc_mapping = {}
            for doc, _ in vector_results + bm25_results:
                doc_mapping[self._get_doc_key(doc)] = doc
            
            # Sort by fused score
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            return [doc_mapping[doc_key] for doc_key, _ in sorted_docs if doc_key in doc_mapping]
            
        except Exception as e:
            self.logger.error(f"RRF fusion failed: {e}")
            return []
    
    def _weighted_fusion(self, vector_results: List[Tuple[Document, float]], 
                        bm25_results: List[Tuple[Document, float]]) -> List[Document]:
        """Simple weighted fusion of results."""
        try:
            scores = {}
            
            # Normalize and weight vector scores
            if vector_results:
                max_vector_score = max(score for _, score in vector_results)
                for doc, score in vector_results:
                    doc_key = self._get_doc_key(doc)
                    normalized_score = score / max_vector_score if max_vector_score > 0 else 0
                    scores[doc_key] = scores.get(doc_key, 0) + self.config.vector_weight * normalized_score
            
            # Weight BM25 results (uniform scoring)
            for doc, _ in bm25_results:
                doc_key = self._get_doc_key(doc)
                scores[doc_key] = scores.get(doc_key, 0) + self.config.bm25_weight
            
            # Create document mapping
            doc_mapping = {}
            for doc, _ in vector_results + bm25_results:
                doc_mapping[self._get_doc_key(doc)] = doc
            
            # Sort by fused score
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            return [doc_mapping[doc_key] for doc_key, _ in sorted_docs if doc_key in doc_mapping]
            
        except Exception as e:
            self.logger.error(f"Weighted fusion failed: {e}")
            return []
    
    def _get_doc_key(self, doc: Document) -> str:
        """Generate unique key for document deduplication."""
        try:
            # Use source + chunk_index for uniqueness
            source = doc.metadata.get('source', '')
            chunk_idx = doc.metadata.get('chunk_index', 0)
            return f"{source}_{chunk_idx}"
        except:
            # Fallback to content hash
            return str(hash(doc.page_content[:100]))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_documents": len(self.documents),
            "has_vector_store": self.vector_store is not None,
            "has_bm25_retriever": self.bm25_retriever is not None,
            "has_ensemble": self.ensemble_retriever is not None,
            "config": {
                "vector_weight": self.config.vector_weight,
                "bm25_weight": self.config.bm25_weight,
                "fusion_method": self.config.fusion_method,
                "top_k": self.config.top_k
            }
        }


# Factory function for easy integration
def create_hybrid_retriever(embeddings=None, **config_kwargs) -> HybridRetriever:
    """
    Factory function to create a hybrid retriever with custom configuration.
    
    Args:
        embeddings: Embedding model for vector search
        **config_kwargs: Configuration parameters to override defaults
        
    Returns:
        HybridRetriever instance
    """
    config = HybridRetrieverConfig(**config_kwargs)
    return HybridRetriever(embeddings=embeddings, config=config)


logging.info("hybrid_retriever.py loaded successfully")