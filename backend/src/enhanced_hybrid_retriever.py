#!/usr/bin/env python3
"""
Enhanced HybridRetriever with Server-Friendly Cross-Encoder Integration
Demonstrates practical integration without overloading the server
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    # Fallback for testing
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class BaseRetriever:
        pass

# Import our server-friendly cross-encoder
from .server_friendly_cross_encoder import (
    ServerFriendlyCrossEncoder, 
    CrossEncoderConfig,
    create_server_friendly_cross_encoder
)

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrievalConfig:
    """Configuration for enhanced hybrid retrieval"""
    
    # Cross-encoder settings
    enable_cross_encoder: bool = True
    cross_encoder_tier: str = "balanced"  # fast, balanced, quality
    cross_encoder_threshold: int = 10     # Minimum docs to trigger reranking
    
    # Performance safeguards
    max_total_docs: int = 100             # Overall limit
    fallback_gracefully: bool = True      # Never break retrieval
    
    # Quality settings
    enable_score_fusion: bool = True      # Combine BM25 + semantic + cross-encoder scores
    score_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.score_weights is None:
            self.score_weights = {
                "bm25": 0.3,
                "semantic": 0.4, 
                "cross_encoder": 0.3
            }


class EnhancedHybridRetriever:
    """
    Enhanced hybrid retriever with server-friendly cross-encoder reranking
    """
    
    def __init__(self, 
                 bm25_retriever: BaseRetriever,
                 semantic_retriever: BaseRetriever,
                 config: HybridRetrievalConfig = None):
        """
        Initialize enhanced hybrid retriever
        
        Args:
            bm25_retriever: BM25/keyword-based retriever
            semantic_retriever: Dense vector retriever 
            config: Configuration for hybrid retrieval
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.config = config or HybridRetrievalConfig()
        
        # Initialize cross-encoder if enabled
        self.cross_encoder = None
        if self.config.enable_cross_encoder:
            self.cross_encoder = create_server_friendly_cross_encoder(
                self.config.cross_encoder_tier
            )
            if self.cross_encoder:
                logger.info(f"Cross-encoder initialized: {self.config.cross_encoder_tier}")
            else:
                logger.warning("Cross-encoder initialization failed, continuing without reranking")
    
    def retrieve(self, query: str, k: int = 20) -> List[Document]:
        """
        Enhanced hybrid retrieval with optional cross-encoder reranking
        """
        try:
            # Step 1: Get initial results from both retrievers
            bm25_docs = self._safe_retrieve(self.bm25_retriever, query, k)
            semantic_docs = self._safe_retrieve(self.semantic_retriever, query, k)
            
            # Step 2: Combine and deduplicate
            combined_docs = self._combine_and_deduplicate(bm25_docs, semantic_docs)
            
            # Step 3: Limit total documents for performance
            if len(combined_docs) > self.config.max_total_docs:
                combined_docs = combined_docs[:self.config.max_total_docs]
                logger.debug(f"Limited to {self.config.max_total_docs} documents for performance")
            
            # Step 4: Apply cross-encoder reranking if available and beneficial
            if self._should_use_cross_encoder(query, combined_docs):
                logger.debug("Applying cross-encoder reranking")
                reranked_docs = self.cross_encoder.rerank(query, combined_docs)
                
                # Step 5: Optionally fuse scores for final ranking
                if self.config.enable_score_fusion:
                    final_docs = self._fuse_scores(query, reranked_docs, bm25_docs, semantic_docs)
                else:
                    final_docs = reranked_docs
            else:
                # Use original hybrid ranking
                final_docs = combined_docs
            
            # Step 6: Return top k results
            return final_docs[:k]
            
        except Exception as e:
            logger.error(f"Error in enhanced hybrid retrieval: {e}")
            if self.config.fallback_gracefully:
                # Fallback to basic hybrid retrieval
                return self._basic_fallback_retrieval(query, k)
            else:
                raise
    
    def _safe_retrieve(self, retriever: BaseRetriever, query: str, k: int) -> List[Document]:
        """Safely retrieve documents with error handling"""
        try:
            if hasattr(retriever, 'invoke'):
                return retriever.invoke(query, k=k) or []
            elif hasattr(retriever, 'get_relevant_documents'):
                return retriever.get_relevant_documents(query)[:k] or []
            else:
                logger.warning(f"Unknown retriever interface: {type(retriever)}")
                return []
        except Exception as e:
            logger.warning(f"Retriever failed: {e}")
            return []
    
    def _combine_and_deduplicate(self, bm25_docs: List[Document], semantic_docs: List[Document]) -> List[Document]:
        """Combine documents from both retrievers and remove duplicates"""
        # Simple deduplication by content hash
        seen_content = set()
        combined = []
        
        # Add BM25 docs first (they tend to be more keyword-relevant)
        for doc in bm25_docs:
            content_hash = hash(doc.page_content[:200])  # Use first 200 chars
            if content_hash not in seen_content:
                doc.metadata['retrieval_source'] = 'bm25'
                doc.metadata['bm25_rank'] = len(combined)
                combined.append(doc)
                seen_content.add(content_hash)
        
        # Add semantic docs that aren't duplicates
        semantic_rank = 0
        for doc in semantic_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                doc.metadata['retrieval_source'] = 'semantic'
                doc.metadata['semantic_rank'] = semantic_rank
                combined.append(doc)
                seen_content.add(content_hash)
            else:
                # Mark existing doc as found by both retrievers
                for existing_doc in combined:
                    if hash(existing_doc.page_content[:200]) == content_hash:
                        existing_doc.metadata['retrieval_source'] = 'both'
                        existing_doc.metadata['semantic_rank'] = semantic_rank
                        break
            semantic_rank += 1
        
        logger.debug(f"Combined {len(bm25_docs)} BM25 + {len(semantic_docs)} semantic -> {len(combined)} unique docs")
        return combined
    
    def _should_use_cross_encoder(self, query: str, documents: List[Document]) -> bool:
        """Determine if cross-encoder reranking should be applied"""
        if not self.cross_encoder:
            return False
        
        if len(documents) < self.config.cross_encoder_threshold:
            logger.debug(f"Too few documents ({len(documents)}) for cross-encoder reranking")
            return False
        
        # Use cross-encoder's intelligent decision making
        return self.cross_encoder.should_use_reranking(query, documents)
    
    def _fuse_scores(self, query: str, cross_encoder_docs: List[Document], 
                     bm25_docs: List[Document], semantic_docs: List[Document]) -> List[Document]:
        """
        Fuse scores from different retrieval methods for final ranking
        """
        if not self.config.enable_score_fusion:
            return cross_encoder_docs
        
        # Create score mappings
        bm25_scores = {hash(doc.page_content[:200]): 1.0 / (i + 1) 
                      for i, doc in enumerate(bm25_docs)}
        semantic_scores = {hash(doc.page_content[:200]): 1.0 / (i + 1) 
                          for i, doc in enumerate(semantic_docs)}
        
        weights = self.config.score_weights
        scored_docs = []
        
        for doc in cross_encoder_docs:
            content_hash = hash(doc.page_content[:200])
            
            # Get individual scores
            bm25_score = bm25_scores.get(content_hash, 0.0)
            semantic_score = semantic_scores.get(content_hash, 0.0)
            cross_encoder_score = doc.metadata.get('cross_encoder_score', 0.0)
            
            # Normalize cross-encoder score to 0-1 range (assuming it's typically -10 to 10)
            normalized_ce_score = max(0, min(1, (cross_encoder_score + 10) / 20))
            
            # Compute fused score
            fused_score = (
                weights['bm25'] * bm25_score +
                weights['semantic'] * semantic_score +
                weights['cross_encoder'] * normalized_ce_score
            )
            
            # Add to metadata
            doc.metadata['fused_score'] = fused_score
            doc.metadata['bm25_score'] = bm25_score
            doc.metadata['semantic_score'] = semantic_score
            doc.metadata['cross_encoder_score_normalized'] = normalized_ce_score
            
            scored_docs.append((doc, fused_score))
        
        # Sort by fused score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]
    
    def _basic_fallback_retrieval(self, query: str, k: int) -> List[Document]:
        """Fallback to basic hybrid retrieval without cross-encoder"""
        logger.info("Using fallback hybrid retrieval")
        
        bm25_docs = self._safe_retrieve(self.bm25_retriever, query, k)
        semantic_docs = self._safe_retrieve(self.semantic_retriever, query, k)
        
        # Simple interleaving of results
        combined = []
        max_docs = max(len(bm25_docs), len(semantic_docs))
        
        for i in range(max_docs):
            if i < len(bm25_docs) and len(combined) < k:
                combined.append(bm25_docs[i])
            if i < len(semantic_docs) and len(combined) < k:
                # Avoid duplicates
                content_hash = hash(semantic_docs[i].page_content[:200])
                if not any(hash(doc.page_content[:200]) == content_hash for doc in combined):
                    combined.append(semantic_docs[i])
        
        return combined[:k]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "cross_encoder_enabled": self.cross_encoder is not None,
            "config": {
                "cross_encoder_tier": self.config.cross_encoder_tier,
                "cross_encoder_threshold": self.config.cross_encoder_threshold,
                "max_total_docs": self.config.max_total_docs,
                "score_fusion_enabled": self.config.enable_score_fusion
            }
        }
        
        if self.cross_encoder:
            stats["cross_encoder_performance"] = self.cross_encoder.get_performance_summary()
        
        return stats
    
    def update_config(self, new_config: HybridRetrievalConfig):
        """Update configuration and reinitialize cross-encoder if needed"""
        old_tier = self.config.cross_encoder_tier
        self.config = new_config
        
        # Reinitialize cross-encoder if tier changed
        if (new_config.enable_cross_encoder and 
            new_config.cross_encoder_tier != old_tier):
            logger.info(f"Reinitializing cross-encoder: {old_tier} -> {new_config.cross_encoder_tier}")
            self.cross_encoder = create_server_friendly_cross_encoder(
                new_config.cross_encoder_tier
            )


# Factory function for easy creation
def create_enhanced_hybrid_retriever(bm25_retriever: BaseRetriever, 
                                   semantic_retriever: BaseRetriever,
                                   tier: str = "balanced") -> EnhancedHybridRetriever:
    """
    Create an enhanced hybrid retriever with optimal settings for production
    """
    config = HybridRetrievalConfig(
        enable_cross_encoder=True,
        cross_encoder_tier=tier,
        cross_encoder_threshold=8,  # Conservative threshold
        max_total_docs=80,          # Conservative limit
        enable_score_fusion=True
    )
    
    return EnhancedHybridRetriever(bm25_retriever, semantic_retriever, config)


# Example usage and testing
def demo_enhanced_retrieval():
    """Demonstrate the enhanced retrieval system"""
    # This would normally use your actual BM25 and semantic retrievers
    
    class MockRetriever:
        def __init__(self, name):
            self.name = name
        
        def invoke(self, query, k=10):
            # Mock documents for demonstration
            return [
                Document(f"{self.name} result for {query} - document {i}", 
                        {"source": f"{self.name}_{i}"})
                for i in range(k)
            ]
    
    bm25_retriever = MockRetriever("BM25")
    semantic_retriever = MockRetriever("Semantic")
    
    # Create enhanced retriever
    enhanced_retriever = create_enhanced_hybrid_retriever(
        bm25_retriever, semantic_retriever, tier="fast"
    )
    
    # Test retrieval
    query = "machine learning algorithms"
    results = enhanced_retriever.retrieve(query, k=10)
    
    print(f"Enhanced retrieval for '{query}':")
    print(f"Found {len(results)} documents")
    
    # Show performance stats
    stats = enhanced_retriever.get_performance_stats()
    print("Performance stats:", stats)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_enhanced_retrieval()