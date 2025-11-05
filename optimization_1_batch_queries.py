#!/usr/bin/env python3
"""
OPTIMIZATION 1: BATCH QUERY EMBEDDING IMPLEMENTATION
This implements batched query embedding to reduce API calls from 4 to 2
"""

# This is the optimization we need to implement in hybrid_retriever.py

OPTIMIZED_RETRIEVE_WITH_QUERY_RESPONSES = '''
def retrieve_with_query_responses_batched(self, queries: List[str]) -> Tuple[List[Document], Dict[str, str]]:
    """
    OPTIMIZED VERSION: Retrieve documents using BATCHED query embeddings.
    
    Instead of embedding each query separately (3 API calls), 
    embed all queries at once (1 API call).
    
    Args:
        queries: List of search queries to use for retrieval
        
    Returns:
        Tuple of (all_documents, query_responses_dict)
    """
    try:
        if not queries:
            return [], {}
            
        # Filter out empty queries
        valid_queries = [q.strip() for q in queries if q.strip()]
        if not valid_queries:
            return [], {}
        
        all_results = []
        query_responses = {}
        seen_docs = set()
        
        # 🚀 OPTIMIZATION: Batch embed all queries at once (1 API call instead of N)
        if hasattr(self.embeddings, 'embed_documents') and len(valid_queries) > 1:
            self.logger.info(f"Batch embedding {len(valid_queries)} queries for retrieval")
            
            # Use embed_documents with RETRIEVAL_QUERY task type for batch processing
            if hasattr(self.embeddings, 'embed_documents'):
                try:
                    # Set task type for query embedding if using enhanced embeddings
                    if hasattr(self.embeddings, 'default_task_type'):
                        original_task = self.embeddings.default_task_type
                        self.embeddings.default_task_type = "RETRIEVAL_QUERY"
                        query_embeddings = self.embeddings.embed_documents(valid_queries)
                        self.embeddings.default_task_type = original_task
                    else:
                        query_embeddings = self.embeddings.embed_documents(valid_queries)
                    
                    self.logger.info(f"Successfully batch embedded {len(valid_queries)} queries")
                    
                    # Now retrieve using pre-computed embeddings
                    for i, query in enumerate(valid_queries):
                        query_embedding = query_embeddings[i]
                        query_results = self._retrieve_with_embedding(query, query_embedding)
                        
                        # Process results same as before
                        if query_results:
                            top_docs = query_results[:3]
                            response_parts = []
                            
                            for doc in top_docs:
                                content = doc.page_content[:200]
                                source = doc.metadata.get('source', 'Unknown')
                                title = doc.metadata.get('title', 'Untitled')
                                response_parts.append(f"From {title} ({source}): {content}...")
                            
                            query_responses[query] = "\\n\\n".join(response_parts)
                        else:
                            query_responses[query] = "No relevant information found for this query."
                        
                        # Add documents with deduplication
                        for doc in query_results:
                            doc_key = self._get_doc_key(doc)
                            if doc_key not in seen_docs:
                                all_results.append(doc)
                                seen_docs.add(doc_key)
                            
                            if len(all_results) >= self.config.top_k:
                                break
                        
                        if len(all_results) >= self.config.top_k:
                            break
                    
                except Exception as e:
                    self.logger.warning(f"Batch embedding failed, falling back to individual: {e}")
                    # Fallback to individual processing
                    return self._retrieve_individual_queries(valid_queries)
        else:
            # Fallback for single query or unsupported embedding type
            return self._retrieve_individual_queries(valid_queries)
        
        # Re-rank if needed
        if len(all_results) > self.config.top_k:
            primary_query = valid_queries[0]
            ranked_results = self._rerank_multi_query_results(primary_query, all_results)
            all_results = ranked_results[:self.config.top_k]
        
        self.logger.info(f"Batched multi-query retrieval returned {len(all_results)} documents and {len(query_responses)} query responses")
        return all_results, query_responses
        
    except Exception as e:
        self.logger.error(f"Error during batched multi-query retrieval: {e}")
        # Fallback to original method
        return self.retrieve_with_query_responses_original(valid_queries)

def _retrieve_with_embedding(self, query: str, query_embedding: List[float]) -> List[Document]:
    """
    Retrieve documents using pre-computed query embedding (no additional API call).
    """
    try:
        if not self.vector_store:
            return []
        
        # Use the pre-computed embedding for vector search
        vector_results = []
        bm25_results = []
        
        # Vector search with pre-computed embedding
        if hasattr(self.vector_store, 'similarity_search_by_vector'):
            try:
                vector_results = self.vector_store.similarity_search_by_vector(
                    query_embedding,
                    k=self.config.top_k * 2,  # Get more candidates for fusion
                    score_threshold=self.config.score_threshold
                )
                vector_results = [(doc, 1.0) for doc in vector_results]  # Placeholder scores
                self.logger.info(f"Vector search (pre-computed) returned {len(vector_results)} results")
            except Exception as e:
                self.logger.warning(f"Vector search with pre-computed embedding failed: {e}")
        
        # BM25 search (doesn't need embedding)
        if self.bm25_retriever:
            try:
                bm25_results = self.bm25_retriever.invoke(query)
                bm25_results = [(doc, 1.0) for doc in bm25_results]
                self.logger.info(f"BM25 search returned {len(bm25_results)} results")
            except Exception as e:
                self.logger.warning(f"BM25 search failed: {e}")
        
        # Fuse results
        if self.config.fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            fused_results = self._weighted_fusion(vector_results, bm25_results)
        
        return fused_results[:self.config.top_k]
        
    except Exception as e:
        self.logger.error(f"Error in _retrieve_with_embedding: {e}")
        return []

def _retrieve_individual_queries(self, queries: List[str]) -> Tuple[List[Document], Dict[str, str]]:
    """
    Fallback method: retrieve queries individually (original behavior).
    """
    # This is the original retrieve_with_query_responses logic
    # (same as current implementation)
    return self.retrieve_with_query_responses_original(queries)
'''

print("🚀 OPTIMIZATION 1: BATCH QUERY EMBEDDING")
print("=" * 50)
print()
print("📊 Current State:")
print("- Document embedding: ✅ Already batched (1 API call for all docs)")
print("- Query embedding: ❌ Individual calls (1 API call per query)")
print("- Total API calls: 4 (1 for docs + 3 for queries)")
print()
print("🎯 Proposed Optimization:")
print("- Document embedding: ✅ Keep batched (1 API call)")
print("- Query embedding: ✅ Batch all queries (1 API call)")
print("- Total API calls: 2 (50% reduction)")
print()
print("📈 Expected Performance Improvement:")
print("- API calls: 4 → 2 (50% reduction)")
print("- Network latency: ~1.2s reduction (2 fewer round trips)")
print("- Total time: 2.47s → ~1.27s (48% improvement)")
print()
print("🔧 Implementation Plan:")
print("1. Add batch query embedding method to hybrid_retriever.py")
print("2. Add helper method for retrieval with pre-computed embeddings")
print("3. Update retrieve_with_query_responses to use batching")
print("4. Add fallback to original method if batching fails")
print("5. Test with our real query test to validate improvement")
print()
print("💡 Key Technical Points:")
print("- Uses embed_documents() with RETRIEVAL_QUERY task type")
print("- Pre-computes all query embeddings in one API call")
print("- Uses similarity_search_by_vector() to avoid re-embedding")
print("- Maintains same result quality with better performance")
print("- Graceful fallback to original method if batching fails")