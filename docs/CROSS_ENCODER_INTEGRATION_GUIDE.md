# Cross-Encoder Integration Guide for INTELLISEARCH

This guide shows how to integrate server-friendly cross-encoder reranking into your existing hybrid retrieval system using the proper LangChain ContextualCompressionRetriever pattern.

## LangChain Pattern Implementation

Your existing pattern:
```python
# --- Contextual Compression Retriever (using Ensemble and Reranker if available) ---
logging.info("Building final Retriever...")
if reranker:
    self.retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever # Use Ensemble retriever as the base
    )
    logging.info("Contextual Compression Retriever initialized with Reranker.")
else:
     self.retriever = ensemble_retriever # Use the Ensemble retriever directly if reranker fails
     logging.warning("Final Retriever initialized without Reranker (due to failure).")
```

## Updated Implementation

### 1. Install Dependencies

```bash
pip install sentence-transformers
```

### 2. Import the Cross-Encoder Modules

```python
# Add these imports at the top of your hybrid_retriever.py
from .langchain_cross_encoder import create_langchain_cross_encoder_retriever
from .langchain_cross_encoder import create_standard_langchain_cross_encoder
```

### 3. Update Your Retriever Building Code

Replace your existing retriever building section with:

```python
def _build_retriever(self):
    """Build retriever following LangChain ContextualCompressionRetriever pattern"""
    
    # --- Build Ensemble Retriever (Base) ---
    logging.info("Building Ensemble Retriever...")
    ensemble_retriever = self._create_ensemble_retriever()
    
    if not ensemble_retriever:
        logging.warning("Failed to create ensemble retriever")
        return None
    
    # --- Contextual Compression Retriever (using Ensemble and Reranker if available) ---
    logging.info("Building final Retriever...")
    
    # Try to create cross-encoder reranker
    reranker = self._create_cross_encoder_reranker()
    
    if reranker:
        # Create ContextualCompressionRetriever with cross-encoder
        self.retriever = create_langchain_cross_encoder_retriever(
            base_retriever=ensemble_retriever,  # Use Ensemble retriever as the base
            model_tier="balanced"  # or "fast" for production
        )
        
        if self.retriever:
            logging.info("Contextual Compression Retriever initialized with Reranker.")
        else:
            # Fallback if compression retriever creation fails
            self.retriever = ensemble_retriever
            logging.warning("Failed to create compression retriever, using ensemble only")
    else:
        # Use the Ensemble retriever directly if reranker fails
        self.retriever = ensemble_retriever
        logging.warning("Final Retriever initialized without Reranker (due to failure).")
    
    return self.retriever

def _create_cross_encoder_reranker(self):
    """Create cross-encoder reranker with server-friendly settings"""
    try:
        reranker = create_standard_langchain_cross_encoder(
            model_tier="balanced"  # Start conservative
        )
        
        if reranker and reranker.is_available():
            logging.info("Cross-encoder reranker created successfully")
            return reranker
        else:
            logging.warning("Cross-encoder reranker not available")
            return None
            
    except Exception as e:
        logging.error(f"Failed to create cross-encoder reranker: {e}")
        return None
```

### 4. Complete Integration Example

Here's how to integrate into your existing `HybridRetriever` class:

```python
class HybridRetriever:
    def __init__(self, documents, config=None):
        self.documents = documents
        self.config = config or HybridRetrieverConfig()
        
        # Your existing initialization code...
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.retriever = None  # This will be the final retriever
        
        # Build the retriever pipeline
        if self.build_index(documents):
            self.retriever = self._build_final_retriever()
    
    def _build_final_retriever(self):
        """Build the final retriever with optional cross-encoder reranking"""
        
        # Create ensemble retriever from your existing BM25 and vector retrievers
        if self.vector_store and self.bm25_retriever:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    self.vector_store.as_retriever(search_kwargs={"k": self.config.top_k}),
                    self.bm25_retriever
                ],
                weights=[self.config.vector_weight, self.config.bm25_weight]
            )
            
            # --- Contextual Compression Retriever (using Ensemble and Reranker if available) ---
            logging.info("Building final Retriever...")
            
            # Try to create reranker
            reranker = self._create_cross_encoder_reranker()
            
            if reranker:
                compression_retriever = create_langchain_cross_encoder_retriever(
                    base_retriever=ensemble_retriever,  # Use Ensemble retriever as the base
                    model_tier="balanced"
                )
                
                if compression_retriever:
                    logging.info("Contextual Compression Retriever initialized with Reranker.")
                    return compression_retriever
                else:
                    logging.warning("Failed to create compression retriever")
            
            # Use the Ensemble retriever directly if reranker fails
            logging.warning("Final Retriever initialized without Reranker (due to failure).")
            return ensemble_retriever
        
        return None
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve documents using the configured retriever pipeline"""
        if not self.retriever:
            logging.error("No retriever available")
            return []
        
        k = k or self.config.top_k
        
        try:
            if hasattr(self.retriever, 'invoke'):
                results = self.retriever.invoke(query)
            else:
                results = self.retriever.get_relevant_documents(query)
            
            return results[:k] if results else []
            
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return []
```

## Model Tier Selection Guide

Choose based on your server capacity and quality needs:

### "fast" Tier (Recommended for High Traffic)
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Size**: ~90MB
- **Latency**: ~50ms per batch
- **Use When**: High request volume, limited resources
- **Quality**: Good for most queries

### "balanced" Tier (Recommended for Most Cases)
- **Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Size**: ~110MB  
- **Latency**: ~100ms per batch
- **Use When**: Moderate traffic, balanced needs
- **Quality**: Better than fast, still efficient

### "quality" Tier (Use Sparingly)
- **Model**: `cross-encoder/ms-marco-electra-base`
- **Size**: ~440MB
- **Latency**: ~200ms per batch
- **Use When**: Critical queries, low volume
- **Quality**: Highest accuracy

## Performance Monitoring

Check performance regularly:

```python
# Get performance stats
stats = enhanced_retriever.get_performance_stats()
print(f"Average latency: {stats['cross_encoder_performance']['avg_latency_ms']:.1f}ms")
print(f"Cache hit rate: {stats['cross_encoder_performance']['cache_hit_rate']:.1%}")

# Adjust if needed
if stats['cross_encoder_performance']['avg_latency_ms'] > 1000:
    print("⚠️ Consider switching to 'fast' tier")
```

## Gradual Deployment Strategy

### Phase 1: Testing (Week 1)
```python
config = HybridRetrievalConfig(
    enable_cross_encoder=True,
    cross_encoder_tier="fast",
    cross_encoder_threshold=15,  # Only for queries with many results
    max_total_docs=50           # Conservative limit
)
```

### Phase 2: Production (Week 2-3)
```python
config = HybridRetrievalConfig(
    enable_cross_encoder=True,
    cross_encoder_tier="balanced",
    cross_encoder_threshold=10,  # More aggressive
    max_total_docs=80
)
```

### Phase 3: Optimization (Week 4+)
```python
# Monitor and adjust based on performance data
if avg_latency < 500:  # If server handles it well
    config.cross_encoder_tier = "quality"
    config.cross_encoder_threshold = 8
```

## Safety Features Built-In

The implementation includes multiple safety nets:

1. **Automatic Fallback**: If reranking fails, returns original ranking
2. **Timeout Protection**: Max 3 seconds per reranking operation
3. **Memory Limits**: Prevents excessive memory usage
4. **Smart Filtering**: Skips reranking for simple/specific queries
5. **Caching**: Reduces redundant computation
6. **Performance Monitoring**: Tracks metrics for optimization

## Example Integration with Your Current Code

If you have this pattern:

```python
def answer_question(query: str):
    # Retrieve documents
    docs = hybrid_retriever.retrieve(query, k=20)
    
    # Generate answer
    answer = llm.invoke(context=docs, question=query)
    return answer
```

Simply replace your retriever:

```python
def answer_question(query: str):
    # Retrieve documents (now with cross-encoder reranking!)
    docs = enhanced_hybrid_retriever.retrieve(query, k=20)
    
    # Generate answer (same as before)
    answer = llm.invoke(context=docs, question=query)
    return answer
```

## Troubleshooting

### Cross-encoder not loading?
```bash
# Install sentence-transformers
pip install sentence-transformers

# Check available models
python -c "from sentence_transformers import CrossEncoder; print('Available')"
```

### High latency?
- Switch to "fast" tier: `tier="fast"`
- Reduce candidates: `cross_encoder_threshold=20`
- Check performance: `get_performance_stats()`

### Memory issues?
- The implementation automatically limits memory usage
- Reduce `max_total_docs` if needed
- Monitor with built-in metrics

## Benefits You'll See

1. **Better Relevance**: Cross-encoder considers query-document relationships
2. **No Breaking Changes**: Graceful fallback if anything fails
3. **Performance Aware**: Automatically adjusts to your server capacity
4. **Easy Monitoring**: Built-in metrics and suggestions
5. **Incremental Adoption**: Start conservative, scale up gradually

The implementation is designed to enhance your retrieval quality while maintaining system stability and performance.