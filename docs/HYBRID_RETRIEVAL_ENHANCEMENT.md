# Hybrid Retrieval Enhancement

## Overview

INTELLISEARCH now features a **Hybrid Retrieval System** that combines the strengths of both sparse (BM25) and dense (vector) search methods for significantly improved document retrieval quality.

## Why Hybrid Retrieval?

### Traditional Approaches Limitations:
- **Vector Search Only**: Great for semantic similarity but may miss exact keyword matches
- **BM25/Keyword Search Only**: Excellent for exact matches but misses conceptual relationships

### Hybrid Approach Benefits:
- **Best of Both Worlds**: Combines exact keyword matching with semantic understanding
- **Improved Relevance**: Better ranking of truly relevant documents
- **Query Flexibility**: Handles both specific technical queries and broad conceptual questions
- **Reduced Noise**: Ensemble methods filter out less relevant results

## Technical Implementation

### Components:
1. **BM25 Retriever**: Sparse retrieval for keyword-based matching
2. **Vector Store (FAISS)**: Dense retrieval using Google embeddings
3. **Ensemble Fusion**: Multiple fusion strategies for combining results

### Fusion Methods:

#### 1. Reciprocal Rank Fusion (RRF) - Default
```
score = Σ(weight / (k + rank))
```
- **Pros**: Robust, handles different score scales well
- **Use Case**: General-purpose retrieval

#### 2. Weighted Fusion
```
score = (vector_weight × vector_score) + (bm25_weight × bm25_score)
```
- **Pros**: Direct score combination, tunable weights
- **Use Case**: When you want explicit control over method importance

## Configuration

### Environment Variables (.env file):
```bash
# Enable/disable hybrid retrieval
USE_HYBRID_RETRIEVAL=true

# Fusion method: "rrf" or "weighted"
HYBRID_FUSION_METHOD=rrf

# Weights for fusion (should sum to 1.0)
HYBRID_VECTOR_WEIGHT=0.6
HYBRID_BM25_WEIGHT=0.4

# RRF parameter (higher = less emphasis on rank)
HYBRID_RRF_K=60

# Retrieval quality settings
RETRIEVAL_TOP_K=20
VECTOR_SCORE_THRESHOLD=0.1
MIN_CHUNK_LENGTH=50
MIN_WORD_COUNT=10
```

### Default Configuration:
- **Vector Weight**: 0.6 (slightly favors semantic search)
- **BM25 Weight**: 0.4 (ensures keyword matching)
- **Fusion Method**: RRF (more robust)
- **Top K**: 20 chunks retrieved
- **Quality Filters**: Minimum 50 characters, 10 words per chunk

## Performance Comparison

### Before (Vector Only):
```
Query: "Tesla quarterly earnings Q3 2025"
Results: Broad articles about Tesla, some irrelevant quarterly reports from other companies
```

### After (Hybrid):
```
Query: "Tesla quarterly earnings Q3 2025" 
Results: Tesla Q3 2025 specific earnings, Tesla quarterly comparisons, relevant financial metrics
```

### Example Improvements:

1. **Technical Queries**: 
   - Query: "BERT transformer architecture attention mechanism"
   - Improvement: Finds both exact technical papers AND conceptually related content

2. **Financial Research**:
   - Query: "Apple iPhone revenue impact Q4 2024"
   - Improvement: Combines specific quarterly mentions with semantic revenue analysis

3. **Legal/Regulatory**:
   - Query: "GDPR compliance requirements data protection"
   - Improvement: Exact regulation text + related compliance guidance

## Implementation Details

### Automatic Fallbacks:
1. **Hybrid → Vector Only**: If BM25 fails
2. **Hybrid → BM25 Only**: If vector search fails  
3. **Hybrid → Simple Text Search**: If both specialized methods fail

### Quality Assurance:
- Chunk deduplication using source + index keys
- Minimum quality thresholds for retrieved content
- Configurable score thresholds to filter poor matches

### Performance Optimizations:
- Lazy loading of retrieval components
- Caching of document indices
- Parallel processing where possible

## Usage Examples

### In Code:
```python
from hybrid_retriever import create_hybrid_retriever

# Create retriever with custom config
retriever = create_hybrid_retriever(
    embeddings=embeddings,
    vector_weight=0.7,  # Favor vector search more
    bm25_weight=0.3,
    fusion_method="rrf",
    top_k=15
)

# Build indices from scraped content
retriever.build_index(relevant_contexts)

# Retrieve relevant chunks
results = retriever.retrieve("your search query")
```

### Configuration Tuning:

**For Technical Queries** (favor exact matches):
```bash
HYBRID_VECTOR_WEIGHT=0.4
HYBRID_BM25_WEIGHT=0.6
```

**For Conceptual Queries** (favor semantic similarity):
```bash
HYBRID_VECTOR_WEIGHT=0.8
HYBRID_BM25_WEIGHT=0.2
```

**For Balanced Research**:
```bash
HYBRID_VECTOR_WEIGHT=0.6
HYBRID_BM25_WEIGHT=0.4
```

## Monitoring and Debugging

### Logs to Monitor:
```
INFO - Using hybrid retriever (BM25 + Vector Search)
INFO - Hybrid retrieval stats: {...}
INFO - Retrieved 18 chunks using hybrid approach
```

### Statistics Available:
- Total documents indexed
- Active retrieval methods
- Fusion method used
- Retrieval success metrics

## Future Enhancements

### Planned Features:
1. **Smart Query Routing**: Automatic method selection based on query type
2. **Reranking**: Post-retrieval reranking using cross-encoders
3. **Query Expansion**: Automatic query expansion for better recall
4. **Adaptive Weights**: Dynamic weight adjustment based on query performance

### Advanced Configurations (Future):
```bash
# Query type detection
AUTO_QUERY_ROUTING=true

# Advanced reranking
USE_CROSS_ENCODER_RERANKING=true
RERANKER_MODEL=ms-marco-MiniLM-L-6-v2

# Query expansion
USE_QUERY_EXPANSION=true
EXPANSION_TERMS=3
```

## Troubleshooting

### Common Issues:

1. **Import Errors**:
   ```
   Solution: Ensure rank-bm25 and langchain-community are installed
   pip install rank-bm25 langchain-community
   ```

2. **Low Quality Results**:
   ```
   Solution: Adjust MIN_CHUNK_LENGTH and MIN_WORD_COUNT
   Check VECTOR_SCORE_THRESHOLD (lower = more permissive)
   ```

3. **Performance Issues**:
   ```
   Solution: Reduce RETRIEVAL_TOP_K
   Enable caching: CACHE_ENABLED=true
   ```

### Fallback Behavior:
- System automatically falls back to vector-only search if hybrid fails
- Logs indicate which retrieval method is actually used
- No breaking changes - existing functionality preserved

## Conclusion

The hybrid retrieval system significantly improves INTELLISEARCH's ability to find relevant information by combining the precision of keyword matching with the flexibility of semantic search. This results in more accurate, comprehensive, and useful research reports.

The system is designed to be:
- **Backwards Compatible**: Works with existing configurations
- **Configurable**: Extensive customization options
- **Robust**: Multiple fallback mechanisms
- **Performant**: Optimized for production use

Enable it today by setting `USE_HYBRID_RETRIEVAL=true` in your environment!