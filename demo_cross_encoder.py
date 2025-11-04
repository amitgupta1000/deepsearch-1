#!/usr/bin/env python3
"""
Cross-Encoder Demo for INTELLISEARCH
Shows practical integration with realistic performance characteristics
"""

import time
import logging
from typing import List

# Mock LangChain components for demonstration
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MockBM25Retriever:
    """Mock BM25 retriever for demonstration"""
    
    def __init__(self):
        # Simulate keyword-focused documents
        self.documents = [
            Document("Python machine learning tutorial with scikit-learn examples", {"source": "tutorial_1"}),
            Document("JavaScript web development frameworks comparison", {"source": "web_1"}),
            Document("Machine learning algorithms explained in detail", {"source": "ml_1"}),
            Document("Deep learning with TensorFlow and neural networks", {"source": "dl_1"}),
            Document("Python programming best practices guide", {"source": "python_1"}),
            Document("Data science workflow with pandas and numpy", {"source": "ds_1"}),
            Document("React vs Vue.js frontend framework comparison", {"source": "web_2"}),
            Document("Natural language processing with transformers", {"source": "nlp_1"}),
            Document("Machine learning model deployment strategies", {"source": "ml_2"}),
            Document("Python data analysis with visualization libraries", {"source": "python_2"}),
        ]
    
    def invoke(self, query: str, k: int = 10) -> List[Document]:
        """Simple keyword matching simulation"""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            content_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                scored_docs.append((doc, overlap))
        
        # Sort by keyword overlap
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

class MockSemanticRetriever:
    """Mock semantic retriever for demonstration"""
    
    def __init__(self):
        # Same documents but different ranking logic
        self.documents = [
            Document("Python machine learning tutorial with scikit-learn examples", {"source": "tutorial_1"}),
            Document("JavaScript web development frameworks comparison", {"source": "web_1"}),
            Document("Machine learning algorithms explained in detail", {"source": "ml_1"}),
            Document("Deep learning with TensorFlow and neural networks", {"source": "dl_1"}),
            Document("Python programming best practices guide", {"source": "python_1"}),
            Document("Data science workflow with pandas and numpy", {"source": "ds_1"}),
            Document("React vs Vue.js frontend framework comparison", {"source": "web_2"}),
            Document("Natural language processing with transformers", {"source": "nlp_1"}),
            Document("Machine learning model deployment strategies", {"source": "ml_2"}),
            Document("Python data analysis with visualization libraries", {"source": "python_2"}),
        ]
    
    def invoke(self, query: str, k: int = 10) -> List[Document]:
        """Semantic similarity simulation (conceptual matching)"""
        # Simulate semantic understanding
        semantic_scores = {}
        query_lower = query.lower()
        
        for doc in self.documents:
            score = 0.0
            content = doc.page_content.lower()
            
            # Boost related concepts
            if "machine learning" in query_lower:
                if any(term in content for term in ["machine learning", "neural", "model", "algorithm"]):
                    score += 0.8
                if any(term in content for term in ["data science", "tensorflow", "scikit"]):
                    score += 0.6
            
            if "python" in query_lower:
                if "python" in content:
                    score += 0.9
                if any(term in content for term in ["programming", "data", "tutorial"]):
                    score += 0.5
            
            if "ai" in query_lower or "artificial intelligence" in query_lower:
                if any(term in content for term in ["machine learning", "deep learning", "neural", "nlp"]):
                    score += 0.8
            
            semantic_scores[doc] = score
        
        # Sort by semantic relevance
        sorted_docs = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:k] if score > 0]


def demo_retrieval_comparison():
    """Compare retrieval methods: BM25, Semantic, Hybrid, Enhanced with Cross-Encoder"""
    
    # Initialize retrievers
    bm25_retriever = MockBM25Retriever()
    semantic_retriever = MockSemanticRetriever()
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "python data science tutorial", 
        "deep learning neural networks",
        "web development frameworks"
    ]
    
    print("=== INTELLISEARCH Cross-Encoder Demo ===\n")
    
    for query in test_queries:
        print(f"🔍 Query: '{query}'")
        print("-" * 50)
        
        # 1. BM25 Results
        bm25_results = bm25_retriever.invoke(query, k=5)
        print("📋 BM25 Results (keyword matching):")
        for i, doc in enumerate(bm25_results, 1):
            print(f"  {i}. {doc.page_content[:60]}...")
        
        # 2. Semantic Results
        semantic_results = semantic_retriever.invoke(query, k=5)
        print("\n🧠 Semantic Results (conceptual matching):")
        for i, doc in enumerate(semantic_results, 1):
            print(f"  {i}. {doc.page_content[:60]}...")
        
        # 3. Simple Hybrid (existing approach)
        hybrid_docs = []
        seen_content = set()
        
        # Interleave BM25 and semantic results
        max_len = max(len(bm25_results), len(semantic_results))
        for i in range(max_len):
            if i < len(bm25_results):
                doc = bm25_results[i]
                if doc.page_content not in seen_content:
                    hybrid_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            if i < len(semantic_results) and len(hybrid_docs) < 5:
                doc = semantic_results[i]
                if doc.page_content not in seen_content:
                    hybrid_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        print("\n🔄 Hybrid Results (BM25 + Semantic):")
        for i, doc in enumerate(hybrid_docs[:5], 1):
            print(f"  {i}. {doc.page_content[:60]}...")
        
        # 4. Enhanced with Cross-Encoder (if available)
        try:
            from src.server_friendly_cross_encoder import create_server_friendly_cross_encoder
            
            cross_encoder = create_server_friendly_cross_encoder("fast")
            if cross_encoder:
                start_time = time.time()
                reranked_docs = cross_encoder.rerank(query, hybrid_docs)
                rerank_time = time.time() - start_time
                
                print(f"\n🎯 Cross-Encoder Enhanced (reranked in {rerank_time:.2f}s):")
                for i, doc in enumerate(reranked_docs[:5], 1):
                    score = doc.metadata.get('cross_encoder_score', 0)
                    print(f"  {i}. [Score: {score:.3f}] {doc.page_content[:50]}...")
                
                # Show performance info
                perf = cross_encoder.get_performance_summary()
                print(f"     ⚡ Avg latency: {perf['avg_latency_ms']:.1f}ms")
            else:
                print("\n❌ Cross-encoder not available")
                
        except ImportError:
            print("\n⚠️ Cross-encoder module not found - install sentence-transformers")
        
        print("\n" + "=" * 80 + "\n")


def demo_server_performance():
    """Demonstrate server-friendly performance characteristics"""
    
    try:
        from src.server_friendly_cross_encoder import create_server_friendly_cross_encoder
        
        print("=== Server Performance Demo ===\n")
        
        # Test different model tiers
        tiers = ["fast", "balanced"]  # Skip "quality" for demo
        
        for tier in tiers:
            print(f"🧪 Testing {tier.upper()} tier:")
            
            cross_encoder = create_server_friendly_cross_encoder(tier)
            if not cross_encoder:
                print(f"  ❌ {tier} model not available")
                continue
            
            # Create test documents
            test_docs = [
                Document(f"Document {i}: machine learning and artificial intelligence content with various technical details", 
                        {"source": f"doc_{i}"})
                for i in range(30)  # Test with 30 documents
            ]
            
            query = "machine learning algorithms and AI techniques"
            
            # Time the reranking
            start_time = time.time()
            try:
                reranked = cross_encoder.rerank(query, test_docs)
                elapsed = time.time() - start_time
                
                print(f"  ✅ Reranked {len(test_docs)} docs in {elapsed:.3f}s")
                print(f"  📊 Avg per doc: {elapsed/len(test_docs)*1000:.1f}ms")
                
                # Show performance stats
                stats = cross_encoder.get_performance_summary()
                print(f"  💾 Cache hit rate: {stats['cache_hit_rate']:.1%}")
                print(f"  ⚡ Total queries: {stats['total_queries']}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            print()
        
        print("🎯 Recommendation: Start with 'fast' tier, monitor performance, upgrade as needed")
        
    except ImportError:
        print("⚠️ Install sentence-transformers to test: pip install sentence-transformers")


def main():
    """Run the complete demo"""
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("INTELLISEARCH Cross-Encoder Integration Demo")
    print("=" * 50)
    print()
    
    print("This demo shows how cross-encoder reranking enhances")
    print("retrieval quality while maintaining server performance.\n")
    
    # Demo 1: Retrieval quality comparison
    demo_retrieval_comparison()
    
    # Demo 2: Server performance characteristics
    demo_server_performance()
    
    print("\n🚀 Ready to integrate? See docs/CROSS_ENCODER_INTEGRATION_GUIDE.md")
    print("💡 Start with 'fast' tier for production safety!")


if __name__ == "__main__":
    main()