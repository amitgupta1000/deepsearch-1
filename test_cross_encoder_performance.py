#!/usr/bin/env python3
"""
Performance Comparison: Simple Keyword vs Cross-Encoder Reranking
"""

import logging
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.hybrid_retriever import HybridRetrieverConfig, HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_documents():
    """Create a larger set of test documents"""
    from langchain_core.documents import Document
    
    documents = [
        Document(page_content="Python is a programming language used for web development and data science.", metadata={"source": "doc1", "title": "Python Programming"}),
        Document(page_content="Machine learning algorithms can analyze large datasets to find patterns.", metadata={"source": "doc2", "title": "ML Algorithms"}),
        Document(page_content="JavaScript is primarily used for frontend web development and browser scripting.", metadata={"source": "doc3", "title": "JavaScript Guide"}),
        Document(page_content="Deep learning neural networks are a subset of machine learning techniques.", metadata={"source": "doc4", "title": "Deep Learning"}),
        Document(page_content="Data science involves extracting insights from structured and unstructured data.", metadata={"source": "doc5", "title": "Data Science"}),
        Document(page_content="Natural language processing enables computers to understand human language.", metadata={"source": "doc6", "title": "NLP Overview"}),
        Document(page_content="Computer vision algorithms can identify objects and patterns in images.", metadata={"source": "doc7", "title": "Computer Vision"}),
        Document(page_content="Statistical analysis is fundamental to data science and machine learning.", metadata={"source": "doc8", "title": "Statistics"}),
        Document(page_content="Neural networks are inspired by biological brain structures and functions.", metadata={"source": "doc9", "title": "Neural Networks"}),
        Document(page_content="Big data technologies handle massive volumes of information efficiently.", metadata={"source": "doc10", "title": "Big Data"}),
        Document(page_content="Artificial intelligence aims to create systems that can think and learn.", metadata={"source": "doc11", "title": "AI Overview"}),
        Document(page_content="Supervised learning uses labeled data to train predictive models.", metadata={"source": "doc12", "title": "Supervised Learning"}),
        Document(page_content="Unsupervised learning finds patterns in data without labeled examples.", metadata={"source": "doc13", "title": "Unsupervised Learning"}),
        Document(page_content="Reinforcement learning teaches agents through rewards and penalties.", metadata={"source": "doc14", "title": "Reinforcement Learning"}),
        Document(page_content="Feature engineering involves selecting and transforming data attributes.", metadata={"source": "doc15", "title": "Feature Engineering"}),
    ]
    
    return documents


def benchmark_reranking_methods():
    """Compare performance of simple vs cross-encoder reranking"""
    print("🏃 PERFORMANCE COMPARISON: Simple vs Cross-Encoder Reranking")
    print("=" * 60)
    
    documents = create_test_documents()
    query = "machine learning algorithms for data science analysis"
    
    print(f"📊 Test Setup:")
    print(f"   Documents: {len(documents)}")
    print(f"   Query: '{query}'")
    print(f"   Iterations: 5 (for averaging)")
    
    # Test 1: Simple keyword reranking
    print(f"\n🔤 Testing Simple Keyword Reranking...")
    config_simple = HybridRetrieverConfig(use_cross_encoder=False)
    retriever_simple = HybridRetriever(embeddings=None, config=config_simple)
    
    simple_times = []
    for i in range(5):
        start_time = time.time()
        simple_results = retriever_simple._rerank_multi_query_results(query, documents)
        end_time = time.time()
        simple_times.append(end_time - start_time)
    
    simple_avg_time = sum(simple_times) / len(simple_times)
    print(f"   ⏱️ Average time: {simple_avg_time:.4f}s")
    print(f"   📋 Top 3 results:")
    for i, doc in enumerate(simple_results[:3]):
        title = doc.metadata.get('title', 'No title')
        print(f"      {i+1}. {title}")
    
    # Test 2: Cross-encoder reranking
    print(f"\n🧠 Testing Cross-Encoder Reranking...")
    config_cross = HybridRetrieverConfig(
        use_cross_encoder=True,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_k=15
    )
    retriever_cross = HybridRetriever(embeddings=None, config=config_cross)
    
    # First run might be slower due to model loading
    print("   Loading cross-encoder model...")
    _ = retriever_cross._rerank_multi_query_results(query, documents)
    
    cross_times = []
    for i in range(5):
        start_time = time.time()
        cross_results = retriever_cross._rerank_multi_query_results(query, documents)
        end_time = time.time()
        cross_times.append(end_time - start_time)
    
    cross_avg_time = sum(cross_times) / len(cross_times)
    print(f"   ⏱️ Average time: {cross_avg_time:.4f}s")
    print(f"   📋 Top 3 results:")
    for i, doc in enumerate(cross_results[:3]):
        title = doc.metadata.get('title', 'No title')
        print(f"      {i+1}. {title}")
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Simple Reranking:     {simple_avg_time:.4f}s")
    print(f"   Cross-Encoder:        {cross_avg_time:.4f}s")
    
    slowdown = cross_avg_time / simple_avg_time
    print(f"   Slowdown Factor:      {slowdown:.2f}x")
    
    if slowdown < 2:
        print(f"   ✅ Acceptable performance impact (< 2x slower)")
    elif slowdown < 5:
        print(f"   ⚠️ Moderate performance impact (2-5x slower)")
    else:
        print(f"   ❌ Significant performance impact (> 5x slower)")
    
    # Quality analysis
    print(f"\n🎯 QUALITY ANALYSIS:")
    
    # Check if ML/data science docs are ranked higher with cross-encoder
    simple_top3_titles = [doc.metadata.get('title', '') for doc in simple_results[:3]]
    cross_top3_titles = [doc.metadata.get('title', '') for doc in cross_results[:3]]
    
    ml_keywords = ['ML', 'Machine', 'Data Science', 'Deep Learning', 'Statistics']
    
    simple_ml_count = sum(1 for title in simple_top3_titles 
                         if any(keyword in title for keyword in ml_keywords))
    cross_ml_count = sum(1 for title in cross_top3_titles 
                        if any(keyword in title for keyword in ml_keywords))
    
    print(f"   Simple - ML-related in top 3:    {simple_ml_count}/3")
    print(f"   Cross-Encoder - ML-related in top 3: {cross_ml_count}/3")
    
    if cross_ml_count > simple_ml_count:
        print(f"   ✅ Cross-encoder shows better semantic understanding")
    elif cross_ml_count == simple_ml_count:
        print(f"   ➖ Both methods show similar relevance")
    else:
        print(f"   ❓ Simple method performed better (unexpected)")
    
    return {
        'simple_time': simple_avg_time,
        'cross_time': cross_avg_time,
        'slowdown': slowdown,
        'simple_ml_count': simple_ml_count,
        'cross_ml_count': cross_ml_count
    }


def main():
    """Run performance comparison"""
    try:
        results = benchmark_reranking_methods()
        
        print(f"\n🏆 RECOMMENDATION:")
        if results['slowdown'] < 3 and results['cross_ml_count'] >= results['simple_ml_count']:
            print(f"   ✅ Enable cross-encoder reranking - good balance of performance and quality")
        elif results['slowdown'] < 2:
            print(f"   ✅ Enable cross-encoder reranking - minimal performance impact")
        elif results['cross_ml_count'] > results['simple_ml_count']:
            print(f"   ⚠️ Consider enabling cross-encoder - better quality but slower")
        else:
            print(f"   ❓ Evaluate based on your use case - performance vs quality tradeoff")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during benchmarking: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)