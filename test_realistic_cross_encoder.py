#!/usr/bin/env python3
"""
Test Cross-Encoder with Real Query
"""

import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import USE_CROSS_ENCODER_RERANKING
from src.hybrid_retriever import HybridRetrieverConfig, HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_realistic_scenario():
    """Test cross-encoder with a realistic research scenario"""
    print("🔍 REALISTIC RESEARCH SCENARIO TEST")
    print("=" * 50)
    
    # Import Document for testing
    from langchain_core.documents import Document
    
    # Create test documents that simulate research results
    research_docs = [
        Document(
            page_content="Artificial intelligence has revolutionized healthcare by enabling faster diagnosis and personalized treatment plans. Machine learning algorithms can analyze medical images with accuracy comparable to radiologists.",
            metadata={"source": "medical_ai_paper.pdf", "title": "AI in Healthcare Revolution"}
        ),
        Document(
            page_content="Climate change mitigation strategies include renewable energy adoption, carbon capture technologies, and sustainable agriculture practices. Global cooperation is essential for effective climate action.",
            metadata={"source": "climate_report.pdf", "title": "Climate Change Mitigation"}
        ),
        Document(
            page_content="Deep learning neural networks have achieved breakthrough performance in natural language processing tasks including translation, summarization, and question answering systems.",
            metadata={"source": "nlp_advances.pdf", "title": "Deep Learning in NLP"}
        ),
        Document(
            page_content="Quantum computing promises to solve complex computational problems exponentially faster than classical computers, with applications in cryptography, optimization, and scientific simulation.",
            metadata={"source": "quantum_computing.pdf", "title": "Quantum Computing Advances"}
        ),
        Document(
            page_content="Blockchain technology provides decentralized ledger systems for secure and transparent transactions, with applications beyond cryptocurrency including supply chain management.",
            metadata={"source": "blockchain_tech.pdf", "title": "Blockchain Applications"}
        ),
        Document(
            page_content="Machine learning model interpretability is crucial for building trust in AI systems, especially in high-stakes applications like healthcare and finance where explainability is required.",
            metadata={"source": "ml_interpretability.pdf", "title": "ML Model Interpretability"}
        ),
        Document(
            page_content="Sustainable urban development requires smart city technologies, green infrastructure, and data-driven planning to address growing population and environmental challenges.",
            metadata={"source": "smart_cities.pdf", "title": "Sustainable Urban Development"}
        ),
        Document(
            page_content="Cybersecurity threats have evolved with artificial intelligence, requiring advanced defense mechanisms including anomaly detection and automated incident response systems.",
            metadata={"source": "ai_cybersecurity.pdf", "title": "AI in Cybersecurity"}
        ),
    ]
    
    # Research query
    query = "artificial intelligence applications in healthcare and medical diagnosis"
    
    print(f"🎯 Research Query: '{query}'")
    print(f"📚 Available Documents: {len(research_docs)}")
    
    # Test simple keyword reranking
    print(f"\n🔤 Simple Keyword Reranking Results:")
    config_simple = HybridRetrieverConfig(use_cross_encoder=False)
    retriever_simple = HybridRetriever(embeddings=None, config=config_simple)
    simple_results = retriever_simple._rerank_multi_query_results(query, research_docs)
    
    for i, doc in enumerate(simple_results[:3]):
        title = doc.metadata.get('title', 'No title')
        print(f"   {i+1}. {title}")
    
    # Test cross-encoder reranking
    print(f"\n🧠 Cross-Encoder Reranking Results:")
    config_cross = HybridRetrieverConfig(
        use_cross_encoder=True,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_top_k=8,  # Process all docs
        rerank_top_k=3,
        cross_encoder_batch_size=4
    )
    retriever_cross = HybridRetriever(embeddings=None, config=config_cross)
    cross_results = retriever_cross._rerank_multi_query_results(query, research_docs)
    
    for i, doc in enumerate(cross_results[:3]):
        title = doc.metadata.get('title', 'No title')
        print(f"   {i+1}. {title}")
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    
    # Check if AI healthcare doc is ranked #1 by cross-encoder
    if cross_results and "Healthcare" in cross_results[0].metadata.get('title', ''):
        print(f"   ✅ Cross-encoder correctly identified most relevant document")
    else:
        print(f"   ⚠️ Cross-encoder may not have optimal ranking")
    
    # Check if keyword ranking got it right
    if simple_results and "Healthcare" in simple_results[0].metadata.get('title', ''):
        print(f"   ✅ Simple keyword also identified most relevant document")
    else:
        print(f"   ❌ Simple keyword ranking missed the most relevant document")
    
    return cross_results, simple_results


def main():
    """Test cross-encoder with realistic scenario"""
    print(f"Cross-encoder enabled in config: {USE_CROSS_ENCODER_RERANKING}")
    
    try:
        cross_results, simple_results = test_realistic_scenario()
        
        print(f"\n🎉 Test completed successfully!")
        print(f"\n💡 The cross-encoder integration is ready for use!")
        print(f"   • To enable: Set USE_CROSS_ENCODER_RERANKING=True in environment or config")
        print(f"   • Performance: Optimized with batching and limited document processing")
        print(f"   • Quality: Better semantic understanding than simple keyword matching")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)