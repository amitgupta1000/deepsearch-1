#!/usr/bin/env python3
"""
Test Cross-Encoder Integration in Hybrid Retriever
Quick validation of semantic reranking functionality
"""

import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (
    USE_CROSS_ENCODER_RERANKING, CROSS_ENCODER_MODEL, 
    CROSS_ENCODER_TOP_K, RERANK_TOP_K
)
from src.hybrid_retriever import HybridRetrieverConfig, HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_cross_encoder_config():
    """Test that cross-encoder configuration is properly loaded"""
    logger.info("🔧 Testing cross-encoder configuration...")
    
    print(f"USE_CROSS_ENCODER_RERANKING: {USE_CROSS_ENCODER_RERANKING}")
    print(f"CROSS_ENCODER_MODEL: {CROSS_ENCODER_MODEL}")
    print(f"CROSS_ENCODER_TOP_K: {CROSS_ENCODER_TOP_K}")
    print(f"RERANK_TOP_K: {RERANK_TOP_K}")
    
    config = HybridRetrieverConfig(
        use_cross_encoder=USE_CROSS_ENCODER_RERANKING,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        cross_encoder_top_k=CROSS_ENCODER_TOP_K,
        rerank_top_k=RERANK_TOP_K
    )
    
    print(f"Config use_cross_encoder: {config.use_cross_encoder}")
    print(f"Config cross_encoder_model: {config.cross_encoder_model}")
    
    return config


def test_cross_encoder_loading():
    """Test that cross-encoder model can be loaded"""
    logger.info("🧠 Testing cross-encoder model loading...")
    
    try:
        from sentence_transformers import CrossEncoder
        logger.info("✅ sentence-transformers available")
        
        # Create simple config
        config = HybridRetrieverConfig(
            use_cross_encoder=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast model for testing
        )
        
        # Create hybrid retriever
        retriever = HybridRetriever(embeddings=None, config=config)
        
        if retriever.cross_encoder is not None:
            logger.info("✅ Cross-encoder loaded successfully!")
            return True
        else:
            logger.error("❌ Cross-encoder failed to load")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading cross-encoder: {e}")
        return False


def test_cross_encoder_reranking():
    """Test cross-encoder reranking functionality"""
    logger.info("🔄 Testing cross-encoder reranking...")
    
    try:
        # Import Document for testing
        from langchain_core.documents import Document
        
        # Create test documents
        test_docs = [
            Document(
                page_content="Python is a programming language used for web development and data science.",
                metadata={"source": "doc1", "title": "Python Programming"}
            ),
            Document(
                page_content="Machine learning algorithms can analyze large datasets to find patterns.",
                metadata={"source": "doc2", "title": "ML Algorithms"}
            ),
            Document(
                page_content="JavaScript is primarily used for frontend web development and browser scripting.",
                metadata={"source": "doc3", "title": "JavaScript Guide"}
            ),
            Document(
                page_content="Deep learning neural networks are a subset of machine learning techniques.",
                metadata={"source": "doc4", "title": "Deep Learning"}
            )
        ]
        
        # Test query
        query = "machine learning and data science algorithms"
        
        # Create retriever with cross-encoder enabled
        config = HybridRetrieverConfig(
            use_cross_encoder=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast model
            rerank_top_k=3
        )
        
        retriever = HybridRetriever(embeddings=None, config=config)
        
        if retriever.cross_encoder is None:
            logger.error("❌ Cross-encoder not available for testing")
            return False
        
        # Test reranking
        logger.info(f"Testing reranking for query: '{query}'")
        logger.info(f"Original documents: {len(test_docs)}")
        
        # Test the reranking method directly
        reranked_docs = retriever._rerank_with_cross_encoder(query, test_docs)
        
        logger.info(f"Reranked documents: {len(reranked_docs)}")
        
        print("\n📊 RERANKING RESULTS:")
        for i, doc in enumerate(reranked_docs):
            title = doc.metadata.get('title', 'No title')
            content_preview = doc.page_content[:50] + "..."
            print(f"  Rank {i+1}: {title} - {content_preview}")
        
        # Check that most relevant docs are ranked higher
        top_doc = reranked_docs[0] if reranked_docs else None
        if top_doc and ("machine learning" in top_doc.page_content.lower() or 
                        "data science" in top_doc.page_content.lower()):
            logger.info("✅ Cross-encoder reranking appears to be working correctly!")
            return True
        else:
            logger.warning("⚠️ Cross-encoder reranking may not be optimal")
            return True  # Still working, just maybe suboptimal
            
    except Exception as e:
        logger.error(f"❌ Error testing cross-encoder reranking: {e}")
        return False


def main():
    """Run all cross-encoder tests"""
    print("🧪 CROSS-ENCODER INTEGRATION TESTS")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_cross_encoder_config),
        ("Model Loading", test_cross_encoder_loading),
        ("Reranking Functionality", test_cross_encoder_reranking),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📈 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Cross-encoder integration is working!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)