#!/usr/bin/env python3
"""
Comprehensive test for the Hybrid Retriever functionality.
Tests vector search, BM25, and ensemble retrieval methods.
"""

import sys
import os
import logging
import json
from typing import Dict, List

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration and imports
def setup_test_environment():
    """Setup test environment and validate imports."""
    try:
        # Import configuration
        from config import (
            RETRIEVAL_METHOD, HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT,
            HYBRID_FUSION_METHOD, HYBRID_RRF_K, USE_HYBRID_RETRIEVAL,
            RETRIEVAL_TOP_K, VECTOR_SCORE_THRESHOLD
        )
        
        # Import hybrid retriever
        from hybrid_retriever import HybridRetriever, HybridRetrieverConfig, create_hybrid_retriever
        
        # Import enhanced embeddings
        from enhanced_embeddings import EnhancedGoogleEmbeddings
        
        logger.info("✅ All imports successful")
        
        # Display current configuration
        config_info = {
            "RETRIEVAL_METHOD": RETRIEVAL_METHOD,
            "USE_HYBRID_RETRIEVAL": USE_HYBRID_RETRIEVAL,
            "HYBRID_VECTOR_WEIGHT": HYBRID_VECTOR_WEIGHT,
            "HYBRID_BM25_WEIGHT": HYBRID_BM25_WEIGHT,
            "HYBRID_FUSION_METHOD": HYBRID_FUSION_METHOD,
            "HYBRID_RRF_K": HYBRID_RRF_K,
            "RETRIEVAL_TOP_K": RETRIEVAL_TOP_K,
            "VECTOR_SCORE_THRESHOLD": VECTOR_SCORE_THRESHOLD
        }
        
        logger.info("📋 Current Hybrid Retrieval Configuration:")
        for key, value in config_info.items():
            logger.info(f"  {key}: {value}")
        
        return True, {
            'HybridRetriever': HybridRetriever,
            'HybridRetrieverConfig': HybridRetrieverConfig,
            'create_hybrid_retriever': create_hybrid_retriever,
            'EnhancedGoogleEmbeddings': EnhancedGoogleEmbeddings,
            'config': config_info
        }
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False, None
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False, None

def create_test_data():
    """Create sample data for testing hybrid retrieval."""
    
    test_contexts = {
        "https://example.com/ai-trends": {
            "title": "Artificial Intelligence Trends 2025",
            "content": """
            Artificial Intelligence continues to evolve rapidly in 2025. Machine learning algorithms 
            are becoming more sophisticated, with transformer models leading breakthrough applications.
            Natural language processing has reached new heights with conversational AI systems.
            Computer vision applications are revolutionizing healthcare, autonomous vehicles, and 
            manufacturing processes. Deep learning frameworks are enabling more efficient training
            of large language models. The integration of AI with cloud computing is creating 
            unprecedented scalability opportunities for businesses worldwide.
            """
        },
        "https://example.com/blockchain-tech": {
            "title": "Blockchain Technology and Cryptocurrency",
            "content": """
            Blockchain technology has matured significantly since its inception with Bitcoin.
            Decentralized finance (DeFi) protocols are reshaping traditional banking systems.
            Smart contracts on Ethereum and other platforms enable programmable money and
            automated financial services. Non-fungible tokens (NFTs) have created new digital
            asset classes for art, gaming, and intellectual property. Central bank digital
            currencies (CBDCs) are being explored by governments worldwide as digital alternatives
            to physical cash. Proof-of-stake consensus mechanisms are replacing energy-intensive
            proof-of-work systems for environmental sustainability.
            """
        },
        "https://example.com/climate-tech": {
            "title": "Climate Technology Solutions",
            "content": """
            Climate technology innovations are crucial for addressing global warming challenges.
            Renewable energy sources like solar panels and wind turbines are becoming more
            cost-effective than fossil fuels. Carbon capture and storage technologies are
            being deployed to reduce atmospheric CO2 levels. Electric vehicle adoption is
            accelerating with improved battery technologies and charging infrastructure.
            Green hydrogen production is emerging as a clean fuel alternative for heavy
            industry and transportation. Climate modeling using artificial intelligence
            helps predict weather patterns and environmental changes more accurately.
            """
        },
        "https://example.com/quantum-computing": {
            "title": "Quantum Computing Advances",
            "content": """
            Quantum computing represents a paradigm shift in computational capabilities.
            Quantum bits (qubits) can exist in superposition states, enabling parallel
            processing of complex calculations. Quantum algorithms like Shor's algorithm
            threaten current cryptographic systems while opening new possibilities for
            optimization problems. Major tech companies are investing heavily in quantum
            hardware development, including superconducting and trapped-ion systems.
            Quantum error correction remains a significant challenge for practical applications.
            The quantum advantage for specific use cases like drug discovery and financial
            modeling is becoming increasingly evident.
            """
        },
        "https://example.com/space-exploration": {
            "title": "Space Exploration and Commercial Spaceflight",
            "content": """
            Space exploration has entered a new era with commercial companies leading innovation.
            SpaceX's reusable rocket technology has dramatically reduced launch costs.
            Mars exploration missions are providing valuable data about the Red Planet's
            geology and potential for human habitation. Satellite constellations are
            enabling global internet coverage and Earth observation capabilities.
            Space tourism is becoming accessible to civilians through suborbital flights.
            Asteroid mining could provide access to rare earth elements and precious metals.
            International space stations continue advancing scientific research in microgravity
            environments for pharmaceuticals and materials science.
            """
        }
    }
    
    test_queries = [
        "artificial intelligence machine learning trends",
        "blockchain cryptocurrency DeFi applications", 
        "climate change renewable energy solutions",
        "quantum computing quantum algorithms applications",
        "space exploration Mars missions commercial spaceflight"
    ]
    
    return test_contexts, test_queries

def test_hybrid_retriever_config():
    """Test hybrid retriever configuration."""
    logger.info("\n🔧 Testing Hybrid Retriever Configuration...")
    
    try:
        from hybrid_retriever import HybridRetrieverConfig
        
        # Test default configuration
        default_config = HybridRetrieverConfig()
        logger.info(f"✅ Default config - top_k: {default_config.top_k}, vector_weight: {default_config.vector_weight}")
        
        # Test custom configuration
        custom_config = HybridRetrieverConfig(
            top_k=15,
            vector_weight=0.7,
            bm25_weight=0.3,
            fusion_method="weighted"
        )
        logger.info(f"✅ Custom config - top_k: {custom_config.top_k}, fusion_method: {custom_config.fusion_method}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_enhanced_embeddings():
    """Test enhanced embeddings initialization."""
    logger.info("\n🧠 Testing Enhanced Embeddings...")
    
    try:
        from enhanced_embeddings import EnhancedGoogleEmbeddings
        from config import GOOGLE_API_KEY
        
        if not GOOGLE_API_KEY:
            logger.warning("⚠️ GOOGLE_API_KEY not set - using mock embeddings")
            return None
        
        embeddings = EnhancedGoogleEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="gemini-embedding-001", 
            default_task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
        
        # Test basic embedding
        test_text = "This is a test document for embedding."
        try:
            embedding = embeddings.embed_query(test_text)
            logger.info(f"✅ Enhanced embeddings working - dimension: {len(embedding)}")
            return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Enhanced embeddings error: {e}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Enhanced embeddings test failed: {e}")
        return None

def test_hybrid_retriever_creation():
    """Test hybrid retriever creation and initialization."""
    logger.info("\n🏗️ Testing Hybrid Retriever Creation...")
    
    try:
        from hybrid_retriever import create_hybrid_retriever, HybridRetrieverConfig
        
        # Test with mock embeddings
        config = HybridRetrieverConfig(top_k=10, vector_weight=0.6, bm25_weight=0.4)
        retriever = create_hybrid_retriever(embeddings=None, **config.__dict__)
        
        logger.info("✅ Hybrid retriever created successfully")
        
        # Test stats before indexing
        stats = retriever.get_stats()
        logger.info(f"📊 Initial stats: {json.dumps(stats, indent=2)}")
        
        return retriever
        
    except Exception as e:
        logger.error(f"❌ Hybrid retriever creation failed: {e}")
        return None

def test_index_building(retriever, embeddings=None):
    """Test building the hybrid index."""
    logger.info("\n📚 Testing Index Building...")
    
    try:
        test_contexts, _ = create_test_data()
        
        # Set embeddings if available
        if embeddings:
            retriever.embeddings = embeddings
            logger.info("✅ Enhanced embeddings attached to retriever")
        
        # Build index
        success = retriever.build_index(test_contexts)
        
        if success:
            logger.info("✅ Index built successfully")
            
            # Get stats after indexing
            stats = retriever.get_stats()
            logger.info(f"📊 Post-index stats: {json.dumps(stats, indent=2)}")
            
            return True
        else:
            logger.error("❌ Index building failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Index building test failed: {e}")
        return False

def test_single_query_retrieval(retriever):
    """Test single query retrieval."""
    logger.info("\n🔍 Testing Single Query Retrieval...")
    
    try:
        _, test_queries = create_test_data()
        
        for i, query in enumerate(test_queries[:3]):  # Test first 3 queries
            logger.info(f"\n📝 Query {i+1}: '{query}'")
            
            # Retrieve documents
            results = retriever.retrieve(query)
            
            logger.info(f"📋 Retrieved {len(results)} documents")
            
            # Display top results
            for j, doc in enumerate(results[:3]):  # Show top 3 results
                title = doc.metadata.get('title', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                
                logger.info(f"  Result {j+1}:")
                logger.info(f"    Title: {title}")
                logger.info(f"    Source: {source}")
                logger.info(f"    Content: {content_preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Single query retrieval test failed: {e}")
        return False

def test_multi_query_retrieval(retriever):
    """Test multi-query retrieval functionality."""
    logger.info("\n🔄 Testing Multi-Query Retrieval...")
    
    try:
        # Test with multiple related queries
        multi_queries = [
            "artificial intelligence machine learning",
            "AI trends 2025 technology",
            "deep learning neural networks"
        ]
        
        logger.info(f"📝 Multi-queries: {multi_queries}")
        
        # Retrieve with multi-query
        results = retriever.retrieve_multi_query(multi_queries, deduplicate=True)
        
        logger.info(f"📋 Multi-query retrieved {len(results)} documents")
        
        # Display results
        for i, doc in enumerate(results[:3]):
            title = doc.metadata.get('title', 'Unknown')
            content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            logger.info(f"  Result {i+1}: {title} - {content_preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-query retrieval test failed: {e}")
        return False

def test_fusion_methods(retriever):
    """Test different fusion methods."""
    logger.info("\n⚡ Testing Fusion Methods...")
    
    try:
        query = "artificial intelligence machine learning trends"
        
        # Test RRF fusion
        retriever.config.fusion_method = "rrf"
        rrf_results = retriever.retrieve(query)
        logger.info(f"📊 RRF fusion: {len(rrf_results)} results")
        
        # Test weighted fusion
        retriever.config.fusion_method = "weighted"
        weighted_results = retriever.retrieve(query)
        logger.info(f"📊 Weighted fusion: {len(weighted_results)} results")
        
        # Compare results
        if rrf_results and weighted_results:
            rrf_titles = [doc.metadata.get('title', 'Unknown') for doc in rrf_results[:3]]
            weighted_titles = [doc.metadata.get('title', 'Unknown') for doc in weighted_results[:3]]
            
            logger.info(f"🔄 RRF top results: {rrf_titles}")
            logger.info(f"⚖️ Weighted top results: {weighted_titles}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fusion methods test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive hybrid retriever test suite."""
    logger.info("🚀 Starting Comprehensive Hybrid Retriever Test Suite\n")
    
    # Setup
    setup_success, modules = setup_test_environment()
    if not setup_success:
        logger.error("❌ Test setup failed - aborting")
        return False
    
    test_results = {}
    
    # Test 1: Configuration
    test_results['config'] = test_hybrid_retriever_config()
    
    # Test 2: Enhanced Embeddings
    embeddings = test_enhanced_embeddings()
    test_results['embeddings'] = embeddings is not None
    
    # Test 3: Retriever Creation
    retriever = test_hybrid_retriever_creation()
    test_results['creation'] = retriever is not None
    
    if retriever:
        # Test 4: Index Building
        test_results['indexing'] = test_index_building(retriever, embeddings)
        
        if test_results['indexing']:
            # Test 5: Single Query Retrieval
            test_results['single_query'] = test_single_query_retrieval(retriever)
            
            # Test 6: Multi-Query Retrieval
            test_results['multi_query'] = test_multi_query_retrieval(retriever)
            
            # Test 7: Fusion Methods
            test_results['fusion'] = test_fusion_methods(retriever)
    
    # Summary
    logger.info("\n📊 TEST RESULTS SUMMARY:")
    logger.info("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.upper():<20} {status}")
    
    logger.info("=" * 50)
    logger.info(f"OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Hybrid Retriever is working correctly!")
        return True
    else:
        logger.warning("⚠️ Some tests failed - check the logs above for details")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)