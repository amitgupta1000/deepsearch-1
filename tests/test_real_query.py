#!/usr/bin/env python3
"""
Real query test for embed_and_retrieve and create_qa_pairs functions
Tests with actual content and API calls
"""
import asyncio
import logging
import sys
import time

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Add current directory to path
sys.path.append('.')

async def test_full_embedding_retrieval():
    """Test the full embedding and retrieval pipeline with real content"""
    try:
        from src.nodes import embed_and_retrieve, create_qa_pairs, Document
        
        print("🚀 Starting real query test for embedding and retrieval...")
        start_time = time.time()
        
        # Create test state with real content about AI/ML in the correct format
        # hybrid_retriever expects: Dict[str, Dict[str, str]] where inner dict has 'content' and 'metadata'
        test_content = {
            'https://ai-fundamentals.com/intro': {
                'content': "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction. Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                'metadata': {'title': 'AI Fundamentals', 'source': 'ai-fundamentals.com'}
            },
            'https://ai-applications.com/overview': {
                'content': "Natural Language Processing (NLP) enables machines to understand, interpret, and generate human language in a valuable way. Applications include chatbots, language translation, sentiment analysis, and text summarization. Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Autonomous vehicles use AI to navigate roads safely without human intervention.",
                'metadata': {'title': 'AI Applications', 'source': 'ai-applications.com'}
            },
            'https://future-ai.org/trends': {
                'content': "The future of AI includes developments in artificial general intelligence (AGI), which would match or exceed human cognitive abilities across all domains. Current AI systems are narrow, designed for specific tasks. Ethical AI development focuses on creating systems that are fair, transparent, and beneficial to humanity. AI will continue to transform industries including healthcare, education, finance, and manufacturing.",
                'metadata': {'title': 'Future of AI', 'source': 'future-ai.org'}
            }
        }
        
        state = {
            'new_query': 'What is artificial intelligence and how does machine learning relate to it?',
            'relevant_contexts': test_content,
            'search_queries': ['ai fundamentals', 'ai applications', 'future ai'],
            'error': None
        }
        
        print(f"📝 Test query: {state['new_query']}")
        print(f"📚 Content URLs: {len(test_content)}")
        print(f"🔍 Search queries: {state['search_queries']}")
        
        # Test embed_and_retrieve
        print("\n🔄 Testing embed_and_retrieve...")
        embed_start = time.time()
        result = await embed_and_retrieve(state)
        embed_time = time.time() - embed_start
        
        error = result.get('error')
        chunks = result.get('relevant_chunks', [])
        responses = result.get('retriever_responses', {})
        
        print(f"⏱️  Embedding/retrieval took: {embed_time:.2f}s")
        print(f"❌ Error: {error if error else 'None'}")
        print(f"📄 Chunks retrieved: {len(chunks)}")
        print(f"💬 Retriever responses: {len(responses)}")
        
        if error:
            print(f"❌ Embedding/retrieval failed: {error}")
            return False
            
        if chunks:
            print(f"📋 Sample chunk preview: {str(chunks[0])[:200]}...")
            print(f"📋 Chunk types: {[type(chunk).__name__ for chunk in chunks[:3]]}")
        
        # Test create_qa_pairs if embedding succeeded
        if not error and chunks:
            print("\n🔄 Testing create_qa_pairs...")
            qa_start = time.time()
            
            # Update state with results from embed_and_retrieve
            state.update(result)
            
            qa_result = await create_qa_pairs(state)
            qa_time = time.time() - qa_start
            
            qa_error = qa_result.get('error')
            qa_pairs = qa_result.get('qa_pairs', [])
            
            print(f"⏱️  QA pair creation took: {qa_time:.2f}s")
            print(f"❌ Error: {qa_error if qa_error else 'None'}")
            print(f"❓ QA pairs created: {len(qa_pairs)}")
            
            if qa_error:
                print(f"❌ QA pair creation failed: {qa_error}")
                return False
                
            if qa_pairs:
                print(f"💡 Sample QA pair:")
                sample_qa = qa_pairs[0]
                print(f"   Query: {sample_qa.get('query', 'N/A')}")
                print(f"   Answer: {sample_qa.get('answer', 'N/A')[:200]}...")
                print(f"   Citations: {len(sample_qa.get('citations', []))}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Full test completed in {total_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"💥 Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_hybrid_retriever_stats():
    """Test hybrid retriever statistics and performance"""
    try:
        from src.hybrid_retriever import create_hybrid_retriever
        from src.llm_utils import embeddings  # Correct import
        from src.config import CHUNK_SIZE, CHUNK_OVERLAP
        
        print("\n📊 Testing hybrid retriever directly...")
        
        # Check if embeddings are available
        if not embeddings:
            print("❌ No embeddings available - skipping hybrid retriever test")
            return False
            
        print(f"🔤 Embeddings type: {type(embeddings).__name__}")
        
        # Create hybrid retriever
        hybrid_retriever = create_hybrid_retriever(
            embeddings=embeddings,
            top_k=5,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Test content in the correct format for hybrid retriever
        test_docs = {
            'https://example.com/ai': {
                'content': "Artificial intelligence is transforming the world through machine learning and deep learning technologies.",
                'metadata': {'title': 'AI Overview', 'source': 'example.com'}
            },
            'https://example.com/nlp': {
                'content': "Natural language processing enables computers to understand and generate human language effectively.",
                'metadata': {'title': 'NLP Guide', 'source': 'example.com'}
            },
            'https://example.com/cv': {
                'content': "Computer vision allows machines to interpret visual information from the world around them.",
                'metadata': {'title': 'Computer Vision', 'source': 'example.com'}
            }
        }
        
        print(f"📚 Building index with {len(test_docs)} documents...")
        success = hybrid_retriever.build_index(test_docs)
        
        if success:
            print("✅ Index built successfully")
            
            # Test retrieval
            query = "How does AI use machine learning?"
            results = hybrid_retriever.retrieve(query)
            
            print(f"🔍 Query: {query}")
            print(f"📄 Retrieved {len(results)} results")
            
            # Get stats
            stats = hybrid_retriever.get_stats()
            print(f"📊 Retriever stats: {stats}")
            
            return True
        else:
            print("❌ Failed to build index")
            return False
            
    except Exception as e:
        print(f"💥 Hybrid retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run comprehensive real query tests"""
    print("=" * 60)
    print("🧪 REAL QUERY TEST - EMBEDDING AND RETRIEVAL FUNCTIONS")
    print("=" * 60)
    
    # Test 1: Full pipeline
    test1 = await test_full_embedding_retrieval()
    
    # Test 2: Hybrid retriever directly
    test2 = await test_hybrid_retriever_stats()
    
    print(f"\n" + "=" * 60)
    print(f"📋 FINAL TEST RESULTS")
    print(f"=" * 60)
    print(f"Full Pipeline Test: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Hybrid Retriever Test: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Overall Result: {'🎉 ALL TESTS PASSED' if test1 and test2 else '⚠️  SOME TESTS FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())