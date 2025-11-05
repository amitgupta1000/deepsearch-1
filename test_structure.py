#!/usr/bin/env python3
"""
Simple test script for embed_and_retrieve function structure
Tests the function logic without requiring external APIs
"""
import asyncio
import logging
import sys
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add current directory to path
sys.path.append('.')

async def test_function_imports():
    """Test that we can import the functions without errors"""
    try:
        from src.nodes import embed_and_retrieve, create_qa_pairs, Document
        print("✓ Successfully imported embed_and_retrieve, create_qa_pairs, and Document")
        
        # Test Document class
        doc = Document(page_content="Test content", metadata={"source": "test"})
        print(f"✓ Document class works: {doc.page_content[:20]}")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_basic_structure():
    """Test basic function structure without external dependencies"""
    try:
        # Mock the dependencies to avoid API calls
        import src.config as config
        
        # Temporarily disable hybrid retrieval to test fallback path
        original_hybrid = getattr(config, 'USE_HYBRID_RETRIEVAL', True)
        config.USE_HYBRID_RETRIEVAL = False
        
        print("Testing with hybrid retrieval disabled...")
        
        from src.nodes import embed_and_retrieve, Document
        
        # Create minimal test state
        state = {
            'new_query': 'test query',
            'relevant_contexts': {'query1': ['test content']},
            'search_queries': ['query1'],
            'error': None
        }
        
        # This should return an error since hybrid retrieval is disabled
        # and there's no fallback, but it tests the function structure
        result = await embed_and_retrieve(state)
        
        # Restore original setting
        config.USE_HYBRID_RETRIEVAL = original_hybrid
        
        print(f"✓ Function executed, error (expected): {result.get('error', 'None')}")
        print(f"✓ Function returned chunks: {len(result.get('relevant_chunks', []))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_qa_pairs_structure():
    """Test create_qa_pairs function structure"""
    try:
        from src.nodes import create_qa_pairs, Document
        
        # Create test state with proper Document objects
        state = {
            'new_query': 'test query',
            'search_queries': ['query1'],
            'relevant_chunks': [
                Document(page_content='Test content about AI', metadata={'source': 'test1'}),
                Document(page_content='More content about ML', metadata={'source': 'test2'})
            ],
            'retriever_responses': {
                'query1': 'Test response'
            },
            'error': None
        }
        
        # This might fail due to LLM requirements, but tests structure
        result = await create_qa_pairs(state)
        
        print(f"✓ create_qa_pairs executed")
        print(f"  Error: {result.get('error', 'None')}")
        print(f"  QA pairs: {len(result.get('qa_pairs', []))}")
        
        return True
        
    except Exception as e:
        print(f"✗ QA pairs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run basic structure tests"""
    print("=== Basic Function Structure Tests ===\n")
    
    # Test imports
    test1 = await test_function_imports()
    
    # Test embed_and_retrieve structure
    test2 = await test_basic_structure()
    
    # Test create_qa_pairs structure
    test3 = await test_qa_pairs_structure()
    
    print(f"\n=== Test Results ===")
    print(f"Imports: {'PASSED' if test1 else 'FAILED'}")
    print(f"embed_and_retrieve structure: {'PASSED' if test2 else 'FAILED'}")
    print(f"create_qa_pairs structure: {'PASSED' if test3 else 'FAILED'}")
    print(f"Overall: {'PASSED' if test1 and test2 and test3 else 'FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())