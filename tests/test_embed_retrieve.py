#!/usr/bin/env python3
"""
Test script for embed_and_retrieve function
"""
import asyncio
import logging
import sys
import os

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add current directory to path
sys.path.append('.')

async def test_embed_and_retrieve():
    """Test the embed_and_retrieve function with sample data"""
    try:
        from src.nodes import embed_and_retrieve
        
        # Create test state with sample data
        state = {
            'new_query': 'test query about artificial intelligence',
            'relevant_contexts': {
                'ai_query': ['Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.'],
                'ml_query': ['Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.']
            },
            'search_queries': ['ai_query', 'ml_query'],
            'error': None
        }
        
        print('Testing embed_and_retrieve function...')
        result = await embed_and_retrieve(state)
        
        error = result.get('error')
        chunks = result.get('relevant_chunks', [])
        responses = result.get('retriever_responses', {})
        
        print(f'Result error: {error}')
        print(f'Number of chunks retrieved: {len(chunks)}')
        print(f'Retriever responses: {len(responses)} queries processed')
        
        if error:
            print(f'Function returned error: {error}')
            return False
        else:
            print('Function completed successfully!')
            if chunks:
                print(f'Sample chunk: {chunks[0][:100]}...' if len(str(chunks[0])) > 100 else chunks[0])
            return True
            
    except Exception as e:
        print(f'Exception occurred: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_create_qa_pairs():
    """Test the create_qa_pairs function"""
    try:
        from src.nodes import create_qa_pairs, Document
        
        # Create test state with chunks and retriever responses
        # Chunks need to be Document objects with page_content
        state = {
            'new_query': 'test query about artificial intelligence',
            'search_queries': ['ai_query', 'ml_query'],
            'relevant_chunks': [
                Document(page_content='Artificial intelligence is the simulation of human intelligence processes by machines.', metadata={'source': 'test1'}),
                Document(page_content='Machine learning is a subset of artificial intelligence.', metadata={'source': 'test2'})
            ],
            'retriever_responses': {
                'ai_query': 'Found information about AI',
                'ml_query': 'Found information about ML'
            },
            'error': None
        }
        
        print('\nTesting create_qa_pairs function...')
        result = await create_qa_pairs(state)
        
        error = result.get('error')
        qa_pairs = result.get('qa_pairs', [])
        
        print(f'Result error: {error}')
        print(f'Number of QA pairs created: {len(qa_pairs)}')
        
        if error:
            print(f'Function returned error: {error}')
            return False
        else:
            print('Function completed successfully!')
            if qa_pairs:
                print(f'Sample QA pair: {str(qa_pairs[0])[:200]}...' if len(str(qa_pairs[0])) > 200 else qa_pairs[0])
            return True
            
    except Exception as e:
        print(f'Exception occurred: {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=== Testing Embedding and Retrieval Functions ===\n")
    
    # Test embed_and_retrieve
    success1 = await test_embed_and_retrieve()
    
    # Test create_qa_pairs
    success2 = await test_create_qa_pairs()
    
    print(f"\n=== Test Results ===")
    print(f"embed_and_retrieve: {'PASSED' if success1 else 'FAILED'}")
    print(f"create_qa_pairs: {'PASSED' if success2 else 'FAILED'}")
    print(f"Overall: {'PASSED' if success1 and success2 else 'FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())