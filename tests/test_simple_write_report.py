#!/usr/bin/env python3
"""
Simple test to check if the write_report function works after modifications.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from datetime import datetime

def create_minimal_test_state():
    """Create a minimal test state."""
    return {
        'new_query': 'Test query about renewable energy',
        'search_queries': ['renewable energy'],
        'qa_pairs': [
            {
                'question': 'What is renewable energy?',
                'answer': 'Renewable energy comes from natural sources like solar and wind [1].',
                'citations': [
                    {
                        'number': 1,
                        'source': 'Test Source',
                        'url': 'https://test.com',
                        'content': 'Test content'
                    }
                ]
            }
        ],
        'prompt_type': 'general',
        'error': None
    }

async def test_basic_functionality():
    """Test basic functionality."""
    print("Testing write_report modification...")
    
    try:
        # Import the function
        from nodes import write_report
        
        # Create test state
        test_state = create_minimal_test_state()
        
        # Run the function
        result = await write_report(test_state)
        
        # Check results
        print("✅ Function executed successfully!")
        
        # Check if we have the new content types
        display_content = result.get('report', '')
        full_content = result.get('full_report_content', '')
        appendix_content = result.get('appendix_content', '')
        analysis_content = result.get('analysis_content', '')
        
        print(f"Display content length: {len(display_content)}")
        print(f"Full content length: {len(full_content)}")
        print(f"Appendix content length: {len(appendix_content)}")
        print(f"Analysis content length: {len(analysis_content)}")
        
        # Check content structure
        print(f"Display has analysis only: {'IntelliSearch Response' in display_content and 'Original User Query' not in display_content}")
        print(f"Full has all sections: {'Original User Query' in full_content and 'IntelliSearch Response' in full_content and 'Appendix' in full_content}")
        
        if result.get('error'):
            print(f"Errors: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if success:
        print("\n🎉 Basic test passed!")
    else:
        print("\n💥 Basic test failed!")