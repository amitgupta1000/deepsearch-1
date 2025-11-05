"""
Test script to verify the write_report modification works correctly.
This tests that:
1. Display content shows Part 1 (User Query) + Part 2 (IntelliSearch Analysis) 
2. Appendix content contains Part 3 (Q&A pairs and citations) separately
3. Full report content contains all three parts
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock dependencies
import logging
logging.basicConfig(level=logging.INFO)

# Mock the required modules
class MockLLM:
    async def __call__(self, messages):
        return "This is a mock IntelliSearch analysis response based on the research findings."

class MockSystemMessage:
    def __init__(self, content):
        self.content = content

class MockHumanMessage:
    def __init__(self, content):
        self.content = content

# Import the modules after setting up mocks
sys.modules['src.llm_calling'] = type('MockModule', (), {
    'llm_call_async': MockLLM()
})()

sys.modules['src.llm_utils'] = type('MockModule', (), {
    'SystemMessage': MockSystemMessage,
    'HumanMessage': MockHumanMessage
})()

sys.modules['src.utils'] = type('MockModule', (), {
    'save_report_to_text': lambda content, filename: f"mock_saved_{filename}",
    'generate_pdf_from_md': lambda content, filename: "PDF generated successfully",
    'get_current_date': lambda: "November 5, 2025"
})()

sys.modules['src.config'] = type('MockModule', (), {
    'REPORT_FILENAME_TEXT': "test_report.txt",
    'REPORT_FILENAME_PDF': "test_report.pdf"
})()

# Import the write_report function
from src.nodes import write_report

async def test_write_report_modification():
    """Test the write_report function with the new structure"""
    
    # Create test state with sample data
    test_state = {
        'new_query': 'What are the latest developments in AI technology?',
        'prompt_type': 'general',
        'qa_pairs': [
            {
                'question': 'What is the current state of AI technology?',
                'answer': 'AI technology has advanced significantly with developments in machine learning [1] and neural networks [2].',
                'citations': [
                    {'number': 1, 'source': 'AI Research Journal 2024', 'url': 'https://example.com/ai-journal', 'content': 'Sample content'},
                    {'number': 2, 'source': 'Neural Networks Today', 'url': 'https://example.com/neural-networks', 'content': 'Sample content'}
                ]
            },
            {
                'question': 'What are the emerging trends in AI?',
                'answer': 'Key trends include generative AI [3], large language models [4], and multimodal AI systems [5].',
                'citations': [
                    {'number': 3, 'source': 'Tech Trends Report 2024', 'url': 'https://example.com/tech-trends', 'content': 'Sample content'},
                    {'number': 4, 'source': 'LLM Survey Paper', 'url': 'https://example.com/llm-survey', 'content': 'Sample content'},
                    {'number': 5, 'source': 'Multimodal AI Research', 'url': 'https://example.com/multimodal', 'content': 'Sample content'}
                ]
            }
        ]
    }
    
    print("🧪 Testing write_report modification...")
    print("=" * 60)
    
    # Run the write_report function
    result_state = await write_report(test_state)
    
    # Test 1: Check that display content contains Part 1 + Part 2
    print("\n📋 Test 1: Display Content Structure")
    display_content = result_state.get('report', '')
    
    # Check for Part 1 (User Query)
    part1_present = "## 1. Original User Query" in display_content and test_state['new_query'] in display_content
    print(f"✓ Part 1 (User Query) present: {part1_present}")
    
    # Check for Part 2 (IntelliSearch Response)  
    part2_present = "## 2. IntelliSearch Response" in display_content
    print(f"✓ Part 2 (IntelliSearch Analysis) present: {part2_present}")
    
    # Check that Part 3 (Appendix) is NOT in display content
    part3_absent = "## 3. Appendix" not in display_content
    print(f"✓ Part 3 (Appendix) absent from display: {part3_absent}")
    
    # Test 2: Check appendix content structure
    print("\n📑 Test 2: Appendix Content Structure")
    appendix_content = result_state.get('appendix_content', '')
    
    appendix_has_qa = "### Research Questions and Detailed Answers" in appendix_content
    appendix_has_citations = "### Sources and Citations" in appendix_content
    appendix_has_part3 = "## 3. Appendix" in appendix_content
    
    print(f"✓ Appendix contains Q&A section: {appendix_has_qa}")
    print(f"✓ Appendix contains citations section: {appendix_has_citations}")
    print(f"✓ Appendix has Part 3 header: {appendix_has_part3}")
    
    # Test 3: Check full report content
    print("\n📚 Test 3: Full Report Content Structure")
    full_content = result_state.get('full_report_content', '')
    
    full_has_part1 = "## 1. Original User Query" in full_content
    full_has_part2 = "## 2. IntelliSearch Response" in full_content  
    full_has_part3 = "## 3. Appendix" in full_content
    
    print(f"✓ Full report contains Part 1: {full_has_part1}")
    print(f"✓ Full report contains Part 2: {full_has_part2}")
    print(f"✓ Full report contains Part 3: {full_has_part3}")
    
    # Test 4: Check that state contains all required fields
    print("\n🔧 Test 4: State Fields")
    required_fields = ['report', 'full_report_content', 'appendix_content', 'analysis_content']
    
    for field in required_fields:
        present = field in result_state and result_state[field]
        print(f"✓ State has {field}: {present}")
    
    # Summary
    print("\n" + "=" * 60)
    all_tests_passed = all([
        part1_present, part2_present, part3_absent,  # Display tests
        appendix_has_qa, appendix_has_citations, appendix_has_part3,  # Appendix tests
        full_has_part1, full_has_part2, full_has_part3,  # Full report tests
        all(field in result_state and result_state[field] for field in required_fields)  # State tests
    ])
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! write_report modification is working correctly.")
        print("\n📊 Content Summary:")
        print(f"   Display content length: {len(display_content)} characters")
        print(f"   Appendix content length: {len(appendix_content)} characters")
        print(f"   Full report content length: {len(full_content)} characters")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = asyncio.run(test_write_report_modification())
    sys.exit(0 if success else 1)
