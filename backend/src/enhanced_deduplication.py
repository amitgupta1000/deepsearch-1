"""
Enhanced Deduplication System for INTELLISEARCH Reports

This module provides intelligent deduplication of report content using both
rule-based and LLM-powered approaches with caching for performance.
"""

import hashlib
import logging
import re
import time
from typing import List, Tuple, Optional, Dict, Any
from difflib import SequenceMatcher

try:
    from .config import (
        USE_LLM_DEDUPLICATION,
        DEDUPLICATION_CACHE_ENABLED,
        DEDUPLICATION_CACHE_TTL,
        SIMILARITY_THRESHOLD,
        MIN_SENTENCE_LENGTH,
        DEDUPLICATION_BATCH_SIZE,
        LLM_DEDUP_MIN_WORDS,
        LLM_DEDUP_DETAILED_ONLY
    )

except ImportError:
    # Fallback imports for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import (
        USE_LLM_DEDUPLICATION,
        DEDUPLICATION_CACHE_ENABLED, 
        DEDUPLICATION_CACHE_TTL,
        SIMILARITY_THRESHOLD,
        MIN_SENTENCE_LENGTH,
        DEDUPLICATION_BATCH_SIZE,
        LLM_DEDUP_MIN_WORDS,
        LLM_DEDUP_DETAILED_ONLY
    )
    from llm_utils import llm_call_async
    from prompt import get_current_date

# Simple in-memory cache for deduplication results
_deduplication_cache: Dict[str, Tuple[str, float]] = {}

def _get_cache_key(content: str) -> str:
    """Generate a cache key for content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def _is_cache_valid(timestamp: float) -> bool:
    """Check if cached result is still valid."""
    if not DEDUPLICATION_CACHE_ENABLED:
        return False
    return (time.time() - timestamp) < DEDUPLICATION_CACHE_TTL

def _get_cached_result(content: str) -> Optional[str]:
    """Get cached deduplication result if available and valid."""
    cache_key = _get_cache_key(content)
    if cache_key in _deduplication_cache:
        cached_content, timestamp = _deduplication_cache[cache_key]
        if _is_cache_valid(timestamp):
            logging.debug("Using cached deduplication result")
            return cached_content
        else:
            # Remove expired cache entry
            del _deduplication_cache[cache_key]
    return None

def _cache_result(original_content: str, deduplicated_content: str) -> None:
    """Cache the deduplication result."""
    if not DEDUPLICATION_CACHE_ENABLED:
        return
        
    cache_key = _get_cache_key(original_content)
    _deduplication_cache[cache_key] = (deduplicated_content, time.time())
    logging.debug(f"Cached deduplication result for content length {len(original_content)}")

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences with improved handling of citations and formatting."""
    # Handle common sentence endings with citation patterns
    text = re.sub(r'(\[[0-9]+\])\s*([A-Z])', r'\1\n\2', text)
    
    # Split on sentence endings, preserving structure
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z#])', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence.split()) >= MIN_SENTENCE_LENGTH:
            cleaned_sentences.append(sentence)
        elif sentence.startswith('#') or sentence.strip().endswith(':'):
            # Keep headers and important structural elements
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text segments."""
    # Normalize text for comparison
    norm1 = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text1.lower())).strip()
    norm2 = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text2.lower())).strip()
    
    if not norm1 or not norm2:
        return 0.0
    
    # Use SequenceMatcher for more accurate similarity
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity

def basic_deduplicate(sentences: List[str]) -> List[str]:
    """Apply rule-based deduplication to sentences."""
    unique_sentences = []
    seen_content = []
    
    for sentence in sentences:
        # Skip very short sentences except headers
        if len(sentence.split()) < MIN_SENTENCE_LENGTH and not sentence.startswith('#'):
            continue
            
        # Always include headers and structural elements
        if sentence.startswith('#') or sentence.strip().endswith(':'):
            unique_sentences.append(sentence)
            continue
        
        # Check for similarity with existing sentences
        is_duplicate = False
        for seen_sentence in seen_content:
            similarity = calculate_similarity(sentence, seen_sentence)
            if similarity > SIMILARITY_THRESHOLD:
                is_duplicate = True
                logging.debug(f"Found duplicate (similarity: {similarity:.2f}): {sentence[:100]}...")
                break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
            seen_content.append(sentence)
    
    return unique_sentences

async def llm_intelligent_deduplicate(content: str) -> str:
    """Use LLM to intelligently deduplicate and improve content flow."""
    # Word count check
    word_count = len(content.split())
    if word_count < LLM_DEDUP_MIN_WORDS:
        logging.info(f"Content too short for LLM deduplication ({word_count} words)")
        return content
    # Check cache first
    cached_result = _get_cached_result(content)
    if cached_result:
        return cached_result
    # Prepare LLM prompt for intelligent deduplication
    deduplication_prompt = f"""
You are an expert editor tasked with improving a research report by eliminating redundancy while preserving all important information.

**TASK**: Remove redundant sentences and improve flow, but NEVER remove important facts or data.

**GUIDELINES**:
1. **Preserve Important Information**: Keep all facts, data, statistics, and key insights
2. **Remove True Duplicates**: Delete sentences that repeat the same information
3. **Merge Similar Content**: If two sentences cover similar points but add different value, merge them intelligently
4. **Maintain Structure**: Preserve headers, sections, and logical flow
5. **Keep Citations**: Maintain all citation references [1], [2], etc.
6. **Preserve Formatting**: Maintain proper spacing around punctuation, citations, and headings
7. **Improve Transitions**: Ensure smooth flow between sentences and paragraphs

**FORMATTING REQUIREMENTS**:
- Maintain proper spacing after periods, commas, colons, and semicolons
- Keep proper spacing around citations: "text [1] more text" 
- Preserve markdown headers with proper spacing
- Keep paragraph breaks and bullet point formatting

**CURRENT DATE**: {get_current_date()}

**CONTENT TO DEDUPLICATE**:
{content}

**OUTPUT**: Return the improved content with redundancy removed but all important information and formatting preserved.
"""
    try:
        try:
            # Try to use message classes from llm_utils first
            from .llm_utils import SystemMessage, HumanMessage
        except ImportError:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
            except ImportError:
                # Fallback message classes
                class SystemMessage:
                    def __init__(self, content): 
                        self.content = content
                    def __str__(self):
                        return f"SystemMessage(content='{self.content}')"
                class HumanMessage:
                    def __init__(self, content): 
                        self.content = content
                    def __str__(self):
                        return f"HumanMessage(content='{self.content}')"
        messages = [
            SystemMessage(content="You are an expert research report editor focused on eliminating redundancy while preserving all important information."),
            HumanMessage(content=deduplication_prompt)
        ]
        logging.info(f"Applying LLM-powered deduplication to {word_count} words...")
        response = await llm_call_async(messages)
        if response and hasattr(response, 'content'):
            deduplicated_content = response.content.strip()
            new_word_count = len(deduplicated_content.split())
            reduction_ratio = (word_count - new_word_count) / word_count
            if reduction_ratio > 0.5:  # More than 50% reduction might be too aggressive
                logging.warning(f"LLM deduplication removed {reduction_ratio:.1%} of content, using fallback")
                return content
            logging.info(f"LLM deduplication: {word_count} → {new_word_count} words ({reduction_ratio:.1%} reduction)")
            # Cache the result
            _cache_result(content, deduplicated_content)
            return deduplicated_content
        else:
            logging.warning("LLM deduplication returned no content, using original")
            return content
    except Exception as e:
        logging.error(f"Error in LLM deduplication: {e}")
        logging.debug(f"Falling back to basic deduplication")
        # Fallback to basic deduplication if LLM fails
        sentences = split_into_sentences(content)
        basic_deduplicated_sentences = basic_deduplicate(sentences)
        return ' '.join(basic_deduplicated_sentences)

async def enhanced_deduplicate_content(text: str) -> str:
    """
    Enhanced deduplication combining rule-based and LLM approaches.
    Args:
        text: The content to deduplicate
    Returns:
        Deduplicated content
    """
    if not text or len(text.strip()) < 100:
        return text
    logging.info("Starting enhanced content deduplication...")
    # Step 1: Basic rule-based deduplication
    sentences = split_into_sentences(text)
    basic_deduplicated_sentences = basic_deduplicate(sentences)
    basic_result = ' '.join(basic_deduplicated_sentences)
    # Calculate basic reduction
    original_words = len(text.split())
    basic_words = len(basic_result.split())
    basic_reduction = (original_words - basic_words) / original_words if original_words > 0 else 0
    logging.info(f"Basic deduplication: {original_words} → {basic_words} words ({basic_reduction:.1%} reduction)")
    # Step 2: LLM-powered intelligent deduplication (if enabled and appropriate)
    if USE_LLM_DEDUPLICATION and basic_words >= LLM_DEDUP_MIN_WORDS:
        final_result = await llm_intelligent_deduplicate(basic_result)
        final_words = len(final_result.split())
        total_reduction = (original_words - final_words) / original_words if original_words > 0 else 0
        logging.info(f"Total deduplication: {original_words} → {final_words} words ({total_reduction:.1%} reduction)")
        return final_result
    else:
        logging.info("Using basic deduplication only")
        return basic_result

# Legacy function for backward compatibility
def deduplicate_content(text: str) -> str:
    """Legacy function that calls the basic deduplication."""
    sentences = split_into_sentences(text)
    deduplicated_sentences = basic_deduplicate(sentences)
    return ' '.join(deduplicated_sentences)

# Cache management functions
def clear_deduplication_cache():
    """Clear the deduplication cache."""
    global _deduplication_cache
    _deduplication_cache.clear()
    logging.info("Cleared deduplication cache")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    valid_entries = sum(1 for _, (_, timestamp) in _deduplication_cache.items() 
                       if _is_cache_valid(timestamp))
    
    return {
        "total_entries": len(_deduplication_cache),
        "valid_entries": valid_entries,
        "cache_enabled": DEDUPLICATION_CACHE_ENABLED,
        "cache_ttl": DEDUPLICATION_CACHE_TTL
    }

if __name__ == "__main__":
    # Test the deduplication system
    test_content = """
    # Test Report
    
    Artificial intelligence is transforming many industries. AI is changing how businesses operate.
    The technology has significant potential. This technology offers great possibilities.
    Machine learning algorithms are being used widely. ML algorithms are becoming popular.
    Companies are investing in AI solutions. Organizations are putting money into artificial intelligence.
    """
    
    print("Original content:")
    print(test_content)
    print("\nDeduplicated content:")
    print(deduplicate_content(test_content))