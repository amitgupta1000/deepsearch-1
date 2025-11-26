# utils.py
import re
import json
import logging
import asyncio
import time # Added time for cache timestamp

# Define SearchResult and PDF classes here for simplicity or ensure they are imported
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import REPORT_FILENAME_TEXT

@dataclass
class SearchResult:
    """
    Dataclass to store individual search results.
    """
    url: str
    title: str
    snippet: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source
        }

# Utility function to safely format prompts with content that may contain curly braces
def safe_format(template: str, **kwargs: Any) -> str:
    """
    Safely format a template string, escaping any curly braces in the values.
    This prevents ValueError when content contains unexpected curly braces.
    """
    # Escape any curly braces in the values
    safe_kwargs = {k: v.replace('{', '{{').replace('}', '}}') if isinstance(v, str) else v
                  for k, v in kwargs.items()}
    return template.format(**safe_kwargs)

# Get current date in a readable format
from datetime import datetime
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


# Helper function to clean extracted text
def clean_extracted_text(text: str) -> str:
       """Cleans extracted text by removing extra whitespaces, image files, and boilerplate text."""
       if text is None:
           return ""
       # Remove extra whitespaces and newlines
       text = re.sub(r'\s+', ' ', text).strip()

       # Remove HTML tags (a more general approach)
       text = re.sub(r'<[^>]+>', '', text)

       # Remove common boilerplate patterns (can be expanded)
       text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # Remove HTML comments
       text = re.sub(r'<(script|style).*?>.*?</(script|style)>', '', text, flags=re.DOTALL | re.IGNORECASE) # Remove script and style tags
       text = re.sub(r'\[document\][^\]]*\]', '', text) # Remove patterns like [document]...
       text = re.sub(r'\(Source:.*?\)', '', text) # Remove source citations if present from previous steps

       return text


# Helper function to rank URLs
from rank_bm25 import BM25Okapi # Assuming rank_bm25 is installed

def rank_urls(query: str, urls: List[str], relevant_contexts: Dict[str, Dict[str, str]]) -> List[str]:
    """Ranks URLs based on their relevance to the query using BM25."""
    if not urls or not relevant_contexts or not query:
        return urls # Return original order if ranking not possible

    # Extract content from the new dictionary structure
    corpus = []
    for url in urls:
        context_data = relevant_contexts.get(url, {})
        if isinstance(context_data, dict):
            content = context_data.get('content', '')
        else:
            # Handle backward compatibility with old string format
            content = context_data if isinstance(context_data, str) else ''
        corpus.append(content)
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    if not any(tokenized_corpus): # Check if corpus is empty after tokenization
        return urls # Return original order

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    if not tokenized_query:
         return urls # Return original order

    scores = bm25.get_scores(tokenized_query)
    # Pair scores with URLs and sort in descending order of scores
    scored_urls = sorted(zip(scores, urls), reverse=True)
    ranked_urls = [url for score, url in scored_urls]

    return ranked_urls

# Function to save report to text

def save_report_to_text(report_content: str, filename: str) -> str:
    """Saves the report content to Firestore only."""
    try:
        from google.cloud import firestore
        db = firestore.Client()
        # Use filename as document id for simplicity
        doc_ref = db.collection("report_files").document(filename)
        doc_ref.set({
            "filename": filename,
            "content": report_content,
            "saved_at": __import__('datetime').datetime.now().isoformat()
        })
        logging.info(f"Report saved to Firestore: {filename}")
        return filename  # Return filename for reference
    except Exception as e:
        logging.warning(f"Could not save report to Firestore: {e}")
        return ""


def format_research_report(report_content: str) -> str:
    """
    Format a research report with proper markdown structure and readability improvements.
    
    Args:
        report_content (str): The raw report content to format
        
    Returns:
        str: Formatted report with improved structure and readability
    """
    if not report_content or not report_content.strip():
        return report_content

    # Enhance the conclusion section before further formatting
    report_content = enhance_conclusion_section(report_content)

    # Split into lines for processing
    lines = report_content.split('\n')
    formatted_lines = []
    
    last_was_empty = False
    for i, line in enumerate(lines):
        line = line.strip()
        # Skip multiple consecutive empty lines
        if not line:
            if not last_was_empty:
                formatted_lines.append('')
                last_was_empty = True
            continue
        last_was_empty = False
        # Format main headings (# Title)
        if line.startswith('# '):
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            formatted_lines.append('')
            continue
        # Format sub-headings (## Title)
        if line.startswith('## '):
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            formatted_lines.append('')
            continue
        # Format sub-sub-headings (### Title)
        if line.startswith('### '):
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            formatted_lines.append('')
            continue
        # Format bullet points
        if line.startswith('- ') or line.startswith('* '):
            if formatted_lines and not formatted_lines[-1].startswith(('- ', '* ')) and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            continue
        # Format numbered lists
        if re.match(r'^\d+\. ', line):
            if formatted_lines and not re.match(r'^\d+\. ', formatted_lines[-1] or '') and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            continue
        # Regular paragraph text
        formatted_lines.append(line)
    
    # Join lines back together
    formatted_content = '\n'.join(formatted_lines)
    
    # Clean up excessive whitespace (more than 2 consecutive empty lines)
    formatted_content = re.sub(r'\n{4,}', '\n\n\n', formatted_content)
    
    # Ensure report starts and ends cleanly
    formatted_content = formatted_content.strip()
    
    return formatted_content


def enhance_report_readability(report_content: str) -> str:
    """
    Enhance report readability with additional formatting improvements.
    
    Args:
        report_content (str): The report content to enhance
        
    Returns:
        str: Enhanced report with improved readability
    """
    if not report_content or not report_content.strip():
        return report_content
    
    content = report_content
    
    # Add proper spacing around citations
    content = re.sub(r'(\[\d+\])(?=[A-Za-z])', r'\1 ', content)  # Space after citation if followed by letter
    content = re.sub(r'([A-Za-z])(\[\d+\])', r'\1 \2', content)  # Space before citation if preceded by letter
    
    # Improve sentence structure
    content = re.sub(r'\.([A-Z])', r'. \1', content)  # Ensure space after periods
    content = re.sub(r',([A-Za-z])', r', \1', content)  # Ensure space after commas
    content = re.sub(r';([A-Za-z])', r'; \1', content)  # Ensure space after semicolons
    content = re.sub(r':([A-Za-z])', r': \1', content)  # Ensure space after colons
    
    # Fix multiple spaces
    content = re.sub(r' {2,}', ' ', content)
    
    # Ensure proper paragraph breaks
    content = re.sub(r'\.([A-Z][a-z])', r'.\n\n\1', content)  # Add paragraph breaks after sentences that end paragraphs
    
    # Format the content with the main formatting function
    content = format_research_report(content)
    
    return content


# Replace all '## Conclusion' with a bold section header for conclusion
import re

def enhance_conclusion_section(text: str) -> str:
    # Replace markdown '## Conclusion' with bolded 'Conclusion' section
    return re.sub(r'## Conclusion\s*', '\n**Conclusion**\n', text)

# Example usage in formatting pipeline:
# formatted_text = enhance_conclusion_section(generated_text)


logging.info("utils.py loaded with utility functions.")
