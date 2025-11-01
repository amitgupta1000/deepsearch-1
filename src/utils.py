# utils.py
import re
import json
import logging
import asyncio
import time # Added time for cache timestamp
import fpdf

# Define SearchResult and PDF classes here for simplicity or ensure they are imported
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import REPORT_FILENAME_TEXT, REPORT_FILENAME_PDF

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

from fpdf import FPDF # Assuming fpdf is installed
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


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

# Helper function to fetch PDF content
def fetch_pdf_content(url: str) -> str:
    """Fetches the text content of a PDF given its URL.
    Downloads the PDF content as bytes and then uses PyMuPDF.
    Returns the text content or an error message.
    Fails gracefully if download takes longer than 90 seconds.
    """
    try:
        # Import requests and fitz locally to avoid global dependency unless needed
        import requests
        import fitz

        # Download the PDF content with 90-second timeout
        # Using both connect and read timeouts for comprehensive coverage
        response = requests.get(
            url, 
            stream=True, 
            timeout=(30, 90)  # (connect_timeout, read_timeout)
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Open the PDF from bytes using PyMuPDF
        doc = fitz.open(stream=response.content, filetype="pdf")
        text_content = ""
        for page in doc:
            text_content += page.get_text()
        return text_content
    except requests.exceptions.Timeout as e:
        return f"PDF download from {url} timed out after 90 seconds: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF from {url}: {e}"
    except Exception as e:
        return f"Error processing PDF content from {url}: {e}"


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

def save_report_to_text(report_content: str, filename: str = REPORT_FILENAME_TEXT) -> str:
    """Saves the report content to a text file."""
    try:
        with open(filename, "w", encoding='utf-8') as f: # Specify utf-8 encoding
            f.write(report_content)
        logging.info(f"Report saved to: {filename}")
        return filename
    except IOError as e:
        logging.exception(f"Error saving report to text file {filename}: {e}")
        return ""

# Function to generate PDF from markdown

def generate_pdf_from_md(content, filename=REPORT_FILENAME_PDF):
    """Generates a PDF from Markdown-like content."""
    if content is None:
        content = ""

    try:
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font('Arial', '', 12)

        sanitized_content = sanitize_content(content)
        sanitized_content = replace_problematic_characters(sanitized_content)

        # Basic Markdown parsing
        lines = sanitized_content.split('\n')

        for line in lines:
            line = line.strip() # Strip leading/trailing whitespace
            if not line: # Skip empty lines
                pdf.ln(5) # Add a small gap for empty lines
                continue

            if line.startswith('#'):
                header_level = min(line.count('#'), 4)
                # Remove markdown bold indicators within headers for cleaner PDF
                header_text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', line.strip('# ').strip())
                font_size = 16 - (header_level - 1) * 2 # Vary font size by header level
                pdf.set_font('Arial', 'B', font_size)
                pdf.multi_cell(0, 10, header_text)
                pdf.ln(2) # Add space after header
                pdf.set_font('Arial', '', 12) # Reset font
            elif line.startswith('* ') or line.startswith('- '): # Basic list items
                 list_item_text = line[2:].strip()
                 # Replace the bullet point character explicitly with a hyphen for multi_cell compatibility
                 list_item_text = list_item_text.replace('\u2022', '-')
                 pdf.set_font('Arial', '', 12)
                 pdf.multi_cell(0, 10, f"- {list_item_text}") # Use hyphen for bullet points in PDF
            else:
                # Handle inline formatting (bold, italic, links)
                # Ensure the regex correctly handles nested markdown if needed, but current seems basic
                parts = re.split(r'(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|\[.*?\]\(.*?\)|\([^ ]+?\))', line)
                for part in parts:
                    if not part:
                        continue
                    if re.match(r'\*\*\*.*?\*\*\*', part):  # Bold Italic
                        text = part.strip('***')
                        pdf.set_font('Arial', 'BI', 12)
                        pdf.write(10, text)
                    elif re.match(r'\*\*.*?\*\*', part):  # Bold
                        text = part.strip('**')
                        pdf.set_font('Arial', 'B', 12)
                        pdf.write(10, text)
                    elif re.match(r'\*.*?\*', part):  # Italic
                        text = part.strip('*')
                        pdf.set_font('Arial', 'I', 12)
                        pdf.write(10, text)
                    elif re.match(r'\[.*?\]\(.*?\)', part):  # Markdown-style link
                        match = re.search(r'\[(.*?)\]\((.*?)\)', part)
                        if match:
                            display_text = match.group(1)
                            url = match.group(2)
                            pdf.set_text_color(0, 0, 255)  # Set text color to blue
                            pdf.set_font('', 'U') # Underline
                            pdf.write(10, display_text, url)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            pdf.set_font('Arial', '', 12) # Reset font
                        else:
                             pdf.write(10, part) # Write as plain text if regex fails
                    elif re.match(r'\([^ ]+?\)', part):  # Plain URL enclosed in parentheses
                        url = part[1:-1] # Remove parentheses
                        pdf.set_text_color(0, 0, 255)  # Set text color to blue
                        pdf.set_font('', 'U') # Underline
                        pdf.write(10, url, url)
                        pdf.set_text_color(0, 0, 0)  # Reset text color
                        pdf.set_font('Arial', '', 12) # Reset font
                    else:
                        pdf.write(10, part)
                    pdf.set_text_color(0, 0, 0) # Reset text color after each part
                    pdf.set_font('Arial', '', 12) # Reset font after each part

                pdf.ln(10) # New line after processing a line of text


        pdf.output(filename)
        return f"PDF generated successfully: {filename}"

    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        return f"Error generating PDF: {e}"

# PDF specific helper functions (moved from PDF Generator cell)
def sanitize_content(content):
    """Sanitizes content for PDF generation, handling encoding errors."""
    if content is None:
        return ""
    try:
        # Use 'utf-8' encoding to handle Unicode characters
        encoded_content = content.encode('utf-8', 'ignore').decode('utf-8')
        return encoded_content
    except UnicodeEncodeError as e:
        logging.warning(f"Encoding error during sanitization: {e}")
        # Attempt a stricter encoding if utf-8 fails on some characters
        sanitized_content = content.encode('ascii', 'ignore').decode('ascii')
        return sanitized_content

def replace_problematic_characters(content):
    """Replaces common problematic Unicode characters for PDF compatibility."""
    if content is None:
        return ""
    
    # Common emoji replacements with text equivalents
    emoji_replacements = {
        '🔍': '[SEARCH]',
        '📊': '[CHART]',
        '📈': '[GROWTH]',
        '📉': '[DECLINE]',
        '💡': '[IDEA]',
        '⚡': '[FAST]',
        '🎯': '[TARGET]',
        '🔑': '[KEY]',
        '📝': '[NOTE]',
        '📋': '[LIST]',
        '🏆': '[AWARD]',
        '🚀': '[ROCKET]',
        '💰': '[MONEY]',
        '📱': '[MOBILE]',
        '💻': '[COMPUTER]',
        '🌟': '[STAR]',
        '✅': '[CHECK]',
        '❌': '[X]',
        '⭐': '[STAR]',
        '🎊': '[CELEBRATION]',
        '🎉': '[PARTY]',
        '🔥': '[FIRE]',
        '💎': '[DIAMOND]',
        '🌍': '[WORLD]',
        '🌎': '[WORLD]',
        '🌏': '[WORLD]',
        '🔒': '[LOCK]',
        '🔓': '[UNLOCK]',
        '📦': '[PACKAGE]',
        '🏁': '[FINISH]'
    }
    
    # Apply emoji replacements first
    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)
    
    # Remove any remaining emojis (fallback for unlisted emojis)
    import re
    # Remove emojis using Unicode ranges
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    content = emoji_pattern.sub(r'', content)
    
    # Standard Unicode character replacements
    replacements = {
        '\u2013': '-',  # en dash to hyphen
        '\u2014': '--',  # em dash to double hyphen
        '\u2018': "'",  # left single quotation mark to apostrophe
        '\u2019': "'",  # right single quotation mark to apostrophe
        '\u201c': '"',  # left double quotation mark to double quote
        '\u201d': '"',  # right double quotation mark to double quote
        '\u2026': '...',  # horizontal ellipsis
        '\u2010': '-',   # hyphen
        '\u2022': '*',   # bullet
        '\u2122': 'TM',  # TradeMark Symbol
        '\u00A0': ' ',  # Non-breaking space
        '\u200B': ''   # Zero-width space
    }

    for char, replacement in replacements.items():
        content = content.replace(char, replacement)

    return content

logging.info("utils.py loaded with utility functions.")
