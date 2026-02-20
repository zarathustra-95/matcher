import logging
import sys
import re
import unicodedata

# Default logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("textreuse")

def normalize_text(text: str) -> str:
    """Normalize text for processing: NFC, de-hyphenation, whitespace collapse."""
    if not text:
        return ""
    
    # 1. Unicode Normalization
    text = unicodedata.normalize('NFC', text)
    
    # 2. De-hyphenation: Join words split by hyphen and whitespace/newline
    # e.g., "pa- ciencia" -> "paciencia", "con- \n tinuar" -> "continuar"
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # 3. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_for_matching(text: str) -> str:
    """Aggressive cleaning for shingles: lowercase, remove punctuation."""
    text = normalize_text(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while avoiding common abbreviations and initials."""
    if not text:
        return []
    
    # 1. Protect common abbreviations by temporarily replacing their periods
    # Focus on those followed by a space
    protected = text
    # Protect initials like "M. ", "A. "
    protected = re.sub(r'\b([A-Z])\.\s', r'\1_DOT_SPACE_', protected)
    # Protect honorifics
    for abb in ['Sr', 'Sra', 'Dr', 'Dra', 'Monsieur', 'Prof']:
        protected = re.sub(rf'\b{abb}\.\s', rf'{abb}_DOT_SPACE_', protected)
    
    # 2. Split on remaining sentence boundaries
    # Split on period/exclamation/question followed by space or newline
    # Also split on double+ newlines
    sents = re.split(r'(?<=[.!?])\s+|\n{2,}', protected)
    
    # 3. Restore periods and clean up
    final_sents = []
    for s in sents:
        s = s.replace('_DOT_SPACE_', '. ')
        s = s.strip()
        if len(s) > 5:
            final_sents.append(s)
            
    return final_sents

def clean_input_text(text: str) -> str:
    """
    Clean raw input text by removing metadata headers, page breaks, 
    and OCR/editorial artifacts (e.g. parenthesized numbers).
    """
    if not text:
        return ""
    
    # 1. Remove Header Block (TITLE: ... ---)
    # Match from start of string until the first separator line (dashes)
    # Using specific markers to avoid false positives if file doesn't have this header
    if text.startswith("TITLE:"):
        text = re.sub(r'\ATITLE:.*?(?:\r?\n)+-{4,}(?:\r?\n)+', '', text, flags=re.DOTALL)
        
    # 2. Remove Page Breaks
    text = re.sub(r'--- PAGE BREAK ---', '', text)
    
    # 3. Remove parenthesized numbers like (10), (11 [missing closing]
    # Matches (10) or (11 followed by space
    text = re.sub(r'\(\d+(?:\)|\s)', ' ', text)
    
    # 4. Remove number + paren at start of line (e.g. "24) ") commonly page numbers
    text = re.sub(r'^\s*\d+\)\s', '', text, flags=re.MULTILINE)
    
    return text
