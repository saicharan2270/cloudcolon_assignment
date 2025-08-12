"""
Utility functions for the resume screening system.
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def save_results(data: Dict[str, Any], filename: str = None) -> str:
    """
    Save processing results to a JSON file.
    
    Args:
        data (Dict[str, Any]): The data to save
        filename (str, optional): Custom filename. If None, generates timestamped filename
        
    Returns:
        str: Path to the saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resume_screening_results_{timestamp}.json"
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def load_results(filename: str) -> Dict[str, Any]:
    """
    Load processing results from a JSON file.
    
    Args:
        filename (str): Name of the file to load
        
    Returns:
        Dict[str, Any]: The loaded data
    """
    # Look in data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        # Try with .json extension if not provided
        if not filename.endswith('.json'):
            filepath = os.path.join(data_dir, f"{filename}.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Results loaded from: {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise

def get_latest_results() -> Dict[str, Any]:
    """
    Get the most recent results file.
    
    Returns:
        Dict[str, Any]: The most recent results data
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all JSON files in data directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and 'resume_screening_results' in f]
    
    if not json_files:
        raise FileNotFoundError("No results files found in data directory")
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    
    latest_file = json_files[0]
    return load_results(latest_file)

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\@\#\$\&\(\)]', '', text)
    
    # Normalize line breaks
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text (str): Text to search for emails
        
    Returns:
        List[str]: List of found email addresses
    """
    if not text:
        return []
    
    # Email regex pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    emails = re.findall(email_pattern, text)
    return list(set(emails))  # Remove duplicates

def extract_phones(text: str) -> List[str]:
    """
    Extract phone numbers from text.
    
    Args:
        text (str): Text to search for phone numbers
        
    Returns:
        List[str]: List of found phone numbers
    """
    if not text:
        return []
    
    # Phone number patterns
    phone_patterns = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b',  # (123) 456-7890
        r'\b\d{10}\b',  # 1234567890
        r'\b\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # +1-123-456-7890
    ]
    
    phones = []
    for pattern in phone_patterns:
        found = re.findall(pattern, text)
        phones.extend(found)
    
    return list(set(phones))  # Remove duplicates

def extract_years_of_experience(text: str) -> int:
    """
    Extract years of experience from text.
    
    Args:
        text (str): Text to search for experience information
        
    Returns:
        int: Years of experience (0 if not found)
    """
    if not text:
        return 0
    
    # Patterns for years of experience
    patterns = [
        r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:in\s*)?(?:the\s*)?field',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:working\s*)?(?:as|in)',
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                # Return the highest number found
                return max(int(match) for match in matches)
            except ValueError:
                continue
    
    return 0

def classify_relevance(score: float) -> str:
    """
    Classify resume relevance based on score.
    
    Args:
        score (float): Relevance score (0-1)
        
    Returns:
        str: Classification category
    """
    if score >= 0.8:
        return "Highly Relevant"
    elif score >= 0.6:  # 60% and above = Moderate
        return "Moderate"
    elif score >= 0.3:  # Lowered threshold for Low Relevance
        return "Low Relevance"
    else:
        return "Irrelevant"

def format_score(score: float) -> str:
    """
    Format score as percentage string.
    
    Args:
        score (float): Score value (0-1)
        
    Returns:
        str: Formatted percentage string
    """
    return f"{score:.1%}"

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        
    Returns:
        float: Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Clean both texts
    clean_text1 = clean_text(text1)
    clean_text2 = clean_text(text2)
    
    if not clean_text1 or not clean_text2:
        return 0.0
    
    # Simple similarity using set intersection
    words1 = set(clean_text1.lower().split())
    words2 = set(clean_text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union 