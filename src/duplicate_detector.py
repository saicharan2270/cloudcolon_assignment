"""
Duplicate detection module for identifying similar resumes.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from utils import clean_text, calculate_text_similarity

logger = logging.getLogger(__name__)

class DuplicateDetector:
    """
    Detects duplicate or similar resumes using various similarity measures.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize the duplicate detector.
        
        Args:
            similarity_threshold (float): Threshold for considering resumes as duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def detect_duplicates(self, resumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect duplicate resumes in the list.
        
        Args:
            resumes (List[Dict[str, Any]]): List of parsed resumes
            
        Returns:
            List[Dict[str, Any]]: List of duplicate groups
        """
        if len(resumes) < 2:
            return []
        
        # Extract text content for comparison
        texts = [resume.get('raw_text', '') for resume in resumes]
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(cleaned_texts)
        
        # Find duplicate groups
        duplicate_groups = self._find_duplicate_groups(similarity_matrix, resumes)
        
        return duplicate_groups
    
    def _calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between all texts.
        
        Args:
            texts (List[str]): List of cleaned texts
            
        Returns:
            np.ndarray: Similarity matrix
        """
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        # TF-IDF similarity
        if any(text.strip() for text in texts):
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                tfidf_similarity = cosine_similarity(tfidf_matrix)
                similarity_matrix += tfidf_similarity * 0.6  # Weight for TF-IDF
            except Exception as e:
                logger.warning(f"Error calculating TF-IDF similarity: {e}")
        
        # Fuzzy string similarity
        for i in range(n):
            for j in range(i + 1, n):
                if texts[i] and texts[j]:
                    # Calculate various fuzzy similarity measures
                    ratio = fuzz.ratio(texts[i], texts[j]) / 100.0
                    partial_ratio = fuzz.partial_ratio(texts[i], texts[j]) / 100.0
                    token_sort_ratio = fuzz.token_sort_ratio(texts[i], texts[j]) / 100.0
                    token_set_ratio = fuzz.token_set_ratio(texts[i], texts[j]) / 100.0
                    
                    # Average of fuzzy measures
                    fuzzy_similarity = (ratio + partial_ratio + token_sort_ratio + token_set_ratio) / 4.0
                    similarity_matrix[i, j] += fuzzy_similarity * 0.4  # Weight for fuzzy
                    similarity_matrix[j, i] = similarity_matrix[i, j]
        
        return similarity_matrix
    
    def _find_duplicate_groups(self, similarity_matrix: np.ndarray, resumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find groups of duplicate resumes based on similarity matrix.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            resumes (List[Dict[str, Any]]): Original resumes
            
        Returns:
            List[Dict[str, Any]]: Groups of duplicate resumes
        """
        n = len(resumes)
        visited = [False] * n
        duplicate_groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Find all resumes similar to resume i
            similar_indices = []
            for j in range(n):
                if i != j and similarity_matrix[i, j] >= self.similarity_threshold:
                    similar_indices.append(j)
            
            if similar_indices:
                # Create a group with the main resume and its duplicates
                group = {
                    'main_resume': resumes[i],
                    'duplicates': [resumes[j] for j in similar_indices],
                    'similarity_scores': [similarity_matrix[i, j] for j in similar_indices],
                    'average_similarity': np.mean([similarity_matrix[i, j] for j in similar_indices])
                }
                duplicate_groups.append(group)
                
                # Mark all resumes in this group as visited
                visited[i] = True
                for j in similar_indices:
                    visited[j] = True
        
        return duplicate_groups
    
    def find_similar_resumes(self, target_resume: Dict[str, Any], all_resumes: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find resumes similar to a target resume.
        
        Args:
            target_resume (Dict[str, Any]): Target resume to find similar ones for
            all_resumes (List[Dict[str, Any]]): All resumes to search in
            top_k (int): Number of similar resumes to return
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Similar resumes with similarity scores
        """
        target_text = clean_text(target_resume.get('raw_text', ''))
        
        similarities = []
        for resume in all_resumes:
            if resume == target_resume:
                continue
            
            resume_text = clean_text(resume.get('raw_text', ''))
            similarity = self._calculate_pair_similarity(target_text, resume_text)
            similarities.append((resume, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _calculate_pair_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using multiple methods.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Combined similarity score
        """
        if not text1 or not text2:
            return 0.0
        
        # TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            tfidf_similarity = cosine_similarity(tfidf_matrix)[0, 1]
        except Exception:
            tfidf_similarity = 0.0
        
        # Fuzzy string similarity
        ratio = fuzz.ratio(text1, text2) / 100.0
        partial_ratio = fuzz.partial_ratio(text1, text2) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(text1, text2) / 100.0
        token_set_ratio = fuzz.token_set_ratio(text1, text2) / 100.0
        
        fuzzy_similarity = (ratio + partial_ratio + token_sort_ratio + token_set_ratio) / 4.0
        
        # Weighted combination
        combined_similarity = tfidf_similarity * 0.6 + fuzzy_similarity * 0.4
        
        return combined_similarity
    
    def analyze_duplicate_statistics(self, resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze duplicate statistics across all resumes.
        
        Args:
            resumes (List[Dict[str, Any]]): List of parsed resumes
            
        Returns:
            Dict[str, Any]: Duplicate analysis statistics
        """
        duplicate_groups = self.detect_duplicates(resumes)
        
        total_duplicates = sum(len(group['duplicates']) for group in duplicate_groups)
        total_groups = len(duplicate_groups)
        
        # Calculate average similarity scores
        avg_similarities = [group['average_similarity'] for group in duplicate_groups]
        
        statistics = {
            'total_resumes': len(resumes),
            'total_duplicate_groups': total_groups,
            'total_duplicate_resumes': total_duplicates,
            'duplicate_percentage': (total_duplicates / len(resumes)) * 100 if resumes else 0,
            'average_similarity_in_duplicates': np.mean(avg_similarities) if avg_similarities else 0,
            'duplicate_groups': duplicate_groups
        }
        
        return statistics
    
    def get_unique_resumes(self, resumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get list of unique resumes by removing duplicates.
        
        Args:
            resumes (List[Dict[str, Any]]): List of parsed resumes
            
        Returns:
            List[Dict[str, Any]]: List of unique resumes
        """
        duplicate_groups = self.detect_duplicates(resumes)
        
        # Create set of duplicate resume indices
        duplicate_indices = set()
        for group in duplicate_groups:
            for duplicate in group['duplicates']:
                # Find index of duplicate resume
                for i, resume in enumerate(resumes):
                    if resume == duplicate:
                        duplicate_indices.add(i)
                        break
        
        # Return resumes that are not duplicates
        unique_resumes = [resume for i, resume in enumerate(resumes) if i not in duplicate_indices]
        
        return unique_resumes
    
    def suggest_resume_improvements(self, resume: Dict[str, Any], job_description: str) -> List[str]:
        """
        Suggest improvements for a resume based on job description.
        
        Args:
            resume (Dict[str, Any]): Resume to analyze
            job_description (str): Job description text
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        resume_text = resume.get('raw_text', '').lower()
        jd_text = job_description.lower()
        
        # Check for missing key skills
        key_skills = [
            'apex', 'visualforce', 'lightning', 'soql', 'sosl', 'salesforce',
            'javascript', 'html', 'css', 'python', 'java', 'sql', 'rest', 'soap',
            'api', 'git', 'agile', 'scrum', 'testing', 'integration'
        ]
        
        missing_skills = []
        for skill in key_skills:
            if skill in jd_text and skill not in resume_text:
                missing_skills.append(skill.title())
        
        if missing_skills:
            suggestions.append(f"Consider adding these skills: {', '.join(missing_skills)}")
        
        # Check for experience level
        years_exp = resume.get('years_experience')
        if years_exp is not None and years_exp < 3:
            suggestions.append("Consider highlighting more years of experience or relevant projects")
        
        # Check for certifications
        certifications = resume.get('certifications', [])
        if not certifications:
            suggestions.append("Consider adding Salesforce certifications")
        
        # Check for education
        education = resume.get('education', [])
        if not education:
            suggestions.append("Consider adding education information")
        
        return suggestions 
