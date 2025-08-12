"""
Job matching module using NLP techniques to rank resumes against job descriptions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using TF-IDF only.")

from utils import clean_text, classify_relevance, format_score

logger = logging.getLogger(__name__)

class JobMatcher:
    """
    Matches resumes against job descriptions using NLP techniques.
    """
    
    def __init__(self, job_description: str = None):
        """
        Initialize the job matcher.
        
        Args:
            job_description (str): Job description text
        """
        self.job_description = job_description or self._load_default_jd()
        self.jd_cleaned = clean_text(self.job_description)
        
        # Initialize models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Using BERT embeddings for matching")
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Pre-compute job description vectors
        self._prepare_jd_vectors()
    
    def _load_default_jd(self) -> str:
        """Load default Salesforce Engineer job description."""
        try:
            with open('data/job_description.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Job description file not found. Using default.")
            return """
            Salesforce Engineer with experience in Apex, Visualforce, Lightning Web Components,
            SOQL, SOSL, REST APIs, SOAP APIs, JavaScript, HTML, CSS, Git, Agile methodologies,
            testing frameworks, and Salesforce certifications.
            """
    
    def _prepare_jd_vectors(self):
        """Pre-compute vectors for job description."""
        # TF-IDF vector
        self.jd_tfidf_vector = self.tfidf_vectorizer.fit_transform([self.jd_cleaned])
        
        # BERT vector
        if self.sentence_model:
            self.jd_bert_vector = self.sentence_model.encode([self.jd_cleaned])
    
    def calculate_relevance_score(self, resume_text: str, resume_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate relevance score between resume and job description.
        
        Args:
            resume_text (str): Cleaned resume text
            resume_data (Dict[str, Any]): Structured resume data
            
        Returns:
            Dict[str, Any]: Relevance scores and analysis
        """
        if not resume_text:
            return {
                'tfidf_score': 0.0,
                'bert_score': 0.0,
                'skill_score': 0.0,
                'experience_score': 0.0,
                'certification_score': 0.0,
                'overall_score': 0.0,
                'classification': 'Irrelevant'
            }
        
        # Calculate different similarity scores
        scores = {}
        
        # TF-IDF similarity
        resume_tfidf = self.tfidf_vectorizer.transform([resume_text])
        tfidf_similarity = cosine_similarity(resume_tfidf, self.jd_tfidf_vector)[0][0]
        scores['tfidf_score'] = float(tfidf_similarity)
        
        # BERT similarity
        if self.sentence_model:
            resume_bert = self.sentence_model.encode([resume_text])
            bert_similarity = cosine_similarity(resume_bert, self.jd_bert_vector)[0][0]
            scores['bert_score'] = float(bert_similarity)
        else:
            scores['bert_score'] = 0.0
        
        # Skill-based scoring
        skill_score = self._calculate_skill_score(resume_data) if resume_data else 0.0
        scores['skill_score'] = skill_score
        
        # Experience scoring
        experience_score = self._calculate_experience_score(resume_data) if resume_data else 0.0
        scores['experience_score'] = experience_score
        
        # Certification scoring
        certification_score = self._calculate_certification_score(resume_data) if resume_data else 0.0
        scores['certification_score'] = certification_score
        
        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(scores)
        scores['overall_score'] = overall_score
        scores['classification'] = classify_relevance(overall_score)
        
        return scores
    
    def _calculate_skill_score(self, resume_data: Dict[str, Any]) -> float:
        """Calculate score based on relevant skills."""
        if not resume_data or 'skills' not in resume_data:
            return 0.0
        
        # Salesforce-specific skills with weights
        skill_weights = {
            'apex': 0.9,
            'visualforce': 0.8,
            'lightning': 0.8,
            'soql': 0.7,
            'sosl': 0.7,
            'salesforce': 0.6,
            'javascript': 0.6,
            'html': 0.5,
            'css': 0.5,
            'python': 0.4,
            'java': 0.4,
            'sql': 0.5,
            'rest': 0.6,
            'soap': 0.6,
            'api': 0.6,
            'git': 0.5,
            'agile': 0.4,
            'scrum': 0.4,
            'testing': 0.5,
            'integration': 0.6,
            'cpq': 0.7,
            'service cloud': 0.7,
            'marketing cloud': 0.7,
            'heroku': 0.6,
            'dx': 0.6
        }
        
        skills = [skill.lower() for skill in resume_data['skills']]
        total_score = 0.0
        max_possible = sum(skill_weights.values())
        
        for skill in skills:
            for skill_key, weight in skill_weights.items():
                if skill_key in skill or skill in skill_key:
                    total_score += weight
                    break
        
        return min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _calculate_experience_score(self, resume_data: Dict[str, Any]) -> float:
        """Calculate score based on years of experience."""
        if not resume_data or 'years_experience' not in resume_data:
            return 0.0
        
        years = resume_data['years_experience']
        if years is None:
            return 0.0
        
        # Score based on experience requirements (3+ years preferred)
        if years >= 5:
            return 1.0
        elif years >= 3:
            return 0.8
        elif years >= 2:
            return 0.6
        elif years >= 1:
            return 0.4
        else:
            return 0.2
    
    def _calculate_certification_score(self, resume_data: Dict[str, Any]) -> float:
        """Calculate score based on Salesforce certifications."""
        if not resume_data or 'certifications' not in resume_data:
            return 0.0
        
        certifications = [cert.lower() for cert in resume_data['certifications']]
        
        # Certification weights
        cert_weights = {
            'platform developer ii': 1.0,
            'platform developer i': 0.8,
            'administrator': 0.7,
            'sales cloud consultant': 0.7,
            'service cloud consultant': 0.7,
            'cpq specialist': 0.6
        }
        
        total_score = 0.0
        for cert in certifications:
            for cert_key, weight in cert_weights.items():
                if cert_key in cert:
                    total_score += weight
                    break
        
        return min(total_score, 1.0)
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = {
            'tfidf_score': 0.3,
            'bert_score': 0.2,
            'skill_score': 0.25,
            'experience_score': 0.15,
            'certification_score': 0.1
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for score_name, weight in weights.items():
            if score_name in scores:
                overall_score += scores[score_name] * weight
                total_weight += weight
        
        return overall_score / total_weight if total_weight > 0 else 0.0
    
    def rank_resumes(self, resumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank resumes by relevance to job description.
        
        Args:
            resumes (List[Dict[str, Any]]): List of parsed resumes
            
        Returns:
            List[Dict[str, Any]]: Ranked resumes with scores
        """
        ranked_resumes = []
        
        for resume in resumes:
            resume_text = resume.get('raw_text', '')
            scores = self.calculate_relevance_score(resume_text, resume)
            
            # Add scores to resume data
            resume_with_scores = resume.copy()
            resume_with_scores.update(scores)
            
            ranked_resumes.append(resume_with_scores)
        
        # Sort by overall score (descending)
        ranked_resumes.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        return ranked_resumes
    
    def get_top_candidates(self, resumes: List[Dict[str, Any]], top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get top N candidates based on relevance.
        
        Args:
            resumes (List[Dict[str, Any]]): List of parsed resumes
            top_n (int): Number of top candidates to return
            
        Returns:
            List[Dict[str, Any]]: Top N candidates
        """
        ranked_resumes = self.rank_resumes(resumes)
        return ranked_resumes[:top_n]
    
    def analyze_skills_distribution(self, resumes: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze skill distribution across all resumes.
        
        Args:
            resumes (List[Dict[str, Any]]): List of parsed resumes
            
        Returns:
            Dict[str, int]: Skill frequency count
        """
        skill_counts = {}
        
        for resume in resumes:
            skills = resume.get('skills', [])
            for skill in skills:
                skill_lower = skill.lower()
                skill_counts[skill_lower] = skill_counts.get(skill_lower, 0) + 1
        
        return skill_counts
    
    def generate_matching_report(self, resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive matching report.
        
        Args:
            resumes (List[Dict[str, Any]]): List of parsed resumes
            
        Returns:
            Dict[str, Any]: Matching analysis report
        """
        if not resumes:
            return {}
        
        ranked_resumes = self.rank_resumes(resumes)
        
        # Calculate statistics
        scores = [r.get('overall_score', 0) for r in ranked_resumes]
        
        report = {
            'total_resumes': len(resumes),
            'top_candidates': ranked_resumes[:20],
            'score_statistics': {
                'mean_score': np.mean(scores),
                'median_score': np.median(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            },
            'classification_distribution': self._get_classification_distribution(ranked_resumes),
            'skill_distribution': self.analyze_skills_distribution(resumes),
            'experience_distribution': self._get_experience_distribution(resumes),
            'certification_distribution': self._get_certification_distribution(resumes)
        }
        
        return report
    
    def _get_classification_distribution(self, resumes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of relevance classifications."""
        distribution = {}
        for resume in resumes:
            classification = resume.get('classification', 'Unknown')
            distribution[classification] = distribution.get(classification, 0) + 1
        return distribution
    
    def _get_experience_distribution(self, resumes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of years of experience."""
        distribution = {}
        for resume in resumes:
            years = resume.get('years_experience')
            if years is not None:
                if years >= 5:
                    category = "5+ years"
                elif years >= 3:
                    category = "3-4 years"
                elif years >= 1:
                    category = "1-2 years"
                else:
                    category = "Less than 1 year"
                distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _get_certification_distribution(self, resumes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of certifications."""
        distribution = {}
        for resume in resumes:
            certs = resume.get('certifications', [])
            for cert in certs:
                cert_lower = cert.lower()
                distribution[cert_lower] = distribution.get(cert_lower, 0) + 1
        return distribution 
