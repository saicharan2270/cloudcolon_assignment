"""
Resume parsing module for extracting structured data from various file formats.
"""

import os
import re
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Document processing imports
try:
    import PyPDF2
    from docx import Document
except ImportError:
    print("Please install PyPDF2 and python-docx: pip install PyPDF2 python-docx")

# NLP imports
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

from utils import clean_text, extract_emails, extract_phones, extract_years_of_experience

logger = logging.getLogger(__name__)

class ResumeParser:
    """
    Parser for extracting structured information from resumes.
    """
    
    def __init__(self):
        """Initialize the resume parser with NLP models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize matcher for named entity recognition
        if self.nlp:
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_matchers()
    
    def _setup_matchers(self):
        """Setup spaCy matchers for extracting specific information."""
        if not self.nlp:
            return
        
        # Education patterns
        education_patterns = [
            [{"LOWER": {"IN": ["bachelor", "master", "phd", "mba", "bs", "ms", "ph.d"]}}],
            [{"LOWER": {"IN": ["university", "college", "institute"]}}],
            [{"LOWER": "degree"}, {"LOWER": "in"}],
        ]
        
        # Skills patterns
        skills_patterns = [
            [{"LOWER": {"IN": ["skills", "technologies", "programming", "languages"]}}],
            [{"LOWER": "proficient"}, {"LOWER": "in"}],
            [{"LOWER": "experience"}, {"LOWER": "with"}],
        ]
        
        # Add patterns to matcher
        for pattern in education_patterns + skills_patterns:
            self.matcher.add("EDUCATION_SKILLS", [pattern])
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a resume file and extract structured information.
        
        Args:
            file_path (str): Path to the resume file
            
        Returns:
            Dict[str, Any]: Extracted resume information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text based on file type
        text = self._extract_text(file_path)
        if not text:
            return {}
        
        # Parse the extracted text
        return self._parse_text(text, file_path)
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from different file formats.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: Extracted text
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
        
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
        
        return text
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading TXT {file_path}: {str(e)}")
                return ""
    
    def _parse_text(self, text: str, file_path: str) -> Dict[str, Any]:
        """
        Parse extracted text to extract structured information.
        
        Args:
            text (str): Raw text from resume
            file_path (str): Original file path
            
        Returns:
            Dict[str, Any]: Structured resume information
        """
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Extract basic information
        result = {
            'file_path': file_path,
            'raw_text': cleaned_text,
            'name': self._extract_name(cleaned_text),
            'email': extract_emails(cleaned_text),
            'phone': extract_phones(cleaned_text),
            'years_experience': extract_years_of_experience(cleaned_text),
            'education': self._extract_education(cleaned_text),
            'skills': self._extract_skills(cleaned_text),
            'experience': self._extract_experience(cleaned_text),
            'certifications': self._extract_certifications(cleaned_text),
            'languages': self._extract_languages(cleaned_text),
        }
        
        return result
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from resume."""
        # Look for name patterns at the beginning of the resume
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 0 and len(line.split()) <= 4:
                # Simple heuristic: name is usually 2-4 words at the top
                if not any(word.lower() in ['resume', 'cv', 'curriculum', 'vitae'] for word in line.split()):
                    return line
        return None
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information."""
        education = []
        
        # Education keywords
        edu_keywords = [
            'bachelor', 'master', 'phd', 'mba', 'bs', 'ms', 'ph.d',
            'university', 'college', 'institute', 'school'
        ]
        
        # Find education section
        lines = text.lower().split('\n')
        in_education = False
        edu_lines = []
        
        for line in lines:
            if any(keyword in line for keyword in ['education', 'academic', 'degree']):
                in_education = True
                continue
            elif in_education and any(keyword in line for keyword in ['experience', 'work', 'employment', 'skills']):
                break
            elif in_education:
                edu_lines.append(line)
        
        # Parse education lines
        for line in edu_lines:
            if any(keyword in line for keyword in edu_keywords):
                education.append({
                    'degree': line.strip(),
                    'institution': '',  # Would need more complex parsing
                    'year': ''
                })
        
        return education
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume using focused approaches."""
        skills = set()  # Use set to avoid duplicates
        text_lower = text.lower()
        
        # Method 1: Look for "Technical Skills and Interests" section (your first resume format)
        skills_start = text_lower.find('technical skills and interests')
        if skills_start == -1:
            skills_start = text_lower.find('skills and interests')
        
        if skills_start != -1:
            # Find where the skills section ends
            skills_end = len(text)
            end_markers = ['positions of responsibility', 'experience', 'achievements', 'awards', 'work experience', 'employment']
            for marker in end_markers:
                marker_pos = text_lower.find(marker, skills_start)
                if marker_pos != -1 and marker_pos < skills_end:
                    skills_end = marker_pos
            
            # Extract the skills section text
            skills_section = text[skills_start:skills_end]
            
            # Define patterns for structured skills sections
            patterns = [
                r'Skills\s+([^\n]+?)(?=\s+Languages|\s+Course Work|\s+Areas of Interest|\s+Soft Skills|\s+Positions of|$)',
                r'Languages\s+([^\n]+?)(?=\s+Frameworks|\s+Course Work|\s+Areas of Interest|\s+Soft Skills|\s+Positions of|$)',
                r'Frameworks\s+([^\n]+?)(?=\s+Course Work|\s+Areas of Interest|\s+Soft Skills|\s+Positions of|$)',
                r'Course Work\s+([^\n]+?)(?=\s+Areas of Interest|\s+Soft Skills|\s+Positions of|$)',
                r'Areas of Interest\s+([^\n]+?)(?=\s+Soft Skills|\s+Positions of|$)',
                r'Soft Skills\s+([^\n]+?)(?=\s+Positions of|$)'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, skills_section, re.IGNORECASE)
                for match in matches:
                    skills_text = match.group(1).strip()
                    self._extract_skills_from_text(skills_text, skills)
            
            return list(skills)  # Return immediately if we found the structured section
        
        # Method 2: Look for dedicated "Skills" sections only (more conservative)
        skill_section_found = False
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Look for skill section headers that are likely to be section titles
            if (line_lower.startswith('technical skills') or 
                line_lower.startswith('skills') or 
                line_lower.startswith('core competencies') or
                line_lower.startswith('technologies') or
                line_lower == 'technical skills' or
                line_lower == 'skills' or
                re.match(r'^skills?\s*[:|-]', line_lower) or
                re.match(r'^technical\s+skills?\s*[:|-]?', line_lower)):
                
                skill_section_found = True
                
                # Check if skills are on the same line
                if ':' in line_stripped:
                    skills_part = line_stripped.split(':', 1)[1].strip()
                    if skills_part:
                        self._extract_skills_from_text(skills_part, skills)
                
                # Look at the next few lines for skills
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Stop if we hit another section
                    next_lower = next_line.lower()
                    if (any(section in next_lower for section in 
                           ['experience', 'education', 'projects', 'work', 'employment', 
                            'achievements', 'certifications', 'summary']) and
                        len(next_line) < 50):  # Likely a section header
                        break
                    
                    # Extract skills from this line
                    self._extract_skills_from_text(next_line, skills)
                
                break  # Found skills section, stop looking
        
        return list(skills)
    
    def _extract_skills_from_text(self, skills_text: str, skills_set: set):
        """Helper method to extract individual skills from a text string."""
        # Split by common delimiters
        skill_items = re.split(r'[,;|/\n]', skills_text)
        
        for skill in skill_items:
            skill = skill.strip()
            if skill and len(skill) > 1:
                # Clean up the skill
                skill = re.sub(r'^[\s\-•●▪▫\*]+|[\s\-•●▪▫\*]+$|\.$', '', skill)
                skill = re.sub(r'\([^)]*\)', '', skill)  # Remove parenthetical content
                skill = skill.strip()
                
                # Filter out common words that aren't skills
                excluded_words = {
                    'and', 'or', 'with', 'the', 'in', 'at', 'of', 'to', 'for', 'on', 'a', 'an',
                    'is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'could',
                    'should', 'may', 'might', 'can', 'must', 'years', 'experience', 'knowledge'
                }
                
                if (skill and len(skill) > 1 and 
                    skill.lower() not in excluded_words and
                    not skill.isdigit() and
                    not re.match(r'^[^a-zA-Z]*$', skill)):
                    skills_set.add(skill)
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience information."""
        experience = []
        
        # Look for experience section
        lines = text.split('\n')
        in_experience = False
        exp_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['experience', 'employment', 'work history']):
                in_experience = True
                continue
            elif in_experience and any(keyword in line.lower() for keyword in ['education', 'skills', 'certifications']):
                break
            elif in_experience:
                exp_lines.append(line)
        
        # Simple parsing - look for date patterns and company names
        for line in exp_lines:
            if re.search(r'\d{4}', line):  # Contains year
                experience.append({
                    'title': line.strip(),
                    'company': '',
                    'duration': '',
                    'description': ''
                })
        
        return experience
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications from resume."""
        certifications = []
        
        # Salesforce certifications
        salesforce_certs = [
            'platform developer i', 'platform developer ii', 'administrator',
            'sales cloud consultant', 'service cloud consultant', 'cpq specialist'
        ]
        
        text_lower = text.lower()
        for cert in salesforce_certs:
            if cert in text_lower:
                certifications.append(cert.title())
        
        return certifications
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract programming languages from resume."""
        languages = []
        
        # Common programming languages
        prog_languages = [
            'python', 'java', 'javascript', 'html', 'css', 'sql', 'apex',
            'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin'
        ]
        
        text_lower = text.lower()
        for lang in prog_languages:
            if lang in text_lower:
                languages.append(lang.title())
        
        return languages
    
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Parse all supported files in a directory.
        
        Args:
            directory_path (str): Path to directory containing resumes
            
        Returns:
            List[Dict[str, Any]]: List of parsed resume information
        """
        results = []
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return results
        
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_ext = Path(filename).suffix.lower()
                if file_ext in supported_extensions:
                    try:
                        parsed_resume = self.parse_file(file_path)
                        if parsed_resume:
                            results.append(parsed_resume)
                            logger.info(f"Successfully parsed: {filename}")
                    except Exception as e:
                        logger.error(f"Error parsing {filename}: {str(e)}")
        
        return results 