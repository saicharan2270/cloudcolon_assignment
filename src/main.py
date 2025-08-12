"""
Main script for the automated resume screening system.
"""

import os
import sys
import logging
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from resume_parser import ResumeParser
from job_matcher import JobMatcher
from duplicate_detector import DuplicateDetector
from utils import save_results, load_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResumeScreeningPipeline:
    """
    Main pipeline for automated resume screening.
    """
    
    def __init__(self, job_description_path: str = None):
        """
        Initialize the screening pipeline.
        
        Args:
            job_description_path (str): Path to job description file
        """
        self.parser = ResumeParser()
        self.matcher = JobMatcher()
        self.duplicate_detector = DuplicateDetector()
        
        # Load job description
        if job_description_path and os.path.exists(job_description_path):
            with open(job_description_path, 'r', encoding='utf-8') as f:
                self.job_description = f.read()
        else:
            self.job_description = None
    
    def process_resumes(self, resume_directory: str) -> Dict[str, Any]:
        """
        Process all resumes in a directory.
        
        Args:
            resume_directory (str): Directory containing resumes
            
        Returns:
            Dict[str, Any]: Processing results
        """
        logger.info(f"Processing resumes from: {resume_directory}")
        
        # Parse resumes
        parsed_resumes = self.parser.parse_directory(resume_directory)
        logger.info(f"Successfully parsed {len(parsed_resumes)} resumes")
        
        if not parsed_resumes:
            logger.warning("No resumes found or successfully parsed")
            return {}
        
        # Detect duplicates
        duplicate_groups = self.duplicate_detector.detect_duplicates(parsed_resumes)
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        
        # Get unique resumes (remove duplicates)
        unique_resumes = self.duplicate_detector.get_unique_resumes(parsed_resumes)
        logger.info(f"Processing {len(unique_resumes)} unique resumes")
        
        # Rank resumes
        ranked_resumes = self.matcher.rank_resumes(unique_resumes)
        
        # Get top candidates
        top_candidates = self.matcher.get_top_candidates(ranked_resumes, top_n=20)
        
        # Generate comprehensive report
        report = self.matcher.generate_matching_report(ranked_resumes)
        
        # Add duplicate analysis
        duplicate_stats = self.duplicate_detector.analyze_duplicate_statistics(parsed_resumes)
        
        # Compile final results
        results = {
            'processing_summary': {
                'total_resumes_processed': len(parsed_resumes),
                'unique_resumes': len(unique_resumes),
                'duplicate_groups_found': len(duplicate_groups),
                'duplicate_percentage': duplicate_stats.get('duplicate_percentage', 0)
            },
            'top_candidates': top_candidates,
            'ranking_report': report,
            'duplicate_analysis': duplicate_stats,
            'all_ranked_resumes': ranked_resumes
        }
        
        return results
    
    def analyze_single_resume(self, resume_path: str) -> Dict[str, Any]:
        """
        Analyze a single resume.
        
        Args:
            resume_path (str): Path to the resume file
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info(f"Analyzing single resume: {resume_path}")
        
        # Parse resume
        parsed_resume = self.parser.parse_file(resume_path)
        if not parsed_resume:
            logger.error(f"Failed to parse resume: {resume_path}")
            return {}
        
        # Calculate relevance scores
        scores = self.matcher.calculate_relevance_score(
            parsed_resume.get('raw_text', ''),
            parsed_resume
        )
        
        # Find similar resumes (if we have others to compare against)
        similar_resumes = []
        
        # Generate improvement suggestions
        suggestions = self.duplicate_detector.suggest_resume_improvements(
            parsed_resume, 
            self.matcher.job_description
        )
        
        results = {
            'resume_data': parsed_resume,
            'relevance_scores': scores,
            'similar_resumes': similar_resumes,
            'improvement_suggestions': suggestions
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate a comprehensive report from results.
        
        Args:
            results (Dict[str, Any]): Processing results
            output_path (str): Path to save report
            
        Returns:
            str: Path to saved report
        """
        if not results:
            logger.warning("No results to generate report from")
            return ""
        
        # Save results to JSON
        report_path = save_results(results, output_path)
        
        # Print summary
        self._print_summary(results)
        
        return report_path
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of the results."""
        if not results:
            return
        
        summary = results.get('processing_summary', {})
        print("\n" + "="*50)
        print("RESUME SCREENING SUMMARY")
        print("="*50)
        print(f"Total resumes processed: {summary.get('total_resumes_processed', 0)}")
        print(f"Unique resumes: {summary.get('unique_resumes', 0)}")
        print(f"Duplicate groups found: {summary.get('duplicate_groups_found', 0)}")
        print(f"Duplicate percentage: {summary.get('duplicate_percentage', 0):.1f}%")
        
        # Top candidates
        top_candidates = results.get('top_candidates', [])
        if top_candidates:
            print(f"\nTop 5 Candidates:")
            for i, candidate in enumerate(top_candidates[:5], 1):
                name = candidate.get('name', 'Unknown')
                score = candidate.get('overall_score', 0)
                classification = candidate.get('classification', 'Unknown')
                print(f"{i}. {name} - {score:.1%} ({classification})")
        
        # Classification distribution
        ranking_report = results.get('ranking_report', {})
        classification_dist = ranking_report.get('classification_distribution', {})
        if classification_dist:
            print(f"\nClassification Distribution:")
            for classification, count in classification_dist.items():
                print(f"  {classification}: {count}")
        
        print("="*50)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Automated Resume Screening System')
    parser.add_argument('--resume-dir', type=str, default='data/sample_resumes',
                       help='Directory containing resumes to process')
    parser.add_argument('--job-description', type=str, default='data/job_description.txt',
                       help='Path to job description file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--single-resume', type=str, default=None,
                       help='Analyze a single resume file')
    parser.add_argument('--generate-samples', action='store_true',
                       help='Generate sample resumes for testing')
    
    args = parser.parse_args()
    
    # Generate sample resumes if requested
    if args.generate_samples:
        from generate_samples import main as generate_samples
        generate_samples()
        return
    
    # Initialize pipeline
    pipeline = ResumeScreeningPipeline(args.job_description)
    
    # Process resumes
    if args.single_resume:
        # Analyze single resume
        if not os.path.exists(args.single_resume):
            logger.error(f"Resume file not found: {args.single_resume}")
            return
        
        results = pipeline.analyze_single_resume(args.single_resume)
        if results:
            pipeline.generate_report(results, args.output)
    else:
        # Process directory of resumes
        if not os.path.exists(args.resume_dir):
            logger.error(f"Resume directory not found: {args.resume_dir}")
            return
        
        results = pipeline.process_resumes(args.resume_dir)
        if results:
            pipeline.generate_report(results, args.output)

if __name__ == "__main__":
    main() 