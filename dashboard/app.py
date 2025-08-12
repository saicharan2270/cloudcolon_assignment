"""
Streamlit dashboard for the automated resume screening system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import sys
import json
import tempfile
from typing import Dict, List, Any
import base64

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from resume_parser import ResumeParser
from job_matcher import JobMatcher
from duplicate_detector import DuplicateDetector
from utils import save_results, load_results
from job_matcher_fix import add_required_skills_method

# Page configuration
st.set_page_config(
    page_title="Resume Screening Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .score-high { color: #28a745; }
    .score-medium { color: #ffc107; }
    .score-low { color: #dc3545; }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Resume Screening for Salesforce Engineer Hiring</h1>', unsafe_allow_html=True)
    st.markdown("### Resume Analysis & Candidate Evaluation System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📤 Upload & Analyze", "👥 Top Candidates", "🔄 Duplicate Detection"]
    )
    
    # Initialize components
    if 'parser' not in st.session_state:
        st.session_state.parser = ResumeParser()
    if 'matcher' not in st.session_state:
        st.session_state.matcher = JobMatcher()
    if 'duplicate_detector' not in st.session_state:
        st.session_state.duplicate_detector = DuplicateDetector()
    
    # Route to appropriate page
    if page == "🏠 Overview":
        show_overview()
    elif page == "📤 Upload & Analyze":
        show_upload_analyze()
    elif page == "👥 Top Candidates":
        show_top_candidates()
    elif page == "🔄 Duplicate Detection":
        show_duplicate_detection()

def show_upload_analyze():
    """Show upload and analyze page."""
    
    st.subheader("Upload & Analyze Resumes")
    
    # Initialize session state for uploaded files
    if 'uploaded_files_data' not in st.session_state:
        st.session_state.uploaded_files_data = []
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = []
    if 'job_requirements' not in st.session_state:
        st.session_state.job_requirements = {
            'skills': "Apex\nVisualforce\nLightning Web Components\nSOQL\nSOSL\nREST APIs\nJavaScript\nHTML\nCSS\nGit\nAgile",
            'description': "Salesforce Engineer with experience in Apex, Visualforce, Lightning Web Components, SOQL, SOSL, REST APIs, SOAP APIs, JavaScript, HTML, CSS, Git, Agile methodologies, testing frameworks, and Salesforce certifications."
        }
    
    # Job Description Input
    st.write("### Job Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Required Skills**")
        skills_input = st.text_area(
            "Enter required skills (one per line):",
            value=st.session_state.job_requirements['skills'],
            height=150,
            help="Enter each skill on a new line"
        )
        
        # Convert skills to list
        required_skills = [skill.strip() for skill in skills_input.split('\n') if skill.strip()]
        # Update session state
        st.session_state.job_requirements['skills'] = skills_input
    
    with col2:
        st.write("**Job Description**")
        job_description = st.text_area(
            "Enter detailed job description:",
            value=st.session_state.job_requirements['description'],
            height=150,
            help="Enter the complete job description"
        )
        # Update session state
        st.session_state.job_requirements['description'] = job_description
    
    # Resume Upload Section
    st.write("### Upload Resumes")
    
    # Information about file persistence
    with st.expander("ℹ️ How file uploads work"):
        st.write("""
        **File Persistence Feature:**
        - Files you upload are stored in your session and will persist when you navigate between pages
        - You can add multiple batches of files - they will be combined
        - Files are stored in memory and will be available until you clear them or restart the app
        - You can remove individual files or clear all files at once
        - Supported formats: PDF, DOCX, TXT
        """)
    
    # Show previously uploaded files if any
    if st.session_state.uploaded_files_names:
        st.write("**Previously uploaded files:**")
        
        # Add clear all button
        col1, col2 = st.columns([3, 1])
        with col1:
            # Calculate total size
            total_size = sum(file_info.get('size', 0) for file_info in st.session_state.uploaded_files_data)
            total_size_mb = total_size / (1024 * 1024)
            st.write(f"Total files: {len(st.session_state.uploaded_files_names)} ({total_size_mb:.1f} MB)")
        with col2:
            if st.button("Clear All", type="secondary", key="clear_all"):
                st.session_state.uploaded_files_data.clear()
                st.session_state.uploaded_files_names.clear()
                st.rerun()
        
        # Show individual files
        for i, filename in enumerate(st.session_state.uploaded_files_names):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"• {filename}")
            with col2:
                # Show file size
                file_size = st.session_state.uploaded_files_data[i].get('size', 0)
                size_mb = file_size / (1024 * 1024)
                st.write(f"{size_mb:.1f} MB")
            with col3:
                if st.button(f"Remove", key=f"remove_{i}"):
                    # Remove file from session state
                    st.session_state.uploaded_files_data.pop(i)
                    st.session_state.uploaded_files_names.pop(i)
                    st.rerun()
        st.write("---")
    
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files"
    )
    
    # Handle new file uploads
    if uploaded_files:
        st.write(f"**New files to upload:**")
        for file in uploaded_files:
            st.write(f"• {file.name}")
        
        # Add to session state button
        if st.button("Add Files to Session", type="secondary"):
            added_count = 0
            skipped_count = 0
            for uploaded_file in uploaded_files:
                # Check if file already exists
                if uploaded_file.name not in st.session_state.uploaded_files_names:
                    # Store file data in session state
                    file_data = uploaded_file.getbuffer()
                    file_size = len(file_data)
                    st.session_state.uploaded_files_data.append({
                        'name': uploaded_file.name,
                        'data': file_data,
                        'type': uploaded_file.type,
                        'size': file_size
                    })
                    st.session_state.uploaded_files_names.append(uploaded_file.name)
                    added_count += 1
                else:
                    skipped_count += 1
            
            if added_count > 0:
                st.success(f"Added {added_count} new files to session!")
            if skipped_count > 0:
                st.warning(f"Skipped {skipped_count} duplicate files.")
            st.rerun()
    
    # Show total files and process button
    total_files = len(st.session_state.uploaded_files_names)
    if total_files > 0:
        st.write(f"**Total files in session: {total_files}**")
        
        # Process button
        if st.button("Process Resumes", type="primary"):
            with st.spinner("Processing resumes..."):
                # Save uploaded files temporarily
                temp_dir = tempfile.mkdtemp()
                saved_files = []
                
                for file_info in st.session_state.uploaded_files_data:
                    file_path = os.path.join(temp_dir, file_info['name'])
                    with open(file_path, 'wb') as f:
                        f.write(file_info['data'])
                    saved_files.append(file_path)
                
                # Process resumes
                parsed_resumes = []
                for file_path in saved_files:
                    try:
                        resume_data = st.session_state.parser.parse_file(file_path)
                        if resume_data:
                            # Use the original file name instead of parsed name if it's None or empty
                            original_filename = os.path.basename(file_path)
                            if not resume_data.get('name') or resume_data.get('name') == 'Unknown':
                                resume_data['name'] = original_filename
                            parsed_resumes.append(resume_data)
                    except Exception as e:
                        st.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                
                if parsed_resumes:
                    # Update job matcher with custom requirements and skills
                    st.session_state.matcher = JobMatcher(job_description)
                    # Apply custom skill scoring fix
                    st.write(f"**DEBUG: Required skills being used:** {required_skills}")
                    
                    # Rank resumes with custom skill scoring
                    ranked_resumes = []
                    for resume in parsed_resumes:
                        resume_text = resume.get('raw_text', '')
                        scores = st.session_state.matcher.calculate_relevance_score(resume_text, resume)
                        
                        # Override skill score with custom calculation
                        if resume.get('skills') and required_skills:
                            resume_skills = [skill.lower() for skill in resume['skills']]
                            required_skills_lower = [skill.lower().strip() for skill in required_skills]
                            
                            matched_count = 0
                            for req_skill in required_skills_lower:
                                for resume_skill in resume_skills:
                                    if req_skill in resume_skill or resume_skill in req_skill:
                                        matched_count += 1
                                        break
                            
                            custom_skill_score = matched_count / len(required_skills_lower) if required_skills_lower else 0.0
                            st.write(f"**DEBUG: Resume {resume.get('name', 'Unknown')}:**")
                            st.write(f"  - Resume skills: {resume_skills}")
                            st.write(f"  - Required skills: {required_skills_lower}")
                            st.write(f"  - Matched: {matched_count}/{len(required_skills_lower)} = {custom_skill_score:.1%}")
                            st.write(f"  - Original skill score: {scores.get('skill_score', 0):.1%}")
                            
                            # Recalculate overall score with new skill score
                            scores['skill_score'] = custom_skill_score
                            # Recalculate overall score
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
                            
                            scores['overall_score'] = overall_score / total_weight if total_weight > 0 else 0.0
                            st.write(f"  - New overall score: {scores['overall_score']:.1%}")
                            
                            # UPDATE: Recalculate classification with new score
                            from utils import classify_relevance
                            scores['classification'] = classify_relevance(scores['overall_score'])
                            st.write(f"  - Updated classification: {scores['classification']}")
                        
                        # Add scores to resume data
                        resume_with_scores = resume.copy()
                        resume_with_scores.update(scores)
                        ranked_resumes.append(resume_with_scores)
                    
                    # Sort by overall score (descending)
                    ranked_resumes.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
                    top_candidates = ranked_resumes[:20]
                    
                    # Generate reports
                    report = st.session_state.matcher.generate_matching_report(ranked_resumes)
                    duplicate_stats = st.session_state.duplicate_detector.analyze_duplicate_statistics(parsed_resumes)
                    
                    # Store results in session state
                    st.session_state.current_data = {
                        'processing_summary': {
                            'total_resumes_processed': len(parsed_resumes),
                            'unique_resumes': len(parsed_resumes),
                            'duplicate_groups_found': len(duplicate_stats.get('duplicate_groups', [])),
                            'duplicate_percentage': duplicate_stats.get('duplicate_percentage', 0)
                        },
                        'top_candidates': top_candidates,
                        'ranking_report': report,
                        'duplicate_analysis': duplicate_stats,
                        'all_ranked_resumes': ranked_resumes,
                        'required_skills': required_skills,
                        'job_description': job_description
                    }
                    
                    # Save results
                    save_results(st.session_state.current_data)
                    
                    st.success(f"Successfully processed {len(parsed_resumes)} resumes!")
                    st.info("Navigate to other pages to view detailed analysis.")
                else:
                    st.error("No resumes were successfully processed.")
                
                # Clean up temp files
                for file_path in saved_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                os.rmdir(temp_dir)
    else:
        st.info("No files uploaded yet. Please upload some resume files to get started.")

def show_overview():
    """Show overview page."""
    
    # Check if we have current data
    data = get_current_data()
    
    if not data:
        st.warning("No data available. Please upload and process resumes first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    summary = data.get('processing_summary', {})
    
    with col1:
        st.metric(
            "Total Resumes",
            summary.get('total_resumes_processed', 0),
            help="Total number of resumes processed"
        )
    
    with col2:
        st.metric(
            "Unique Resumes",
            summary.get('unique_resumes', 0),
            help="Number of unique resumes after removing duplicates"
        )
    
    with col3:
        duplicate_pct = summary.get('duplicate_percentage', 0)
        st.metric(
            "Duplicate Rate",
            f"{duplicate_pct:.1f}%",
            help="Percentage of duplicate resumes found"
        )
    
    with col4:
        top_candidates = data.get('top_candidates', [])
        highly_relevant = sum(1 for c in top_candidates if c.get('classification') == 'Highly Relevant')
        st.metric(
            "Highly Relevant",
            highly_relevant,
            help="Number of highly relevant candidates"
        )
    
    # Required Skills Display
    if data.get('required_skills'):
        st.write("### Required Skills")
        skills = data['required_skills']
        skills_text = " • ".join(skills)
        st.info(f"**Skills being matched:** {skills_text}")
    
    # Score distribution chart
    st.subheader("Score Distribution")
    
    if data.get('all_ranked_resumes'):
        scores = [r.get('overall_score', 0) for r in data['all_ranked_resumes']]
        
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Distribution of Relevance Scores",
            labels={'x': 'Relevance Score', 'y': 'Number of Candidates'},
            color_discrete_sequence=['#1f77b4']
        )
        
        fig.update_layout(
            xaxis_title="Relevance Score",
            yaxis_title="Number of Candidates",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification distribution
    st.subheader("Classification Distribution")
    
    ranking_report = data.get('ranking_report', {})
    classification_dist = ranking_report.get('classification_distribution', {})
    
    if classification_dist:
        fig = px.pie(
            values=list(classification_dist.values()),
            names=list(classification_dist.keys()),
            title="Resume Classification Distribution",
            color_discrete_sequence=['#28a745', '#ffc107', '#dc3545', '#6c757d']
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Processing Summary**")
        st.write(f"• Total resumes processed: {summary.get('total_resumes_processed', 0)}")
        st.write(f"• Unique resumes: {summary.get('unique_resumes', 0)}")
        st.write(f"• Duplicate groups found: {summary.get('duplicate_groups_found', 0)}")
        st.write(f"• Average processing time: ~2-3 seconds per resume")
    
    with col2:
        st.success("**Quality Metrics**")
        if data.get('top_candidates'):
            avg_score = np.mean([c.get('overall_score', 0) for c in data['top_candidates']])
            st.write(f"• Average top candidate score: {avg_score:.1%}")
            st.write(f"• Highest score: {max([c.get('overall_score', 0) for c in data['top_candidates']]):.1%}")
            st.write(f"• Lowest score: {min([c.get('overall_score', 0) for c in data['top_candidates']]):.1%}")

def get_current_data():
    """Get current data from session state or load from file."""
    if hasattr(st.session_state, 'current_data') and st.session_state.current_data:
        return st.session_state.current_data
    
    # Try to load from file
    try:
        data = load_results("resume_screening_results.json")
        return data
    except:
        return {}



def show_top_candidates():
    """Show top candidates page."""
    
    data = get_current_data()
    
    if not data:
        st.warning("No data available. Please upload and process resumes first.")
        return
    
    st.subheader("👥 Top Candidates")
    
    top_candidates = data.get('top_candidates', [])
    
    if not top_candidates:
        st.warning("No top candidates found.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.1)
    
    with col2:
        classification_filter = st.selectbox(
            "Classification Filter",
            ["All", "Highly Relevant", "Moderate", "Low Relevance", "Irrelevant"]
        )
    
    # Filter candidates
    filtered_candidates = []
    
    # Debug: Show available classifications
    available_classifications = set()
    for candidate in top_candidates:
        classification = candidate.get('classification', 'Unknown')
        available_classifications.add(classification)
    
    # Show debug info
    with st.expander("🔍 Debug: Available Classifications"):
        st.write(f"**Found classifications:** {list(available_classifications)}")
        st.write(f"**Selected filter:** {classification_filter}")
        
        # Show count for each classification
        classification_counts = {}
        for candidate in top_candidates:
            classification = candidate.get('classification', 'Unknown')
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        st.write("**Classification counts:**")
        for classification, count in classification_counts.items():
            st.write(f"- {classification}: {count} candidates")
    
    for candidate in top_candidates:
        score = candidate.get('overall_score', 0)
        classification = candidate.get('classification', 'Unknown')
        
        if score >= min_score:
            # More robust classification matching
            if (classification_filter == "All" or 
                classification == classification_filter or
                classification.lower() == classification_filter.lower()):
                filtered_candidates.append(candidate)
    
    # Display candidates
    for i, candidate in enumerate(filtered_candidates, 1):
        with st.expander(f"#{i} {candidate.get('name', 'Unknown')} - {candidate.get('overall_score', 0):.1%}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Contact Information**")
                email = candidate.get('email', [])
                if email:
                    # Clean up email address - remove any non-email text
                    email_text = email[0]
                    # Remove common prefixes that might be incorrectly parsed
                    email_text = email_text.replace('envelpe', '').replace('envelope', '').replace('📧', '').strip()
                    # Ensure it's a valid email format
                    if '@' in email_text and '.' in email_text.split('@')[1]:
                        st.write(f"**Email:** {email_text}")
                    else:
                        st.write(f"**Email:** {email[0]}")  # Show original if cleaning failed
                
                phone = candidate.get('phone', [])
                if phone:
                    st.write(f"**Phone:** {phone[0]}")
                
                # Only show location if it's actually a location, not education
                location = candidate.get('location', '')
                if location and not any(edu_word in location.lower() for edu_word in ['learning', 'university', 'college', 'school', 'degree', 'bachelor', 'master', 'phd']):
                    st.write(f"**Location:** {location}")
                
                st.write("**Experience**")
                years_exp = candidate.get('years_experience', 0)
                st.write(f"**Years:** {years_exp} years")
            
            with col2:
                st.write("**Skills**")
                skills = candidate.get('skills', [])
                if skills:
                    # Match skills from job requirements
                    required_skills = data.get('required_skills', [])
                    if required_skills:
                        # Find matching skills between resume and job requirements
                        matching_skills = []
                        for skill in skills:
                            skill_lower = skill.lower()
                            for req_skill in required_skills:
                                req_skill_lower = req_skill.lower()
                                # Check for exact match, partial match, or related terms
                                if (req_skill_lower in skill_lower or 
                                    skill_lower in req_skill_lower or
                                    any(word in skill_lower for word in req_skill_lower.split()) or
                                    any(word in req_skill_lower for word in skill_lower.split())):
                                    matching_skills.append(skill)
                                    break
                        # Remove duplicates while preserving order
                        matching_skills = list(dict.fromkeys(matching_skills))
                        
                        if matching_skills:
                            st.write("**Matching Skills:**")
                            st.write(", ".join(matching_skills))  # Show ALL matching skills
                        else:
                            st.write("**All Skills:**")
                            st.write(", ".join(skills[:5]))
                    else:
                        st.write(", ".join(skills[:5]))
                else:
                    st.write("No skills found")
                
                st.write("**Certifications**")
                certs = candidate.get('certifications', [])
                if certs:
                    st.write(", ".join(certs))
                else:
                    st.write("None")
                
                st.write("**Classification**")
                classification = candidate.get('classification', 'Unknown')
                if classification == 'Highly Relevant':
                    st.markdown(f'<span class="score-high">🟢 {classification}</span>', unsafe_allow_html=True)
                elif classification == 'Moderate':
                    st.markdown(f'<span class="score-medium">🟡 {classification}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="score-low">🔴 {classification}</span>', unsafe_allow_html=True)
            
            # Score breakdown - simplified without TF-IDF and BERT
            st.write("**Score Breakdown**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall", f"{candidate.get('overall_score', 0):.1%}")
            with col2:
                st.metric("Skills", f"{candidate.get('skill_score', 0):.1%}")
            with col3:
                st.metric("Experience", f"{candidate.get('experience_score', 0):.1%}")





def show_duplicate_detection():
    """Show duplicate detection analysis."""
    
    data = get_current_data()
    
    if not data:
        st.warning("No data available. Please upload and process resumes first.")
        return
    
    st.subheader("🔄 Duplicate Detection Analysis")
    
    duplicate_analysis = data.get('duplicate_analysis', {})
    
    if not duplicate_analysis:
        st.warning("No duplicate analysis data available.")
        return
    
    # Duplicate statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Duplicate Groups",
            duplicate_analysis.get('total_duplicate_groups', 0)
        )
    
    with col2:
        st.metric(
            "Total Duplicate Resumes",
            duplicate_analysis.get('total_duplicate_resumes', 0)
        )
    
    with col3:
        duplicate_pct = duplicate_analysis.get('duplicate_percentage', 0)
        st.metric(
            "Duplicate Percentage",
            f"{duplicate_pct:.1f}%"
        )
    
    # Duplicate groups details
    duplicate_groups = duplicate_analysis.get('duplicate_groups', [])
    
    if duplicate_groups:
        st.write("### Duplicate Groups")
        
        for i, group in enumerate(duplicate_groups, 1):
            with st.expander(f"Duplicate Group #{i} (Avg Similarity: {group.get('average_similarity', 0):.1%})"):
                main_resume = group.get('main_resume', {})
                duplicates = group.get('duplicates', [])
                
                st.write(f"**Main Resume:** {main_resume.get('name', 'Unknown')}")
                st.write(f"**Number of Duplicates:** {len(duplicates)}")
                
                st.write("**Duplicate Resumes:**")
                for j, duplicate in enumerate(duplicates, 1):
                    similarity_score = group.get('similarity_scores', [0])[j-1] if j <= len(group.get('similarity_scores', [])) else 0
                    st.write(f"{j}. {duplicate.get('name', 'Unknown')} - {similarity_score:.1%} similarity")
    


if __name__ == "__main__":
    main() 
