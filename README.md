# Automated Resume Screening System for Salesforce Engineer Hiring

## Overview
This project implements an automated resume screening system that processes resumes, extracts relevant information, and matches them against a Salesforce Engineer job description using NLP techniques.

## Features
- **Resume Parsing**: Extracts structured data from PDF, DOCX, and text files
- **NLP-based Matching**: Uses TF-IDF and BERT embeddings for job matching
- **Ranking System**: Ranks resumes by relevance score
- **Interactive Dashboard**: Streamlit-based visualization of results
- **Skill Extraction**: Identifies technical skills and certifications
- **Duplicate Detection**: Finds similar resumes

## Project Structure
```
├── src/
│   ├── resume_parser.py      # Resume parsing and extraction
│   ├── job_matcher.py        # Job matching and ranking logic
│   ├── skill_extractor.py    # Skill and keyword extraction
│   ├── duplicate_detector.py # Duplicate resume detection
│   └── utils.py             # Utility functions
├── data/
│   ├── resumes/             # Sample resumes (PDF/DOCX)
│   ├── job_description.txt  # Salesforce Engineer JD
│   └── sample_resumes/      # Generated sample resumes
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── notebooks/
│   └── analysis.ipynb      # Jupyter notebook for analysis
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   
## 🚀 Live Demo

You can access the live demo of this application deployed on Streamlit Community Cloud:

https://salesforcehiring-app.streamlit.app/



## Technologies Used
- **NLP**: spaCy, NLTK, Transformers
- **ML**: scikit-learn, sentence-transformers
- **Visualization**: Streamlit, Plotly, WordCloud
- **Document Processing**: PyPDF2, python-docx 


link:- https://salesforcehiring-app.streamlit.app/
