#!/usr/bin/env python3
"""
Setup script for the Automated Resume Screening System.
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def download_spacy_model():
    """Download spaCy model."""
    print("🤖 Downloading spaCy model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading spaCy model: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "data",
        "data/sample_resumes",
        "data/resumes",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")

def generate_sample_data():
    """Generate sample resumes for testing."""
    print("📄 Generating sample resumes...")
    try:
        from src.generate_samples import main as generate_samples
        generate_samples()
        print("✅ Sample resumes generated successfully")
        return True
    except Exception as e:
        print(f"❌ Error generating sample resumes: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    required_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'nltk',
        'spacy',
        'plotly',
        'PyPDF2',
        'docx',
        'fuzzywuzzy'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Automated Resume Screening System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        return False
    
    # Download spaCy model
    if not download_spacy_model():
        return False
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        return False
    
    # Generate sample data
    if not generate_sample_data():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the dashboard: streamlit run dashboard/app.py")
    print("2. Or process resumes: python src/main.py --generate-samples")
    print("3. Check the README.md for more information")
    print("\n📚 Documentation:")
    print("- README.md: Project overview and usage")
    print("- src/: Core modules")
    print("- dashboard/: Streamlit dashboard")
    print("- notebooks/: Jupyter notebooks for analysis")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 