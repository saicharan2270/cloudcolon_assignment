"""
Main Streamlit app entry point for deployment.
This file imports and runs the dashboard from the dashboard directory.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the main dashboard
from dashboard.app import main

if __name__ == "__main__":
    main()
