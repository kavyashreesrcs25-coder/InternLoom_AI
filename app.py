"""
InternLoom AI — Main Entry Point
Run with:  streamlit run app.py
"""

import sys
import os

# Ensure project root is on the Python path so all imports resolve
# regardless of working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.dashboard import run_dashboard  # noqa: E402

if __name__ == "__main__" or True:
    run_dashboard()
