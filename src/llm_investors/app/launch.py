"""`investor-viewer`: start the Streamlit viewer (equivalent to streamlit run <this dir>/streamlit_app.py)."""
import sys
from pathlib import Path


def main():
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("streamlit_app.py")), *sys.argv[1:]]
    sys.exit(cli.main())
