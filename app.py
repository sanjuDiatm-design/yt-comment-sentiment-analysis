"""
Hugging Face Spaces / Streamlit Entrypoint
Forwards execution to streamlit_app.py
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("streamlit_app.py")
