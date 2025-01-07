"""Header component for the Browser Use WebUI"""

import gradio as gr

def create_header():
    """Creates and returns the header section of the UI"""
    return gr.Markdown(
        """
        # 🌐 Browser Use WebUI
        ### Control your browser with AI assistance
        """,
        elem_classes=["header-text"]
    )