"""Header component for the Browser Use WebUI"""

import gradio as gr


def create_header():
    """Creates and returns the header section of the UI"""
    return gr.Markdown(
        """
        # browser-use with Default Deepseek V3 settings
        ### Control your browser with AI assistance
        """,
        elem_classes=["header-text"]
    )
