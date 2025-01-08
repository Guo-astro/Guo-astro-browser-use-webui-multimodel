"""Header component for the Browser Use WebUI"""

import gradio as gr


def create_header():
    """Creates and returns the header section of the UI"""
    return gr.Markdown(
        """
        # browser-use powered by Reinforcement learning
        ### Control your browser with AI assistance
        """,
        elem_classes=["header-text"]
    )
