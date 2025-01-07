"""Recordings component for the Browser Use WebUI"""

import gradio as gr

def create_recordings_tab():
    """Creates and returns the recordings tab"""
    with gr.TabItem("🎬 Recordings", id="recordings_tab"):
        recording_display = gr.Video(label="Latest Recording")
        
        with gr.Group():
            gr.Markdown("### Results")
            with gr.Row():
                with gr.Column():
                    final_result_output = gr.Textbox(
                        label="Final Result",
                        lines=3,
                        show_label=True
                    )
                with gr.Column():
                    errors_output = gr.Textbox(
                        label="Errors",
                        lines=3,
                        show_label=True
                    )
            with gr.Row():
                with gr.Column():
                    model_actions_output = gr.Textbox(
                        label="Model Actions",
                        lines=3,
                        show_label=True
                    )
                with gr.Column():
                    model_thoughts_output = gr.Textbox(
                        label="Model Thoughts",
                        lines=3,
                        show_label=True
                    )
                    
    return {
        "recording_display": recording_display,
        "final_result_output": final_result_output,
        "errors_output": errors_output,
        "model_actions_output": model_actions_output,
        "model_thoughts_output": model_thoughts_output
    }