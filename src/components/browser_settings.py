"""Browser settings component for the Browser Use WebUI"""

import gradio as gr

def create_browser_settings_tab():
    """Creates and returns the browser settings tab"""
    with gr.TabItem("🌐 Browser Settings", id="browser_settings_tab"):
        with gr.Group():
            with gr.Row():
                use_own_browser = gr.Checkbox(
                    label="Use Own Browser",
                    value=False,
                    info="Use your existing browser instance"
                )
                headless = gr.Checkbox(
                    label="Headless Mode",
                    value=False,
                    info="Run browser without GUI"
                )
                disable_security = gr.Checkbox(
                    label="Disable Security",
                    value=True,
                    info="Disable browser security features"
                )
            with gr.Row():
                window_width = gr.Number(
                    label="Window Width",
                    value=1920,
                    info="Browser window width"
                )
                window_height = gr.Number(
                    label="Window Height",
                    value=1080,
                    info="Browser window height"
                )
            save_recording_path = gr.Textbox(
                label="Recording Path",
                placeholder="e.g. ./tmp/record_videos",
                value="./tmp/record_videos",
                info="Path to save browser recordings"
            )
            
    return {
        "use_own_browser": use_own_browser,
        "headless": headless,
        "disable_security": disable_security,
        "window_width": window_width,
        "window_height": window_height,
        "save_recording_path": save_recording_path
    }