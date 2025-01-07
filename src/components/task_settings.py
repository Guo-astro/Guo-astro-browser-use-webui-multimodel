"""Task settings component for the Browser Use WebUI"""

import gradio as gr

def create_task_settings_tab():
    """Creates and returns the task settings tab"""
    with gr.TabItem("📝 Task Settings", id="task_settings_tab"):
        task_description = gr.Textbox(
            label="Task Description",
            lines=4,
            placeholder="Enter your task here...",
            value="go to google.com and type 'OpenAI', click search and give me the first URL",
            info="Describe what you want the agent to do"
        )
        additional_information = gr.Textbox(
            label="Additional Information",
            lines=3,
            placeholder="Add any helpful context or instructions...",
            info="Optional hints to help the LLM complete the task"
        )
        
        with gr.Row():
            run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2, elem_id="run_button")
            stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1, elem_id="stop_button")
            
    return {
        "task_description": task_description,
        "additional_information": additional_information,
        "run_button": run_button,
        "stop_button": stop_button
    }