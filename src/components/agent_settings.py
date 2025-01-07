"""Agent settings component for the Browser Use WebUI"""

import gradio as gr

def create_agent_settings_tab():
    """Creates and returns the agent settings tab"""
    with gr.TabItem("🤖 Agent Settings", id="agent_settings_tab"):
        with gr.Group():
            agent_type = gr.Radio(
                ["org", "custom"],
                label="Agent Type",
                value="custom",
                info="Select the type of agent to use"
            )
            max_steps = gr.Slider(
                minimum=1,
                maximum=200,
                value=100,
                step=1,
                label="Max Run Steps",
                info="Maximum number of steps the agent will take"
            )
            use_vision = gr.Checkbox(
                label="Use Vision",
                value=False,
                info="Enable visual processing capabilities"
            )
            
    return {
        "agent_type": agent_type,
        "max_steps": max_steps,
        "use_vision": use_vision
    }