"""Main UI module for Browser Use WebUI"""

import gradio as gr
from typing import Dict, Any

from src.components.job_search import create_job_search_tab
from src.job_agent_runner import run_job_agent
from themes import THEME_MAP, BASE_CSS, BASE_JS
from components.header import create_header
from components.agent_settings import create_agent_settings_tab
from components.llm_configuration import create_llm_configuration_tab
from components.browser_settings import create_browser_settings_tab
from components.task_settings import create_task_settings_tab
from components.recordings import create_recordings_tab
from agent_runner import run_browser_agent


def _create_application_tabs() -> Dict[str, Any]:
    """Create and return all application tabs"""
    return {
        "agent_settings": create_agent_settings_tab(),
        "llm_configuration": create_llm_configuration_tab(),
        "browser_settings": create_browser_settings_tab(),
        "task_settings": create_task_settings_tab(),
        "recordings": create_recordings_tab(),
        "job_search": create_job_search_tab(),

    }


def _wire_up_event_handlers(interface: gr.Blocks, tabs: Dict[str, Any]) -> None:
    """Configure all event handlers for the application"""
    task_settings_tab = tabs["task_settings"]
    agent_settings_tab = tabs["agent_settings"]
    llm_configuration_tab = tabs["llm_configuration"]
    browser_settings_tab = tabs["browser_settings"]
    recordings_tab = tabs["recordings"]

    task_settings_tab["run_button"].click(
        fn=run_browser_agent,
        inputs=[
            agent_settings_tab["agent_type"],
            llm_configuration_tab["llm_provider"],
            llm_configuration_tab["llm_model_name"],
            llm_configuration_tab["llm_temperature"],
            llm_configuration_tab["llm_base_url"],
            llm_configuration_tab["llm_api_key"],
            browser_settings_tab["use_own_browser"],
            browser_settings_tab["headless"],
            browser_settings_tab["disable_security"],
            browser_settings_tab["window_width"],
            browser_settings_tab["window_height"],
            browser_settings_tab["save_recording_path"],
            task_settings_tab["task_description"],
            task_settings_tab["additional_information"],
            agent_settings_tab["max_steps"],
            agent_settings_tab["use_vision"]
        ],
        outputs=[
            recordings_tab["final_result_output"],
            recordings_tab["errors_output"],
            recordings_tab["model_actions_output"],
            recordings_tab["model_thoughts_output"],
            recordings_tab["recording_display"]
        ]
    )
    job_search_tab = tabs["job_search"]

    # The new "Search & Apply" button => run_job_agent
    job_search_tab["run_button"].click(
        fn=run_job_agent,
        inputs=[
            agent_settings_tab["agent_type"],
            llm_configuration_tab["llm_provider"],
            llm_configuration_tab["llm_model_name"],
            llm_configuration_tab["llm_temperature"],
            llm_configuration_tab["llm_base_url"],
            llm_configuration_tab["llm_api_key"],
            browser_settings_tab["use_own_browser"],
            browser_settings_tab["headless"],
            browser_settings_tab["disable_security"],
            browser_settings_tab["window_width"],
            browser_settings_tab["window_height"],
            browser_settings_tab["save_recording_path"],
            # We pass 'task' from the job tab
            job_search_tab["task"],
            # Instead of a simple string, we pass the entire CV file as 'add_infos'
            # so we can parse it in run_job_agent:
            job_search_tab["cv_file"],
            agent_settings_tab["max_steps"],
            agent_settings_tab["use_vision"]
        ],
        outputs=[
            job_search_tab["final_result_output"],
            job_search_tab["errors_output"],
            job_search_tab["logs_output"]
        ]
    )


def create_application_ui(theme_name: str = "Production") -> gr.Blocks:
    """
    Creates and returns the main application UI.
    
    Args:
        theme_name: Name of the theme to use (must be in THEME_MAP)
        
    Returns:
        Configured Gradio interface
    """
    if theme_name not in THEME_MAP:
        raise ValueError(f"Invalid theme name: {theme_name}")

    with gr.Blocks(
            title="Browser Use WebUI",
            theme=THEME_MAP[theme_name],
            css=BASE_CSS,
            js=BASE_JS
    ) as interface:
        # Create header section
        create_header()

        # Create main tabs
        tabs = _create_application_tabs()

        # Configure event handlers
        _wire_up_event_handlers(interface, tabs)

    return interface
