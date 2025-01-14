# file: src/components/job_search.py

import gradio as gr

def create_job_search_tab():
    """
    Creates a new tab for 'Find & Apply to Jobs' with PDF upload for the CV.
    Returns a dictionary of UI elements.
    """
    with gr.TabItem("🧑‍💼 Find & Apply to Jobs"):
        with gr.Group():
            cv_file = gr.File(
                label="Upload your CV (PDF)",
                file_types=[".pdf"],
                type="filepath",     # Returns a dict with { name: ..., data: ... }
                visible=True
            )
            task = gr.Textbox(
                label="Job Search Instructions",
                lines=3,
                placeholder="e.g. 'Please find ML internships at Google, focusing on Data Science roles.'"
            )

            # The run button
            run_button = gr.Button("🔍 Search & Apply", variant="primary")

        # Output boxes
        final_result_output = gr.Textbox(
            label="Final Result",
            lines=3,
            show_label=True
        )
        errors_output = gr.Textbox(
            label="Errors",
            lines=3,
            show_label=True
        )
        logs_output = gr.Textbox(
            label="Agent Logs (Actions & Thoughts)",
            lines=5,
            show_label=True
        )

    return {
        "cv_file": cv_file,
        "task": task,
        "run_button": run_button,
        "final_result_output": final_result_output,
        "errors_output": errors_output,
        "logs_output": logs_output,
    }
