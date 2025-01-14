# file: src/agent_runner.py

import os
import glob
import traceback
from typing import Tuple, Optional

from org_agent import run_org_agent
from custom_agent import run_custom_agent
from src.utils import utils  # Adjust import if needed


# -------------------------
# Original run_browser_agent
# -------------------------
async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        add_infos,
        max_steps,
        use_vision
):
    # Ensure the recording directory exists
    os.makedirs(save_recording_path, exist_ok=True)

    # Get the list of existing videos before the agent runs
    existing_videos = set(
        glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) +
        glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]'))
    )

    # Prepare the LLM using your shared utils
    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key
    )

    # Run the appropriate agent
    try:
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts = await run_org_agent(
                llm=llm,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")
    except Exception as e:
        # Catch any unexpected exceptions
        traceback.print_exc()
        final_result = ""
        errors = str(e) + "\n" + traceback.format_exc()
        model_actions = ""
        model_thoughts = ""

    # Get the list of videos after the agent runs
    new_videos = set(
        glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) +
        glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]'))
    )

    # Find the newly created video
    latest_video = None
    created_videos = new_videos - existing_videos
    if created_videos:
        # Grab the first new video (or modify logic if multiple recordings possible)
        latest_video = list(created_videos)[0]

    return final_result, errors, model_actions, model_thoughts, latest_video


# -------------------------
# New run_job_agent function
# -------------------------
async def run_job_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        add_infos,
        max_steps,
        use_vision
) -> Tuple[str, str, str, str, Optional[str]]:
    """
    Similar signature to run_browser_agent, but specialized for
    reading an uploaded CV PDF and applying to jobs.

    Returns:
      (final_result, errors, model_actions, model_thoughts, latest_video)
    """

    import logging
    from PyPDF2 import PdfReader

    logger = logging.getLogger(__name__)
    os.makedirs(save_recording_path, exist_ok=True)

    # 1) Track existing videos
    existing_videos = set(
        glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) +
        glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]'))
    )

    # 2) Build the LLM
    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key
    )

    # 3) If user uploaded a PDF CV, read it and incorporate into 'task'
    cv_text = ""
    if isinstance(add_infos, dict) and "name" in add_infos:
        # Gradio file object => add_infos["name"] is the actual path
        pdf_path = add_infos["name"]
        if os.path.isfile(pdf_path):
            try:
                pdf = PdfReader(pdf_path)
                for page in pdf.pages:
                    cv_text += page.extract_text() or ""
                logger.info(f"Read {len(cv_text)} characters from user-uploaded CV.")
            except Exception as e:
                logger.warning("Error reading user-uploaded CV PDF: %s", str(e))
        else:
            logger.info("Uploaded file path not found on disk.")
    elif isinstance(add_infos, str) and os.path.isfile(add_infos):
        # Possibly user typed a path or your code passed a raw string path
        pdf_path = add_infos
        try:
            pdf = PdfReader(pdf_path)
            for page in pdf.pages:
                cv_text += page.extract_text() or ""
            logger.info(f"Read {len(cv_text)} characters from local CV PDF path.")
        except Exception as e:
            logger.warning("Error reading local CV PDF: %s", str(e))
    else:
        logger.info("No CV PDF found in 'add_infos' - proceeding without CV text.")

    # Incorporate CV text into the 'task' instructions
    full_task = (
        f"You are a professional job finder with my CV:\n\n{cv_text}\n\n"
        "Please find to most relevant positions as much as possible as instructed.\n\n"
        "Best working conditions and salaries is a must. Only can be ignored if it is big tech company \n\n"
        "Please do not try to sign in or sign up as Linkedin or X.\n\n"
        "Also for final result, please list the url and corresponding summary including salary and  working conditions a numbered list\n\n"
        f"Main instructions: {task}\n"
    )

    # 4) Run the appropriate agent
    try:
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts = await run_org_agent(
                llm=llm,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                task=full_task,  # pass the augmented task
                max_steps=max_steps,
                use_vision=use_vision
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                task=full_task,  # pass the augmented task
                add_infos="(CV read internally)",  # or keep add_infos if you want
                max_steps=max_steps,
                use_vision=use_vision
            )
        else:
            raise ValueError(f"Invalid agent type for job search: {agent_type}")

    except Exception as e:
        traceback.print_exc()
        final_result = ""
        errors = str(e) + "\n" + traceback.format_exc()
        model_actions = ""
        model_thoughts = ""

    # 5) Compare videos after run
    new_videos = set(
        glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) +
        glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]'))
    )
    created_videos = new_videos - existing_videos

    latest_video = None
    if created_videos:
        latest_video = list(created_videos)[0]

    return final_result, errors, model_actions, model_thoughts, latest_video
