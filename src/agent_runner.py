# agent_runner.py
import os
import glob
from typing import Tuple, Optional
import traceback

from org_agent import run_org_agent
from custom_agent import run_custom_agent
from src.utils import utils  # Adjust import if needed


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

    # Prepare the LLM
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
