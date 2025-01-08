# -*- coding: utf-8 -*-
"""
Filename: test_browser_use.py

Description:
    Test scripts for verifying browser automation using both the 'org' agent
    (original Browser + Agent) and the 'custom' agent (CustomBrowser, CustomContext, etc.).
"""

import os
import sys
import logging
import asyncio
from pprint import pprint
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Make local modules available if needed
sys.path.append(".")

from browser_use.agent.views import AgentHistoryList
from src.utils import utils


async def test_browser_use_org() -> None:
    """
    Test the original 'org' browser agent using Browser + Agent from browser_use,
    instructing it to go to Google, search 'OpenAI', and extract the first URL.
    """
    from browser_use.browser.browser import Browser, BrowserConfig
    from browser_use.browser.context import (
        BrowserContextConfig,
        BrowserContextWindowSize,
    )
    from browser_use.agent.service import Agent

    logger.info("Setting up LLM and Browser for 'org' agent test.")
    llm = utils.get_llm_model(
        provider="azure_openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    )

    window_w, window_h = 1920, 1080

    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
            extra_chromium_args=[f"--window-size={window_w},{window_h}"],
        )
    )

    async with await browser.new_context(
            config=BrowserContextConfig(
                trace_path="./tmp/traces",
                save_recording_path="./tmp/record_videos",
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
            )
    ) as browser_context:
        agent = Agent(
            task="go to google.com and type 'OpenAI' click search and give me the first url",
            llm=llm,
            browser_context=browser_context,
        )
        logger.info("Running 'org' agent for up to 10 steps...")
        history: AgentHistoryList = await agent.run(max_steps=10)

        print("Final Result:")
        pprint(history.final_result(), indent=4)

        print("\nErrors:")
        pprint(history.errors(), indent=4)

        print("\nModel Outputs:")
        pprint(history.model_actions(), indent=4)

        print("\nThoughts:")
        pprint(history.model_thoughts(), indent=4)

    # Close the browser
    logger.info("Closing the browser.")
    await browser.close()


async def test_browser_use_custom() -> None:
    """
    Test the custom browser agent using CustomBrowser, CustomContext,
    CustomController, and CustomAgent. The agent is tasked with going to
    Google, searching 'OpenAI', and extracting the first URL.
    """
    from playwright.async_api import async_playwright
    from browser_use.browser.context import BrowserContextWindowSize

    from src.browser.enhanced_playwright_browser import CustomBrowser, BrowserConfig
    from src.browser.enhanced_playwright_browser_context import BrowserContext, BrowserContextConfig
    from src.controller.custom_controller import CustomController
    from src.agent.browser_agent import CustomAgent
    from src.agent.browser_system_prompts import CustomSystemPrompt

    logger.info("Setting up LLM and Custom Browser for 'custom' agent test.")
    llm = utils.get_llm_model(
        provider="ollama",
        model_name="qwen2.5:7b",
        temperature=0.8
    )

    controller = CustomController()
    use_own_browser = False
    disable_security = True
    use_vision = False

    window_w, window_h = 1920, 1080
    playwright = None
    browser_context_ = None

    try:
        if use_own_browser:
            logger.info("Launching a persistent browser context with existing profile.")
            playwright = await async_playwright().start()
            chrome_exe = os.getenv("CHROME_PATH", "")
            chrome_use_data = os.getenv("CHROME_USER_DATA", "")
            browser_context_ = await playwright.chromium.launch_persistent_context(
                user_data_dir=chrome_use_data,
                executable_path=chrome_exe,
                no_viewport=False,
                headless=False,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
                ),
                java_script_enabled=True,
                bypass_csp=disable_security,
                ignore_https_errors=disable_security,
                record_video_dir="./tmp/record_videos",
                record_video_size={"width": window_w, "height": window_h},
            )
        else:
            browser_context_ = None

        browser = CustomBrowser(
            config=BrowserConfig(
                headless=False,
                disable_security=True,
                extra_chromium_args=[f"--window-size={window_w},{window_h}"],
            )
        )

        async with await browser.new_context(
                config=BrowserContextConfig(
                    trace_path="./tmp/result_processing",
                    save_recording_path="./tmp/record_videos",
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                ),
                context=browser_context_
        ) as browser_context:
            agent = CustomAgent(
                task="go to google.com and type 'OpenAI' click search and give me the first url",
                add_infos="",  # additional info/hints
                llm=llm,
                browser_context=browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                use_vision=use_vision,
            )

            logger.info("Running 'custom' agent for up to 10 steps...")
            history: AgentHistoryList = await agent.run(max_steps=10)

            print("Final Result:")
            pprint(history.final_result(), indent=4)

            print("\nErrors:")
            pprint(history.errors(), indent=4)

            print("\nModel Outputs:")
            pprint(history.model_actions(), indent=4)

            print("\nThoughts:")
            pprint(history.model_thoughts(), indent=4)

    except Exception as exc:
        logger.error("An exception occurred:", exc_info=exc)
    finally:
        # Close persistent context if used
        if browser_context_:
            logger.info("Closing persistent browser context.")
            await browser_context_.close()

        # Stop the Playwright object
        if playwright:
            logger.info("Stopping Playwright.")
            await playwright.stop()

        logger.info("Closing the custom browser.")
        await browser.close()


if __name__ == "__main__":
    # Uncomment the desired test:
    # asyncio.run(test_browser_use_org())
    asyncio.run(test_browser_use_custom())
