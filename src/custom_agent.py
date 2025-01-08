import os
import traceback

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextWindowSize, BrowserContextConfig
from playwright.async_api import async_playwright

from src.agent.browser_agent import BrowserAgent
from src.agent.browser_system_prompts import BrowserSystemPrompt
from src.browser.enhanced_playwright_browser import EnhancedPlaywrightBrowser
from src.controller.custom_controller import CustomController

async def run_custom_agent(
        llm,
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
    """Run the 'custom' agent using a specialized browser and custom agent."""
    controller = CustomController()
    playwright = None
    browser_context_ = None

    try:
        # Optional: Use existing Chrome profile
        if use_own_browser:
            playwright = await async_playwright().start()
            chrome_exe = os.getenv("CHROME_PATH", "")
            chrome_use_data = os.getenv("CHROME_USER_DATA", "")
            browser_context_ = await playwright.chromium.launch_persistent_context(
                user_data_dir=chrome_use_data,
                executable_path=chrome_exe,
                no_viewport=False,
                headless=headless,
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
                ),
                java_script_enabled=True,
                bypass_csp=disable_security,
                ignore_https_errors=disable_security,
                record_video_dir=save_recording_path if save_recording_path else None,
                record_video_size={'width': window_w, 'height': window_h}
            )
        else:
            browser_context_ = None

        # Create a new specialized browser with your custom logic
        browser = EnhancedPlaywrightBrowser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                extra_chromium_args=[f'--window-size={window_w},{window_h}'],
            )
        )

        async with await browser.new_context(
                config=BrowserContextConfig(
                    trace_path='./tmp/result_processing',
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                ),
                context=browser_context_
        ) as browser_context:
            agent = BrowserAgent(
                task=task,
                additional_info=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser_context=browser_context,
                controller=controller,
                system_prompt_class=BrowserSystemPrompt
            )
            history = await agent.run(max_steps=max_steps)

            final_result = history.final_result()
            errors = history.errors()
            model_actions = history.model_actions()
            model_thoughts = history.model_thoughts()

    except Exception as e:
        traceback.print_exc()
        final_result = ""
        errors = str(e) + "\n" + traceback.format_exc()
        model_actions = ""
        model_thoughts = ""
    finally:
        # Close persistent context if used
        if browser_context_:
            await browser_context_.close()

        # Stop the Playwright object
        if playwright:
            await playwright.stop()

        # Close the specialized browser
        await browser.close()

    return final_result, errors, model_actions, model_thoughts
