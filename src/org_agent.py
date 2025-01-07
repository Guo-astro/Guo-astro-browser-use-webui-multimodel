# org_agent.py
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContext,
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from browser_use.agent.service import Agent


async def run_org_agent(
    llm,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    task,
    max_steps,
    use_vision
):
    """Run the 'org' agent using the standard Browser and Agent classes."""
    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            disable_security=disable_security,
            extra_chromium_args=[f'--window-size={window_w},{window_h}'],
        )
    )

    async with await browser.new_context(
        config=BrowserContextConfig(
            trace_path='./tmp/traces',
            save_recording_path=save_recording_path if save_recording_path else None,
            no_viewport=False,
            browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
        )
    ) as browser_context:
        agent = Agent(
            task=task,
            llm=llm,
            use_vision=use_vision,
            browser_context=browser_context,
        )

        history = await agent.run(max_steps=max_steps)
        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

    await browser.close()
    return final_result, errors, model_actions, model_thoughts
