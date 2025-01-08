# enhanced_browser_context.py
"""
Description:
    This module defines an EnhancedBrowserContext class that extends the
    base BrowserContext to include anti-detection measures, cookie loading,
    and optional reuse of an existing context.
"""

import json
import logging
import os
from typing import Optional

from playwright.async_api import Browser as PlaywrightBrowser

from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.browser import Browser

logger = logging.getLogger(__name__)


class EnhancedBrowserContext(BrowserContext):
    """
    An enhanced BrowserContext that includes anti-detection measures,
    cookie management, and optional reuse of an existing context.
    """

    def __init__(
        self,
        browser: Browser,
        config: BrowserContextConfig = BrowserContextConfig(),
        context: Optional[BrowserContext] = None
    ):
        """
        Initialize an EnhancedBrowserContext.

        Parameters
        ----------
        browser : Browser
            The parent Browser instance.
        config : BrowserContextConfig, optional
            Configuration for the browser context.
        context : Optional[BrowserContext], optional
            An existing BrowserContext to reuse, if available.
        """
        super().__init__(browser, config)
        self.context = context

    async def _create_context(self, browser: PlaywrightBrowser) -> BrowserContext:
        """
        Create or reuse a browser context with added anti-detection measures.
        Loads cookies if available and starts tracing if configured.

        Parameters
        ----------
        browser : PlaywrightBrowser
            The underlying Playwright Browser object.

        Returns
        -------
        BrowserContext
            The newly created or existing browser context.
        """
        # Reuse provided context if available
        if self.context:
            logger.debug("Using provided existing browser context.")
            return self.context

        # Connect to an existing Chrome instance or create a new context
        if self.browser.config.chrome_instance_path and browser.contexts:
            logger.debug("Connecting to existing Chrome instance.")
            context = browser.contexts[0]
        else:
            logger.debug("Creating a new Playwright browser context.")
            context = await browser.new_context(
                viewport=self.config.browser_window_size,
                no_viewport=False,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
                ),
                java_script_enabled=True,
                bypass_csp=self.config.disable_security,
                ignore_https_errors=self.config.disable_security,
                record_video_dir=self.config.save_recording_path,
                record_video_size=self.config.browser_window_size
            )

        # Start tracing if configured
        if self.config.trace_path:
            logger.info("Starting trace recording for the browser context.")
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Load cookies if they exist
        if self.config.cookies_file and os.path.exists(self.config.cookies_file):
            with open(self.config.cookies_file, 'r') as f:
                cookies = json.load(f)
                logger.info(f"Loaded {len(cookies)} cookies from {self.config.cookies_file}")
                await context.add_cookies(cookies)

        # Inject anti-detection scripts
        logger.debug("Injecting anti-detection scripts into the browser context.")
        await context.add_init_script(
            """
            // Hide the 'webdriver' property from navigator
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // Spoof languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });

            // Fake plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // Mock chrome runtime object
            window.chrome = { runtime: {} };

            // Override permissions query for notifications
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters)
            );
            """
        )

        return context
