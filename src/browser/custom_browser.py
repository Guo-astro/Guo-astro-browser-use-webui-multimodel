# specialized_browser.py
"""
Description:
    This module defines a SpecializedBrowser class that overrides the `new_context`
    method to return a CustomBrowserContext.
"""

import logging
from typing import Optional

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContext

from src.browser.custom_context import EnhancedBrowserContext

logger = logging.getLogger(__name__)


class SpecializedBrowser(Browser):
    """
    A specialized Browser implementation that yields a CustomBrowserContext
    upon requesting a new context.
    """

    async def new_context(
            self,
            config: BrowserContextConfig = BrowserContextConfig(),
            context: Optional[BrowserContext] = None
    ) -> BrowserContext:
        """
        Create and return a CustomBrowserContext.

        Parameters
        ----------
        config : BrowserContextConfig, optional
            Configuration settings for the browser context.
        context : Optional[BrowserContext], optional
            An existing BrowserContext-like instance, if reusing contexts.

        Returns
        -------
        BrowserContext
            A newly created or existing CustomBrowserContext.
        """
        logger.debug("Creating a new specialized browser context.")
        return EnhancedBrowserContext(
            config=config,
            browser=self,
            context=context
        )
