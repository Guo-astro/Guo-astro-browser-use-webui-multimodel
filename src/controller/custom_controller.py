# -*- coding: utf-8 -*-
"""
Filename: custom_action.py

Description:
    This module implements a CustomController that extends the base Controller
    to register custom actions for copying text to the system clipboard and
    pasting text from the clipboard into a browser context.
"""

import logging
import pyperclip

from browser_use.controller.service import Controller
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

logger = logging.getLogger(__name__)


class CustomController(Controller):
    """
    A custom controller extending the base Controller to register unique actions
    such as copying and pasting text using the system clipboard via pyperclip.
    """

    def __init__(self):
        super().__init__()
        self._register_custom_actions()

    def _register_custom_actions(self) -> None:
        """
        Register all custom browser actions supported by this controller.
        """

        @self.registry.action('Copy text to clipboard')
        def copy_to_clipboard(text: str) -> ActionResult:
            """
            Copy the given text to the system clipboard.

            Parameters
            ----------
            text : str
                The text content to be copied.

            Returns
            -------
            ActionResult
                Contains the extracted content, which is the text that was copied.
            """
            logger.debug("Copying text to clipboard: %r", text)
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action('Paste text from clipboard', requires_browser=True)
        async def paste_from_clipboard(browser: BrowserContext) -> ActionResult:
            """
            Paste the text from the system clipboard into the currently active
            element of the browser page (where the cursor is focused).

            Parameters
            ----------
            browser : BrowserContext
                The current browser context. Used to retrieve the active page
                and type the clipboard content.

            Returns
            -------
            ActionResult
                Contains the extracted content, which is the text that was pasted.
            """
            text = pyperclip.paste()
            logger.debug("Pasting text from clipboard: %r", text)

            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)
