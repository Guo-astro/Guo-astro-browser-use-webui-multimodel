# browser_message_manager.py
import logging
from typing import List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState

from src.agent.browser_system_prompts import BrowserMessagePrompt

logger = logging.getLogger(__name__)


class BrowserMessageManager(MessageManager):
    def __init__(
        self,
        llm: BaseChatModel,
        task: str,
        action_descriptions: str,
        system_prompt_class: Type[SystemPrompt],
        max_input_tokens: int = 128000,
        estimated_tokens_per_character: int = 3,
        image_tokens: int = 800,
        include_attributes: Optional[List[str]] = None,
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
    ):
        super().__init__(
            llm,
            task,
            action_descriptions,
            system_prompt_class,
            max_input_tokens,
            estimated_tokens_per_character,
            image_tokens,
            include_attributes or [],
            max_error_length,
            max_actions_per_step,
        )
        self.history = MessageHistory()
        self._add_message_with_tokens(self.system_prompt)

    def add_state_message(
        self,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        step_info: Optional[AgentStepInfo] = None,
    ) -> None:
        """
        Add the current browser state and optional results as a message.

        This method stores action results in memory if provided, then
        constructs a message based on the current state.
        """
        if result:
            self._store_results_in_memory(result)
            result = None

        state_message = BrowserMessagePrompt(
            state=state,
            result=result,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            step_info=step_info,
        ).get_user_message()

        self._add_message_with_tokens(state_message)

    def _store_results_in_memory(self, results: List[ActionResult]) -> None:
        """Store action results' content and errors in memory as messages."""
        for result in results:
            if result.include_in_memory:
                if result.extracted_content:
                    self._add_message_with_tokens(HumanMessage(content=str(result.extracted_content)))
                if result.error:
                    truncated_error = str(result.error)[-self.max_error_length:]
                    self._add_message_with_tokens(HumanMessage(content=truncated_error))
