# browser_prompts.py
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from browser_use.agent.views import ActionResult
from browser_use.browser.views import BrowserState
from browser_use.agent.prompts import SystemPrompt

from src.agent.browser_agent_output import StepInfo


class BrowserSystemPrompt(SystemPrompt):
    """System prompt containing rules and instructions for browser automation agent."""

    def important_rules(self) -> str:
        """Provides key rules for JSON response format and handling actions."""
        rules = (
            "1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:\n"
            "   {\n"
            "     \"current_state\": {\n"
            "       \"prev_action_evaluation\": \"Success|Failed|Unknown - ...\",\n"
            "       \"important_contents\": \"...\",\n"
            "       \"completed_contents\": \"...\",\n"
            "       \"thought\": \"...\",\n"
            "       \"summary\": \"...\"\n"
            "     },\n"
            "     \"action\": [\n"
            "       {\n"
            "         \"action_name\": {\n"
            "           // action-specific parameters\n"
            "         }\n"
            "       }\n"
            "     ]\n"
            "   }\n\n"
            "2. ACTIONS: You can specify multiple actions to be executed in sequence.\n"
            "3. ELEMENT INTERACTION: (See documentation for more details.)\n"
            "4. NAVIGATION & ERROR HANDLING: (See documentation for more details.)\n"
            "5. TASK COMPLETION: (See documentation for more details.)\n"
            "6. VISUAL CONTEXT: (See documentation for more details.)\n"
            "7. Form filling: (See documentation for more details.)\n"
            "8. ACTION SEQUENCING: (See documentation for more details.)\n"
        )
        rules += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return rules

    def input_format(self) -> str:
        """Describes the input structure provided to the agent."""
        return (
            "INPUT STRUCTURE:\n"
            "1. Task: The user's instructions to complete.\n"
            "2. Hints(Optional): Additional hints for guidance.\n"
            "3. Memory: Important historical contents.\n"
            "4. Task Progress: Items completed so far.\n"
            "5. Current URL: The webpage currently being viewed.\n"
            "6. Available Tabs: List of open browser tabs.\n"
            "7. Interactive Elements: List in the format index[:]<element_type> ... ."
        )

    def get_system_message(self) -> SystemMessage:
        """Creates the final system prompt message with current date and rules."""
        current_time = self.current_date.strftime('%Y-%m-%d %H:%M')
        prompt_text = (
            f"You are a precise browser automation agent that interacts with websites.\n"
            f"Current date and time: {current_time}\n\n"
            f"{self.input_format()}\n"
            f"{self.important_rules()}\n"
            f"Functions:\n{self.default_action_description}\n\n"
            "Remember: Your responses must be valid JSON matching the specified format."
        )
        return SystemMessage(content=prompt_text)


class BrowserMessagePrompt:
    """
    Formats the browser state and previous action results into a HumanMessage
    that the language model can process.
    """

    def __init__(
        self,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        include_attributes: Optional[List[str]] = None,
        max_error_length: int = 400,
        step_info: Optional[StepInfo] = None,
    ):
        self.state = state
        self.result = result
        self.include_attributes = include_attributes or []
        self.max_error_length = max_error_length
        self.step_info = step_info

    def _compose_state_description(self) -> str:
        """Creates a description of current task, memory, and browser state."""
        return (
            f"1. Task: {self.step_info.task}\n"
            f"2. Hints(Optional):\n{self.step_info.additional_info}\n"  # updated line
            f"3. Memory:\n{self.step_info.memory}\n"
            f"4. Task Progress:\n{self.step_info.task_progress}\n"
            f"5. Current url: {self.state.url}\n"
            f"6. Available tabs:\n{self.state.tabs}\n"
            f"7. Interactive elements:\n"
            f"{self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)}"
        )

    def _append_results(self, description: str) -> str:
        """Appends results of previous actions to the state description."""
        if not self.result:
            return description
        for idx, action_result in enumerate(self.result, start=1):
            if action_result.extracted_content:
                description += f"\nResult of action {idx}/{len(self.result)}: {action_result.extracted_content}"
            if action_result.error:
                snippet = action_result.error[-self.max_error_length:]
                description += f"\nError of action {idx}/{len(self.result)}: ...{snippet}"
        return description

    def get_user_message(self) -> HumanMessage:
        """
        Combines state information and previous action results
        into a single HumanMessage.
        """
        description = self._compose_state_description()
        description = self._append_results(description)

        if self.state.screenshot:
            content = [
                {'type': 'text', 'text': description},
                {
                    'type': 'image_url',
                    'image_url': {'url': f"data:image/png;base64,{self.state.screenshot}"}
                },
            ]
            return HumanMessage(content=content)
        return HumanMessage(content=description)
