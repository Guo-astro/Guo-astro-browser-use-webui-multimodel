# browser_agent.py
import asyncio
import json
import logging
from typing import Any, Optional, Type, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentOutput, AgentStepInfo, AgentHistoryList
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepErrorTelemetryEvent,
)
from browser_use.utils import time_execution_async

from src.agent.browser_massage_manager import BrowserMessageManager
from src.agent.browser_agent_output import BrowserAgentOutput, StepInfo

logger = logging.getLogger(__name__)


class BrowserAgent(Agent):
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        additional_info: str = "",
        browser: Optional[Browser] = None,
        browser_context: Optional[BrowserContext] = None,
        controller: Controller = Controller(),
        use_vision: bool = True,
        save_conversation_path: Optional[str] = None,
        max_failures: int = 5,
        retry_delay: int = 10,
        system_prompt_class: Type = None,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        include_attributes: Optional[List[str]] = None,
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
    ):
        super().__init__(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            use_vision=use_vision,
            save_conversation_path=save_conversation_path,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            include_attributes=include_attributes or [
                "title", "type", "name", "role", "tabindex",
                "aria-label", "placeholder", "value", "alt",
                "aria-expanded",
            ],
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
        )
        self.additional_info = additional_info
        self.message_manager = BrowserMessageManager(
            llm=self.llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=self.system_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
        )
        self._setup_action_models()

    def _setup_action_models(self) -> None:
        """Set up dynamic action and output models."""
        self.ActionModel = self.controller.registry.create_action_model()
        self.AgentOutput = BrowserAgentOutput.with_dynamic_actions(self.ActionModel)

    def _log_response(self, response: BrowserAgentOutput) -> None:
        """Log details of the agent's response."""
        evaluation = response.current_state.prev_action_evaluation
        emoji = "✅" if "Success" in evaluation else "❌" if "Failed" in evaluation else "🤷"
        logger.info("%s Eval: %s", emoji, evaluation)
        logger.info("🧠 New Memory: %s", response.current_state.important_contents)
        logger.info("⏳ Task Progress: %s", response.current_state.completed_contents)
        logger.info("🤔 Thought: %s", response.current_state.thought)
        logger.info("🎯 Summary: %s", response.current_state.summary)
        for i, action in enumerate(response.action, start=1):
            logger.info("🛠️  Action %d/%d: %s", i, len(response.action), action.model_dump_json(exclude_unset=True))

    def update_step_info(self, model_output: BrowserAgentOutput, step_info: Optional[StepInfo] = None) -> None:
        """Update step info based on model output."""
        if not step_info:
            return
        step_info.step_number += 1
        important_content = model_output.current_state.important_contents
        if important_content and "None" not in important_content and important_content not in step_info.memory:
            step_info.memory += important_content + "\n"
        completed_content = model_output.current_state.completed_contents
        if completed_content and "None" not in completed_content:
            step_info.task_progress = completed_content

    @time_execution_async("--get_next_action")
    async def get_next_action(self, input_messages: List[BaseMessage]) -> AgentOutput:
        """Fetch the next action from the LLM."""
        response = self.llm.invoke(input_messages)
        json_str = response.content.replace("```json", "").replace("```", "")
        parsed_json = json.loads(json_str)
        parsed: AgentOutput = self.AgentOutput(**parsed_json)
        parsed.action = parsed.action[: self.max_actions_per_step]
        self._log_response(parsed)
        self.n_steps += 1
        return parsed

    @time_execution_async("--step")
    async def step(self, step_info: Optional[StepInfo] = None) -> None:
        """Execute one step of the agent's workflow."""
        logger.info("\n📍 Step %d", self.n_steps)
        state = model_output = None
        result: List[ActionResult] = []

        try:
            state = await self.browser_context.get_state(use_vision=self.use_vision)
            self.message_manager.add_state_message(state, self._last_result, step_info)
            input_messages = self.message_manager.get_messages()
            model_output = await self.get_next_action(input_messages)
            self.update_step_info(model_output, step_info)
            logger.info("🧠 All Memory: %s", step_info.memory)
            self._save_conversation(input_messages, model_output)
            self.message_manager._remove_last_state_message()
            self.message_manager.add_model_output(model_output)
            result = await self.controller.multi_act(model_output.action, self.browser_context)
            self._last_result = result

            if result and result[-1].is_done:
                logger.info("📄 Result: %s", result[-1].extracted_content)
            self.consecutive_failures = 0

        except Exception as exc:
            result = self._handle_step_error(exc)
            self._last_result = result

        finally:
            await self._handle_telemetry_and_history(model_output, state, result)

    async def _handle_telemetry_and_history(self, model_output, state, result):
        """Handle telemetry events and history creation after a step."""
        if not result:
            return
        for action_result in result:
            if action_result.error:
                self.telemetry.capture(AgentStepErrorTelemetryEvent(agent_id=self.agent_id, error=action_result.error))
        if state:
            self._make_history_item(model_output, state, result)

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Run the agent for a maximum of steps and return history."""
        logger.info("🚀 Starting task: %s", self.task)
        self.telemetry.capture(AgentRunTelemetryEvent(agent_id=self.agent_id, task=self.task))
        step_info = StepInfo(
            task=self.task,
            additional_info=self.additional_info,
            step_number=1,
            max_steps=max_steps,
            memory="",
            task_progress="",
        )
        try:
            for step_index in range(max_steps):
                if self._too_many_failures():
                    break
                await self.step(step_info)
                if self.history.is_done():
                    if self.validate_output and step_index < max_steps - 1:
                        if not await self._validate_output():
                            continue
                    logger.info("✅ Task completed successfully")
                    break
            else:
                logger.info("❌ Failed to complete task in maximum steps")
            return self.history
        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.agent_id,
                    task=self.task,
                    success=self.history.is_done(),
                    steps=len(self.history.history),
                )
            )
            if not self.injected_browser_context:
                await self.browser_context.close()
            if not self.injected_browser and self.browser:
                await self.browser.close()
