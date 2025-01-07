# browser_models.py
from dataclasses import dataclass
from typing import Type, List

from pydantic import BaseModel, ConfigDict, Field, create_model
from browser_use.controller.registry.views import ActionModel
from browser_use.agent.views import AgentOutput


@dataclass
class StepInfo:
    """
    Contains information about the current step in the browser automation process.
    """
    step_number: int
    max_steps: int
    task: str
    additional_info: str
    memory: str
    task_progress: str


class AgentBrain(BaseModel):
    """
    Represents the agent's internal state after evaluating an action.
    """
    prev_action_evaluation: str
    important_contents: str
    completed_contents: str
    thought: str
    summary: str


class BrowserAgentOutput(AgentOutput):
    """
    Output model for the browser automation agent. Extends AgentOutput to
    include the agent's brain state and a list of dynamic actions.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_state: AgentBrain
    action: List[ActionModel]

    @staticmethod
    def with_dynamic_actions(action_model: Type[ActionModel]) -> Type["BrowserAgentOutput"]:
        """
        Creates a new output model class with a specified dynamic action model.
        """
        return create_model(
            "BrowserAgentOutput",
            __base__=BrowserAgentOutput,
            action=(List[action_model], Field(...)),
            __module__=BrowserAgentOutput.__module__,
        )
