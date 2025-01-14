src/__init__.py
```
1 | # -*- coding: utf-8 -*-
```

src/agent/__init__.py
```
1 | # -*- coding: utf-8 -*-
```

src/agent/browser_agent.py
```
1 | # browser_agent.py
2 | import asyncio
3 | import json
4 | import logging
5 | from typing import Any, Optional, Type, List
6 | 
7 | from langchain_core.language_models.chat_models import BaseChatModel
8 | from langchain_core.messages import BaseMessage
9 | from pydantic import BaseModel
10 | 
11 | from browser_use.agent.service import Agent
12 | from browser_use.agent.views import ActionResult, AgentOutput, AgentStepInfo, AgentHistoryList
13 | from browser_use.browser.browser import Browser
14 | from browser_use.browser.context import BrowserContext
15 | from browser_use.controller.service import Controller
16 | from browser_use.telemetry.views import (
17 |     AgentEndTelemetryEvent,
18 |     AgentRunTelemetryEvent,
19 |     AgentStepErrorTelemetryEvent,
20 | )
21 | from browser_use.utils import time_execution_async
22 | 
23 | from src.agent.browser_massage_manager import BrowserMessageManager
24 | from src.agent.browser_agent_output import BrowserAgentOutput, StepInfo
25 | 
26 | logger = logging.getLogger(__name__)
27 | 
28 | 
29 | class BrowserAgent(Agent):
30 |     def __init__(
31 |         self,
32 |         task: str,
33 |         llm: BaseChatModel,
34 |         additional_info: str = "",
35 |         browser: Optional[Browser] = None,
36 |         browser_context: Optional[BrowserContext] = None,
37 |         controller: Controller = Controller(),
38 |         use_vision: bool = True,
39 |         save_conversation_path: Optional[str] = None,
40 |         max_failures: int = 5,
41 |         retry_delay: int = 10,
42 |         system_prompt_class: Type = None,
43 |         max_input_tokens: int = 128000,
44 |         validate_output: bool = False,
45 |         include_attributes: Optional[List[str]] = None,
46 |         max_error_length: int = 400,
47 |         max_actions_per_step: int = 10,
48 |     ):
49 |         super().__init__(
50 |             task=task,
51 |             llm=llm,
52 |             browser=browser,
53 |             browser_context=browser_context,
54 |             controller=controller,
55 |             use_vision=use_vision,
56 |             save_conversation_path=save_conversation_path,
57 |             max_failures=max_failures,
58 |             retry_delay=retry_delay,
59 |             system_prompt_class=system_prompt_class,
60 |             max_input_tokens=max_input_tokens,
61 |             validate_output=validate_output,
62 |             include_attributes=include_attributes or [
63 |                 "title", "type", "name", "role", "tabindex",
64 |                 "aria-label", "placeholder", "value", "alt",
65 |                 "aria-expanded",
66 |             ],
67 |             max_error_length=max_error_length,
68 |             max_actions_per_step=max_actions_per_step,
69 |         )
70 |         self.additional_info = additional_info
71 |         self.message_manager = BrowserMessageManager(
72 |             llm=self.llm,
73 |             task=self.task,
74 |             action_descriptions=self.controller.registry.get_prompt_description(),
75 |             system_prompt_class=self.system_prompt_class,
76 |             max_input_tokens=self.max_input_tokens,
77 |             include_attributes=self.include_attributes,
78 |             max_error_length=self.max_error_length,
79 |             max_actions_per_step=self.max_actions_per_step,
80 |         )
81 |         self._setup_action_models()
82 | 
83 |     def _setup_action_models(self) -> None:
84 |         """Set up dynamic action and output models."""
85 |         self.ActionModel = self.controller.registry.create_action_model()
86 |         self.AgentOutput = BrowserAgentOutput.with_dynamic_actions(self.ActionModel)
87 | 
88 |     def _log_response(self, response: BrowserAgentOutput) -> None:
89 |         """Log details of the agent's response."""
90 |         evaluation = response.current_state.prev_action_evaluation
91 |         emoji = "✅" if "Success" in evaluation else "❌" if "Failed" in evaluation else "🤷"
92 |         logger.info("%s Eval: %s", emoji, evaluation)
93 |         logger.info("🧠 New Memory: %s", response.current_state.important_contents)
94 |         logger.info("⏳ Task Progress: %s", response.current_state.completed_contents)
95 |         logger.info("🤔 Thought: %s", response.current_state.thought)
96 |         logger.info("🎯 Summary: %s", response.current_state.summary)
97 |         for i, action in enumerate(response.action, start=1):
98 |             logger.info("🛠️  Action %d/%d: %s", i, len(response.action), action.model_dump_json(exclude_unset=True))
99 | 
100 |     def update_step_info(self, model_output: BrowserAgentOutput, step_info: Optional[StepInfo] = None) -> None:
101 |         """Update step info based on model output."""
102 |         if not step_info:
103 |             return
104 |         step_info.step_number += 1
105 |         important_content = model_output.current_state.important_contents
106 |         if important_content and "None" not in important_content and important_content not in step_info.memory:
107 |             step_info.memory += important_content + "\n"
108 |         completed_content = model_output.current_state.completed_contents
109 |         if completed_content and "None" not in completed_content:
110 |             step_info.task_progress = completed_content
111 | 
112 |     @time_execution_async("--get_next_action")
113 |     async def get_next_action(self, input_messages: List[BaseMessage]) -> AgentOutput:
114 |         """Fetch the next action from the LLM."""
115 |         response = self.llm.invoke(input_messages)
116 |         json_str = response.content.replace("```json", "").replace("```", "")
117 |         parsed_json = json.loads(json_str)
118 |         parsed: AgentOutput = self.AgentOutput(**parsed_json)
119 |         parsed.action = parsed.action[: self.max_actions_per_step]
120 |         self._log_response(parsed)
121 |         self.n_steps += 1
122 |         return parsed
123 | 
124 |     @time_execution_async("--step")
125 |     async def step(self, step_info: Optional[StepInfo] = None) -> None:
126 |         """Execute one step of the agent's workflow."""
127 |         logger.info("\n📍 Step %d", self.n_steps)
128 |         state = model_output = None
129 |         result: List[ActionResult] = []
130 | 
131 |         try:
132 |             state = await self.browser_context.get_state(use_vision=self.use_vision)
133 |             self.message_manager.add_state_message(state, self._last_result, step_info)
134 |             input_messages = self.message_manager.get_messages()
135 |             model_output = await self.get_next_action(input_messages)
136 |             self.update_step_info(model_output, step_info)
137 |             logger.info("🧠 All Memory: %s", step_info.memory)
138 |             self._save_conversation(input_messages, model_output)
139 |             self.message_manager._remove_last_state_message()
140 |             self.message_manager.add_model_output(model_output)
141 |             result = await self.controller.multi_act(model_output.action, self.browser_context)
142 |             self._last_result = result
143 | 
144 |             if result and result[-1].is_done:
145 |                 logger.info("📄 Result: %s", result[-1].extracted_content)
146 |             self.consecutive_failures = 0
147 | 
148 |         except Exception as exc:
149 |             result = self._handle_step_error(exc)
150 |             self._last_result = result
151 | 
152 |         finally:
153 |             await self._handle_telemetry_and_history(model_output, state, result)
154 | 
155 |     async def _handle_telemetry_and_history(self, model_output, state, result):
156 |         """Handle telemetry events and history creation after a step."""
157 |         if not result:
158 |             return
159 |         for action_result in result:
160 |             if action_result.error:
161 |                 self.telemetry.capture(AgentStepErrorTelemetryEvent(agent_id=self.agent_id, error=action_result.error))
162 |         if state:
163 |             self._make_history_item(model_output, state, result)
164 | 
165 |     async def run(self, max_steps: int = 100) -> AgentHistoryList:
166 |         """Run the agent for a maximum of steps and return history."""
167 |         logger.info("🚀 Starting task: %s", self.task)
168 |         self.telemetry.capture(AgentRunTelemetryEvent(agent_id=self.agent_id, task=self.task))
169 |         step_info = StepInfo(
170 |             task=self.task,
171 |             additional_info=self.additional_info,
172 |             step_number=1,
173 |             max_steps=max_steps,
174 |             memory="",
175 |             task_progress="",
176 |         )
177 |         try:
178 |             for step_index in range(max_steps):
179 |                 if self._too_many_failures():
180 |                     break
181 |                 await self.step(step_info)
182 |                 if self.history.is_done():
183 |                     if self.validate_output and step_index < max_steps - 1:
184 |                         if not await self._validate_output():
185 |                             continue
186 |                     logger.info("✅ Task completed successfully")
187 |                     break
188 |             else:
189 |                 logger.info("❌ Failed to complete task in maximum steps")
190 |             return self.history
191 |         finally:
192 |             self.telemetry.capture(
193 |                 AgentEndTelemetryEvent(
194 |                     agent_id=self.agent_id,
195 |                     task=self.task,
196 |                     success=self.history.is_done(),
197 |                     steps=len(self.history.history),
198 |                 )
199 |             )
200 |             if not self.injected_browser_context:
201 |                 await self.browser_context.close()
202 |             if not self.injected_browser and self.browser:
203 |                 await self.browser.close()
```

src/agent/browser_agent_output.py
```
1 | # browser_models.py
2 | from dataclasses import dataclass
3 | from typing import Type, List
4 | 
5 | from pydantic import BaseModel, ConfigDict, Field, create_model
6 | from browser_use.controller.registry.views import ActionModel
7 | from browser_use.agent.views import AgentOutput
8 | 
9 | 
10 | @dataclass
11 | class StepInfo:
12 |     """
13 |     Contains information about the current step in the browser automation process.
14 |     """
15 |     step_number: int
16 |     max_steps: int
17 |     task: str
18 |     additional_info: str
19 |     memory: str
20 |     task_progress: str
21 | 
22 | 
23 | class AgentBrain(BaseModel):
24 |     """
25 |     Represents the agent's internal state after evaluating an action.
26 |     """
27 |     prev_action_evaluation: str
28 |     important_contents: str
29 |     completed_contents: str
30 |     thought: str
31 |     summary: str
32 | 
33 | 
34 | class BrowserAgentOutput(AgentOutput):
35 |     """
36 |     Output model for the browser automation agent. Extends AgentOutput to
37 |     include the agent's brain state and a list of dynamic actions.
38 |     """
39 |     model_config = ConfigDict(arbitrary_types_allowed=True)
40 | 
41 |     current_state: AgentBrain
42 |     action: List[ActionModel]
43 | 
44 |     @staticmethod
45 |     def with_dynamic_actions(action_model: Type[ActionModel]) -> Type["BrowserAgentOutput"]:
46 |         """
47 |         Creates a new output model class with a specified dynamic action model.
48 |         """
49 |         return create_model(
50 |             "BrowserAgentOutput",
51 |             __base__=BrowserAgentOutput,
52 |             action=(List[action_model], Field(...)),
53 |             __module__=BrowserAgentOutput.__module__,
54 |         )
```

src/agent/browser_massage_manager.py
```
1 | # browser_message_manager.py
2 | import logging
3 | from typing import List, Optional, Type
4 | 
5 | from langchain_core.language_models import BaseChatModel
6 | from langchain_core.messages import HumanMessage
7 | 
8 | from browser_use.agent.message_manager.service import MessageManager
9 | from browser_use.agent.message_manager.views import MessageHistory
10 | from browser_use.agent.prompts import SystemPrompt
11 | from browser_use.agent.views import ActionResult, AgentStepInfo
12 | from browser_use.browser.views import BrowserState
13 | 
14 | from src.agent.browser_system_prompts import BrowserMessagePrompt
15 | 
16 | logger = logging.getLogger(__name__)
17 | 
18 | 
19 | class BrowserMessageManager(MessageManager):
20 |     def __init__(
21 |         self,
22 |         llm: BaseChatModel,
23 |         task: str,
24 |         action_descriptions: str,
25 |         system_prompt_class: Type[SystemPrompt],
26 |         max_input_tokens: int = 128000,
27 |         estimated_tokens_per_character: int = 3,
28 |         image_tokens: int = 800,
29 |         include_attributes: Optional[List[str]] = None,
30 |         max_error_length: int = 400,
31 |         max_actions_per_step: int = 10,
32 |     ):
33 |         super().__init__(
34 |             llm,
35 |             task,
36 |             action_descriptions,
37 |             system_prompt_class,
38 |             max_input_tokens,
39 |             estimated_tokens_per_character,
40 |             image_tokens,
41 |             include_attributes or [],
42 |             max_error_length,
43 |             max_actions_per_step,
44 |         )
45 |         self.history = MessageHistory()
46 |         self._add_message_with_tokens(self.system_prompt)
47 | 
48 |     def add_state_message(
49 |         self,
50 |         state: BrowserState,
51 |         result: Optional[List[ActionResult]] = None,
52 |         step_info: Optional[AgentStepInfo] = None,
53 |     ) -> None:
54 |         """
55 |         Add the current browser state and optional results as a message.
56 | 
57 |         This method stores action results in memory if provided, then
58 |         constructs a message based on the current state.
59 |         """
60 |         if result:
61 |             self._store_results_in_memory(result)
62 |             result = None
63 | 
64 |         state_message = BrowserMessagePrompt(
65 |             state=state,
66 |             result=result,
67 |             include_attributes=self.include_attributes,
68 |             max_error_length=self.max_error_length,
69 |             step_info=step_info,
70 |         ).get_user_message()
71 | 
72 |         self._add_message_with_tokens(state_message)
73 | 
74 |     def _store_results_in_memory(self, results: List[ActionResult]) -> None:
75 |         """Store action results' content and errors in memory as messages."""
76 |         for result in results:
77 |             if result.include_in_memory:
78 |                 if result.extracted_content:
79 |                     self._add_message_with_tokens(HumanMessage(content=str(result.extracted_content)))
80 |                 if result.error:
81 |                     truncated_error = str(result.error)[-self.max_error_length:]
82 |                     self._add_message_with_tokens(HumanMessage(content=truncated_error))
```

src/agent/browser_system_prompts.py
```
1 | # browser_prompts.py
2 | from datetime import datetime
3 | from typing import List, Optional
4 | 
5 | from langchain_core.messages import HumanMessage, SystemMessage
6 | 
7 | from browser_use.agent.views import ActionResult
8 | from browser_use.browser.views import BrowserState
9 | from browser_use.agent.prompts import SystemPrompt
10 | 
11 | from src.agent.browser_agent_output import StepInfo
12 | 
13 | 
14 | class BrowserSystemPrompt(SystemPrompt):
15 |     """System prompt containing rules and instructions for browser automation agent."""
16 | 
17 |     def important_rules(self) -> str:
18 |         """Provides key rules for JSON response format and handling actions."""
19 |         rules = (
20 |             "1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:\n"
21 |             "   {\n"
22 |             "     \"current_state\": {\n"
23 |             "       \"prev_action_evaluation\": \"Success|Failed|Unknown - ...\",\n"
24 |             "       \"important_contents\": \"...\",\n"
25 |             "       \"completed_contents\": \"...\",\n"
26 |             "       \"thought\": \"...\",\n"
27 |             "       \"summary\": \"...\"\n"
28 |             "     },\n"
29 |             "     \"action\": [\n"
30 |             "       {\n"
31 |             "         \"action_name\": {\n"
32 |             "           // action-specific parameters\n"
33 |             "         }\n"
34 |             "       }\n"
35 |             "     ]\n"
36 |             "   }\n\n"
37 |             "2. ACTIONS: You can specify multiple actions to be executed in sequence.\n"
38 |             "3. ELEMENT INTERACTION: (See documentation for more details.)\n"
39 |             "4. NAVIGATION & ERROR HANDLING: (See documentation for more details.)\n"
40 |             "5. TASK COMPLETION: (See documentation for more details.)\n"
41 |             "6. VISUAL CONTEXT: (See documentation for more details.)\n"
42 |             "7. Form filling: (See documentation for more details.)\n"
43 |             "8. ACTION SEQUENCING: (See documentation for more details.)\n"
44 |         )
45 |         rules += f"   - use maximum {self.max_actions_per_step} actions per sequence"
46 |         return rules
47 | 
48 |     def input_format(self) -> str:
49 |         """Describes the input structure provided to the agent."""
50 |         return (
51 |             "INPUT STRUCTURE:\n"
52 |             "1. Task: The user's instructions to complete.\n"
53 |             "2. Hints(Optional): Additional hints for guidance.\n"
54 |             "3. Memory: Important historical contents.\n"
55 |             "4. Task Progress: Items completed so far.\n"
56 |             "5. Current URL: The webpage currently being viewed.\n"
57 |             "6. Available Tabs: List of open browser tabs.\n"
58 |             "7. Interactive Elements: List in the format index[:]<element_type> ... ."
59 |         )
60 | 
61 |     def get_system_message(self) -> SystemMessage:
62 |         """Creates the final system prompt message with current date and rules."""
63 |         current_time = self.current_date.strftime('%Y-%m-%d %H:%M')
64 |         prompt_text = (
65 |             f"You are a precise browser automation agent that interacts with websites.\n"
66 |             f"Current date and time: {current_time}\n\n"
67 |             f"{self.input_format()}\n"
68 |             f"{self.important_rules()}\n"
69 |             f"Functions:\n{self.default_action_description}\n\n"
70 |             "Remember: Your responses must be valid JSON matching the specified format."
71 |         )
72 |         return SystemMessage(content=prompt_text)
73 | 
74 | 
75 | class BrowserMessagePrompt:
76 |     """
77 |     Formats the browser state and previous action results into a HumanMessage
78 |     that the language model can process.
79 |     """
80 | 
81 |     def __init__(
82 |         self,
83 |         state: BrowserState,
84 |         result: Optional[List[ActionResult]] = None,
85 |         include_attributes: Optional[List[str]] = None,
86 |         max_error_length: int = 400,
87 |         step_info: Optional[StepInfo] = None,
88 |     ):
89 |         self.state = state
90 |         self.result = result
91 |         self.include_attributes = include_attributes or []
92 |         self.max_error_length = max_error_length
93 |         self.step_info = step_info
94 | 
95 |     def _compose_state_description(self) -> str:
96 |         """Creates a description of current task, memory, and browser state."""
97 |         return (
98 |             f"1. Task: {self.step_info.task}\n"
99 |             f"2. Hints(Optional):\n{self.step_info.additional_info}\n"  # updated line
100 |             f"3. Memory:\n{self.step_info.memory}\n"
101 |             f"4. Task Progress:\n{self.step_info.task_progress}\n"
102 |             f"5. Current url: {self.state.url}\n"
103 |             f"6. Available tabs:\n{self.state.tabs}\n"
104 |             f"7. Interactive elements:\n"
105 |             f"{self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)}"
106 |         )
107 | 
108 |     def _append_results(self, description: str) -> str:
109 |         """Appends results of previous actions to the state description."""
110 |         if not self.result:
111 |             return description
112 |         for idx, action_result in enumerate(self.result, start=1):
113 |             if action_result.extracted_content:
114 |                 description += f"\nResult of action {idx}/{len(self.result)}: {action_result.extracted_content}"
115 |             if action_result.error:
116 |                 snippet = action_result.error[-self.max_error_length:]
117 |                 description += f"\nError of action {idx}/{len(self.result)}: ...{snippet}"
118 |         return description
119 | 
120 |     def get_user_message(self) -> HumanMessage:
121 |         """
122 |         Combines state information and previous action results
123 |         into a single HumanMessage.
124 |         """
125 |         description = self._compose_state_description()
126 |         description = self._append_results(description)
127 | 
128 |         if self.state.screenshot:
129 |             content = [
130 |                 {'type': 'text', 'text': description},
131 |                 {
132 |                     'type': 'image_url',
133 |                     'image_url': {'url': f"data:image/png;base64,{self.state.screenshot}"}
134 |                 },
135 |             ]
136 |             return HumanMessage(content=content)
137 |         return HumanMessage(content=description)
```

src/agent_runner.py
```
1 | # agent_runner.py
2 | import os
3 | import glob
4 | from typing import Tuple, Optional
5 | import traceback
6 | 
7 | from org_agent import run_org_agent
8 | from custom_agent import run_custom_agent
9 | from src.utils import utils  # Adjust import if needed
10 | 
11 | 
12 | async def run_browser_agent(
13 |     agent_type,
14 |     llm_provider,
15 |     llm_model_name,
16 |     llm_temperature,
17 |     llm_base_url,
18 |     llm_api_key,
19 |     use_own_browser,
20 |     headless,
21 |     disable_security,
22 |     window_w,
23 |     window_h,
24 |     save_recording_path,
25 |     task,
26 |     add_infos,
27 |     max_steps,
28 |     use_vision
29 | ):
30 |     # Ensure the recording directory exists
31 |     os.makedirs(save_recording_path, exist_ok=True)
32 | 
33 |     # Get the list of existing videos before the agent runs
34 |     existing_videos = set(
35 |         glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) +
36 |         glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]'))
37 |     )
38 | 
39 |     # Prepare the LLM
40 |     llm = utils.get_llm_model(
41 |         provider=llm_provider,
42 |         model_name=llm_model_name,
43 |         temperature=llm_temperature,
44 |         base_url=llm_base_url,
45 |         api_key=llm_api_key
46 |     )
47 | 
48 |     # Run the appropriate agent
49 |     try:
50 |         if agent_type == "org":
51 |             final_result, errors, model_actions, model_thoughts = await run_org_agent(
52 |                 llm=llm,
53 |                 headless=headless,
54 |                 disable_security=disable_security,
55 |                 window_w=window_w,
56 |                 window_h=window_h,
57 |                 save_recording_path=save_recording_path,
58 |                 task=task,
59 |                 max_steps=max_steps,
60 |                 use_vision=use_vision
61 |             )
62 |         elif agent_type == "custom":
63 |             final_result, errors, model_actions, model_thoughts = await run_custom_agent(
64 |                 llm=llm,
65 |                 use_own_browser=use_own_browser,
66 |                 headless=headless,
67 |                 disable_security=disable_security,
68 |                 window_w=window_w,
69 |                 window_h=window_h,
70 |                 save_recording_path=save_recording_path,
71 |                 task=task,
72 |                 add_infos=add_infos,
73 |                 max_steps=max_steps,
74 |                 use_vision=use_vision
75 |             )
76 |         else:
77 |             raise ValueError(f"Invalid agent type: {agent_type}")
78 |     except Exception as e:
79 |         # Catch any unexpected exceptions
80 |         traceback.print_exc()
81 |         final_result = ""
82 |         errors = str(e) + "\n" + traceback.format_exc()
83 |         model_actions = ""
84 |         model_thoughts = ""
85 | 
86 |     # Get the list of videos after the agent runs
87 |     new_videos = set(
88 |         glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) +
89 |         glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]'))
90 |     )
91 | 
92 |     # Find the newly created video
93 |     latest_video = None
94 |     created_videos = new_videos - existing_videos
95 |     if created_videos:
96 |         # Grab the first new video (or modify logic if multiple recordings possible)
97 |         latest_video = list(created_videos)[0]
98 | 
99 |     return final_result, errors, model_actions, model_thoughts, latest_video
```

src/browser/__init__.py
```
1 | # -*- coding: utf-8 -*-
```

src/browser/enhanced_playwright_browser.py
```
1 | # specialized_browser.py
2 | """
3 | Description:
4 |     This module defines a SpecializedBrowser class that overrides the `new_context`
5 |     method to return a CustomBrowserContext.
6 | """
7 | 
8 | import logging
9 | from typing import Optional
10 | 
11 | from browser_use.browser.browser import Browser, BrowserConfig
12 | from browser_use.browser.context import BrowserContextConfig, BrowserContext
13 | 
14 | from src.browser.enhanced_playwright_browser_context import EnhancedBrowserContext
15 | 
16 | logger = logging.getLogger(__name__)
17 | 
18 | 
19 | class EnhancedPlaywrightBrowser(Browser):
20 |     """
21 |     A specialized Browser implementation that yields a CustomBrowserContext
22 |     upon requesting a new context.
23 |     """
24 | 
25 |     async def new_context(
26 |             self,
27 |             config: BrowserContextConfig = BrowserContextConfig(),
28 |             context: Optional[BrowserContext] = None
29 |     ) -> BrowserContext:
30 |         """
31 |         Create and return a CustomBrowserContext.
32 | 
33 |         Parameters
34 |         ----------
35 |         config : BrowserContextConfig, optional
36 |             Configuration settings for the browser context.
37 |         context : Optional[BrowserContext], optional
38 |             An existing BrowserContext-like instance, if reusing contexts.
39 | 
40 |         Returns
41 |         -------
42 |         BrowserContext
43 |             A newly created or existing CustomBrowserContext.
44 |         """
45 |         logger.debug("Creating a new specialized browser context.")
46 |         return EnhancedBrowserContext(
47 |             config=config,
48 |             browser=self,
49 |             context=context
50 |         )
```

src/browser/enhanced_playwright_browser_context.py
```
1 | # enhanced_browser_context.py
2 | """
3 | Description:
4 |     This module defines an EnhancedBrowserContext class that extends the
5 |     base BrowserContext to include anti-detection measures, cookie loading,
6 |     and optional reuse of an existing context.
7 | """
8 | 
9 | import json
10 | import logging
11 | import os
12 | from typing import Optional
13 | 
14 | from playwright.async_api import Browser as PlaywrightBrowser
15 | 
16 | from browser_use.browser.context import BrowserContext, BrowserContextConfig
17 | from browser_use.browser.browser import Browser
18 | 
19 | logger = logging.getLogger(__name__)
20 | 
21 | 
22 | class EnhancedBrowserContext(BrowserContext):
23 |     """
24 |     An enhanced BrowserContext that includes anti-detection measures,
25 |     cookie management, and optional reuse of an existing context.
26 |     """
27 | 
28 |     def __init__(
29 |         self,
30 |         browser: Browser,
31 |         config: BrowserContextConfig = BrowserContextConfig(),
32 |         context: Optional[BrowserContext] = None
33 |     ):
34 |         """
35 |         Initialize an EnhancedBrowserContext.
36 | 
37 |         Parameters
38 |         ----------
39 |         browser : Browser
40 |             The parent Browser instance.
41 |         config : BrowserContextConfig, optional
42 |             Configuration for the browser context.
43 |         context : Optional[BrowserContext], optional
44 |             An existing BrowserContext to reuse, if available.
45 |         """
46 |         super().__init__(browser, config)
47 |         self.context = context
48 | 
49 |     async def _create_context(self, browser: PlaywrightBrowser) -> BrowserContext:
50 |         """
51 |         Create or reuse a browser context with added anti-detection measures.
52 |         Loads cookies if available and starts tracing if configured.
53 | 
54 |         Parameters
55 |         ----------
56 |         browser : PlaywrightBrowser
57 |             The underlying Playwright Browser object.
58 | 
59 |         Returns
60 |         -------
61 |         BrowserContext
62 |             The newly created or existing browser context.
63 |         """
64 |         # Reuse provided context if available
65 |         if self.context:
66 |             logger.debug("Using provided existing browser context.")
67 |             return self.context
68 | 
69 |         # Connect to an existing Chrome instance or create a new context
70 |         if self.browser.config.chrome_instance_path and browser.contexts:
71 |             logger.debug("Connecting to existing Chrome instance.")
72 |             context = browser.contexts[0]
73 |         else:
74 |             logger.debug("Creating a new Playwright browser context.")
75 |             context = await browser.new_context(
76 |                 viewport=self.config.browser_window_size,
77 |                 no_viewport=False,
78 |                 user_agent=(
79 |                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
80 |                     "(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
81 |                 ),
82 |                 java_script_enabled=True,
83 |                 bypass_csp=self.config.disable_security,
84 |                 ignore_https_errors=self.config.disable_security,
85 |                 record_video_dir=self.config.save_recording_path,
86 |                 record_video_size=self.config.browser_window_size
87 |             )
88 | 
89 |         # Start tracing if configured
90 |         if self.config.trace_path:
91 |             logger.info("Starting trace recording for the browser context.")
92 |             await context.tracing.start(screenshots=True, snapshots=True, sources=True)
93 | 
94 |         # Load cookies if they exist
95 |         if self.config.cookies_file and os.path.exists(self.config.cookies_file):
96 |             with open(self.config.cookies_file, 'r') as f:
97 |                 cookies = json.load(f)
98 |                 logger.info(f"Loaded {len(cookies)} cookies from {self.config.cookies_file}")
99 |                 await context.add_cookies(cookies)
100 | 
101 |         # Inject anti-detection scripts
102 |         logger.debug("Injecting anti-detection scripts into the browser context.")
103 |         await context.add_init_script(
104 |             """
105 |             // Hide the 'webdriver' property from navigator
106 |             Object.defineProperty(navigator, 'webdriver', {
107 |                 get: () => undefined
108 |             });
109 | 
110 |             // Spoof languages
111 |             Object.defineProperty(navigator, 'languages', {
112 |                 get: () => ['en-US', 'en']
113 |             });
114 | 
115 |             // Fake plugins
116 |             Object.defineProperty(navigator, 'plugins', {
117 |                 get: () => [1, 2, 3, 4, 5]
118 |             });
119 | 
120 |             // Mock chrome runtime object
121 |             window.chrome = { runtime: {} };
122 | 
123 |             // Override permissions query for notifications
124 |             const originalQuery = window.navigator.permissions.query;
125 |             window.navigator.permissions.query = (parameters) => (
126 |                 parameters.name === 'notifications'
127 |                     ? Promise.resolve({ state: Notification.permission })
128 |                     : originalQuery(parameters)
129 |             );
130 |             """
131 |         )
132 | 
133 |         return context
```

src/components/agent_settings.py
```
1 | """Agent settings component for the Browser Use WebUI"""
2 | 
3 | import gradio as gr
4 | 
5 | def create_agent_settings_tab():
6 |     """Creates and returns the agent settings tab"""
7 |     with gr.TabItem("🤖 Agent Settings", id="agent_settings_tab"):
8 |         with gr.Group():
9 |             agent_type = gr.Radio(
10 |                 ["org", "custom"],
11 |                 label="Agent Type",
12 |                 value="custom",
13 |                 info="Select the type of agent to use"
14 |             )
15 |             max_steps = gr.Slider(
16 |                 minimum=1,
17 |                 maximum=200,
18 |                 value=100,
19 |                 step=1,
20 |                 label="Max Run Steps",
21 |                 info="Maximum number of steps the agent will take"
22 |             )
23 |             use_vision = gr.Checkbox(
24 |                 label="Use Vision",
25 |                 value=False,
26 |                 info="Enable visual processing capabilities"
27 |             )
28 |             
29 |     return {
30 |         "agent_type": agent_type,
31 |         "max_steps": max_steps,
32 |         "use_vision": use_vision
33 |     }
```

src/components/browser_settings.py
```
1 | """Browser settings component for the Browser Use WebUI"""
2 | 
3 | import gradio as gr
4 | 
5 | def create_browser_settings_tab():
6 |     """Creates and returns the browser settings tab"""
7 |     with gr.TabItem("🌐 Browser Settings", id="browser_settings_tab"):
8 |         with gr.Group():
9 |             with gr.Row():
10 |                 use_own_browser = gr.Checkbox(
11 |                     label="Use Own Browser",
12 |                     value=False,
13 |                     info="Use your existing browser instance"
14 |                 )
15 |                 headless = gr.Checkbox(
16 |                     label="Headless Mode",
17 |                     value=False,
18 |                     info="Run browser without GUI"
19 |                 )
20 |                 disable_security = gr.Checkbox(
21 |                     label="Disable Security",
22 |                     value=True,
23 |                     info="Disable browser security features"
24 |                 )
25 |             with gr.Row():
26 |                 window_width = gr.Number(
27 |                     label="Window Width",
28 |                     value=1920,
29 |                     info="Browser window width"
30 |                 )
31 |                 window_height = gr.Number(
32 |                     label="Window Height",
33 |                     value=1080,
34 |                     info="Browser window height"
35 |                 )
36 |             save_recording_path = gr.Textbox(
37 |                 label="Recording Path",
38 |                 placeholder="e.g. ./tmp/record_videos",
39 |                 value="./tmp/record_videos",
40 |                 info="Path to save browser recordings"
41 |             )
42 |             
43 |     return {
44 |         "use_own_browser": use_own_browser,
45 |         "headless": headless,
46 |         "disable_security": disable_security,
47 |         "window_width": window_width,
48 |         "window_height": window_height,
49 |         "save_recording_path": save_recording_path
50 |     }
```

src/components/header.py
```
1 | """Header component for the Browser Use WebUI"""
2 | 
3 | import gradio as gr
4 | 
5 | 
6 | def create_header():
7 |     """Creates and returns the header section of the UI"""
8 |     return gr.Markdown(
9 |         """
10 |         # browser-use powered by Reinforcement learning
11 |         ### Control your browser with AI assistance
12 |         """,
13 |         elem_classes=["header-text"]
14 |     )
```

src/components/llm_configuration.py
```
1 | import gradio as gr
2 | 
3 | def update_model_and_url(selected_provider):
4 |     """Returns default model name and base URL based on the selected provider."""
5 |     defaults = {
6 |         "anthropic": {"model_name": "claude-3-5-haiku-latest", "base_url": "https://api.anthropic.com"},
7 |         "openai": {"model_name": "gpt-3.5-turbo", "base_url": "https://api.openai.com"},
8 |         "gemini": {"model_name": "gemini-2.0-flash-exp", "base_url": "https://api.gemini.com"},
9 |         "deepseek": {"model_name": "deepseek-chat", "base_url": "https://api.deepseek.com"},
10 |         "ollama": {"model_name": "llama3", "base_url": "http://localhost:11434/api/generate"},
11 |     }
12 |     provider_defaults = defaults.get(selected_provider, {"model_name": "", "base_url": ""})
13 |     return provider_defaults["model_name"], provider_defaults["base_url"]
14 | 
15 | def create_llm_configuration_tab():
16 |     """Creates and returns the LLM configuration tab"""
17 |     with gr.TabItem("🔧 LLM Configuration", id="llm_configuration_tab"):
18 |         with gr.Group():
19 |             llm_provider = gr.Dropdown(
20 |                 ["anthropic", "openai", "gemini", "azure_openai", "deepseek", "ollama"],
21 |                 label="LLM Provider",
22 |                 value="deepseek",
23 |                 info="Select your preferred language model provider"
24 |             )
25 |             llm_model_name = gr.Textbox(
26 |                 label="Model Name",
27 |                 value="deepseek-chat",
28 |                 info="Specify the model to use"
29 |             )
30 |             llm_temperature = gr.Slider(
31 |                 minimum=0.0,
32 |                 maximum=2.0,
33 |                 value=0.7,
34 |                 step=0.1,
35 |                 label="Temperature",
36 |                 info="Controls randomness in model outputs"
37 |             )
38 |             with gr.Row():
39 |                 llm_base_url = gr.Textbox(
40 |                     label="Base URL",
41 |                     value="https://api.deepseek.com",
42 |                     info="API endpoint URL (if required)"
43 |                 )
44 |                 llm_api_key = gr.Textbox(
45 |                     label="API Key",
46 |                     type="password",
47 |                     info="Your API key"
48 |                 )
49 | 
50 |             # Set up a change event on the provider dropdown to update model name and base URL
51 |             llm_provider.change(
52 |                 fn=update_model_and_url,
53 |                 inputs=llm_provider,
54 |                 outputs=[llm_model_name, llm_base_url]
55 |             )
56 | 
57 |     return {
58 |         "llm_provider": llm_provider,
59 |         "llm_model_name": llm_model_name,
60 |         "llm_temperature": llm_temperature,
61 |         "llm_base_url": llm_base_url,
62 |         "llm_api_key": llm_api_key
63 |     }
```

src/components/recordings.py
```
1 | """Recordings component for the Browser Use WebUI"""
2 | 
3 | import gradio as gr
4 | 
5 | def create_recordings_tab():
6 |     """Creates and returns the recordings tab"""
7 |     with gr.TabItem("🎬 Recordings", id="recordings_tab"):
8 |         recording_display = gr.Video(label="Latest Recording")
9 |         
10 |         with gr.Group():
11 |             gr.Markdown("### Results")
12 |             with gr.Row():
13 |                 with gr.Column():
14 |                     final_result_output = gr.Textbox(
15 |                         label="Final Result",
16 |                         lines=3,
17 |                         show_label=True
18 |                     )
19 |                 with gr.Column():
20 |                     errors_output = gr.Textbox(
21 |                         label="Errors",
22 |                         lines=3,
23 |                         show_label=True
24 |                     )
25 |             with gr.Row():
26 |                 with gr.Column():
27 |                     model_actions_output = gr.Textbox(
28 |                         label="Model Actions",
29 |                         lines=3,
30 |                         show_label=True
31 |                     )
32 |                 with gr.Column():
33 |                     model_thoughts_output = gr.Textbox(
34 |                         label="Model Thoughts",
35 |                         lines=3,
36 |                         show_label=True
37 |                     )
38 |                     
39 |     return {
40 |         "recording_display": recording_display,
41 |         "final_result_output": final_result_output,
42 |         "errors_output": errors_output,
43 |         "model_actions_output": model_actions_output,
44 |         "model_thoughts_output": model_thoughts_output
45 |     }
```

src/components/task_settings.py
```
1 | """Task settings component for the Browser Use WebUI"""
2 | 
3 | import gradio as gr
4 | 
5 | def create_task_settings_tab():
6 |     """Creates and returns the task settings tab"""
7 |     with gr.TabItem("📝 Task Settings", id="task_settings_tab"):
8 |         task_description = gr.Textbox(
9 |             label="Task Description",
10 |             lines=4,
11 |             placeholder="Enter your task here...",
12 |             value="go to google.com and type 'OpenAI', click search and give me the first URL",
13 |             info="Describe what you want the agent to do"
14 |         )
15 |         additional_information = gr.Textbox(
16 |             label="Additional Information",
17 |             lines=3,
18 |             placeholder="Add any helpful context or instructions...",
19 |             info="Optional hints to help the LLM complete the task"
20 |         )
21 |         
22 |         with gr.Row():
23 |             run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2, elem_id="run_button")
24 |             stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1, elem_id="stop_button")
25 |             
26 |     return {
27 |         "task_description": task_description,
28 |         "additional_information": additional_information,
29 |         "run_button": run_button,
30 |         "stop_button": stop_button
31 |     }
```

src/controller/__init__.py
```
1 | # -*- coding: utf-8 -*-
```

src/controller/custom_controller.py
```
1 | # -*- coding: utf-8 -*-
2 | """
3 | Filename: custom_action.py
4 | 
5 | Description:
6 |     This module implements a CustomController that extends the base Controller
7 |     to register custom actions for copying text to the system clipboard and
8 |     pasting text from the clipboard into a browser context.
9 | """
10 | 
11 | import logging
12 | import pyperclip
13 | 
14 | from browser_use.controller.service import Controller
15 | from browser_use.agent.views import ActionResult
16 | from browser_use.browser.context import BrowserContext
17 | 
18 | logger = logging.getLogger(__name__)
19 | 
20 | 
21 | class CustomController(Controller):
22 |     """
23 |     A custom controller extending the base Controller to register unique actions
24 |     such as copying and pasting text using the system clipboard via pyperclip.
25 |     """
26 | 
27 |     def __init__(self):
28 |         super().__init__()
29 |         self._register_custom_actions()
30 | 
31 |     def _register_custom_actions(self) -> None:
32 |         """
33 |         Register all custom browser actions supported by this controller.
34 |         """
35 | 
36 |         @self.registry.action('Copy text to clipboard')
37 |         def copy_to_clipboard(text: str) -> ActionResult:
38 |             """
39 |             Copy the given text to the system clipboard.
40 | 
41 |             Parameters
42 |             ----------
43 |             text : str
44 |                 The text content to be copied.
45 | 
46 |             Returns
47 |             -------
48 |             ActionResult
49 |                 Contains the extracted content, which is the text that was copied.
50 |             """
51 |             logger.debug("Copying text to clipboard: %r", text)
52 |             pyperclip.copy(text)
53 |             return ActionResult(extracted_content=text)
54 | 
55 |         @self.registry.action('Paste text from clipboard', requires_browser=True)
56 |         async def paste_from_clipboard(browser: BrowserContext) -> ActionResult:
57 |             """
58 |             Paste the text from the system clipboard into the currently active
59 |             element of the browser page (where the cursor is focused).
60 | 
61 |             Parameters
62 |             ----------
63 |             browser : BrowserContext
64 |                 The current browser context. Used to retrieve the active page
65 |                 and type the clipboard content.
66 | 
67 |             Returns
68 |             -------
69 |             ActionResult
70 |                 Contains the extracted content, which is the text that was pasted.
71 |             """
72 |             text = pyperclip.paste()
73 |             logger.debug("Pasting text from clipboard: %r", text)
74 | 
75 |             page = await browser.get_current_page()
76 |             await page.keyboard.type(text)
77 | 
78 |             return ActionResult(extracted_content=text)
```

src/custom_agent.py
```
1 | import os
2 | import traceback
3 | 
4 | from browser_use.browser.browser import BrowserConfig
5 | from browser_use.browser.context import BrowserContextWindowSize, BrowserContextConfig
6 | from playwright.async_api import async_playwright
7 | 
8 | from src.agent.browser_agent import BrowserAgent
9 | from src.agent.browser_system_prompts import BrowserSystemPrompt
10 | from src.browser.enhanced_playwright_browser import EnhancedPlaywrightBrowser
11 | from src.controller.custom_controller import CustomController
12 | 
13 | async def run_custom_agent(
14 |         llm,
15 |         use_own_browser,
16 |         headless,
17 |         disable_security,
18 |         window_w,
19 |         window_h,
20 |         save_recording_path,
21 |         task,
22 |         add_infos,
23 |         max_steps,
24 |         use_vision
25 | ):
26 |     """Run the 'custom' agent using a specialized browser and custom agent."""
27 |     controller = CustomController()
28 |     playwright = None
29 |     browser_context_ = None
30 | 
31 |     try:
32 |         # Optional: Use existing Chrome profile
33 |         if use_own_browser:
34 |             playwright = await async_playwright().start()
35 |             chrome_exe = os.getenv("CHROME_PATH", "")
36 |             chrome_use_data = os.getenv("CHROME_USER_DATA", "")
37 |             browser_context_ = await playwright.chromium.launch_persistent_context(
38 |                 user_data_dir=chrome_use_data,
39 |                 executable_path=chrome_exe,
40 |                 no_viewport=False,
41 |                 headless=headless,
42 |                 user_agent=(
43 |                     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
44 |                     '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
45 |                 ),
46 |                 java_script_enabled=True,
47 |                 bypass_csp=disable_security,
48 |                 ignore_https_errors=disable_security,
49 |                 record_video_dir=save_recording_path if save_recording_path else None,
50 |                 record_video_size={'width': window_w, 'height': window_h}
51 |             )
52 |         else:
53 |             browser_context_ = None
54 | 
55 |         # Create a new specialized browser with your custom logic
56 |         browser = EnhancedPlaywrightBrowser(
57 |             config=BrowserConfig(
58 |                 headless=headless,
59 |                 disable_security=disable_security,
60 |                 extra_chromium_args=[f'--window-size={window_w},{window_h}'],
61 |             )
62 |         )
63 | 
64 |         async with await browser.new_context(
65 |                 config=BrowserContextConfig(
66 |                     trace_path='./tmp/result_processing',
67 |                     save_recording_path=save_recording_path if save_recording_path else None,
68 |                     no_viewport=False,
69 |                     browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
70 |                 ),
71 |                 context=browser_context_
72 |         ) as browser_context:
73 |             agent = BrowserAgent(
74 |                 task=task,
75 |                 additional_info=add_infos,
76 |                 use_vision=use_vision,
77 |                 llm=llm,
78 |                 browser_context=browser_context,
79 |                 controller=controller,
80 |                 system_prompt_class=BrowserSystemPrompt
81 |             )
82 |             history = await agent.run(max_steps=max_steps)
83 | 
84 |             final_result = history.final_result()
85 |             errors = history.errors()
86 |             model_actions = history.model_actions()
87 |             model_thoughts = history.model_thoughts()
88 | 
89 |     except Exception as e:
90 |         traceback.print_exc()
91 |         final_result = ""
92 |         errors = str(e) + "\n" + traceback.format_exc()
93 |         model_actions = ""
94 |         model_thoughts = ""
95 |     finally:
96 |         # Close persistent context if used
97 |         if browser_context_:
98 |             await browser_context_.close()
99 | 
100 |         # Stop the Playwright object
101 |         if playwright:
102 |             await playwright.stop()
103 | 
104 |         # Close the specialized browser
105 |         await browser.close()
106 | 
107 |     return final_result, errors, model_actions, model_thoughts
```

src/main.py
```
1 | """Main entry point for the Browser Use WebUI application."""
2 | 
3 | import argparse
4 | import logging
5 | from typing import Optional
6 | from ui import create_application_ui
7 | 
8 | def _configure_logging() -> None:
9 |     """Configure basic logging for the application."""
10 |     logging.basicConfig(
11 |         level=logging.INFO,
12 |         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
13 |     )
14 | 
15 | def _parse_arguments() -> argparse.Namespace:
16 |     """
17 |     Parse and validate command line arguments.
18 |     
19 |     Returns:
20 |         Parsed command line arguments
21 |     """
22 |     parser = argparse.ArgumentParser(
23 |         description="Launch the Browser Use WebUI application"
24 |     )
25 |     
26 |     parser.add_argument(
27 |         "--ip",
28 |         type=str,
29 |         default="127.0.0.1",
30 |         help="IP address to bind the server to (default: 127.0.0.1)"
31 |     )
32 |     
33 |     parser.add_argument(
34 |         "--port",
35 |         type=int,
36 |         default=7788,
37 |         help="Port number to listen on (default: 7788)"
38 |     )
39 |     
40 |     parser.add_argument(
41 |         "--theme",
42 |         type=str,
43 |         default="Ocean",
44 |         choices=["Default", "Soft", "Monochrome", "Glass", "Origin", "Citrus", "Ocean"],
45 |         help="Visual theme for the application interface (default: Ocean)"
46 |     )
47 |     
48 |     parser.add_argument(
49 |         "--dark-mode",
50 |         action="store_true",
51 |         help="Enable dark mode for the application interface"
52 |     )
53 |     
54 |     return parser.parse_args()
55 | 
56 | def _launch_application(ip: str, port: int, theme: str) -> None:
57 |     """
58 |     Launch the application with the specified configuration.
59 |     
60 |     Args:
61 |         ip: IP address to bind to
62 |         port: Port number to listen on
63 |         theme: Visual theme to use
64 |     """
65 |     try:
66 |         logging.info("Starting Browser Use WebUI application")
67 |         interface = create_application_ui(theme_name=theme)
68 |         interface.launch(
69 |             server_name=ip,
70 |             server_port=port
71 |         )
72 |     except Exception as e:
73 |         logging.error(f"Failed to launch application: {str(e)}")
74 |         raise
75 | 
76 | def main() -> None:
77 |     """Main entry point for the Browser Use WebUI application."""
78 |     _configure_logging()
79 |     args = _parse_arguments()
80 |     _launch_application(args.ip, args.port, args.theme)
81 | 
82 | if __name__ == '__main__':
83 |     main()
```

src/org_agent.py
```
1 | # org_agent.py
2 | from browser_use.browser.browser import Browser, BrowserConfig
3 | from browser_use.browser.context import (
4 |     BrowserContext,
5 |     BrowserContextConfig,
6 |     BrowserContextWindowSize,
7 | )
8 | from browser_use.agent.service import Agent
9 | 
10 | 
11 | async def run_org_agent(
12 |     llm,
13 |     headless,
14 |     disable_security,
15 |     window_w,
16 |     window_h,
17 |     save_recording_path,
18 |     task,
19 |     max_steps,
20 |     use_vision
21 | ):
22 |     """Run the 'org' agent using the standard Browser and Agent classes."""
23 |     browser = Browser(
24 |         config=BrowserConfig(
25 |             headless=headless,
26 |             disable_security=disable_security,
27 |             extra_chromium_args=[f'--window-size={window_w},{window_h}'],
28 |         )
29 |     )
30 | 
31 |     async with await browser.new_context(
32 |         config=BrowserContextConfig(
33 |             trace_path='./tmp/traces',
34 |             save_recording_path=save_recording_path if save_recording_path else None,
35 |             no_viewport=False,
36 |             browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
37 |         )
38 |     ) as browser_context:
39 |         agent = Agent(
40 |             task=task,
41 |             llm=llm,
42 |             use_vision=use_vision,
43 |             browser_context=browser_context,
44 |         )
45 | 
46 |         history = await agent.run(max_steps=max_steps)
47 |         final_result = history.final_result()
48 |         errors = history.errors()
49 |         model_actions = history.model_actions()
50 |         model_thoughts = history.model_thoughts()
51 | 
52 |     await browser.close()
53 |     return final_result, errors, model_actions, model_thoughts
```

src/themes.py
```
1 | """Theme configuration for the Browser Use WebUI"""
2 | 
3 | from gradio.themes import Default, Soft, Monochrome, Glass, Origin, Citrus, Ocean
4 | 
5 | class ProductionTheme(Soft):
6 |     """Custom theme extending the Soft theme with refined styling"""
7 |     
8 |     def __init__(self, primary_hue="blue", secondary_hue="orange", neutral_hue="gray"):
9 |         super().__init__(primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue)
10 |         self.set(
11 |             body_background_fill="#f9fafb",
12 |             panel_background_fill="white",
13 |         )
14 | 
15 | THEME_MAP = {
16 |     "Default": Default(),
17 |     "Soft": Soft(),
18 |     "Monochrome": Monochrome(),
19 |     "Glass": Glass(),
20 |     "Origin": Origin(),
21 |     "Citrus": Citrus(),
22 |     "Ocean": Ocean(),
23 |     "Production": ProductionTheme()
24 | }
25 | 
26 | BASE_CSS = """
27 | /* Container to be responsive, centered, with some padding */
28 | .gradio-container {
29 |     max-width: 1200px !important;
30 |     margin: auto !important;
31 |     padding-top: 20px !important;
32 |     font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
33 | }
34 | 
35 | /* Header text styling */
36 | .header-text {
37 |     text-align: center;
38 |     margin-bottom: 30px;
39 | }
40 | 
41 | /* Tab style improvements */
42 | .gradio-container .tabs .tabitem {
43 |     background-color: #ffffff;
44 |     border-radius: 8px;
45 |     padding: 20px;
46 | }
47 | 
48 | /* Smoother corners for group containers */
49 | .gradio-container .gr-group {
50 |     border-radius: 8px;
51 |     background: #fefefe;
52 |     padding: 16px;
53 |     border: 1px solid #eaeaea;
54 | }
55 | 
56 | /* Style the run and stop buttons */
57 | button#run_button {
58 |     background-color: #0b5ed7 !important;
59 |     color: #fff !important;
60 |     border: none !important;
61 |     padding: 12px 24px !important;
62 |     font-size: 16px !important;
63 |     border-radius: 6px !important;
64 |     cursor: pointer !important;
65 |     transition: background-color 0.3s ease;
66 | }
67 | 
68 | button#run_button:hover {
69 |     background-color: #084ca1 !important;
70 | }
71 | 
72 | button#stop_button {
73 |     background-color: #dc3545 !important;
74 |     color: #fff !important;
75 |     border: none !important;
76 |     padding: 12px 24px !important;
77 |     font-size: 16px !important;
78 |     border-radius: 6px !important;
79 |     cursor: pointer !important;
80 |     transition: background-color 0.3s ease;
81 | }
82 | 
83 | button#stop_button:hover {
84 |     background-color: #b52d36 !important;
85 | }
86 | 
87 | /* Textboxes, checkboxes, radio buttons, etc. */
88 | .gr-textbox textarea {
89 |     background: #fafafa !important;
90 |     border: 1px solid #ccc !important;
91 |     border-radius: 6px !important;
92 |     transition: border-color 0.3s ease;
93 | }
94 | 
95 | .gr-textbox textarea:focus {
96 |     outline: none !important;
97 |     border-color: #888 !important;
98 | }
99 | 
100 | /* Make label texts slightly bolder and spaced */
101 | label {
102 |     font-weight: 600 !important;
103 |     margin-bottom: 4px !important;
104 | }
105 | 
106 | /* A subtle hover effect for textual inputs */
107 | .gr-textbox:hover textarea {
108 |     border-color: #bbb !important;
109 | }
110 | """
111 | 
112 | BASE_JS = """
113 | function refresh() {
114 |     const url = new URL(window.location);
115 |     if (url.searchParams.get('__theme') !== 'dark') {
116 |         url.searchParams.set('__theme', 'dark');
117 |         window.location.href = url.href;
118 |     }
119 | }
120 | """
```

src/ui.py
```
1 | """Main UI module for Browser Use WebUI"""
2 | 
3 | import gradio as gr
4 | from typing import Dict, Any
5 | from themes import THEME_MAP, BASE_CSS, BASE_JS
6 | from components.header import create_header
7 | from components.agent_settings import create_agent_settings_tab
8 | from components.llm_configuration import create_llm_configuration_tab
9 | from components.browser_settings import create_browser_settings_tab
10 | from components.task_settings import create_task_settings_tab
11 | from components.recordings import create_recordings_tab
12 | from agent_runner import run_browser_agent
13 | 
14 | def _create_application_tabs() -> Dict[str, Any]:
15 |     """Create and return all application tabs"""
16 |     return {
17 |         "agent_settings": create_agent_settings_tab(),
18 |         "llm_configuration": create_llm_configuration_tab(),
19 |         "browser_settings": create_browser_settings_tab(),
20 |         "task_settings": create_task_settings_tab(),
21 |         "recordings": create_recordings_tab()
22 |     }
23 | 
24 | def _wire_up_event_handlers(interface: gr.Blocks, tabs: Dict[str, Any]) -> None:
25 |     """Configure all event handlers for the application"""
26 |     task_settings_tab = tabs["task_settings"]
27 |     agent_settings_tab = tabs["agent_settings"]
28 |     llm_configuration_tab = tabs["llm_configuration"]
29 |     browser_settings_tab = tabs["browser_settings"]
30 |     recordings_tab = tabs["recordings"]
31 | 
32 |     task_settings_tab["run_button"].click(
33 |         fn=run_browser_agent,
34 |         inputs=[
35 |             agent_settings_tab["agent_type"],
36 |             llm_configuration_tab["llm_provider"],
37 |             llm_configuration_tab["llm_model_name"],
38 |             llm_configuration_tab["llm_temperature"],
39 |             llm_configuration_tab["llm_base_url"],
40 |             llm_configuration_tab["llm_api_key"],
41 |             browser_settings_tab["use_own_browser"],
42 |             browser_settings_tab["headless"],
43 |             browser_settings_tab["disable_security"],
44 |             browser_settings_tab["window_width"],
45 |             browser_settings_tab["window_height"],
46 |             browser_settings_tab["save_recording_path"],
47 |             task_settings_tab["task_description"],
48 |             task_settings_tab["additional_information"],
49 |             agent_settings_tab["max_steps"],
50 |             agent_settings_tab["use_vision"]
51 |         ],
52 |         outputs=[
53 |             recordings_tab["final_result_output"],
54 |             recordings_tab["errors_output"],
55 |             recordings_tab["model_actions_output"],
56 |             recordings_tab["model_thoughts_output"],
57 |             recordings_tab["recording_display"]
58 |         ]
59 |     )
60 | 
61 | def create_application_ui(theme_name: str = "Production") -> gr.Blocks:
62 |     """
63 |     Creates and returns the main application UI.
64 |     
65 |     Args:
66 |         theme_name: Name of the theme to use (must be in THEME_MAP)
67 |         
68 |     Returns:
69 |         Configured Gradio interface
70 |     """
71 |     if theme_name not in THEME_MAP:
72 |         raise ValueError(f"Invalid theme name: {theme_name}")
73 |         
74 |     with gr.Blocks(
75 |             title="Browser Use WebUI",
76 |             theme=THEME_MAP[theme_name],
77 |             css=BASE_CSS,
78 |             js=BASE_JS
79 |     ) as interface:
80 |         
81 |         # Create header section
82 |         create_header()
83 |         
84 |         # Create main tabs
85 |         tabs = _create_application_tabs()
86 |         
87 |         # Configure event handlers
88 |         _wire_up_event_handlers(interface, tabs)
89 |         
90 |     return interface
```

src/utils/__init__.py
```
1 | # -*- coding: utf-8 -*-
```

src/utils/utils.py
```
1 | # -*- coding: utf-8 -*-
2 | """
3 | Filename: utils.py
4 | 
5 | Description:
6 |     Utility functions for loading language models (LLMs) from various providers
7 |     (OpenAI, Anthropic, DeepSeek, Gemini, Ollama, Azure OpenAI, etc.) and for
8 |     encoding image files into Base64 strings.
9 | """
10 | 
11 | import os
12 | import base64
13 | import logging
14 | from typing import Optional, Union
15 | 
16 | # Your custom langchain-like imports
17 | from langchain_openai import ChatOpenAI, AzureChatOpenAI
18 | from langchain_anthropic import ChatAnthropic
19 | from langchain_google_genai import ChatGoogleGenerativeAI
20 | from langchain_ollama import ChatOllama
21 | 
22 | logger = logging.getLogger(__name__)
23 | 
24 | 
25 | def get_llm_model(provider: str, **kwargs) -> Union[
26 |     ChatOpenAI, AzureChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatOllama
27 | ]:
28 |     """
29 |     Obtain a language model instance based on the specified provider.
30 | 
31 |     Parameters
32 |     ----------
33 |     provider : str
34 |         The LLM provider identifier (e.g., 'openai', 'anthropic', etc.).
35 |     **kwargs :
36 |         Arbitrary keyword arguments, including base_url, api_key, model_name,
37 |         temperature, etc.
38 | 
39 |     Returns
40 |     -------
41 |     Union[ChatOpenAI, AzureChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatOllama]
42 |         A configured LLM client from one of the supported providers.
43 | 
44 |     Raises
45 |     ------
46 |     ValueError
47 |         If an unsupported provider is specified.
48 |     """
49 |     logger.debug("Fetching LLM model for provider: %s with kwargs: %s", provider, kwargs)
50 | 
51 |     # -------------------------
52 |     # Anthropic
53 |     # -------------------------
54 |     if provider == 'anthropic':
55 |         base_url = kwargs.get("base_url") or os.getenv("ANTHROPIC_ENDPOINT", "https://api.anthropic.com")
56 |         api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
57 |         model_name = kwargs.get("model_name", "claude-3-5-sonnet-20240620")
58 |         temperature = kwargs.get("temperature", 0.0)
59 | 
60 |         logger.debug("Configuring ChatAnthropic with model=%s, base_url=%s", model_name, base_url)
61 |         return ChatAnthropic(
62 |             model_name=model_name,
63 |             temperature=temperature,
64 |             base_url=base_url,
65 |             api_key=api_key
66 |         )
67 | 
68 |     # -------------------------
69 |     # OpenAI
70 |     # -------------------------
71 |     elif provider == 'openai':
72 |         base_url = kwargs.get("base_url") or os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
73 |         api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY", "")
74 |         model_name = kwargs.get("model_name", "gpt-4o")
75 |         temperature = kwargs.get("temperature", 0.0)
76 | 
77 |         logger.debug("Configuring ChatOpenAI with model=%s, base_url=%s", model_name, base_url)
78 |         return ChatOpenAI(
79 |             model=model_name,
80 |             temperature=temperature,
81 |             base_url=base_url,
82 |             api_key=api_key
83 |         )
84 | 
85 |     # -------------------------
86 |     # DeepSeek
87 |     # -------------------------
88 |     elif provider == 'deepseek':
89 |         base_url = kwargs.get("base_url") or os.getenv("DEEPSEEK_ENDPOINT", "")
90 |         api_key = kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
91 |         model_name = kwargs.get("model_name", "deepseek-chat")
92 |         temperature = kwargs.get("temperature", 0.0)
93 | 
94 |         logger.debug("Configuring DeepSeek ChatOpenAI with model=%s, base_url=%s", model_name, base_url)
95 |         return ChatOpenAI(
96 |             model=model_name,
97 |             temperature=temperature,
98 |             base_url=base_url,
99 |             api_key=api_key
100 |         )
101 | 
102 |     # -------------------------
103 |     # Gemini (Google Generative AI)
104 |     # -------------------------
105 |     elif provider == 'gemini':
106 |         api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY", "")
107 |         model_name = kwargs.get("model_name", "gemini-2.0-flash-exp")
108 |         temperature = kwargs.get("temperature", 0.0)
109 | 
110 |         logger.debug("Configuring ChatGoogleGenerativeAI with model=%s", model_name)
111 |         return ChatGoogleGenerativeAI(
112 |             model=model_name,
113 |             temperature=temperature,
114 |             google_api_key=api_key
115 |         )
116 | 
117 |     # -------------------------
118 |     # Ollama
119 |     # -------------------------
120 |     elif provider == 'ollama':
121 |         model_name = kwargs.get("model_name", "qwen2.5:7b")
122 |         temperature = kwargs.get("temperature", 0.0)
123 | 
124 |         logger.debug("Configuring ChatOllama with model=%s", model_name)
125 |         return ChatOllama(
126 |             model=model_name,
127 |             temperature=temperature,
128 |         )
129 | 
130 |     # -------------------------
131 |     # Azure OpenAI
132 |     # -------------------------
133 |     elif provider == 'azure_openai':
134 |         base_url = kwargs.get("base_url") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
135 |         api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY", "")
136 |         model_name = kwargs.get("model_name", "gpt-4o")
137 |         temperature = kwargs.get("temperature", 0.0)
138 | 
139 |         logger.debug("Configuring AzureChatOpenAI with model=%s, base_url=%s", model_name, base_url)
140 |         return AzureChatOpenAI(
141 |             model=model_name,
142 |             temperature=temperature,
143 |             api_version="2024-05-01-preview",
144 |             azure_endpoint=base_url,
145 |             api_key=api_key
146 |         )
147 | 
148 |     else:
149 |         error_msg = f"Unsupported provider: {provider}"
150 |         logger.error(error_msg)
151 |         raise ValueError(error_msg)
152 | 
153 | 
154 | def encode_image(img_path: Optional[str]) -> Optional[str]:
155 |     """
156 |     Encode an image file into a Base64 string.
157 | 
158 |     Parameters
159 |     ----------
160 |     img_path : Optional[str]
161 |         The path to the image file on disk. If None or empty, returns None.
162 | 
163 |     Returns
164 |     -------
165 |     Optional[str]
166 |         A Base64-encoded string of the image's contents, or None if the path
167 |         was not provided.
168 |     """
169 |     if not img_path:
170 |         logger.debug("No image path provided; returning None.")
171 |         return None
172 | 
173 |     if not os.path.isfile(img_path):
174 |         logger.warning("Image file not found at path: %s", img_path)
175 |         return None
176 | 
177 |     logger.debug("Encoding image from path: %s", img_path)
178 |     with open(img_path, "rb") as fin:
179 |         image_data = base64.b64encode(fin.read()).decode("utf-8")
180 |     return image_data
```

tests/test_browser_use.py
```
1 | # -*- coding: utf-8 -*-
2 | """
3 | Filename: test_browser_use.py
4 | 
5 | Description:
6 |     Test scripts for verifying browser automation using both the 'org' agent
7 |     (original Browser + Agent) and the 'custom' agent (CustomBrowser, CustomContext, etc.).
8 | """
9 | 
10 | import os
11 | import sys
12 | import logging
13 | import asyncio
14 | from pprint import pprint
15 | from typing import Optional
16 | 
17 | from dotenv import load_dotenv
18 | 
19 | # Load environment variables
20 | load_dotenv()
21 | 
22 | # Optional logging setup
23 | logging.basicConfig(
24 |     level=logging.INFO,
25 |     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
26 | )
27 | logger = logging.getLogger(__name__)
28 | 
29 | # Make local modules available if needed
30 | sys.path.append(".")
31 | 
32 | from browser_use.agent.views import AgentHistoryList
33 | from src.utils import utils
34 | 
35 | 
36 | async def test_browser_use_org() -> None:
37 |     """
38 |     Test the original 'org' browser agent using Browser + Agent from browser_use,
39 |     instructing it to go to Google, search 'OpenAI', and extract the first URL.
40 |     """
41 |     from browser_use.browser.browser import Browser, BrowserConfig
42 |     from browser_use.browser.context import (
43 |         BrowserContextConfig,
44 |         BrowserContextWindowSize,
45 |     )
46 |     from browser_use.agent.service import Agent
47 | 
48 |     logger.info("Setting up LLM and Browser for 'org' agent test.")
49 |     llm = utils.get_llm_model(
50 |         provider="azure_openai",
51 |         model_name="gpt-4o",
52 |         temperature=0.8,
53 |         base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
54 |         api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
55 |     )
56 | 
57 |     window_w, window_h = 1920, 1080
58 | 
59 |     browser = Browser(
60 |         config=BrowserConfig(
61 |             headless=False,
62 |             disable_security=True,
63 |             extra_chromium_args=[f"--window-size={window_w},{window_h}"],
64 |         )
65 |     )
66 | 
67 |     async with await browser.new_context(
68 |             config=BrowserContextConfig(
69 |                 trace_path="./tmp/traces",
70 |                 save_recording_path="./tmp/record_videos",
71 |                 no_viewport=False,
72 |                 browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
73 |             )
74 |     ) as browser_context:
75 |         agent = Agent(
76 |             task="go to google.com and type 'OpenAI' click search and give me the first url",
77 |             llm=llm,
78 |             browser_context=browser_context,
79 |         )
80 |         logger.info("Running 'org' agent for up to 10 steps...")
81 |         history: AgentHistoryList = await agent.run(max_steps=10)
82 | 
83 |         print("Final Result:")
84 |         pprint(history.final_result(), indent=4)
85 | 
86 |         print("\nErrors:")
87 |         pprint(history.errors(), indent=4)
88 | 
89 |         print("\nModel Outputs:")
90 |         pprint(history.model_actions(), indent=4)
91 | 
92 |         print("\nThoughts:")
93 |         pprint(history.model_thoughts(), indent=4)
94 | 
95 |     # Close the browser
96 |     logger.info("Closing the browser.")
97 |     await browser.close()
98 | 
99 | 
100 | async def test_browser_use_custom() -> None:
101 |     """
102 |     Test the custom browser agent using CustomBrowser, CustomContext,
103 |     CustomController, and CustomAgent. The agent is tasked with going to
104 |     Google, searching 'OpenAI', and extracting the first URL.
105 |     """
106 |     from playwright.async_api import async_playwright
107 |     from browser_use.browser.context import BrowserContextWindowSize
108 | 
109 |     from src.browser.enhanced_playwright_browser import CustomBrowser, BrowserConfig
110 |     from src.browser.enhanced_playwright_browser_context import BrowserContext, BrowserContextConfig
111 |     from src.controller.custom_controller import CustomController
112 |     from src.agent.browser_agent import CustomAgent
113 |     from src.agent.browser_system_prompts import CustomSystemPrompt
114 | 
115 |     logger.info("Setting up LLM and Custom Browser for 'custom' agent test.")
116 |     llm = utils.get_llm_model(
117 |         provider="ollama",
118 |         model_name="qwen2.5:7b",
119 |         temperature=0.8
120 |     )
121 | 
122 |     controller = CustomController()
123 |     use_own_browser = False
124 |     disable_security = True
125 |     use_vision = False
126 | 
127 |     window_w, window_h = 1920, 1080
128 |     playwright = None
129 |     browser_context_ = None
130 | 
131 |     try:
132 |         if use_own_browser:
133 |             logger.info("Launching a persistent browser context with existing profile.")
134 |             playwright = await async_playwright().start()
135 |             chrome_exe = os.getenv("CHROME_PATH", "")
136 |             chrome_use_data = os.getenv("CHROME_USER_DATA", "")
137 |             browser_context_ = await playwright.chromium.launch_persistent_context(
138 |                 user_data_dir=chrome_use_data,
139 |                 executable_path=chrome_exe,
140 |                 no_viewport=False,
141 |                 headless=False,
142 |                 user_agent=(
143 |                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
144 |                     "(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
145 |                 ),
146 |                 java_script_enabled=True,
147 |                 bypass_csp=disable_security,
148 |                 ignore_https_errors=disable_security,
149 |                 record_video_dir="./tmp/record_videos",
150 |                 record_video_size={"width": window_w, "height": window_h},
151 |             )
152 |         else:
153 |             browser_context_ = None
154 | 
155 |         browser = CustomBrowser(
156 |             config=BrowserConfig(
157 |                 headless=False,
158 |                 disable_security=True,
159 |                 extra_chromium_args=[f"--window-size={window_w},{window_h}"],
160 |             )
161 |         )
162 | 
163 |         async with await browser.new_context(
164 |                 config=BrowserContextConfig(
165 |                     trace_path="./tmp/result_processing",
166 |                     save_recording_path="./tmp/record_videos",
167 |                     no_viewport=False,
168 |                     browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
169 |                 ),
170 |                 context=browser_context_
171 |         ) as browser_context:
172 |             agent = CustomAgent(
173 |                 task="go to google.com and type 'OpenAI' click search and give me the first url",
174 |                 add_infos="",  # additional info/hints
175 |                 llm=llm,
176 |                 browser_context=browser_context,
177 |                 controller=controller,
178 |                 system_prompt_class=CustomSystemPrompt,
179 |                 use_vision=use_vision,
180 |             )
181 | 
182 |             logger.info("Running 'custom' agent for up to 10 steps...")
183 |             history: AgentHistoryList = await agent.run(max_steps=10)
184 | 
185 |             print("Final Result:")
186 |             pprint(history.final_result(), indent=4)
187 | 
188 |             print("\nErrors:")
189 |             pprint(history.errors(), indent=4)
190 | 
191 |             print("\nModel Outputs:")
192 |             pprint(history.model_actions(), indent=4)
193 | 
194 |             print("\nThoughts:")
195 |             pprint(history.model_thoughts(), indent=4)
196 | 
197 |     except Exception as exc:
198 |         logger.error("An exception occurred:", exc_info=exc)
199 |     finally:
200 |         # Close persistent context if used
201 |         if browser_context_:
202 |             logger.info("Closing persistent browser context.")
203 |             await browser_context_.close()
204 | 
205 |         # Stop the Playwright object
206 |         if playwright:
207 |             logger.info("Stopping Playwright.")
208 |             await playwright.stop()
209 | 
210 |         logger.info("Closing the custom browser.")
211 |         await browser.close()
212 | 
213 | 
214 | if __name__ == "__main__":
215 |     # Uncomment the desired test:
216 |     # asyncio.run(test_browser_use_org())
217 |     asyncio.run(test_browser_use_custom())
```

tests/test_llm_api.py
```
1 | # -*- coding: utf-8 -*-
2 | """
3 | Filename: test_llm_api.py
4 | 
5 | Description:
6 |     Test scripts for verifying functionality of various LLM providers (OpenAI,
7 |     Gemini, Azure OpenAI, DeepSeek, Ollama). Each test function sets up an LLM,
8 |     encodes an image (if needed), invokes the model, and displays the response.
9 | """
10 | 
11 | import os
12 | import sys
13 | import logging
14 | 
15 | from dotenv import load_dotenv
16 | 
17 | # Load environment variables
18 | load_dotenv()
19 | 
20 | # Optionally, set up logging
21 | logging.basicConfig(
22 |     level=logging.INFO,
23 |     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
24 | )
25 | logger = logging.getLogger(__name__)
26 | 
27 | # Ensure local paths are recognized
28 | sys.path.append(".")
29 | 
30 | 
31 | def test_openai_model() -> None:
32 |     """
33 |     Test the OpenAI GPT model, passing an image plus text prompt and printing the LLM response.
34 |     """
35 |     from langchain_core.messages import HumanMessage
36 |     from src.utils import utils
37 | 
38 |     llm = utils.get_llm_model(
39 |         provider="openai",
40 |         model_name="gpt-4o",
41 |         temperature=0.8,
42 |         base_url=os.getenv("OPENAI_ENDPOINT", ""),
43 |         api_key=os.getenv("OPENAI_API_KEY", "")
44 |     )
45 | 
46 |     image_path = "assets/examples/test.png"
47 |     image_data = utils.encode_image(image_path)
48 | 
49 |     message = HumanMessage(
50 |         content=[
51 |             {"type": "text", "text": "describe this image"},
52 |             {
53 |                 "type": "image_url",
54 |                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
55 |             },
56 |         ]
57 |     )
58 | 
59 |     logger.info("Invoking OpenAI GPT model...")
60 |     ai_msg = llm.invoke([message])
61 |     logger.info("Model response: %s", ai_msg.content)
62 |     print(ai_msg.content)
63 | 
64 | 
65 | def test_gemini_model() -> None:
66 |     """
67 |     Test the Google Generative AI (Gemini) model, providing an image-based prompt
68 |     and retrieving its textual description.
69 |     """
70 |     from langchain_core.messages import HumanMessage
71 |     from src.utils import utils
72 | 
73 |     llm = utils.get_llm_model(
74 |         provider="deepseek",  # or 'gemini' if you directly support 'gemini' as a provider
75 |         model_name="gemini-2.0-flash-exp",
76 |         temperature=0.8,
77 |         api_key=os.getenv("GOOGLE_API_KEY", "")
78 |     )
79 | 
80 |     image_path = "assets/examples/test.png"
81 |     image_data = utils.encode_image(image_path)
82 | 
83 |     message = HumanMessage(
84 |         content=[
85 |             {"type": "text", "text": "describe this image"},
86 |             {
87 |                 "type": "image_url",
88 |                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
89 |             },
90 |         ]
91 |     )
92 | 
93 |     logger.info("Invoking Gemini model...")
94 |     ai_msg = llm.invoke([message])
95 |     logger.info("Model response: %s", ai_msg.content)
96 |     print(ai_msg.content)
97 | 
98 | 
99 | def test_azure_openai_model() -> None:
100 |     """
101 |     Test the Azure OpenAI model by sending an image-based prompt and printing the LLM response.
102 |     """
103 |     from langchain_core.messages import HumanMessage
104 |     from src.utils import utils
105 | 
106 |     llm = utils.get_llm_model(
107 |         provider="azure_openai",
108 |         model_name="gpt-4o",
109 |         temperature=0.8,
110 |         base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
111 |         api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
112 |     )
113 | 
114 |     image_path = "assets/examples/test.png"
115 |     image_data = utils.encode_image(image_path)
116 | 
117 |     message = HumanMessage(
118 |         content=[
119 |             {"type": "text", "text": "describe this image"},
120 |             {
121 |                 "type": "image_url",
122 |                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
123 |             },
124 |         ]
125 |     )
126 | 
127 |     logger.info("Invoking Azure OpenAI GPT model...")
128 |     ai_msg = llm.invoke([message])
129 |     logger.info("Model response: %s", ai_msg.content)
130 |     print(ai_msg.content)
131 | 
132 | 
133 | def test_deepseek_model() -> None:
134 |     """
135 |     Test the DeepSeek model, sending a simple textual prompt and printing the LLM response.
136 |     """
137 |     from langchain_core.messages import HumanMessage
138 |     from src.utils import utils
139 | 
140 |     llm = utils.get_llm_model(
141 |         provider="deepseek",
142 |         model_name="deepseek-chat",
143 |         temperature=0.8,
144 |         base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
145 |         api_key=os.getenv("DEEPSEEK_API_KEY", "")
146 |     )
147 | 
148 |     message = HumanMessage(
149 |         content=[
150 |             {"type": "text", "text": "Who are you?"}
151 |         ]
152 |     )
153 | 
154 |     logger.info("Invoking DeepSeek model...")
155 |     ai_msg = llm.invoke([message])
156 |     logger.info("Model response: %s", ai_msg.content)
157 |     print(ai_msg.content)
158 | 
159 | 
160 | def test_ollama_model() -> None:
161 |     """
162 |     Test the Ollama model by sending a textual prompt and printing its response.
163 |     """
164 |     from langchain_ollama import ChatOllama
165 | 
166 |     logger.info("Invoking Ollama model...")
167 |     llm = ChatOllama(model="qwen2.5:7b")
168 |     ai_msg = llm.invoke("Sing a ballad of LangChain.")
169 |     logger.info("Model response: %s", ai_msg.content)
170 |     print(ai_msg.content)
171 | 
172 | 
173 | if __name__ == "__main__":
174 |     # Uncomment the tests you want to run:
175 |     # test_openai_model()
176 |     # test_gemini_model()
177 |     # test_azure_openai_model()
178 |     # test_deepseek_model()
179 |     test_ollama_model()
```

tests/test_playwright.py
```
1 | # -*- coding: utf-8 -*-
2 | """
3 | Filename: test_browser_use.py
4 | 
5 | Description:
6 |     Contains tests for verifying browser automation using both the original
7 |     ("org") agent and the custom agent implementation. Each test launches
8 |     an agent, instructs it to visit Google, search for "OpenAI," and then
9 |     retrieve the first resulting URL.
10 | """
11 | 
12 | import os
13 | import sys
14 | import logging
15 | import asyncio
16 | from pprint import pprint
17 | from typing import Optional, List
18 | 
19 | from dotenv import load_dotenv
20 | 
21 | # Load environment variables
22 | load_dotenv()
23 | 
24 | # Optionally set up logging
25 | logging.basicConfig(
26 |     level=logging.INFO,
27 |     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
28 | )
29 | logger = logging.getLogger(__name__)
30 | 
31 | # Add local paths to Python path if needed
32 | sys.path.append(".")
33 | 
34 | # Project imports
35 | from browser_use import Agent
36 | from browser_use.agent.views import AgentHistoryList
37 | from src.utils import utils
38 | 
39 | 
40 | async def test_browser_use_org() -> None:
41 |     """
42 |     Test the original 'org' agent from browser_use. It will:
43 |       1. Launch a Browser with specified settings.
44 |       2. Create a BrowserContext with the given config.
45 |       3. Instantiate an Agent with a simple task:
46 |          "Go to google.com, search 'OpenAI', give me the first URL."
47 |       4. Run the agent for up to 10 steps and print the results (final result, errors, actions, thoughts).
48 |     """
49 |     from browser_use.browser.browser import Browser, BrowserConfig
50 |     from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
51 |     from browser_use.agent.service import Agent
52 | 
53 |     logger.info("Setting up LLM for 'org' agent test.")
54 |     llm = utils.get_llm_model(
55 |         provider="azure_openai",
56 |         model_name="gpt-4o",
57 |         temperature=0.8,
58 |         base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
59 |         api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
60 |     )
61 | 
62 |     window_w, window_h = 1920, 1080
63 |     logger.info("Initializing the Browser in non-headless mode with window size %dx%d.", window_w, window_h)
64 | 
65 |     browser = Browser(
66 |         config=BrowserConfig(
67 |             headless=False,
68 |             disable_security=True,
69 |             extra_chromium_args=[f"--window-size={window_w},{window_h}"],
70 |         )
71 |     )
72 | 
73 |     async with await browser.new_context(
74 |             config=BrowserContextConfig(
75 |                 trace_path="./tmp/traces",
76 |                 save_recording_path="./tmp/record_videos",
77 |                 no_viewport=False,
78 |                 browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
79 |             )
80 |     ) as browser_context:
81 |         agent = Agent(
82 |             task="go to google.com and type 'OpenAI' click search and give me the first url",
83 |             llm=llm,
84 |             browser_context=browser_context,
85 |         )
86 |         logger.info("Running 'org' agent for up to 10 steps...")
87 |         history: AgentHistoryList = await agent.run(max_steps=10)
88 | 
89 |         print("Final Result:")
90 |         pprint(history.final_result(), indent=4)
91 | 
92 |         print("\nErrors:")
93 |         pprint(history.errors(), indent=4)
94 | 
95 |         print("\nModel Outputs:")
96 |         pprint(history.model_actions(), indent=4)
97 | 
98 |         print("\nThoughts:")
99 |         pprint(history.model_thoughts(), indent=4)
100 | 
101 |     logger.info("Closing the browser.")
102 |     await browser.close()
103 | 
104 | 
105 | async def test_browser_use_custom() -> None:
106 |     """
107 |     Test the custom agent (CustomBrowser, CustomContext, CustomAgent, etc.). It will:
108 |       1. Optionally launch a persistent browser context with user profile (if `use_own_browser` is True).
109 |       2. Create a custom browser context with the specified config.
110 |       3. Instantiate a CustomAgent with a simple task:
111 |          "Go to google.com, search 'OpenAI', give me the first URL."
112 |       4. Run the custom agent for up to 10 steps and print the results (final result, errors, actions, thoughts).
113 |     """
114 |     from playwright.async_api import async_playwright
115 |     from browser_use.browser.context import BrowserContextWindowSize
116 | 
117 |     from src.browser.enhanced_playwright_browser import CustomBrowser, BrowserConfig
118 |     from src.browser.enhanced_playwright_browser_context import BrowserContext, BrowserContextConfig
119 |     from src.controller.custom_controller import CustomController
120 |     from src.agent.browser_agent import CustomAgent
121 |     from src.agent.browser_system_prompts import CustomSystemPrompt
122 | 
123 |     logger.info("Setting up LLM for 'custom' agent test.")
124 |     llm = utils.get_llm_model(
125 |         provider="ollama",
126 |         model_name="qwen2.5:7b",
127 |         temperature=0.8
128 |     )
129 | 
130 |     controller = CustomController()
131 |     use_own_browser = False  # If True, reuses an existing local Chrome profile
132 |     disable_security = True
133 |     use_vision = False
134 | 
135 |     window_w, window_h = 1920, 1080
136 |     playwright = None
137 |     browser_context_: Optional[BrowserContext] = None
138 | 
139 |     try:
140 |         if use_own_browser:
141 |             logger.info("Launching a persistent browser context with existing profile.")
142 |             playwright = await async_playwright().start()
143 |             chrome_exe = os.getenv("CHROME_PATH", "")
144 |             chrome_use_data = os.getenv("CHROME_USER_DATA", "")
145 | 
146 |             browser_context_ = await playwright.chromium.launch_persistent_context(
147 |                 user_data_dir=chrome_use_data,
148 |                 executable_path=chrome_exe,
149 |                 no_viewport=False,
150 |                 headless=False,
151 |                 user_agent=(
152 |                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
153 |                     "(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
154 |                 ),
155 |                 java_script_enabled=True,
156 |                 bypass_csp=disable_security,
157 |                 ignore_https_errors=disable_security,
158 |                 record_video_dir="./tmp/record_videos",
159 |                 record_video_size={"width": window_w, "height": window_h},
160 |             )
161 |         else:
162 |             logger.info("No persistent browser context used (use_own_browser=False).")
163 | 
164 |         logger.info("Initializing CustomBrowser in non-headless mode with window size %dx%d.", window_w, window_h)
165 |         browser = CustomBrowser(
166 |             config=BrowserConfig(
167 |                 headless=False,
168 |                 disable_security=disable_security,
169 |                 extra_chromium_args=[f"--window-size={window_w},{window_h}"],
170 |             )
171 |         )
172 | 
173 |         async with await browser.new_context(
174 |                 config=BrowserContextConfig(
175 |                     trace_path="./tmp/result_processing",
176 |                     save_recording_path="./tmp/record_videos",
177 |                     no_viewport=False,
178 |                     browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
179 |                 ),
180 |                 context=browser_context_
181 |         ) as browser_context:
182 |             agent = CustomAgent(
183 |                 task="go to google.com and type 'OpenAI' click search and give me the first url",
184 |                 add_infos="",
185 |                 llm=llm,
186 |                 browser_context=browser_context,
187 |                 controller=controller,
188 |                 system_prompt_class=CustomSystemPrompt,
189 |                 use_vision=use_vision,
190 |             )
191 | 
192 |             logger.info("Running 'custom' agent for up to 10 steps...")
193 |             history: AgentHistoryList = await agent.run(max_steps=10)
194 | 
195 |             print("Final Result:")
196 |             pprint(history.final_result(), indent=4)
197 | 
198 |             print("\nErrors:")
199 |             pprint(history.errors(), indent=4)
200 | 
201 |             print("\nModel Outputs:")
202 |             pprint(history.model_actions(), indent=4)
203 | 
204 |             print("\nThoughts:")
205 |             pprint(history.model_thoughts(), indent=4)
206 | 
207 |     except Exception as exc:
208 |         logger.error("An exception occurred in test_browser_use_custom:", exc_info=exc)
209 |     finally:
210 |         # Close persistent context if used
211 |         if browser_context_:
212 |             logger.info("Closing persistent browser context.")
213 |             await browser_context_.close()
214 | 
215 |         # Stop the Playwright object
216 |         if playwright:
217 |             logger.info("Stopping Playwright.")
218 |             await playwright.stop()
219 | 
220 |         logger.info("Closing the custom browser.")
221 |         await browser.close()
222 | 
223 | 
224 | if __name__ == "__main__":
225 |     # Uncomment the test you want to run:
226 |     # asyncio.run(test_browser_use_org())
227 |     asyncio.run(test_browser_use_custom())
```

