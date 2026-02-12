from abc import ABC, abstractmethod
from typing import Any

from aidial_client.types.chat import ToolParam, FunctionParam
from aidial_client.types.chat.legacy.chat_completion import Role
from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.models import ToolCallParams


class BaseTool(ABC):

    async def execute(self, tool_call_params: ToolCallParams) -> Message:
        message = Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )
        try:
            result =  await self._execute(tool_call_params)
        except Exception as e:
            message.content = StrictStr(f"Error: {e}")
        else:
            if isinstance(result, Message):
                message = result
            else:
                message.content = StrictStr(result)
        return message

    @abstractmethod
    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        pass

    @property
    def show_in_stage(self) -> bool:
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def schema(self) -> ToolParam:
        """Provides tool schema according to DIAL specification."""
        #TODO:
        # see https://dialx.ai/dial_api#operation/sendChatCompletionRequest -> `tools`
        # or https://platform.openai.com/docs/guides/function-calling#defining-functions
        return ToolParam(
            type="function",
            function=FunctionParam(
                name=self.name,
                description=self.description,
                parameters=self.parameters
            )
        )
