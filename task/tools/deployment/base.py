import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview'
        )

        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt")
        del arguments["prompt"]
        chunks = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            deployment_name=self.deployment_name,
            extra_body={
                "custom_fields": {
                    "configuration": {**arguments}
                }
            },
            **self.tool_parameters,
        )

        content = ''
        custom_content: CustomContent = CustomContent(attachments=[])
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta:
                    if delta.content:
                        tool_call_params.stage.append_content(delta.content)
                        content += delta.content
                    if delta.custom_content and delta.custom_content.attachments:
                        attachments = delta.custom_content.attachments
                        custom_content.attachments.extend(attachments)

                        for attachment in attachments:
                            tool_call_params.stage.add_attachment(
                                type=attachment.type,
                                title=attachment.title,
                                data=attachment.data,
                                url=attachment.url,
                                reference_url=attachment.reference_url,
                                reference_type=attachment.reference_type,
                            )

        return Message(
            role=Role.TOOL,
            content=StrictStr(content),
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )
