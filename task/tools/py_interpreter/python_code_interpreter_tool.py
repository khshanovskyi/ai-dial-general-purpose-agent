import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        self._dial_endpoint = dial_endpoint
        self._mcp_client = mcp_client
        self._code_execute_tool: Optional[MCPToolModel] = None
        for mcp_tool_model in mcp_tool_models:
            if mcp_tool_model.name == tool_name:
                self._code_execute_tool = mcp_tool_model
                break
        if self._code_execute_tool is None:
            raise Exception(f"MCP tool {tool_name} not found.")

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        client = await MCPClient.create(mcp_url)
        tools = await client.get_tools()
        return cls(
            mcp_client=client,
            mcp_tool_models=tools,
            tool_name=tool_name,
            dial_endpoint=dial_endpoint,
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        code = arguments["code"]
        session_id = arguments.get("session_id")

        stage.append_content(f"```python\n\r{code}\n\r```\n\r")
        if session_id:
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")
        stage.append_content("## Response: \n")

        content = await self._mcp_client.call_tool(self.name, arguments)
        execution_result_json = json.loads(content)
        execution_result = _ExecutionResult.model_validate(execution_result_json)

        if execution_result.files:
            dial_client = Dial(
                base_url=self._dial_endpoint,
                api_key=tool_call_params.api_key,
            )
            files_home = dial_client.my_appdata_home()
            for file in execution_result.files:
                name = file.name
                mime_type = file.mime_type
                resource = await self._mcp_client.get_resource(AnyUrl(file.uri))
                if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml']:
                    file_data = resource.encode('utf-8')
                else:
                    file_data = base64.b64decode(resource)
                url = f"files/{(files_home / name).as_posix()}"
                dial_client.files.upload(url=url, file=file_data)
                attachment = Attachment(
                    url=StrictStr(url),
                    type=StrictStr(mime_type),
                    title=StrictStr(name)
                )
                stage.add_attachment(attachment)
                tool_call_params.choice.add_attachment(attachment)

            execution_result_json["instructions"] = (
                "Generates files have been provided to user, "
                "DON'T include links to them in response!"
            )
        if execution_result.output:
            new_output = [output[:200] for output in execution_result.output]
            execution_result.output = new_output
        stage.append_content(f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r")
        return StrictStr(execution_result.model_dump_json())
