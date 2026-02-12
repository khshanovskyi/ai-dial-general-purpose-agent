import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


class FileContentExtractionTool(BaseTool):
    """
    Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.
    PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated.
    USAGE: Start with page=1 (by default)
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "file_content_extraction_tool"

    @property
    def description(self) -> str:
        return (
            "Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM. "
            "PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated. "
            "USAGE: Start with page=1. If paginated, call again with page=2, page=3, etc. to get remaining content. "
            "Always check response end for pagination info before answering user queries about file content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "File URL"
                },
                "page": {
                    "type": "integer",
                    "description": "For large documents pagination is enabled. Each page consists of 10000 characters.",
                    "default": 1
                },
            },
            "required": [
                "file_url",
            ]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        file_url = arguments["file_url"]
        page = arguments.get("page", 1)

        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**File URL**: {file_url}\n\r")
        if page > 1:
            stage.append_content(f"**Page**: {page}\n\r")
        stage.append_content(f"## Response: \n")

        content_extractor = DialFileContentExtractor(
            endpoint=self.endpoint,
            api_key=tool_call_params.api_key,
        )
        content = content_extractor.extract_text(file_url)
        if not content:
            content = "Error: File content not found."

        if len(content) > 10_000:
            page_size = 10_000
            total_pages = (len(content) + page_size - 1) // page_size

            if page < 1:
                page = 1
            elif page > total_pages:
                return f"Error: Page {page} does not exist. Total pages: {total_pages}"

            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            page_content = content[start_index:end_index]
            content = f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**"
        stage.append_content(f"```text\n\r{content}\n\r```\n\r")
        return content
