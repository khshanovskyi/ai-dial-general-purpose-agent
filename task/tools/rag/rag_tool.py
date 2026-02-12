import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You are an expert document analysis assistant. Your task is to accurately answer questions based only on the provided context.

Guidelines:
1. Only use information explicitly stated in the context
2. If the context doesn't contain the information needed to answer completely, acknowledge the limitations
3. Be precise and concise in your answers
4. Cite specific parts of the context when relevant
5. Do not introduce information from outside the provided context
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        self._endpoint = endpoint
        self._deployment_name = deployment_name
        self._document_cache = document_cache
        self._model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2')
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return 'RagTool'

    @property
    def description(self) -> str:
        return (
            "Performs semantic search on documents to find and answer questions based on relevant content. "
            "Supports: PDF, TXT, CSV, HTML. "
            "Use this tool when user asks questions about document content, needs specific information from large files, "
            "or wants to search for particular topics/keywords. "
            "Don't use it when: user wants to read entire document sequentially. "
            "HOW IT WORKS: Splits document into chunks, finds top 3 most relevant sections using semantic search, "
            "then generates answer based only on those sections."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document"
                },
                "file_url": {
                    "type": "string",
                    "description": "File URL"
                },
            },
            "required": ["request", "file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments["request"]
        file_url = arguments["file_url"]
        stage = tool_call_params.stage
    
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")
    
        cache_document_key = f"{tool_call_params.conversation_id}_{file_url}"
    
        cached_data = self._document_cache.get(cache_document_key)
        if cached_data:
            index, chunks = cached_data
        else:
            extractor = DialFileContentExtractor(self._endpoint, tool_call_params.api_key)
            text_content = extractor.extract_text(file_url)
            if not text_content:
                stage.append_content("## Error: \n\rFailed to extract content from the file.\n\r")
                return "Error: Could not extract content from the provided file."
    
            chunks = self._text_splitter.split_text(text_content)
            embeddings = self._model.encode(chunks)
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings).astype('float32'))
            self._document_cache.set(cache_document_key, index, chunks)
    
        query_embedding = self._model.encode([request]).astype('float32')
        distances, indices = index.search(query_embedding, k=3)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        augmented_prompt = self.__augmentation(request, retrieved_chunks)
        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")
        dial_client = AsyncDial(base_url=self._endpoint, api_key=tool_call_params.api_key)
        result_content = ""
    
        async for chunk in dial_client.chat.completions.create(
            deployment_name=self._deployment_name,
            stream=True,
            messages=[
                {"role": Role.SYSTEM, "content": _SYSTEM_PROMPT},
                {"role": Role.USER, "content": augmented_prompt},
            ]
        ):
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                stage.append_content(content)
                result_content += content
        return result_content
    

    def __augmentation(self, request: str, chunks: list[str]) -> str:
            context = "\n\n---\n\n".join(chunks)
            augmented_prompt = f"""I need you to answer a question based solely on the provided context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {request}
    
    Answer the question using only information from the provided context. If the context doesn't contain enough information to answer the question fully, please state that clearly."""
            return augmented_prompt
