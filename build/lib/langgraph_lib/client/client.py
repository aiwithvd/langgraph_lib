# src/my_library/client/client.py

import json
import os
from typing import Any, AsyncGenerator, Generator, Optional, Dict

import httpx

from langgraph_lib.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    StreamInput,
    UserInput,
)


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://localhost:80",
        auth_secret: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            auth_secret (Optional[str]): Authentication secret for the agent service.
            timeout (Optional[float]): Timeout for HTTP requests.
        """
        self.base_url = base_url
        self.auth_secret = auth_secret or os.getenv("AUTH_SECRET")
        self.timeout = timeout

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    async def ainvoke(
        self, message: str, model: Optional[str] = None, thread_id: Optional[str] = None
    ) -> ChatMessage:
        """
        Asynchronously invoke the agent. Only the final message is returned.

        Args:
            message (str): The message to send to the agent.
            model (Optional[str]): LLM model to use for the agent.
            thread_id (Optional[str]): Thread ID for continuing a conversation.

        Returns:
            ChatMessage: The response from the agent.
        """
        request = UserInput(message=message, model=model, thread_id=thread_id)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return ChatMessage.model_validate(response.json())
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def invoke(
        self, message: str, model: Optional[str] = None, thread_id: Optional[str] = None
    ) -> ChatMessage:
        """
        Synchronously invoke the agent. Only the final message is returned.

        Args:
            message (str): The message to send to the agent.
            model (Optional[str]): LLM model to use for the agent.
            thread_id (Optional[str]): Thread ID for continuing a conversation.

        Returns:
            ChatMessage: The response from the agent.
        """
        request = UserInput(message=message, model=model, thread_id=thread_id)
        response = httpx.post(
            f"{self.base_url}/invoke",
            json=request.model_dump(),
            headers=self._headers,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            return ChatMessage.model_validate(response.json())
        raise Exception(f"Error: {response.status_code} - {response.text}")

    def _parse_stream_line(self, line: str) -> Optional[ChatMessage | str]:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error parsing message from server: {e}")
            if parsed["type"] == "message":
                try:
                    return ChatMessage.model_validate(parsed["content"])
                except Exception as e:
                    raise Exception(f"Server returned invalid message: {e}")
            elif parsed["type"] == "token":
                return parsed["content"]
            elif parsed["type"] == "error":
                raise Exception(parsed["content"])
        return None

    def stream(
        self,
        message: str,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Args:
            message (str): The message to send to the agent.
            model (Optional[str]): LLM model to use for the agent.
            thread_id (Optional[str]): Thread ID for continuing a conversation.
            stream_tokens (bool): Stream tokens as they are generated.

        Yields:
            Generator[ChatMessage | str, None, None]: The response from the agent.
        """
        request = StreamInput(
            message=message, model=model, thread_id=thread_id, stream_tokens=stream_tokens
        )
        with httpx.stream(
            "POST",
            f"{self.base_url}/stream",
            json=request.model_dump(),
            headers=self._headers,
            timeout=self.timeout,
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            for line in response.iter_lines():
                if line.strip():
                    parsed = self._parse_stream_line(line)
                    if parsed is None:
                        break
                    yield parsed

    async def astream(
        self,
        message: str,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Args:
            message (str): The message to send to the agent.
            model (Optional[str]): LLM model to use for the agent.
            thread_id (Optional[str]): Thread ID for continuing a conversation.
            stream_tokens (bool): Stream tokens as they are generated.

        Yields:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent.
        """
        request = StreamInput(
            message=message, model=model, thread_id=thread_id, stream_tokens=stream_tokens
        )
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Error: {response.status_code} - {response.text}")
                async for line in response.aiter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a feedback record for a run.

        Args:
            run_id (str): The run ID to record feedback for.
            key (str): The feedback key.
            score (float): The feedback score.
            kwargs (Optional[Dict[str, Any]]): Additional feedback parameters.
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs or {})
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/feedback",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")

    def get_history(self, thread_id: str) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str): Thread ID for identifying a conversation.

        Returns:
            ChatHistory: The chat history for the given thread ID.
        """
        request = ChatHistoryInput(thread_id=thread_id)
        response = httpx.post(
            f"{self.base_url}/history",
            json=request.model_dump(),
            headers=self._headers,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            response_object = response.json()
            return ChatHistory.model_validate(response_object)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
