# src/langgraph_fastapi/service/service.py

import json
import os
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient

from ..schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
    convert_message_content_to_string,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)


def verify_bearer(request: Request) -> None:
    auth_secret = os.getenv("AUTH_SECRET")
    if auth_secret:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        token = auth_header[len("Bearer ") :]
        if token != auth_secret:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def create_app(agent: CompiledStateGraph) -> FastAPI:
    """
    Create a FastAPI app to serve the provided LangGraph agent.

    Args:
        agent (CompiledStateGraph): The LangGraph agent to serve.

    Returns:
        FastAPI: The configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Construct agent with Sqlite checkpointer
        async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
            agent.checkpointer = saver
            app.state.agent = agent
            yield
        # context manager will clean up the AsyncSqliteSaver on exit

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(verify_bearer)])
    router = APIRouter()

    def _parse_input(user_input: UserInput) -> Tuple[Dict[str, Any], str]:
        run_id = str(uuid4())
        thread_id = user_input.thread_id or str(uuid4())
        input_message = ChatMessage(type="human", content=user_input.message)
        kwargs = {
            "input": {"messages": [input_message.to_langchain()]},
            "config": RunnableConfig(
                configurable={"thread_id": thread_id, "model": user_input.model}, run_id=run_id
            ),
        }
        return kwargs, run_id

    def _remove_tool_calls(content: Any) -> Any:
        """Remove tool calls from content."""
        if isinstance(content, str):
            return content
        return [
            content_item
            for content_item in content
            if isinstance(content_item, str) or content_item.get("type") != "tool_use"
        ]

    @router.post("/invoke")
    async def invoke(user_input: UserInput) -> ChatMessage:
        """
        Invoke the agent with user input to retrieve a final response.
        """
        agent_instance: CompiledStateGraph = app.state.agent
        kwargs, run_id = _parse_input(user_input)
        try:
            response = await agent_instance.ainvoke(**kwargs)
            output = ChatMessage.from_langchain(response["messages"][-1])
            output.run_id = run_id
            return output
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
        """
        Generate a stream of messages from the agent.
        """
        agent_instance: CompiledStateGraph = app.state.agent
        kwargs, run_id = _parse_input(user_input)

        async for event in agent_instance.astream_events(**kwargs, version="v2"):
            if not event:
                continue

            if (
                event["event"] == "on_chain_end"
                and any(t.startswith("graph:step:") for t in event.get("tags", []))
                and "messages" in event["data"]["output"]
            ):
                new_messages = event["data"]["output"]["messages"]
                for message in new_messages:
                    try:
                        chat_message = ChatMessage.from_langchain(message)
                        chat_message.run_id = run_id
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                        continue
                    if chat_message.type == "human" and chat_message.content == user_input.message:
                        continue
                    yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if (
                event["event"] == "on_chat_model_stream"
                and user_input.stream_tokens
                and "llama_guard" not in event.get("tags", [])
            ):
                content = _remove_tool_calls(event["data"]["chunk"].content)
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
                continue

        yield "data: [DONE]\n\n"

    @router.post("/stream", response_class=StreamingResponse)
    async def stream_agent(user_input: StreamInput) -> StreamingResponse:
        """
        Stream the agent's response to a user input, including intermediate messages and tokens.
        """
        return StreamingResponse(message_generator(user_input), media_type="text/event-stream")

    @router.post("/feedback")
    async def feedback(feedback: Feedback) -> FeedbackResponse:
        """
        Record feedback for a run to LangSmith.
        """
        client = LangsmithClient()
        kwargs = feedback.kwargs or {}
        client.create_feedback(
            run_id=feedback.run_id,
            key=feedback.key,
            score=feedback.score,
            **kwargs,
        )
        return FeedbackResponse()

    @router.post("/history")
    def history(input: ChatHistoryInput) -> ChatHistory:
        """
        Get chat history.
        """
        agent_instance: CompiledStateGraph = app.state.agent
        try:
            state_snapshot = agent_instance.get_state(
                config=RunnableConfig(
                    configurable={
                        "thread_id": input.thread_id,
                    }
                )
            )
            messages: List[AnyMessage] = state_snapshot.values["messages"]
            chat_messages: List[ChatMessage] = []
            for message in messages:
                chat_messages.append(ChatMessage.from_langchain(message))
            return ChatHistory(messages=chat_messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    app.include_router(router)

    return app
