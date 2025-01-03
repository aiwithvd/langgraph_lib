# src/my_library/schema/schema.py

from typing import Any, Dict, List, Optional, Union, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from pydantic import BaseModel, Field


def convert_message_content_to_string(content: Union[str, List[Union[str, Dict]]]) -> str:
    if isinstance(content, str):
        return content
    text: List[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item.get("type") == "text":
            text.append(content_item.get("text", ""))
    return "".join(text)


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="LLM Model to use for the agent.",
        examples=["gpt-4o-mini", "llama-3.1-70b"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        default=True,
        description="Whether to stream LLM tokens to the client.",
    )


class AgentResponse(BaseModel):
    """Response from the agent when called via /invoke."""

    message: Dict[str, Any] = Field(
        description="Final response from the agent, as a serialized LangChain message.",
        examples=[
            {
                "message": {
                    "type": "ai",
                    "data": {"content": "The weather in Tokyo is 70 degrees.", "type": "ai"},
                }
            }
        ],
    )


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tool calls in the message.",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call that this message is responding to.",
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Run ID of the message.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    original: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original LangChain message in serialized form.",
    )

    @classmethod
    def from_langchain(cls, message: BaseMessage) -> "ChatMessage":
        """Create a ChatMessage from a LangChain message."""
        original = message_to_dict(message)
        content = convert_message_content_to_string(message.content)
        if isinstance(message, HumanMessage):
            return cls(
                type="human",
                content=content,
                original=original,
            )
        elif isinstance(message, AIMessage):
            return cls(
                type="ai",
                content=content,
                tool_calls=message.tool_calls or [],
                original=original,
            )
        elif isinstance(message, ToolMessage):
            return cls(
                type="tool",
                content=content,
                tool_call_id=message.tool_call_id,
                original=original,
            )
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")

    def to_langchain(self) -> BaseMessage:
        """Convert the ChatMessage to a LangChain message."""
        if self.original:
            raw_original = messages_from_dict([self.original])[0]
            raw_original.content = self.content
            return raw_original
        if self.type == "human":
            return HumanMessage(content=self.content)
        elif self.type == "ai":
            return AIMessage(content=self.content, tool_calls=self.tool_calls)
        elif self.type == "tool":
            return ToolMessage(content=self.content, tool_call_id=self.tool_call_id)
        else:
            raise NotImplementedError(f"Unsupported message type: {self.type}")

    def pretty_print(self) -> None:
        """Pretty print the ChatMessage."""
        lc_msg = self.to_langchain()
        lc_msg.pretty_print()


class Feedback(BaseModel):
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional feedback kwargs, passed to LangSmith.",
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    """Chat history containing a list of messages."""

    messages: List[ChatMessage]
