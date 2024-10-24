# src/langgraph_fastapi/__init__.py

"""
LangGraph FastAPI Toolkit

A Python package providing tools and infrastructure to serve LangGraph agents using FastAPI.
"""

__version__ = "0.1.0"

from .client import AgentClient
from .schema import (
    UserInput,
    StreamInput,
    AgentResponse,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ChatHistoryInput,
    ChatHistory,
)
from .service import create_app
