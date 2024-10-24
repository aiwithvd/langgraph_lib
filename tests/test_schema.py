# tests/test_schema.py

from langgraph_lib.schema import ChatMessage, HumanMessage

def test_chat_message_conversion():
    content = "Hello, this is a test message."
    human_msg = HumanMessage(content=content)
    chat_msg = ChatMessage.from_langchain(human_msg)
    assert chat_msg.type == "human"
    assert chat_msg.content == content
