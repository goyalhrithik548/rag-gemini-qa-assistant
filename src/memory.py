"""Conversation memory helpers."""

from __future__ import annotations

from langchain_classic.memory import ConversationBufferMemory


def build_conversation_memory() -> ConversationBufferMemory:
    """Create chat memory for the conversational retrieval chain."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )
