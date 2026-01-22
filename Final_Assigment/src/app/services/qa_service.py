"""Service layer for handling QA requests.

This module provides a simple interface for the FastAPI layer to interact
with the multi-agent RAG pipeline without depending directly on LangGraph
or agent implementation details.
"""

from typing import Dict, Any

from ..core.agents.graph import run_qa_flow


def answer_question(question: str):
    # Ensure this is a simple string key-value pair
    initial_state = {"question": question} 
    result = run_qa_flow(initial_state)
    return result

