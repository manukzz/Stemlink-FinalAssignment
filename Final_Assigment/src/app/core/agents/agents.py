"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List

from .state import QAState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: list) -> str:
    """Helper to get text from the last AIMessage."""
    from langchain_core.messages import AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


# Define agents at module level for reuse
retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)

def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Sends the user's question to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Extracts the tool's content (CONTEXT string) from the ToolMessage.
    - Stores the consolidated context string in `state["context"]`.
    - Also extracts citation metadata from the tool's artifact and stores
    """
    question_data = state.get("question")
    if isinstance(question_data, dict):
        question_text = question_data.get("question", "")
    else:
        question_text = str(question_data)

    result = retrieval_agent.invoke({"messages": [HumanMessage(content=question_text)]})
    
    messages = result.get("messages", [])
    
    context = ""
    citations = {}

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = msg.content
            # Check if artifact is a dict before calling .get()
            if isinstance(msg.artifact, dict):
                citations = msg.artifact.get("citation_map", {})
            break

    return {
        "context": context,
        "citations": citations,
    }

def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: updated to handle citation instructions."""
    question = state["question"]
    context = state.get("context")
    # Carry over citations from the previous state
    citations = state.get("citations", {})

    # Update the user_content to be more explicit about citing the [CX] tags
    user_content = (
        f"Question: {question}\n\n"
        f"Context with Chunk IDs:\n{context}\n\n"
        "Instructions: Answer the question using ONLY the context provided. "
        "You MUST cite the source chunks using their IDs (e.g., [C1], [C2]) "
        "immediately after the relevant information in your answer."
    )

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    # We return the new draft_answer, but we keep 'citations' in the state 
    # so the next node (verification) can use them.
    return {
        "draft_answer": draft_answer,
        "citations": state.get("citations") # Pass it through!
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: Audits both content and citation accuracy."""
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")
    citations = state.get("citations", {}) # Carry forward the map

    # We update the user_content to include strict citation auditing rules
    user_content = f"""Question: {question}

Context (Source Material):
{context}

Draft Answer to Audit:
{draft_answer}

TASKS:
1. FACT CHECK: Ensure every claim in the Draft Answer is supported by the Context.
2. CITATION AUDIT: 
   - Verify that every [CX] tag correctly refers to the chunk containing that information.
   - If a citation is missing for a supported fact, add the correct [CX] tag.
   - If a citation points to the wrong chunk, correct it.
3. CLEANUP: If a claim cannot be verified by the context, remove both the claim and its citation.
4. CONSISTENCY: Ensure the final output maintains the [CX] format for citations.

Please provide the final, verified answer below:"""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    
    messages = result.get("messages", [])
    final_answer = _extract_last_ai_content(messages)

    # Return the final answer and ensure citations map is passed to the final output
    return {
        "answer": final_answer, # If you used 'final_answer' above, use it here
        "citations": state.get("citations", {})
    }
