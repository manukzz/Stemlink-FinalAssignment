"""Tools available to agents in the multi-agent RAG system."""

from langchain_core.tools import tool

from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks



tool(response_format="content_and_artifact")
def retrieval_tool(query: str):
    """
    Search the vector database for relevant document chunks based on a query.
    
    This tool retrieves relevant excerpts from the IKMS system to help answer 
    questions with high precision and verifiable citations.
    """
    # 1. Retrieve raw documents (your existing retrieve function)
    docs = retrieve(query, k=4)

    # 2. Use the serializer to get the [CX] string and the metadata map
    context_str, citation_map = serialize_chunks(docs)

    # 3. Return the tuple (content for LLM, artifact for system state)
    artifact = {
        "citation_map": citation_map,
        "raw_docs": docs
    }
    
    return context_str, artifact