"""Utilities for serializing retrieved document chunks."""

from typing import List

from langchain_core.documents import Document

from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

def serialize_chunks(docs: List[Document]) -> Tuple[str, Dict[str, Any]]:
    """
    Serialize Documents into a cited CONTEXT string AND a citation map.
    
    Returns:
        - str: The context block with [C1], [C2] tags for the LLM.
        - dict: The citation mapping for the final API response.
    """
    context_parts = []
    citation_map = {}

    for idx, doc in enumerate(docs, start=1):
        chunk_id = f"C{idx}"
        
        # Metadata extraction
        page_num = doc.metadata.get("page") or doc.metadata.get("page_number", "unknown")
        source_name = doc.metadata.get("source", "unknown")
        
        # 1. Format for the LLM (Content)
        # Using [C1] instead of 'Chunk 1' helps the LLM cite more accurately.
        chunk_header = f"[{chunk_id}] Chunk from page {page_num}:"
        chunk_content = doc.page_content.strip()
        context_parts.append(f"{chunk_header}\n{chunk_content}")

        # 2. Format for the API (Metadata Map)
        # This is what will populate your final JSON 'citations' field.
        citation_map[chunk_id] = {
            "page": page_num,
            "snippet": chunk_content[:120] + "...", 
            "source": source_name
        }

    return "\n\n".join(context_parts), citation_map

