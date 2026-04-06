from pydantic import BaseModel
from typing import Dict, Any


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """

    question: str

class CitationEntry(BaseModel):
    """The specific metadata for an individual chunk citation."""
    page: Any
    snippet: str
    source: str


class QAResponse(BaseModel):
    """
    Response body for the `/qa` endpoint.
    Exposes the cited answer and the mapping for those citations.
    """
    answer: str
    # Removed the 'context' string to focus only on structured citations
    citations: Dict[str, CitationEntry]







