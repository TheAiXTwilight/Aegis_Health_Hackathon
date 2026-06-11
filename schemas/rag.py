from __future__ import annotations

from pydantic import BaseModel, Field


class RAGPassage(BaseModel):
    """One retrieved evidence passage."""

    text: str
    source: str
    citation: str


class RAGSearchResult(BaseModel):
    """
    Output of MedicalRAGSearch (Step 4).

    retrieval_successful=True even when passages=[] — the mechanism ran
    correctly and found nothing. ToolError(fatal=False) is used only
    when the mechanism itself fails.
    """

    passages: list[RAGPassage] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    query_used: str
    retrieval_successful: bool

    schema_version: str = "1.0"