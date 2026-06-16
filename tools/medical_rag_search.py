"""
tools/rag_search.py — Medical RAG search (Step 4).

Placeholder implementation using keyword retrieval.
Real implementation uses ONNX MiniLM + ChromaDB/FAISS
without changing this module's public interface.

Changes from original:
    - Adds ToolError isinstance guards before accessing result fields.
    - Removes internal state.rag_result assignment (pipeline owns state).
    - Uses TOOL_MEDICAL_RAG_SEARCH from tool_names.py.
"""

from __future__ import annotations

from schemas.errors import ToolError
from schemas.rag import RAGPassage, RAGSearchResult
from schemas.state import AegisState
from tools.tool_names import TOOL_MEDICAL_RAG_SEARCH


_KNOWLEDGE_BASE = [
    {
        "text":     (
            "Chest pain with elevated troponin may indicate "
            "acute coronary syndrome."
        ),
        "source":   "AHA Guidelines",
        "citation": "AHA-ACS-2024",
    },
    {
        "text":     (
            "Persistent fever with productive cough may suggest pneumonia."
        ),
        "source":   "WHO Respiratory Guidelines",
        "citation": "WHO-PNEUMONIA",
    },
    {
        "text":     (
            "Hyperkalemia can produce life-threatening cardiac arrhythmias."
        ),
        "source":   "Merck Manual",
        "citation": "MERCK-HYPERKALEMIA",
    },
    {
        "text":     (
            "Drug interactions should always be evaluated before "
            "prescribing medication."
        ),
        "source":   "BNF",
        "citation": "BNF-DRUGS",
    },
    {
        "text":     (
            "Abnormal chest X-ray findings require clinical correlation."
        ),
        "source":   "Radiology Reference",
        "citation": "RAD-CXR",
    },
]


class MedicalRAGSearch:
    """
    Keyword-based RAG search placeholder.
    Does not write to state — pipeline owns state mutation.
    """

    TOOL_NAME = TOOL_MEDICAL_RAG_SEARCH

    async def run(
        self,
        state: AegisState,
    ) -> RAGSearchResult | ToolError:

        try:
            query_parts: list[str] = []

            if state.raw_symptoms_text:
                query_parts.append(state.raw_symptoms_text)

            # Guard: only access structured fields when not ToolError.
            if (
                state.symptom_result
                and not isinstance(state.symptom_result, ToolError)
                and state.symptom_result.symptoms
            ):
                query_parts.extend(state.symptom_result.symptoms)

            if (
                state.lab_result
                and not isinstance(state.lab_result, ToolError)
                and state.lab_result.abnormal_values
            ):
                query_parts.extend(state.lab_result.abnormal_values)

            query = " ".join(query_parts).lower()

            retrieved: list[RAGPassage] = []
            citations: list[str]        = []

            for record in _KNOWLEDGE_BASE:
                score = sum(
                    1 for word in query.split()
                    if word in record["text"].lower()
                )
                if score > 0:
                    retrieved.append(
                        RAGPassage(
                            text=record["text"],
                            source=record["source"],
                            citation=record["citation"],
                        )
                    )
                    citations.append(record["citation"])

            return RAGSearchResult(
                passages=retrieved,
                citations=citations,
                query_used=query,
                retrieval_successful=True,
            )

        except Exception as exc:
            return ToolError(
                tool=TOOL_MEDICAL_RAG_SEARCH,
                reason=str(exc),
                fatal=False,
            )


async def search(state: AegisState) -> RAGSearchResult | ToolError:
    """Canonical functional entrypoint."""
    return await MedicalRAGSearch().run(state)