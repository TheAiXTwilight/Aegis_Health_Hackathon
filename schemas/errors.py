from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ToolError(BaseModel):
    """
    Standard error contract for all pipeline tools.

    fatal=False  — pipeline continues, tool output is None
    fatal=True   — pipeline halts, report cannot be generated

    See spec section: Fatal vs non-fatal policy.
    """

    tool: str
    code: str | None = None
    reason: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    fatal: bool = False

    model_config = {"frozen": True}


class FatalPipelineError(Exception):
    """
    Raised by AegisPipeline when a step produces ToolError(fatal=True).

    Carries the originating ToolError for diagnostic context.
    The queue worker catches this and marks the job FAILED with the
    ToolError's reason as job.error.

    This exception exists to make pipeline halts EXPLICIT rather than
    inferred from "the async generator stopped yielding." The worker can
    distinguish:
        - normal completion        (async generator exhausted cleanly)
        - fatal pipeline failure   (FatalPipelineError raised)
        - infrastructure timeout   (asyncio.TimeoutError raised)
        - unhandled exception      (everything else)
    """

    def __init__(self, tool_error: ToolError):
        self.tool_error = tool_error
        super().__init__(tool_error.reason)