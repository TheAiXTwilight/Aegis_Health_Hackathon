"""
Base interface for all Aegis Health tools.

Every tool implements one async run() method that accepts
an AegisState and returns either a result schema or ToolError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from schemas.errors import ToolError
from schemas.state import AegisState

T = TypeVar("T")


class BaseTool(ABC, Generic[T]):
    """
    Abstract base class for all tools.

    Concrete tools should inherit from BaseTool and implement
    the async run() method.
    """

    TOOL_NAME = "base_tool"

    @abstractmethod
    async def run(
        self,
        state: AegisState,
    ) -> T | ToolError:
        """
        Execute the tool.

        Parameters
        ----------
        state:
            Shared pipeline state.

        Returns
        -------
        T
            Tool-specific result model.

        ToolError
            Returned for expected/recoverable failures.
        """
        raise NotImplementedError

    async def __call__(
        self,
        state: AegisState,
    ) -> T | ToolError:
        """
        Allows:

            result = await tool(state)

        instead of:

            result = await tool.run(state)
        """
        return await self.run(state)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(TOOL_NAME='{self.TOOL_NAME}')"
        )

    def __str__(self) -> str:
        return self.TOOL_NAME