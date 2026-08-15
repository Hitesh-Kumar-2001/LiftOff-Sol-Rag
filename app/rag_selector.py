"""Placeholder structure only -- no implementation yet."""

from typing import Any


class RagSelector:
    """Suggests a RAG implementation for a document. Single-choice for now;
    will be modularized into a registry of selectable implementations later.
    """

    def __init__(self) -> None:
        pass

    def suggest(self, document_metadata: Any) -> str:
        """Return the name/identifier of the suggested RAG implementation."""
        raise NotImplementedError

    def _score(self, document_metadata: Any) -> Any:
        """Placeholder for future per-implementation scoring logic."""
        raise NotImplementedError

    def _available_implementations(self) -> list[str]:
        """Placeholder for the future registry of selectable implementations."""
        raise NotImplementedError
