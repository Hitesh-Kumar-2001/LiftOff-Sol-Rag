"""Public API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import ErrorResponse, QueryRequest, QueryResponse
from app.security import AuthenticationError, ServerRegistry, get_server_registry

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against a RAG database",
    responses={
        401: {"model": ErrorResponse, "description": "Unknown serverId or serverSecret"},
    },
)
async def query(
    payload: QueryRequest,
    registry: Annotated[ServerRegistry, Depends(get_server_registry)],
) -> QueryResponse:
    """Verify the calling server, then answer its question from one database."""
    try:
        registry.authenticate(payload.server_id, payload.server_secret.get_secret_value())
    except AuthenticationError:
        # Deliberately vague: do not reveal which half of the pair was wrong.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid serverId or serverSecret.",
        ) from None

    # TODO: hand (rag_db_id, question) to the retrieval layer once it exists.
    return QueryResponse(
        answer=f"Retrieval is not implemented yet. Received question: {payload.question}",
        rag_db_id=payload.rag_db_id,
    )
