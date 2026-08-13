"""Public API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.job_manager import JobManager, get_job_manager
from app.schemas import (
    DocumentIngestRequest,
    ErrorResponse,
    JobResponse,
    QueryRequest,
    QueryResponse,
)
from app.security import AuthenticationError, ServerRegistry, get_server_registry

router = APIRouter(prefix="/api/v1")


def _require_server(registry: ServerRegistry, server_id: str, server_secret: str) -> None:
    """Raise the shared 401 if the caller cannot be verified."""
    try:
        registry.authenticate(server_id, server_secret)
    except AuthenticationError:
        # Deliberately vague: do not reveal which half of the pair was wrong.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid serverId or serverSecret.",
        ) from None


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against a RAG database",
    tags=["query"],
    responses={
        401: {"model": ErrorResponse, "description": "Unknown serverId or serverSecret"},
    },
)
async def query(
    payload: QueryRequest,
    registry: Annotated[ServerRegistry, Depends(get_server_registry)],
) -> QueryResponse:
    """Verify the calling server, then answer its question from one database."""
    _require_server(registry, payload.server_id, payload.server_secret.get_secret_value())

    # TODO: hand (rag_db_id, question) to the retrieval layer once it exists.
    return QueryResponse(
        answer=f"Retrieval is not implemented yet. Received question: {payload.question}",
        rag_db_id=payload.rag_db_id,
    )


@router.post(
    "/document",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a document for ingestion",
    tags=["documents"],
    responses={
        401: {"model": ErrorResponse, "description": "Unknown serverId or serverSecret"},
    },
)
async def submit_document(
    payload: DocumentIngestRequest,
    registry: Annotated[ServerRegistry, Depends(get_server_registry)],
    jobs: Annotated[JobManager, Depends(get_job_manager)],
) -> JobResponse:
    """Verify the calling server, queue the document, and return immediately.

    The job's id is the ``ragDbId`` itself, not a generated one -- a document
    submission exists to populate that database, and a second submission for
    the same ragDbId reuses the same job id (see ``JobManager``). What
    processing means downstream is not fully decided yet (see
    ``app.documents.DocumentProcessor``); the response only confirms the job
    was queued.
    """
    _require_server(registry, payload.server_id, payload.server_secret.get_secret_value())

    job = jobs.create(
        server_id=payload.server_id,
        document_link=payload.document_link,
        rag_db_id=payload.rag_db_id,
    )
    return JobResponse(job_id=job.job_id, status=job.status.value)
