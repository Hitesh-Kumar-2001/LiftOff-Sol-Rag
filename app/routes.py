"""Public API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.jobManager import JobManager, getJobManager
from app.ragIngestionPipeline import ChunkingStrategy
from app.schemas import (
    DocumentIngestRequest,
    ErrorResponse,
    JobResponse,
    JobStatusRequest,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
)
from app.security import AuthenticationError, ServerRegistry, getServerRegistry

router = APIRouter(prefix="/api/v1")


def _requireServer(registry: ServerRegistry, serverId: str, serverSecret: str) -> None:
    """Raise the shared 401 if the caller cannot be verified."""
    try:
        registry.authenticate(serverId, serverSecret)
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
    registry: Annotated[ServerRegistry, Depends(getServerRegistry)],
) -> QueryResponse:
    """Verify the calling server, then answer its question from one database."""
    _requireServer(registry, payload.server_id, payload.server_secret.get_secret_value())

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
async def submitDocument(
    payload: DocumentIngestRequest,
    registry: Annotated[ServerRegistry, Depends(getServerRegistry)],
    jobs: Annotated[JobManager, Depends(getJobManager)],
) -> JobResponse:
    """Verify the calling server, queue the document, and return immediately.

    The job's id is the ``ragDbId`` itself, not a generated one -- a document
    submission exists to populate that database, and a second submission for
    the same ragDbId reuses the same job id (see ``JobManager``). What
    processing means downstream is not fully decided yet (see
    ``app.documents.DocumentProcessor``); the response only confirms the job
    was queued.
    """
    _requireServer(registry, payload.server_id, payload.server_secret.get_secret_value())

    job = jobs.create(
        serverId=payload.server_id,
        documentLink=payload.document_link,
        ragDbId=payload.rag_db_id,
    )
    return JobResponse(job_id=job.jobId, status=job.status.value)


@router.post(
    "/document/status",
    response_model=JobStatusResponse,
    response_model_exclude_none=True,
    summary="Check how a document's ingestion is going",
    tags=["documents"],
    responses={
        401: {"model": ErrorResponse, "description": "Unknown serverId or serverSecret"},
        404: {"model": ErrorResponse, "description": "No job for that ragDbId"},
    },
)
async def documentStatus(
    payload: JobStatusRequest,
    registry: Annotated[ServerRegistry, Depends(getServerRegistry)],
    jobs: Annotated[JobManager, Depends(getJobManager)],
) -> JobStatusResponse:
    """Verify the calling server, then report the job's status and where to
    go next -- nothing else.

    A POST rather than a GET so the credentials stay in the body: this API
    carries them there, and a GET would put a secret in a query string, where
    it lands in access logs and browser history.

    Jobs live in memory, so a ragDbId ingested before the last restart reads
    as 404 rather than as done. That distinction matters to a caller deciding
    whether to resubmit, which is why an unknown id is not reported as some
    'unknown' status.
    """
    _requireServer(registry, payload.server_id, payload.server_secret.get_secret_value())

    job = jobs.get(payload.rag_db_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingestion job for ragDbId '{payload.rag_db_id}'.",
        )

    # A document small enough to keep whole was never written to a vector
    # database, so answering with its ragDbId would name something there is
    # nothing behind. That caller is sent back to the source instead.
    if job.strategy is ChunkingStrategy.RAW:
        return JobStatusResponse(status=job.status.value, document_link=job.documentLink)

    return JobStatusResponse(status=job.status.value, rag_db_id=job.jobId)
