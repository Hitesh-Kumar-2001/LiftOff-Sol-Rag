"""Public API routes.

Every endpoint takes a ``projectId``. The ``ragDbId`` behind it is resolved
here, at the edge, and everything inward -- the job manager, the conflict
check, the chunk store -- goes on keying on that id exactly as before. The
indirection stops in this file, which is what keeps it cheap: see
``app.projectStore`` for why it exists at all.

**There is no authentication.** Every endpoint takes a ``serverId``, but
nothing verifies it: there is no secret, no signature, and no registry to check
it against. It is a label the caller picked, recorded in the log so a request
can be traced back to whoever says they made it, and treated as nothing more.
Anyone who can reach this API can therefore ingest into any project and read
any project's chunks, and can name themselves anything while doing it. That is
a deliberate choice for a service that is not yet exposed; putting it behind
anything public means adding authentication back first -- in front of these
routes, or in front of the whole app.

One place still *uses* ``serverId`` for a decision rather than a log line:
``app.jobs.resolveSubmission`` treats the same document from the same
``serverId`` as a retry and a different caller's document as a conflict. With
nothing verifying the id, a caller can put itself on either side of that by
choosing what to send. The consequence is a 409 avoided or provoked, not access
gained -- but it is the one spot where an unverified value still steers
behaviour.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.chunkStoreFactory import getChunkStore
from app.jobManager import JobConflictError, JobDispatchError, JobManager, getJobManager
from app.projectStore import ProjectStore, getProjectStore
from app.ragIngestionPipeline import ChunkingStrategy, ChunkStore
from app.schemas import (
    DocumentIngestRequest,
    ErrorResponse,
    JobResponse,
    JobStatusRequest,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against a project's RAG database",
    tags=["query"],
)
async def query(payload: QueryRequest) -> QueryResponse:
    """Answer a question from one project."""
    # Debug rather than info: a read is not a state change, and one line per
    # question would drown the ingestion events that are worth seeing.
    logger.debug("Query from '%s' on project '%s'.", payload.server_id, payload.project_id)

    # TODO: retrieval. It resolves the project the way /search does --
    # ``await projects.resolve(payload.project_id)``, read-only, because a
    # question does not bring a database into existence, and a project nothing
    # was ingested into answers empty rather than 404. Deliberately not wired
    # up yet: with nothing to hand the ragDbId to, resolving it would be a
    # round trip to Firestore on every call whose result is thrown away.
    return QueryResponse(
        answer=f"Retrieval is not implemented yet. Received question: {payload.question}",
        project_id=payload.project_id,
    )


@router.post(
    "/document",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a document for ingestion into a project",
    tags=["documents"],
    responses={
        409: {
            "model": ErrorResponse,
            "description": "A different document is already being ingested into this project",
        },
        503: {
            "model": ErrorResponse,
            "description": "The queue was unreachable; nothing was started, so resubmit as-is",
        },
    },
)
async def submitDocument(
    payload: DocumentIngestRequest,
    jobs: Annotated[JobManager, Depends(getJobManager)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
) -> JobResponse:
    """Queue the document and return immediately.

    The only endpoint that may *create* a project's database: a first
    submission is what brings one into existence, and every later request for
    that project resolves to the same one. What processing means downstream is
    not fully decided yet (see ``app.documents.DocumentProcessor``); the
    response only confirms the job was queued.
    """
    # Info, unlike the read routes: this one spends money and changes what is
    # stored, so it is the line worth having when working out who filled a
    # project with something unexpected.
    logger.info(
        "Ingestion requested by '%s' for project '%s': %s",
        payload.server_id,
        payload.project_id,
        payload.document_link,
    )

    # Resolved before the claim, because the claim is keyed on the ragDbId. A
    # mapping is permanent and idempotent, so minting one for a submission that
    # then turns out to conflict costs nothing -- the next submission for this
    # project resolves to the same id rather than a second one.
    ragDbId = await projects.resolveOrCreate(payload.project_id)

    try:
        job = await jobs.create(
            # Unverified, and recorded as given. It reaches resolveSubmission,
            # which is the one place it still decides something -- see the
            # module docstring.
            serverId=payload.server_id,
            documentLink=payload.document_link,
            ragDbId=ragDbId,
        )
    except JobConflictError:
        # Not a 202: nothing was queued, and running it anyway would leave the
        # database holding half of each document. The manager's message names
        # the ragDbId, which is internal, so it is reworded here in the terms
        # the caller actually sent.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Another document is already being ingested into project "
                f"'{payload.project_id}'. Wait for it to finish before submitting another."
            ),
        ) from None
    except JobDispatchError:
        # The request was fine; the queue behind it was not. A 503 says
        # "resubmit this exact request later", which a 500 does not.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Project '{payload.project_id}' could not be queued for processing. "
                f"Nothing was started; resubmit when the queue is reachable."
            ),
        ) from None

    return JobResponse(project_id=payload.project_id, status=job.status.value)


@router.post(
    "/document/status",
    response_model=JobStatusResponse,
    response_model_exclude_none=True,
    summary="Check how a project's ingestion is going",
    tags=["documents"],
    responses={
        404: {"model": ErrorResponse, "description": "No ingestion job for that project"},
    },
)
async def documentStatus(
    payload: JobStatusRequest,
    jobs: Annotated[JobManager, Depends(getJobManager)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
) -> JobStatusResponse:
    """Report the job's status and where to go next -- nothing else.

    A POST rather than a GET because the rest of this API takes its input in
    the body and answering a status check the same way keeps one shape across
    every endpoint.

    An unknown project is a 404 rather than some 'unknown' status: that
    distinction is what a caller needs to decide whether to resubmit. How long
    a project stays known depends on where the mapping and the job table live
    -- in memory that is until the next restart, in Firestore it is
    indefinitely (see app.projectStore and app.jobManager).
    """
    logger.debug(
        "Status check from '%s' on project '%s'.", payload.server_id, payload.project_id
    )

    # One 404 for both misses. A project nobody ever submitted and a project
    # whose job has since gone are the same answer to the caller.
    notFound = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No ingestion job for projectId '{payload.project_id}'.",
    )

    ragDbId = await projects.resolve(payload.project_id)
    if ragDbId is None:
        raise notFound

    job = await jobs.get(ragDbId)
    if job is None:
        raise notFound

    # A document small enough to keep whole was never written to a vector
    # database, so answering "this project is queryable" would name something
    # there is nothing behind. That caller is sent back to the source instead.
    if job.strategy is ChunkingStrategy.RAW:
        return JobStatusResponse(status=job.status.value, document_link=job.documentLink)

    return JobStatusResponse(status=job.status.value, project_id=payload.project_id)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search a project's RAG database for the chunks nearest a query",
    tags=["query"],
)
async def search(
    payload: SearchRequest,
    store: Annotated[ChunkStore, Depends(getChunkStore)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
) -> SearchResponse:
    """Return the chunks matching the query.

    Retrieval only -- the matching passages, not an answer written from them.
    Generating an answer is ``/query``'s job, and keeping the two apart means
    a caller can see what was actually retrieved rather than inferring it
    from a summary.

    Which store answers is not decided here: under test mode it is the local
    one and the ranking is keyword overlap, otherwise it is Pinecone and the
    ranking is embedding similarity. Both satisfy the same protocol.

    A project that was never ingested into has nothing to match, so it comes
    back with no hits rather than a 404 -- an empty database and an empty
    result set are the same answer to "what matches this query". Searching
    does not create a database either: minting one here would leave an empty
    namespace behind for every mistyped projectId that ever arrived.
    """
    logger.debug("Search from '%s' on project '%s'.", payload.server_id, payload.project_id)

    ragDbId = await projects.resolve(payload.project_id)
    results = await store.search(ragDbId, payload.query, payload.top_k) if ragDbId else []

    return SearchResponse(
        project_id=payload.project_id,
        hits=[
            SearchHit(text=r.text, chunk_index=r.index, score=r.score) for r in results
        ],
    )
