"""Request and response bodies for the public API.

Callers name a ``projectId`` and never a ``ragDbId``. The database behind a
project is internal -- resolved by ``app.stores.projectStore`` and used from there
inward -- so it appears nowhere in this module. Anything a caller sends back to
us, or polls with, is the project id it already had.

``serverId`` says who is calling, for the log. Nothing verifies it: there is no
secret, no signature, and no registry behind it, so it is a label the caller
chose and not a claim the API can check. See the note in ``app.api.routes``.
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base model that speaks camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class QueryRequest(CamelModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    project_id: str = Field(min_length=1, max_length=128)
    # The conversation this question belongs to. Absent means "start one", and
    # the id of the new chat comes back on the response. A chatId that does not
    # exist is a 404 rather than a new chat: a typo silently opening a fresh
    # conversation is the one failure a caller cannot see, because it looks
    # exactly like a model that has forgotten everything.
    chat_id: str | None = Field(default=None, min_length=1, max_length=128)


class QueryResponse(CamelModel):
    answer: str
    project_id: str
    # Always returned, including on the turn that created it -- it is the only
    # way a caller learns the id of a chat it did not name.
    chat_id: str | None = None


class DocumentIngestRequest(CamelModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    # Not restricted to http(s): left as a plain string so gs://, s3://, or an
    # internal path can be supported later without a schema change.
    document_link: str = Field(min_length=1, max_length=2048)
    # The project this document belongs to. A project's first submission is
    # what brings its RAG database into existence -- see app.stores.projectStore.
    project_id: str = Field(min_length=1, max_length=128)


class JobStatusRequest(CamelModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    # The project whose ingestion is being asked about.
    project_id: str = Field(min_length=1, max_length=128)


class JobResponse(CamelModel):
    """The submission was accepted. Poll ``/document/status`` with the same
    ``projectId`` to follow it."""

    # Deliberately no job id. A job is keyed by the ragDbId behind the project
    # (see app.jobs.job.Job) and that id is internal -- handing it out would make it
    # something callers hold and we could no longer change. The projectId they
    # already sent is what /document/status takes, so it is the only id worth
    # returning. This is a comment rather than part of the docstring above
    # because a docstring here is published as the schema description in the
    # OpenAPI spec, and our storage layout is not the caller's business.
    project_id: str
    status: str


class JobStatusResponse(CamelModel):
    """Deliberately just two fields: the status, and where to go next.

    No detail, no metadata, no chunk counts -- a caller polling this wants to
    know whether it can query yet, and everything else is either noise or a
    description of another server's document.

    "Where to go next" is one of the two below, never both. A document kept
    whole (ChunkingStrategy.RAW) was never written to a vector database, so
    there is nothing to query at all; that caller gets ``documentLink`` back
    and should read the source directly. Everything else gets ``projectId``,
    meaning "this project is queryable now". The absent field is omitted from
    the response rather than sent as null, so which one arrived is unambiguous.
    """

    status: str
    project_id: str | None = None
    document_link: str | None = None


class SearchRequest(CamelModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    # Capped so one request cannot ask for a whole database back.
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(CamelModel):
    text: str
    chunk_index: int
    # Comparable within one response, not across stores: Pinecone scores are
    # embedding similarities, the offline store's are keyword overlap.
    score: float


class SearchResponse(CamelModel):
    project_id: str
    hits: list[SearchHit]


class ErrorResponse(CamelModel):
    detail: str


class CpuStats(CamelModel):
    used_percent: float
    # Logical cores, hyperthreads included.
    count: int


class MemoryStats(CamelModel):
    total_bytes: int
    available_bytes: int
    used_bytes: int
    used_percent: float


class DiskStats(CamelModel):
    # Which filesystem these figures describe -- the mount containing this
    # path, not necessarily the path itself.
    path: str
    total_bytes: int
    free_bytes: int
    used_bytes: int
    used_percent: float


class MachineStats(CamelModel):
    """Whatever could be read about the machine.

    Every field is optional because every reading is taken independently and
    a platform may refuse any of them. A section that could not be read is
    omitted rather than sent as null or zero -- "no answer" and "0%" mean
    very different things to whoever is looking at this.
    """

    cpu: CpuStats | None = None
    memory: MemoryStats | None = None
    disk: DiskStats | None = None


class HealthResponse(CamelModel):
    """``status`` first and unchanged: anything already watching this endpoint
    is checking that field, and the machine figures are an addition to it, not
    a replacement. They are informational only -- a machine under load is still
    ``ok``, because this endpoint answers "am I running", and failing it would
    pull a working service out of a load balancer."""

    status: str
    machine: MachineStats | None = None
