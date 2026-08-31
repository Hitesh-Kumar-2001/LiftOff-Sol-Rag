"""Request and response bodies for the public API.

Callers name a ``projectId`` and never a ``ragDbId``. The database behind a
project is internal -- resolved by ``app.stores.projectStore`` and used from there
inward -- so it appears nowhere in this module. Anything a caller sends back to
us, or polls with, is the project id it already had.

``serverId`` says who is calling, for the log. Nothing verifies it: there is no
secret, no signature, and no registry behind it, so it is a label the caller
chose and not a claim the API can check. See the note in ``app.api.routes``.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# A projectId becomes a Firestore document id, so it has to be one. Firestore
# rejects an id that is longer than 1500 bytes, contains a slash, is exactly "."
# or "..", or is wrapped in double underscores -- and it rejects them with an
# InvalidArgument that escapes as a 500, several layers below the request that
# caused it. Checking here turns every one of those into a 422 naming the field.
#
# 128 rather than Firestore's 1500 because this is an identifier a caller
# chooses, not a payload: nothing legitimate is longer, and the cap is also what
# keeps a pathological id out of a Redis key and a log line.
ID_MAX_CHARS = 128
PROJECT_ID_MAX_CHARS = ID_MAX_CHARS


def checkDocumentId(value: str) -> str:
    """Refuse an id Firestore would refuse. Raises ValueError.

    Every one of these reaches Firestore as a document id, and Firestore turns
    a bad one into an InvalidArgument several layers below the request. What
    that becomes on the way out differs by which id it was, and both outcomes
    are wrong: an unchecked projectId escapes as a 500, and an unchecked
    conversationId is *worse* -- the store wraps it as "unreachable", the route
    degrades to answering without history (invariant 24), and the caller gets a
    cheerful 200 under an id that can never work. Which is precisely the failure
    invariant 25 exists to prevent, arriving through the guard meant to prevent
    it. It also spends a model call on every attempt.
    """
    if not value:
        raise ValueError("must not be empty")
    if len(value) > ID_MAX_CHARS:
        raise ValueError(f"must be at most {ID_MAX_CHARS} characters")
    if "/" in value:
        raise ValueError("must not contain '/'")
    if value in (".", ".."):
        raise ValueError("must not be '.' or '..'")
    if value.startswith("__") and value.endswith("__"):
        raise ValueError("must not be wrapped in double underscores")
    return value


def checkProjectId(projectId: str) -> str:
    """Refuse a projectId Firestore would refuse. Raises ValueError."""
    return checkDocumentId(projectId)


# The wire type. Everything that takes a projectId uses it, in a body or a path,
# so the rule cannot drift between the two -- which is exactly how the path
# routes ended up with no length cap when the project moved out of the body.
ProjectId = Annotated[
    str, Field(min_length=1, max_length=ID_MAX_CHARS), AfterValidator(checkProjectId)
]

# Same rule, and for the same reason: this is a document id too.
ConversationId = Annotated[
    str, Field(min_length=1, max_length=ID_MAX_CHARS), AfterValidator(checkDocumentId)
]


class CamelModel(BaseModel):
    """Base model that speaks camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ConversationCreateRequest(CamelModel):
    """Start a conversation. The project is in the path, not here."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    # Optional, and optional for a reason: left empty, the first question asked
    # here becomes the title (see FirestoreConversationStore._appendTurn). A
    # caller that has a name up front -- a UI where the user typed one -- can
    # say so instead.
    title: str = Field(default="", max_length=200)


class ConversationCreateResponse(CamelModel):
    """The conversation that now exists, and the id every later turn names.

    ``systemPrompt`` is the project's prompt as it stood at this moment,
    snapshotted onto the conversation. It is returned because it is the
    substantive thing this call decided: the conversation will answer under
    these instructions for its whole life, even if the project's prompt is
    edited tomorrow, and a caller cannot otherwise see which one it got.
    """

    conversation_id: str
    project_id: str
    system_prompt: str


class ConversationMessageRequest(CamelModel):
    """A question posted to the ``web`` gateway.

    Only the web gateway uses this shape. WhatsApp and LINE send their own
    envelope, which is validated by its signature rather than by a model here --
    a stronger check, and the only one available, since neither platform will
    add fields for us.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    # Absent means "start one", and the id of the new conversation comes back on
    # the response. An id that does not exist is a 404 rather than a new
    # conversation: a typo silently opening a fresh one is the failure a caller
    # cannot see, because it looks exactly like a model that has forgotten
    # everything.
    #
    # The webhook gateways never send this. Neither WhatsApp nor LINE has any
    # conversation id to offer -- they identify a *person*, not a thread -- so
    # one is minted on their behalf and remembered against them; see
    # app.stores.channelStore.
    conversation_id: ConversationId | None = None


class ConversationMessageResponse(CamelModel):
    answer: str
    project_id: str
    # Always returned, including on the turn that created it -- it is the only
    # way a caller learns the id of a conversation it did not name.
    conversation_id: str | None = None


class WebhookAck(CamelModel):
    """What a messaging platform gets back, and all it gets back.

    Small and uninformative on purpose. Nothing on the far side reads it, and a
    webhook response is not a place to describe this service's internals to
    whoever found the URL. ``accepted`` is for our own logs and tests: how many
    messages in the delivery were queued, after receipts, non-text and
    redeliveries were dropped.
    """

    status: str = "accepted"
    accepted: int = 0


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
    project_id: ProjectId


class JobStatusRequest(CamelModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    # The project whose ingestion is being asked about.
    project_id: ProjectId


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
    project_id: ProjectId
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
