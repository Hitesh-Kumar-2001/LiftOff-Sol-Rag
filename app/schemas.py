"""Request and response bodies for the public API."""

from pydantic import BaseModel, ConfigDict, Field, SecretStr
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
    server_secret: SecretStr = Field(min_length=1, max_length=512)
    question: str = Field(min_length=1, max_length=4000)
    rag_db_id: str = Field(min_length=1, max_length=128)


class QueryResponse(CamelModel):
    answer: str
    rag_db_id: str


class DocumentIngestRequest(CamelModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    server_secret: SecretStr = Field(min_length=1, max_length=512)
    # Not restricted to http(s): left as a plain string so gs://, s3://, or an
    # internal path can be supported later without a schema change.
    document_link: str = Field(min_length=1, max_length=2048)
    # The database this document is ingested into. Also becomes the job's id
    # -- see app.jobs.Job.job_id.
    rag_db_id: str = Field(min_length=1, max_length=128)


class JobStatusRequest(CamelModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    server_id: str = Field(min_length=1, max_length=128)
    server_secret: SecretStr = Field(min_length=1, max_length=512)
    # The database whose ingestion is being asked about. Also the job's id --
    # see app.jobs.Job.job_id.
    rag_db_id: str = Field(min_length=1, max_length=128)


class JobResponse(CamelModel):
    job_id: str
    status: str


class JobStatusResponse(CamelModel):
    """Deliberately just two fields: the status, and where to go next.

    No detail, no metadata, no chunk counts -- a caller polling this wants to
    know whether it can query yet, and everything else is either noise or a
    description of another server's document.

    "Where to go next" is one of the two below, never both. A document kept
    whole (ChunkingStrategy.RAW) was never written to a vector database, so
    there is nothing to query by ragDbId; that caller gets ``documentLink``
    back and should read the source directly. Everything else gets
    ``ragDbId``. The absent field is omitted from the response rather than
    sent as null, so which one arrived is unambiguous.
    """

    status: str
    rag_db_id: str | None = None
    document_link: str | None = None


class ErrorResponse(CamelModel):
    detail: str
