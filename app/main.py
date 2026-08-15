import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.jobManager import getJobManager
from app.routes import router
from app.security import getServerRegistry, refreshing

logger = logging.getLogger(__name__)

# What a validation error may say about itself. Everything else pydantic
# offers -- notably `input` -- is dropped; see validationErrorHandler.
SAFE_ERROR_KEYS = ("type", "loc", "msg")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Resolved the same way the route resolves it, so startup can never fill one
    # registry while requests read another.
    registry = getServerRegistry()

    # Fill memory before the first request lands. Failing here is deliberate: a
    # process that cannot read the credential store should not accept traffic.
    logger.info("Loaded %d server credentials.", await registry.loadAll())

    jobs = getJobManager()
    try:
        async with refreshing(registry):
            yield
    finally:
        # Let in-flight ingestion jobs finish rather than be killed mid-write
        # when the process exits. This waits; it does not cancel.
        await jobs.shutdown()


app = FastAPI(title="RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validationErrorHandler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Answer a malformed request without quoting it back.

    Pydantic reports a missing field by including the whole request body as
    the error's ``input``, which on these endpoints means the plaintext
    ``serverSecret`` lands in the 422 body -- and in any log, proxy, or
    browser console that records the response. ``SecretStr`` does not help
    here: the value never becomes one, because validation is what failed.

    An allowlist rather than dropping ``input`` by name, so a future pydantic
    field carrying request values cannot quietly reintroduce the leak. What
    survives is enough to fix a request: what was wrong, and where.
    """
    detail = [
        {key: value for key, value in error.items() if key in SAFE_ERROR_KEYS}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": detail}
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "RAG API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
