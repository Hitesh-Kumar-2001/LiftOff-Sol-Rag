import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.schemas import HealthResponse
from app.infra.machineStats import machineStats, primeCpuPercent
from app.jobs.jobManager import getJobManager

logger = logging.getLogger(__name__)

# What a validation error may say about itself. Everything else pydantic
# offers -- notably `input` -- is dropped; see validationErrorHandler.
SAFE_ERROR_KEYS = ("type", "loc", "msg")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Start the CPU measurement window now, so the first /health to arrive
    # reports usage since startup rather than the 0.0 psutil returns when it
    # has nothing to compare against. See app.infra.machineStats.
    primeCpuPercent()

    jobs = getJobManager()
    try:
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

    Pydantic reports a missing field by including the whole request body as the
    error's ``input``, so a 422 would otherwise echo whatever was sent into any
    log, proxy, or browser console that records the response. There is no
    secret in these bodies any more, but a request body is still the caller's
    data and not something to reflect back by default.

    An allowlist rather than dropping ``input`` by name, so a future pydantic
    field carrying request values cannot quietly reintroduce the echo. What
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


@app.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health() -> HealthResponse:
    """Liveness, plus what the machine is doing.

    The machine figures never change the outcome: this answers "am I running",
    and a busy host is still running. A section that could not be read is left
    out of the response rather than sent as null.

    ``machineStats`` already takes each reading independently and drops the
    ones that fail, so this catch is for the failures it cannot cover: a
    ``psutil`` build that raises on import-time-resolved state, or a
    ``MachineStats`` validation error from a figure some platform reports out
    of range. Either would 500 a liveness probe and pull a working service out
    of its load balancer over a diagnostic number, so the numbers go missing
    instead of the service.
    """
    try:
        machine = machineStats()
    except Exception:
        logger.warning("Could not read machine stats.", exc_info=True)
        machine = None

    return HealthResponse(status="ok", machine=machine)
