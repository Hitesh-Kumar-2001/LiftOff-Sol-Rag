import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes import router
from app.security import get_server_registry, refreshing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Resolved the same way the route resolves it, so startup can never fill one
    # registry while requests read another.
    registry = get_server_registry()

    # Fill memory before the first request lands. Failing here is deliberate: a
    # process that cannot read the credential store should not accept traffic.
    logger.info("Loaded %d server credentials.", await registry.load_all())

    async with refreshing(registry):
        yield


app = FastAPI(title="RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "RAG API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
