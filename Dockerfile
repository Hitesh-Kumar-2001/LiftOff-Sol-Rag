FROM python:3.12-slim

WORKDIR /app

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Send logs directly to Docker
ENV PYTHONUNBUFFERED=1

# .rar is the one archive format Python cannot read on its own: `rarfile` shells
# out to an external binary, and unlike zip there is nothing in the standard
# library behind it. The corpora here are RAR5, and the choice of binary is not
# a matter of taste -- it was measured against a real 200-file RAR5 archive:
#
#   unrar (this one)  200/200 members read correctly
#   unar              195/200; the other five come back as ZERO bytes, silently
#   bsdtar            0/200; returns 51 bytes for a 39KB member, silently
#
# Both free alternatives fail by *truncating*, not by erroring, and
# app.ingestion.documents._archiveText deliberately skips a member it cannot
# read rather than losing the whole document -- so either of them would build a
# RAG database quietly missing files. Hence the official unrar, which lives in
# Debian's non-free component. Its licence permits redistributing and using the
# decoder; what it forbids is reusing the source to build a RAR *compressor*.
RUN echo "deb http://deb.debian.org/debian bookworm main non-free non-free-firmware" \
      > /etc/apt/sources.list.d/nonfree.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends unrar \
    && rm -rf /var/lib/apt/lists/*

# tiktoken downloads its BPE data on first use. Without a writable cache that
# download happens on every cold start, and on a task with restricted egress it
# fails outright -- taking ingestion down for a reason that looks nothing like
# a network policy.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN mkdir -p /app/.tiktoken

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first
# This allows Docker to cache dependencies
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev

# Fail the build, not a production upload, if the rar backend cannot read RAR5.
# Checks extracted bytes against the size in the archive header, because the
# failure being guarded against is silent truncation rather than an exception.
COPY docker ./docker
RUN uv run python docker/checkRarBackend.py

# Copy application
COPY app ./app
COPY scripts ./scripts

# models.toml (which model answers, grades, summarises and chunks) and
# prompts.toml (who the agent is). app/modelConfig.py and app/promptConfig.py
# locate these relative to the package rather than the working directory -- app/
# sits at /app/app here, so config/ has to be at /app/config for that to
# resolve, which is exactly where this puts it.
#
# Without models.toml the image starts and dies in checkConfiguration with "no
# model is configured for agent", which is at least a loud failure rather than a
# 503 per request. Without prompts.toml it starts and serves the built-in
# fallback prompt, which is quieter and worse: the service answers as a plain
# assistant instead of as the persona somebody configured, and nothing looks
# wrong until a transcript is read.
#
# A deployment that wants different models or a different persona without a
# rebuild sets RAG_*_PROVIDER / RAG_*_MODEL or RAG_PERSONA; those win over these
# files.
COPY config ./config

# FastAPI
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
