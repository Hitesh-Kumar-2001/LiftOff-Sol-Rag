"""Vercel entry point.

Vercel's Python runtime looks for a module-level ASGI application called
``app``; this file exists only to expose the one from ``app.main`` at the
path the platform expects. Everything about the API itself lives there.
"""

from app.main import app

__all__ = ["app"]
