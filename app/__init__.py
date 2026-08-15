"""FastAPI application package."""

from dotenv import load_dotenv

# Runs once, before any other ``app.*`` module reads an env var for config
# (GEMINI_API_KEY, PINECONE_API_KEY, ...) -- every submodule import goes
# through this package init first.
load_dotenv()

