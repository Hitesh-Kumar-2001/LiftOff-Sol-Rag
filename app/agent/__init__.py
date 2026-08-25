"""The answering agent, and everything it needs to run.

``llmManager`` builds a chat model for any supported provider. ``promptStore``
resolves a project's system prompt (Firestore, cached in Redis). ``tools`` is
what the agent may reach for -- this project's documents, and web search.
``reviewer`` grades the draft. ``agent`` is the entry point that runs the loop:
answer, review once, optionally retry once, return.
"""
