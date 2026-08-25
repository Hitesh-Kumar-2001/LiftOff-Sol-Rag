"""Turning a document link into stored chunks.

``documents`` reads the file, ``ragSelector`` picks a chunking strategy,
``ragIngestionPipeline`` chunks and stores, and ``ragProcessor`` is the one
module that knows all three.
"""
