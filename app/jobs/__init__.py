"""Ingestion jobs: the record, the rules, the table, the queue, the worker.

``job`` holds what a job *is*, plus the submission rules every store shares.
``jobManager`` picks how this deployment runs them. ``redisJobStore`` is the
table, ``jobQueue`` the hand-off, and ``worker`` the process on the far end.
"""
