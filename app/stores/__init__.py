"""Where things are kept: chunks, and the projectId -> ragDbId mapping.

Each is a Protocol plus a factory that picks the implementation from the
environment, so nothing above has to know more than one exists.
"""
