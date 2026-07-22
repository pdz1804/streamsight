"""Domain exceptions.

Each carries the HTTP status the API layer should surface, so route handlers stay
free of status-code branching.
"""

from __future__ import annotations


class StreamSightError(Exception):
    """Base class for all recoverable application errors."""

    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidFrameError(StreamSightError):
    """The submitted image could not be decoded."""

    status_code = 400


class BackendUnavailableError(StreamSightError):
    """The requested inference backend cannot run on this host."""

    status_code = 409


class NoBackendError(StreamSightError):
    """No inference backend at all is runnable - the service cannot serve."""

    status_code = 503


class SourceUnavailableError(StreamSightError):
    """A video source could not be opened."""

    status_code = 400
