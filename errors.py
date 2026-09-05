"""Shared exception types for the hhgoa-task3 pipeline.

Every failure that is *expected* (a missing credential, a photo with no face in
it, a chain that is not running) raises a :class:`PipelineError` subclass so
``pipeline.py`` can print a short, actionable message instead of dumping a raw
traceback at the user.

Each error carries an optional ``hint`` (what to do about it) and an ``exit_code``
so the CLI can signal *which* stage failed to a calling script.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all expected, user-facing pipeline failures."""

    #: Process exit code used when this error reaches the CLI boundary.
    exit_code: int = 1

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ConfigError(PipelineError):
    """Something is missing or malformed in the environment / .env file."""

    exit_code = 2


class FaceDetectionError(PipelineError):
    """The image could not be loaded, or did not contain usable face(s)."""

    exit_code = 3


class SearchError(PipelineError):
    """The reverse-image-search call itself failed (network, quota, bad key)."""

    exit_code = 4


class NoMatchFound(PipelineError):
    """The search succeeded but returned no social-media match.

    This is an honest, valid outcome -- not a bug. See constraint 1.2 in the
    task brief: nothing is ever substituted for a real match.
    """

    exit_code = 5


class ChainError(PipelineError):
    """Deploying to / reading from / writing to the blockchain failed."""

    exit_code = 6


class VerificationFailed(PipelineError):
    """On-chain re-verification did not produce the expected MATCH."""

    exit_code = 7
