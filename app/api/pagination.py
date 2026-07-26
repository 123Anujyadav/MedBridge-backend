"""
Shared pagination parameters for list endpoints.

Response shapes are intentionally left as plain arrays. Wrapping them in an
envelope would be a cleaner API but would break every existing frontend caller,
so the bound is applied via optional query parameters with safe defaults —
callers that pass nothing simply get a capped first page instead of the whole
table.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


@dataclass(frozen=True, slots=True)
class Pagination:
    """Validated offset/limit for a list query."""

    skip: int
    limit: int


def pagination_params(
    skip: int = Query(0, ge=0, description="Rows to skip."),
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description=f"Maximum rows to return (max {MAX_LIMIT}).",
    ),
) -> Pagination:
    """FastAPI dependency supplying bounded pagination parameters."""
    return Pagination(skip=skip, limit=limit)
