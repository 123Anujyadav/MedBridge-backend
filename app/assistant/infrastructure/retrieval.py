"""
Knowledge retrieval for the AI Medical Assistant.

Reuses the medical corpus already ingested by the source project into a local
(file-backed) Qdrant collection. Retrieval is strictly optional: if
`qdrant-client` is not installed, the corpus is absent, or the embedding model
cannot load, the adapter reports itself unavailable and the pipeline answers
from model knowledge instead. It never raises into the request path.

Imports of the heavy optional stack are deferred to first use so that neither
the API process nor the test suite pays for them unless retrieval is actually
exercised.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.assistant.application.ports import RetrievedSnippet
from app.assistant.config import AssistantSettings, get_assistant_settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""Must match what the source project ingested with, or vectors won't align."""


class QdrantKnowledgeRetriever:
    """Semantic search over the bundled medical corpus."""

    def __init__(self, settings: AssistantSettings | None = None) -> None:
        self._settings = settings or get_assistant_settings()
        self._client: Any = None
        self._encoder: Any = None
        self._initialised = False
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def _initialise(self) -> None:
        """
        Lazily load the client and encoder.

        Runs once; any failure permanently marks the retriever unavailable
        rather than retrying heavy imports on every message.
        """
        if self._initialised:
            return
        self._initialised = True

        path = self._settings.qdrant_local_path
        if not self._settings.enable_retrieval or not path:
            logger.info("[ASSISTANT_RAG_DISABLED] no corpus configured")
            return

        try:
            from qdrant_client import QdrantClient
        except ImportError:
            logger.info(
                "[ASSISTANT_RAG_UNAVAILABLE] qdrant-client not installed; "
                "answering from model knowledge"
            )
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.info("[ASSISTANT_RAG_UNAVAILABLE] sentence-transformers missing")
            return

        try:
            self._client = QdrantClient(path=path)
            collections = {
                c.name for c in self._client.get_collections().collections
            }
            if self._settings.qdrant_collection not in collections:
                logger.warning(
                    "[ASSISTANT_RAG_NO_COLLECTION] %s not in %s",
                    self._settings.qdrant_collection,
                    sorted(collections),
                )
                self._client = None
                return

            self._encoder = SentenceTransformer(_EMBEDDING_MODEL)

            if not self._is_compatible():
                self._client = None
                self._encoder = None
                return

            self._available = True
            logger.info(
                "[ASSISTANT_RAG_READY] collection=%s",
                self._settings.qdrant_collection,
            )
        except Exception:
            logger.exception("[ASSISTANT_RAG_INIT_FAILED] continuing without retrieval")
            self._client = None
            self._encoder = None
            self._available = False

    def _encoder_dimension(self) -> int:
        """
        Embedding width, across sentence-transformers versions.

        `get_sentence_embedding_dimension` was renamed to
        `get_embedding_dimension` in 5.x and now emits a FutureWarning.
        """
        for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            method = getattr(self._encoder, attr, None)
            if callable(method):
                return int(method())
        return 0

    def _is_compatible(self) -> bool:
        """
        Verify the collection's vectors match this encoder.

        Querying a collection with the wrong dimensionality returns zero hits
        forever without erroring, which is indistinguishable from "nothing
        relevant found". Checking up front turns a silent dead end into an
        explicit, actionable log line.
        """
        try:
            info = self._client.get_collection(self._settings.qdrant_collection)
            params = info.config.params.vectors
            encoder_dim = int(self._encoder_dimension())

            # Named-vector collections need a `using=` selector the ingesting
            # project chose; a plain query cannot address them.
            if isinstance(params, dict):
                names = sorted(params)
                sizes = {n: getattr(params[n], "size", None) for n in names}
                logger.warning(
                    "[ASSISTANT_RAG_INCOMPATIBLE] collection uses named vectors "
                    "%s with sizes %s; this adapter embeds unnamed %d-dim vectors "
                    "(%s). Retrieval disabled — the assistant will answer from "
                    "model knowledge. Re-ingest the corpus with %s to enable RAG.",
                    names,
                    sizes,
                    encoder_dim,
                    _EMBEDDING_MODEL,
                    _EMBEDDING_MODEL,
                )
                return False

            collection_dim = int(getattr(params, "size", 0))
            if collection_dim != encoder_dim:
                logger.warning(
                    "[ASSISTANT_RAG_INCOMPATIBLE] collection vectors are %d-dim "
                    "but %s produces %d-dim. Retrieval disabled.",
                    collection_dim,
                    _EMBEDDING_MODEL,
                    encoder_dim,
                )
                return False

            return True
        except Exception:
            logger.exception("[ASSISTANT_RAG_COMPAT_CHECK_FAILED] disabling retrieval")
            return False

    def _query(self, vector: list[float], limit: int) -> list[Any]:
        """
        Run the vector search across qdrant-client versions.

        `search()` was removed in 1.14 in favour of `query_points()`. Supporting
        both keeps this working against the client the source project pinned
        (1.13) and the current release.
        """
        collection = self._settings.qdrant_collection

        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(response, "points", response) or [])

        return list(
            self._client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        )

    def _search_sync(self, query: str, limit: int) -> list[RetrievedSnippet]:
        """Blocking search, run off the event loop by `retrieve`."""
        self._initialise()
        if not self._available or self._client is None or self._encoder is None:
            return []

        vector = self._encoder.encode(query).tolist()
        hits = self._query(vector, limit)

        snippets: list[RetrievedSnippet] = []
        for hit in hits:
            payload = hit.payload or {}
            text = (
                payload.get("page_content")
                or payload.get("text")
                or payload.get("content")
                or ""
            )
            if not str(text).strip():
                continue
            metadata = payload.get("metadata") or {}
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or payload.get("source")
                or "Medical knowledge base"
            )
            snippets.append(
                RetrievedSnippet(
                    text=str(text), source=str(source), score=float(hit.score or 0.0)
                )
            )
        return snippets

    async def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedSnippet]:
        """
        Search the corpus.

        Offloaded to a worker thread: `qdrant-client` and SentenceTransformer are
        synchronous and would otherwise stall the event loop for every request.
        """
        if not query or not query.strip():
            return []
        try:
            return await asyncio.to_thread(self._search_sync, query, limit)
        except Exception:
            logger.exception("[ASSISTANT_RAG_SEARCH_FAILED] returning no context")
            return []


class NullKnowledgeRetriever:
    """Always-empty retriever. Used in tests and when retrieval is disabled."""

    @property
    def is_available(self) -> bool:
        return False

    async def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedSnippet]:
        return []
