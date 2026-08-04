"""
RxNorm normalisation via the NLM RxNav REST API.

RxNav is public, unauthenticated and free, which is why it is the default
normaliser. It is used for one job only: turning whatever the clinician typed
into an RxCUI and its active ingredients.

It is deliberately *not* used for interactions. NLM retired the RxNav drug
interaction API in January 2024; the endpoint either 404s or returns an empty
payload depending on the path, and an empty payload from a dead endpoint is
indistinguishable from "no interactions found" unless you know it is dead.
Reading that as a clean bill of health would be exactly the failure mode this
context exists to prevent. Interaction evidence comes from openFDA labels
instead, and same-ingredient duplication is checked locally against the
ingredient lists returned here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

import httpx

from app.rxsafety.domain.entities import DrugConcept

logger = logging.getLogger(__name__)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
REQUEST_TIMEOUT_SECONDS = 6.0
MAX_CONCURRENCY = 5
"""RxNav is a shared public service; a prescription of 20 drugs should not
open 20 sockets against it at once."""


class RxNormNormaliser:
    """Implements `DrugNormaliser`."""

    name = "rxnorm"

    def __init__(self, base_url: str = RXNAV_BASE) -> None:
        self._base = base_url.rstrip("/")
        self._failed = False

    async def _get(self, client: httpx.AsyncClient, path: str, params: dict) -> dict | None:
        """One request, every failure flattened to None."""
        try:
            response = await client.get(f"{self._base}{path}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            # Logged once per process. A normaliser that is down is down for
            # every drug in the prescription, and one line says so as well as
            # twenty do.
            if not self._failed:
                logger.warning("[RXNORM_UNAVAILABLE] %s%s: %s", self._base, path, exc)
                self._failed = True
            return None

    async def _resolve(self, client: httpx.AsyncClient, drug_name: str) -> DrugConcept:
        cleaned = (drug_name or "").strip()
        if not cleaned:
            return DrugConcept(original_name=drug_name or "")

        # search=2 turns on normalised matching, which is what lets a brand or a
        # misspelling resolve rather than returning nothing.
        body = await self._get(client, "/rxcui.json", {"name": cleaned, "search": 2})
        rxcui = None
        if body:
            ids = (body.get("idGroup") or {}).get("rxnormId") or []
            rxcui = ids[0] if ids else None

        if not rxcui:
            return DrugConcept(original_name=cleaned)

        properties = await self._get(client, f"/rxcui/{rxcui}/properties.json", {})
        normalised = None
        if properties:
            normalised = (properties.get("properties") or {}).get("name")

        # tty=IN gives the active ingredients. Ingredient overlap is how
        # duplicate therapy is detected, so this is load-bearing rather than
        # decorative: "Crocin" and "Paracetamol" both reduce to acetaminophen.
        related = await self._get(
            client, f"/rxcui/{rxcui}/related.json", {"tty": "IN"}
        )
        ingredients: list[str] = []
        if related:
            for group in (related.get("relatedGroup") or {}).get("conceptGroup") or []:
                for concept in group.get("conceptProperties") or []:
                    ingredient = (concept.get("name") or "").strip().lower()
                    if ingredient and ingredient not in ingredients:
                        ingredients.append(ingredient)

        return DrugConcept(
            original_name=cleaned,
            rxcui=str(rxcui),
            normalised_name=normalised or cleaned,
            ingredients=tuple(ingredients),
        )

    async def normalise(self, drug_name: str) -> DrugConcept:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            return await self._resolve(client, drug_name)

    async def normalise_many(self, drug_names: Sequence[str]) -> list[DrugConcept]:
        if not drug_names:
            return []

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:

            async def one(drug_name: str) -> DrugConcept:
                async with semaphore:
                    return await self._resolve(client, drug_name)

            results = await asyncio.gather(
                *(one(n) for n in drug_names), return_exceptions=True
            )

        concepts: list[DrugConcept] = []
        for drug_name, result in zip(drug_names, results):
            if isinstance(result, BaseException):
                logger.warning("[RXNORM_RESOLVE_FAILED] %s: %s", drug_name, result)
                concepts.append(DrugConcept(original_name=drug_name))
            else:
                concepts.append(result)
        return concepts
