"""
Structured product labels from openFDA.

openFDA republishes FDA-approved labelling. It is free and works without a key,
though `OPENFDA_API_KEY` raises the rate limit from 240 to 240_000 requests per
day and is read from settings when present.

What comes back is the manufacturer's own label text. That matters for how it is
presented: an excerpt cited here is a quotation from an approved label, not an
assertion by MedBridge or by a model. Sections are returned verbatim and the
reader is shown the wording alongside a link to the source record.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

import httpx

from app.core.config import settings
from app.rxsafety.domain.entities import DrugConcept, DrugLabel

logger = logging.getLogger(__name__)

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
REQUEST_TIMEOUT_SECONDS = 8.0
MAX_CONCURRENCY = 4
MAX_EXCERPT_CHARS = 1200
"""Label sections run to thousands of words. Excerpts are truncated for storage
and display; `reference` always points at the full record."""


def _section(payload: dict, *keys: str) -> list[str]:
    """
    Pull the first populated section among `keys`.

    openFDA is inconsistent about which field carries a given topic — renal
    guidance appears under `use_in_specific_populations` on newer labels and
    under `precautions` on older ones — so each topic lists its candidates in
    preference order.
    """
    for key in keys:
        value = payload.get(key)
        if not value:
            continue
        if isinstance(value, str):
            value = [value]
        cleaned = [t.strip()[:MAX_EXCERPT_CHARS] for t in value if isinstance(t, str) and t.strip()]
        if cleaned:
            return cleaned
    return []


class OpenFDALabelSource:
    """Implements `DrugLabelSource`."""

    name = "openfda"

    def __init__(self, base_url: str = OPENFDA_LABEL_URL) -> None:
        self._url = base_url
        self._failed = False

    def _params(self, search: str) -> dict:
        params = {"search": search, "limit": 1}
        api_key = (getattr(settings, "OPENFDA_API_KEY", "") or "").strip()
        if api_key:
            params["api_key"] = api_key
        return params

    async def _query(self, client: httpx.AsyncClient, search: str) -> dict | None:
        try:
            response = await client.get(self._url, params=self._params(search))
            if response.status_code == 404:
                # openFDA returns 404 for "no matching record", which is a
                # legitimate answer rather than a failure: this drug has no
                # published label. Distinct from an outage, which is caught
                # below and leaves the drug marked unchecked.
                return {}
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            if not self._failed:
                logger.warning("[OPENFDA_UNAVAILABLE] %s: %s", self._url, exc)
                self._failed = True
            return None

        results = body.get("results") or []
        return results[0] if results else {}

    async def _fetch(self, client: httpx.AsyncClient, concept: DrugConcept) -> DrugLabel | None:
        if not concept.resolved:
            return None

        payload = await self._query(client, f'openfda.rxcui:"{concept.rxcui}"')

        # Fall back to the ingredient name. Many labels carry no rxcui in their
        # openfda block, and skipping them would silently drop real warnings.
        if payload == {} and concept.ingredients:
            payload = await self._query(
                client, f'openfda.generic_name:"{concept.ingredients[0]}"'
            )

        if payload is None:
            return None
        if not payload:
            return DrugLabel(rxcui=concept.rxcui)

        openfda = payload.get("openfda") or {}
        record_id = payload.get("id") or ""

        return DrugLabel(
            rxcui=concept.rxcui,
            brand_name=(openfda.get("brand_name") or [""])[0],
            generic_name=(openfda.get("generic_name") or [""])[0],
            drug_interactions=_section(payload, "drug_interactions", "drug_and_or_laboratory_test_interactions"),
            contraindications=_section(payload, "contraindications"),
            warnings=_section(payload, "warnings_and_cautions", "warnings", "boxed_warning"),
            pregnancy=_section(payload, "pregnancy", "teratogenic_effects", "nursing_mothers"),
            geriatric_use=_section(payload, "geriatric_use"),
            renal_notes=_section(payload, "use_in_specific_populations", "precautions"),
            hepatic_notes=_section(payload, "hepatic_impairment", "use_in_specific_populations"),
            dosage_and_administration=_section(payload, "dosage_and_administration"),
            food_effect=_section(payload, "food_effect", "information_for_patients"),
            reference=(
                f"{OPENFDA_LABEL_URL}?search=id:{record_id}" if record_id else OPENFDA_LABEL_URL
            ),
        )

    async def fetch_label(self, concept: DrugConcept) -> DrugLabel | None:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            return await self._fetch(client, concept)

    async def fetch_labels(
        self, concepts: Sequence[DrugConcept]
    ) -> dict[str, DrugLabel | None]:
        """
        Batch variant. Keyed by rxcui; a None value means "could not retrieve",
        which the caller records as unchecked rather than as safe.
        """
        resolved = [c for c in concepts if c.resolved]
        if not resolved:
            return {}

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:

            async def one(concept: DrugConcept) -> tuple[str, DrugLabel | None]:
                async with semaphore:
                    try:
                        return concept.rxcui, await self._fetch(client, concept)
                    except Exception as exc:
                        logger.warning(
                            "[OPENFDA_FETCH_FAILED] rxcui=%s: %s", concept.rxcui, exc
                        )
                        return concept.rxcui, None

            pairs = await asyncio.gather(*(one(c) for c in resolved))

        return dict(pairs)
