"""
Composition root for the pharmacy context.

The single place a concrete provider is named. Adding Apollo, Tata 1mg,
PharmEasy, Netmeds or an ONDC gateway is a new adapter plus a branch here —
`PHARMACY_PROVIDER` selects it, and no service, endpoint or model changes,
because none of them mention a vendor.

`local_db` is the only provider that can answer stock and price truthfully
today, so it is the default and the fallback for an unrecognised name.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.pharmacy.application.search import PharmacySearchService
from app.pharmacy.infrastructure.local_db_provider import LocalDbPharmacyProvider
from app.pharmacy.infrastructure.maps_adapters import (
    ORSDistanceSource,
    OSMPharmacyDiscovery,
)

logger = logging.getLogger(__name__)

_search_service: PharmacySearchService | None = None

PROVIDERS = {
    "local_db": LocalDbPharmacyProvider,
}


def build_search_service() -> PharmacySearchService:
    configured = (getattr(settings, "PHARMACY_PROVIDER", "") or "local_db").strip()
    provider_cls = PROVIDERS.get(configured)

    if provider_cls is None:
        # Falling back rather than raising: an unknown provider name in
        # configuration must not take the pharmacy feature offline, and the
        # local network is always a valid answer.
        logger.warning(
            "[PHARMACY_PROVIDER_UNKNOWN] '%s' is not registered; using local_db. "
            "Known providers: %s",
            configured, ", ".join(sorted(PROVIDERS)),
        )
        provider_cls = LocalDbPharmacyProvider

    # Both Maps adapters are optional and self-disable without a key, so they
    # are always wired: switching Maps on is purely a configuration change.
    provider = provider_cls(distance_source=ORSDistanceSource())
    return PharmacySearchService(
        provider=provider, discovery=OSMPharmacyDiscovery()
    )


def get_search_service() -> PharmacySearchService:
    global _search_service
    if _search_service is None:
        _search_service = build_search_service()
        logger.info(
            "[PHARMACY_READY] provider=%s discovery=openstreetmap",
            getattr(settings, "PHARMACY_PROVIDER", "local_db"),
        )
    return _search_service


def set_search_service(service: PharmacySearchService | None) -> None:
    """Override the process-wide service. Passing None restores the default."""
    global _search_service
    _search_service = service
