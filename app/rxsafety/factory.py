"""
Composition root for the rxsafety context.

The single place that names concrete providers. Swapping openFDA for DrugBank,
First Databank or Medi-Span is a change here plus one new adapter — no
application or domain code moves, because both sides only know the ports.

`set_prescription_verifier` exists for tests and for runtime substitution, and
mirrors `set_maps_service` in the maps service.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.rxsafety.application.verifier import PrescriptionVerifier
from app.rxsafety.infrastructure.explainer import GroqSafetyExplainer
from app.rxsafety.infrastructure.openfda import OpenFDALabelSource
from app.rxsafety.infrastructure.rxnorm import RxNormNormaliser

logger = logging.getLogger(__name__)

_verifier: PrescriptionVerifier | None = None


def build_prescription_verifier() -> PrescriptionVerifier:
    """Construct the default provider stack."""
    return PrescriptionVerifier(
        normaliser=RxNormNormaliser(),
        label_source=OpenFDALabelSource(),
        explainer=GroqSafetyExplainer(),
    )


def get_prescription_verifier() -> PrescriptionVerifier:
    global _verifier
    if _verifier is None:
        _verifier = build_prescription_verifier()
        logger.info(
            "[RXSAFETY_READY] normaliser=rxnorm labels=openfda explainer=groq "
            "enabled=%s",
            settings.RXSAFETY_ENABLED,
        )
    return _verifier


def set_prescription_verifier(verifier: PrescriptionVerifier | None) -> None:
    """Override the process-wide verifier. Passing None restores the default."""
    global _verifier
    _verifier = verifier


def is_enabled() -> bool:
    """Read live so the switch is configuration, not a deployment."""
    return bool(settings.RXSAFETY_ENABLED)
