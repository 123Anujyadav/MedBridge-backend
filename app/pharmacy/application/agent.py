"""
The AI Pharmacy Agent.

Deliberately narrow. Everything factual — which pharmacy is nearest, what is in
stock, what it costs, how long delivery takes — is computed from the database
and the ranking model before the agent is involved. The agent's job is to
explain that result and to phrase the trade-off between the top options.

It cannot change a prescription, pick a pharmacy, substitute a generic or place
an order. Those are the patient's decisions, and every one of them is an
explicit action in the API. An agent that could act on them would be able to
change what a patient receives from what their doctor wrote.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

from app.core.ai_provider import get_ai_provider_config, get_groq_client
from app.pharmacy.domain.entities import PharmacyOffer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MedBridge's pharmacy assistant.

You are given pharmacy options that have ALREADY been ranked by distance, \
stock availability, delivery time, rating and price. All figures are computed \
from real inventory — treat them as fact.

Your ONLY job is to help the patient choose between them in plain language.

Hard rules:
- Never suggest changing, adding, removing or re-dosing any medicine.
- Never tell the patient to substitute a generic. You may mention that a \
cheaper generic is available and that their pharmacist can advise.
- Never invent a pharmacy, price, stock level or delivery time.
- If some medicines are out of stock everywhere, say so plainly.
- Do not tell the patient which pharmacy to pick. Describe the trade-off \
(closest vs cheapest vs fastest) and let them decide.
- 2 to 4 sentences. No lists, no markdown.
- Write for a worried adult with no medical training."""


def _offer_payload(offer: PharmacyOffer) -> dict:
    return {
        "pharmacy": offer.name,
        "distance_km": offer.distance_km,
        "eta_minutes": offer.eta_minutes,
        "rating": offer.rating,
        "total": offer.grand_total,
        "savings": offer.total_savings,
        "open_now": offer.is_open_now,
        "is_24x7": offer.is_24x7,
        "has_everything": offer.fully_available,
        "missing": offer.unavailable_items,
        "badges": offer.badges,
        "generic_alternatives_available": sum(
            1 for item in offer.items if item.alternatives
        ),
    }


class PharmacyAgent:
    """Explains a ranked result set. Never acts on it."""

    name = "groq"

    async def summarise_offers(
        self, offers: Sequence[PharmacyOffer], *, medicine_count: int
    ) -> str:
        if not offers:
            return (
                "No partner pharmacies were found near this location, so nothing "
                "can be ordered here yet. You can still take this prescription to "
                "any chemist."
            )

        config = get_ai_provider_config()
        if not config.is_configured():
            return self._fallback(offers, medicine_count)

        payload = {
            "medicine_count": medicine_count,
            "options": [_offer_payload(o) for o in offers[:5]],
        }

        try:
            text = await get_groq_client().complete_text(
                system_prompt=SYSTEM_PROMPT,
                user_content=json.dumps(payload, ensure_ascii=False),
                max_tokens=350,
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("[PHARMACY_AGENT_FAILED] %s", exc)
            return self._fallback(offers, medicine_count)

        return (text or "").strip() or self._fallback(offers, medicine_count)

    @staticmethod
    def _fallback(offers: Sequence[PharmacyOffer], medicine_count: int) -> str:
        """
        Deterministic summary for when the model is unavailable.

        Built from the same computed figures, so it is never less accurate than
        the generated version — only less fluent.
        """
        orderable = [o for o in offers if o.can_order]
        if not orderable:
            missing = sorted({name for o in offers for name in o.unavailable_items})
            detail = f" Out of stock nearby: {', '.join(missing)}." if missing else ""
            return (
                "Pharmacies were found nearby, but none can currently supply this "
                f"prescription.{detail}"
            )

        best = orderable[0]
        parts = [
            f"{best.name} is {best.distance_km} km away and can deliver in about "
            f"{best.eta_minutes} minutes for {best.grand_total:.2f}."
        ]
        if best.total_savings > 0:
            parts.append(f"That is {best.total_savings:.2f} below MRP.")
        if not best.fully_available:
            parts.append(f"It cannot supply: {', '.join(best.unavailable_items)}.")
        if len(orderable) > 1:
            parts.append(f"{len(orderable) - 1} other nearby option(s) are available.")
        return " ".join(parts)


pharmacy_agent = PharmacyAgent()
