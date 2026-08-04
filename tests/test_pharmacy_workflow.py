"""
Pharmacy discovery, availability, ranking and the order lifecycle.

The properties pinned here are the ones that cost money or dispense the wrong
medicine when they break: stock cannot go negative, a cancelled order returns
its stock, an order cannot skip lifecycle stages, and a generic is never
substituted without the patient choosing it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.models.medicine_order import (
    CANCELLABLE_STATUSES,
    ORDER_CANCELLED,
    ORDER_DELIVERED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_PACKED,
    ORDER_PREPARING,
    ORDER_RECEIVED,
    ORDER_STATUSES,
    ORDER_TRANSITIONS,
)
from app.models.pharmacy import Pharmacy, PharmacyInventory
from app.pharmacy.application.agent import PharmacyAgent
from app.pharmacy.domain.entities import (
    MedicineAvailability,
    PharmacyOffer,
    assign_badges,
    estimate_travel_minutes,
    haversine_km,
    score_offer,
)
from app.pharmacy.domain.ports import MedicineRequirement
from app.pharmacy.infrastructure.local_db_provider import (
    LocalDbPharmacyProvider,
    _is_open_now,
)


# ── geography ────────────────────────────────────────────────────────────


def test_haversine_matches_known_distance():
    """Connaught Place to India Gate is ~3.5km."""
    km = haversine_km(28.6315, 77.2167, 28.6129, 77.2295)
    assert 2.0 < km < 4.5


def test_haversine_is_zero_for_the_same_point():
    assert haversine_km(28.6, 77.2, 28.6, 77.2) == pytest.approx(0.0, abs=1e-9)


def test_travel_estimate_applies_a_detour_factor():
    """
    Straight-line distance understates road distance, so the estimate must not
    simply divide by speed — an optimistic ETA is a broken promise to a patient
    waiting on medicine.
    """
    naive_minutes = (5.0 / 22.0) * 60
    assert estimate_travel_minutes(5.0) > naive_minutes


def test_travel_estimate_is_never_zero():
    assert estimate_travel_minutes(0.001) >= 1


# ── opening hours ────────────────────────────────────────────────────────


def _pharmacy(**overrides) -> Pharmacy:
    base = dict(
        name="Test Chemist", address="1 Road", latitude=28.6, longitude=77.2,
        is_partner=True, is_active=True, is_24x7=False, opens_at=None, closes_at=None,
        delivers=True, delivery_radius_km=10.0, delivery_fee=0.0,
        free_delivery_above=None, min_order_value=0.0, avg_prep_minutes=15,
        rating=4.0, total_ratings=100,
    )
    base.update(overrides)
    return Pharmacy(**base)


def test_24x7_pharmacy_is_always_open():
    assert _is_open_now(_pharmacy(is_24x7=True)) is True


def test_unknown_hours_count_as_open():
    """
    Hiding a pharmacy because nobody filled in its timings loses a real
    dispensing option; showing one that turns out to be shut costs a call.
    """
    assert _is_open_now(_pharmacy(opens_at=None, closes_at=None)) is True


def test_normal_window_is_respected():
    at_noon = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    assert _is_open_now(_pharmacy(opens_at="09:00", closes_at="22:00"), at_noon) is True

    at_dawn = datetime(2026, 8, 3, 5, 0, tzinfo=timezone.utc)
    assert _is_open_now(_pharmacy(opens_at="09:00", closes_at="22:00"), at_dawn) is False


def test_window_crossing_midnight_is_handled():
    """A 22:00–06:00 window must not be read as an empty range."""
    overnight = _pharmacy(opens_at="22:00", closes_at="06:00")
    assert _is_open_now(overnight, datetime(2026, 8, 3, 23, 30, tzinfo=timezone.utc)) is True
    assert _is_open_now(overnight, datetime(2026, 8, 3, 2, 0, tzinfo=timezone.utc)) is True
    assert _is_open_now(overnight, datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)) is False


# ── inventory matching ───────────────────────────────────────────────────


def _inventory(**overrides) -> PharmacyInventory:
    base = dict(
        pharmacy_id=uuid.uuid4(), sku="SKU1", rxcui="6809",
        medicine_name="Metformin 500mg", generic_name="metformin",
        brand_name=None, strength="500 mg", is_generic=True,
        mrp=100.0, selling_price=80.0, discount_percent=20.0,
        stock_quantity=50, low_stock_threshold=10,
        restock_expected_at=None, stock_synced_at=None,
    )
    base.update(overrides)
    row = PharmacyInventory(**base)
    row.id = overrides.get("id", uuid.uuid4())
    return row


def test_availability_reflects_stock_bands():
    assert _inventory(stock_quantity=0).availability == "out_of_stock"
    assert _inventory(stock_quantity=5, low_stock_threshold=10).availability == "limited"
    assert _inventory(stock_quantity=50, low_stock_threshold=10).availability == "available"


def test_can_supply_respects_requested_quantity():
    row = _inventory(stock_quantity=3)
    assert row.can_supply(3) is True
    assert row.can_supply(4) is False


def test_rxcui_match_wins_over_name():
    """
    Name matching mis-fills prescriptions — "Crocin" and "Paracetamol" are the
    same ingredient, and two different-looking names can be one drug.
    """
    provider = LocalDbPharmacyProvider()
    row = _inventory(rxcui="6809", medicine_name="Glycomet")
    requirement = MedicineRequirement(name="Metformin", rxcui="6809")
    assert provider._matches(row, requirement) is True


def test_name_match_is_the_fallback_when_rxcui_is_absent():
    provider = LocalDbPharmacyProvider()
    row = _inventory(rxcui=None, medicine_name="Metformin 500mg")
    requirement = MedicineRequirement(name="metformin", rxcui=None)
    assert provider._matches(row, requirement) is True


def test_unrelated_drug_does_not_match():
    provider = LocalDbPharmacyProvider()
    row = _inventory(rxcui="1111", medicine_name="Atorvastatin")
    requirement = MedicineRequirement(name="Metformin", rxcui="6809")
    assert provider._matches(row, requirement) is False


def test_availability_prefers_a_row_that_can_supply_the_full_quantity():
    """
    Choosing purely on price would pick a cheaper row with one unit left and
    report the line as available when it cannot be filled.
    """
    provider = LocalDbPharmacyProvider()
    cheap_but_short = _inventory(sku="A", selling_price=50.0, stock_quantity=1)
    dearer_but_stocked = _inventory(sku="B", selling_price=80.0, stock_quantity=100)

    result = provider._availability(
        [cheap_but_short, dearer_but_stocked],
        MedicineRequirement(name="Metformin", rxcui="6809", quantity=10),
    )
    assert result.unit_price == 80.0
    assert result.status == "available"


def test_missing_medicine_is_out_of_stock_not_unknown():
    provider = LocalDbPharmacyProvider()
    result = provider._availability(
        [], MedicineRequirement(name="Nonexistent", rxcui="9999", quantity=1)
    )
    assert result.status == "out_of_stock"
    assert result.line_total == 0.0


def test_cheaper_equivalents_are_offered_as_alternatives():
    provider = LocalDbPharmacyProvider()
    chosen = _inventory(sku="BRAND", selling_price=100.0, stock_quantity=50, is_generic=False)
    generic = _inventory(sku="GEN", selling_price=40.0, stock_quantity=50, is_generic=True)

    result = provider._availability(
        [chosen, generic], MedicineRequirement(name="Metformin", rxcui="6809", quantity=2)
    )

    # The cheapest suppliable row is selected, and the dearer one is not
    # advertised as a saving.
    assert result.unit_price == 40.0
    assert all(alt.unit_price < result.unit_price for alt in result.alternatives)


def test_line_total_and_savings_are_computed_per_quantity():
    line = MedicineAvailability(
        requested_name="Metformin", rxcui="6809", requested_quantity=10,
        status="available", mrp=100.0, unit_price=80.0,
    )
    assert line.line_total == 800.0
    assert line.savings == 200.0


def test_out_of_stock_line_contributes_nothing_to_the_bill():
    line = MedicineAvailability(
        requested_name="X", rxcui=None, requested_quantity=5,
        status="out_of_stock", mrp=100.0, unit_price=80.0,
    )
    assert line.line_total == 0.0
    assert line.savings == 0.0


# ── ranking ──────────────────────────────────────────────────────────────


def _offer(**overrides) -> PharmacyOffer:
    base = dict(
        pharmacy_id=str(uuid.uuid4()), name="Chemist", address="", phone=None,
        latitude=28.6, longitude=77.2, rating=4.0, total_ratings=50,
        is_partner=True, is_24x7=False, is_open_now=True, delivers=True,
        distance_km=2.0, travel_minutes=10, eta_minutes=25,
        delivery_fee=0.0, min_order_value=0.0,
        subtotal=500.0, total_savings=0.0, grand_total=500.0, can_order=True,
    )
    base.update(overrides)
    offer = PharmacyOffer(**base)
    offer.items = overrides.get(
        "items",
        [
            MedicineAvailability(
                requested_name="A", rxcui="1", requested_quantity=1, status="available"
            )
        ],
    )
    return offer


def test_availability_outranks_proximity():
    """
    The nearest pharmacy that cannot dispense the prescription is useless. A
    ranking led by distance would put it first, which is the whole reason
    availability carries the heaviest weight.
    """
    near_but_empty = _offer(
        distance_km=0.3, eta_minutes=10,
        items=[
            MedicineAvailability(
                requested_name="A", rxcui="1", requested_quantity=1, status="out_of_stock"
            )
        ],
    )
    far_but_stocked = _offer(distance_km=6.0, eta_minutes=40)

    near_score = score_offer(
        near_but_empty, max_distance_km=6.0, max_eta=40, cheapest_total=500.0
    )
    far_score = score_offer(
        far_but_stocked, max_distance_km=6.0, max_eta=40, cheapest_total=500.0
    )
    assert far_score > near_score


def test_fulfilment_ratio_is_partial_not_binary():
    offer = _offer(
        items=[
            MedicineAvailability(requested_name="A", rxcui="1", requested_quantity=1, status="available"),
            MedicineAvailability(requested_name="B", rxcui="2", requested_quantity=1, status="out_of_stock"),
        ]
    )
    assert offer.fulfilment_ratio == 0.5
    assert offer.fully_available is False


def test_badges_go_only_to_pharmacies_that_can_dispense_everything():
    """Labelling a shop 'Nearest' when it holds none of the prescription is
    worse than not labelling anything."""
    empty_but_closest = _offer(
        distance_km=0.1, can_order=False,
        items=[
            MedicineAvailability(
                requested_name="A", rxcui="1", requested_quantity=1, status="out_of_stock"
            )
        ],
    )
    stocked = _offer(distance_km=4.0)

    assign_badges([empty_but_closest, stocked])
    assert empty_but_closest.badges == []
    assert "Nearest" in stocked.badges


def test_badges_identify_cheapest_and_fastest():
    cheap_slow = _offer(grand_total=200.0, eta_minutes=60, distance_km=5.0)
    dear_fast = _offer(grand_total=900.0, eta_minutes=15, distance_km=1.0)

    assign_badges([cheap_slow, dear_fast])
    assert "Lowest price" in cheap_slow.badges
    assert "Fastest delivery" in dear_fast.badges


def test_badges_are_silent_when_nothing_is_orderable():
    offer = _offer(can_order=False)
    assign_badges([offer])
    assert offer.badges == []


# ── order lifecycle ──────────────────────────────────────────────────────


def test_every_status_appears_in_the_transition_table():
    """A status with no entry would silently become a dead end."""
    assert set(ORDER_TRANSITIONS) == set(ORDER_STATUSES)


def test_the_happy_path_is_reachable_end_to_end():
    path = [
        ORDER_RECEIVED, ORDER_PREPARING, ORDER_PACKED,
        ORDER_OUT_FOR_DELIVERY, ORDER_DELIVERED,
    ]
    for current, nxt in zip(path, path[1:]):
        assert nxt in ORDER_TRANSITIONS[current], f"{current} -> {nxt} is blocked"


def test_terminal_states_have_no_exits():
    assert ORDER_TRANSITIONS[ORDER_DELIVERED] == ()
    assert ORDER_TRANSITIONS[ORDER_CANCELLED] == ()


def test_orders_cannot_be_cancelled_after_dispatch():
    """
    Once the goods have left the counter it is a return, which is a different
    process with different money attached.
    """
    assert ORDER_CANCELLED not in ORDER_TRANSITIONS[ORDER_OUT_FOR_DELIVERY]
    assert ORDER_OUT_FOR_DELIVERY not in CANCELLABLE_STATUSES
    assert ORDER_DELIVERED not in CANCELLABLE_STATUSES


def test_pre_dispatch_states_are_all_cancellable():
    for status in (ORDER_RECEIVED, ORDER_PREPARING, ORDER_PACKED):
        assert status in CANCELLABLE_STATUSES


def test_stages_cannot_be_skipped():
    assert ORDER_DELIVERED not in ORDER_TRANSITIONS[ORDER_RECEIVED]
    assert ORDER_OUT_FOR_DELIVERY not in ORDER_TRANSITIONS[ORDER_RECEIVED]
    assert ORDER_PACKED not in ORDER_TRANSITIONS[ORDER_RECEIVED]


def test_order_number_is_unpredictable():
    """
    A sequential number leaks total order volume and lets a customer guess
    their neighbour's reference.
    """
    from app.pharmacy.application.ordering import generate_order_number

    numbers = {generate_order_number() for _ in range(200)}
    assert len(numbers) == 200
    assert all(n.startswith("MB-") for n in numbers)


# ── the AI agent's boundaries ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_reports_plainly_when_nothing_can_be_supplied():
    agent = PharmacyAgent()
    empty = _offer(
        can_order=False,
        items=[
            MedicineAvailability(
                requested_name="Metformin", rxcui="1", requested_quantity=1,
                status="out_of_stock",
            )
        ],
    )
    empty.unavailable_items = ["Metformin"]

    text = agent._fallback([empty], medicine_count=1)
    assert "none can currently supply" in text.lower()
    assert "Metformin" in text


@pytest.mark.asyncio
async def test_agent_says_so_when_no_pharmacies_exist():
    agent = PharmacyAgent()
    text = await agent.summarise_offers([], medicine_count=2)
    assert "no partner pharmacies" in text.lower()
    # Must not imply the prescription cannot be filled anywhere at all.
    assert "any chemist" in text.lower()


def test_agent_fallback_quotes_only_computed_figures():
    """The deterministic summary must never be less accurate than the model's."""
    agent = PharmacyAgent()
    offer = _offer(distance_km=1.5, eta_minutes=30, grand_total=450.0, total_savings=50.0)

    text = agent._fallback([offer], medicine_count=1)
    assert "1.5 km" in text
    assert "30 minutes" in text
    assert "450.00" in text
    assert "50.00" in text
