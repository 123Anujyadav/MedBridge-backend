"""
Clinical safety policies for the Medical Case Intake Agent.

This module is where the accuracy requirements are actually enforced, in three
independent mechanisms:

1. `is_evidence_grounded` — an extracted value survives only if the text it
   claims to come from is genuinely present in the patient's own words. This is
   the structural defence against fabricated symptoms, allergies and
   medications; it does not rely on the model choosing to behave.

2. `detect_red_flags` — deterministic, multilingual pattern matching for
   presentations that must bypass conversational follow-up and escalate
   immediately. Runs independently of the LLM so an unavailable or misbehaving
   model cannot suppress an emergency.

3. `evaluate_readiness` — a structured case is only generated once mandatory
   fields clear a confidence floor. Below it, the agent asks rather than guesses.

Pure functions, no I/O, fully unit-testable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.intake.domain.entities import ExtractedEntity, IntakeSession
from app.intake.domain.enums import EntityKind, UrgencyLevel
from app.intake.domain.value_objects import Confidence

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

MIN_ENTITY_CONFIDENCE = 0.55
"""An individual extraction below this is treated as too weak to rely on."""

MIN_OVERALL_CONFIDENCE = 0.65
"""Aggregate confidence required across mandatory fields before case generation."""

MAX_FOLLOWUP_ROUNDS = 3
"""
Hard ceiling on clarification rounds.

Without this the agent can interrogate a patient indefinitely when a field is
simply unknowable. On hitting the ceiling we generate the case with the gaps
explicitly marked `Unknown` rather than continuing to ask.
"""

MANDATORY_ENTITY_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SYMPTOM,
    EntityKind.DURATION,
    EntityKind.SEVERITY,
)
"""Minimum set a clinician needs before a case is worth routing."""

EVIDENCE_TOKEN_OVERLAP_FLOOR = 0.70
"""
Fraction of an evidence quote's tokens that must appear in the patient
transcript for the citation to count as grounded.

Not 1.0: models legitimately normalise casing, inflection and filler when
quoting. Well above 0.0: a fabricated quote shares few tokens with real text.
"""

_PARTIAL_GROUNDING_PENALTY = 0.75
"""Confidence multiplier for extractions that ground only partially."""


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\sऀ-ॿ]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalise(text: str) -> str:
    """
    Casefold, strip punctuation, collapse whitespace.

    Unicode-normalised to NFKC first so Devanagari and romanised input compare
    consistently regardless of how the client encoded it.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    stripped = _PUNCT_RE.sub(" ", folded)
    return _WS_RE.sub(" ", stripped).strip()


def _tokens(text: str) -> list[str]:
    normalised = normalise(text)
    return [t for t in normalised.split(" ") if t]


# --------------------------------------------------------------------------
# 1. Evidence grounding — anti-fabrication
# --------------------------------------------------------------------------


def is_evidence_grounded(quote: str, transcript: str) -> tuple[bool, float]:
    """
    Check whether `quote` is genuinely traceable to `transcript`.

    Returns `(grounded, overlap_ratio)`.

    Exact normalised containment scores 1.0. Otherwise we fall back to token
    overlap, which tolerates the model tidying a quote while still rejecting one
    it invented. An empty quote is never grounded — absence of evidence is not
    evidence.
    """
    if not quote or not quote.strip():
        return False, 0.0
    if not transcript or not transcript.strip():
        return False, 0.0

    normalised_quote = normalise(quote)
    normalised_transcript = normalise(transcript)
    if not normalised_quote:
        return False, 0.0

    # Fast path: the model quoted verbatim.
    if normalised_quote in normalised_transcript:
        return True, 1.0

    quote_tokens = _tokens(quote)
    if not quote_tokens:
        return False, 0.0

    transcript_tokens = set(_tokens(transcript))
    hits = sum(1 for token in quote_tokens if token in transcript_tokens)
    ratio = hits / len(quote_tokens)
    return ratio >= EVIDENCE_TOKEN_OVERLAP_FLOOR, round(ratio, 4)


def enforce_grounding(
    entities: list[ExtractedEntity], transcript: str
) -> tuple[list[ExtractedEntity], list[ExtractedEntity]]:
    """
    Split extractions into those provably grounded in the transcript and those
    that are not.

    Fully-grounded entities pass through untouched. Partially-grounded ones are
    kept with reduced confidence. Ungrounded ones are rejected outright and
    returned separately so the caller can log exactly what the model tried to
    invent.
    """
    kept: list[ExtractedEntity] = []
    rejected: list[ExtractedEntity] = []

    for entity in entities:
        # An explicit "not stated" carries no claim, so it needs no evidence.
        if entity.is_unknown:
            kept.append(entity)
            continue

        grounded, ratio = is_evidence_grounded(entity.evidence.quote, transcript)
        if not grounded:
            rejected.append(entity)
            continue

        if ratio < 1.0:
            kept.append(
                ExtractedEntity(
                    kind=entity.kind,
                    value=entity.value,
                    confidence=entity.confidence.penalised(_PARTIAL_GROUNDING_PENALTY),
                    evidence=entity.evidence,
                )
            )
        else:
            kept.append(entity)

    return kept, rejected


# --------------------------------------------------------------------------
# 2. Red flags — deterministic emergency detection
# --------------------------------------------------------------------------

# Each entry: (clinical label, regex over normalised text).
# Patterns cover English, romanised Hinglish and Devanagari Hindi, because the
# agent must escalate correctly regardless of which the patient reaches for.
_RED_FLAG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Possible acute coronary syndrome (chest pain)",
        re.compile(
            r"\b(chest pain|chest pressure|crushing chest"
            r"|chest (is )?(hurting|hurts|tight|tightening|heavy|burning)"
            r"|tightness in (my |the )?chest|pressure (in|on) (my |the )?chest"
            r"|pain in (my |the )?chest"
            r"|seene? me[ni]? dard|chaati me[ni]? dard|dil me[ni]? dard)\b"
            r"|सीने\s*में\s*दर्द|छाती\s*में\s*दर्द",
            re.UNICODE,
        ),
    ),
    (
        "Respiratory distress",
        re.compile(
            r"\b((can'?t|cannot|unable to|not able to|hard to|struggling to) breathe"
            r"|(difficulty|trouble|problem) breathing|breathing (difficulty|problem)"
            r"|shortness of breath|short of breath|gasping|choking|suffocating"
            r"|saans nahi|saans nah[ie]n|dam ghut"
            r"|s[aā]ns lene me[ni]? (dikkat|takleef|problem))\b"
            r"|सांस\s*नहीं|साँस\s*नहीं|दम\s*घुट",
            re.UNICODE,
        ),
    ),
    (
        "Altered consciousness",
        re.compile(
            r"\b(unconscious|passed out|fainted|unresponsive|not waking up"
            r"|behosh|behoshi)\b"
            r"|बेहोश",
            re.UNICODE,
        ),
    ),
    (
        "Possible stroke",
        re.compile(
            r"\b(stroke|face drooping|slurred speech|sudden numbness"
            r"|can'?t move (my )?(arm|leg|side)|paralysis|lakwa|falij)\b"
            r"|लकवा|पक्षाघात",
            re.UNICODE,
        ),
    ),
    (
        "Severe haemorrhage",
        re.compile(
            r"\b(severe bleeding|heavy bleeding|bleeding a lot|won'?t stop bleeding"
            r"|bahut khoon|khoon beh raha|khoon ruk nahi)\b"
            r"|बहुत\s*खून|खून\s*बह",
            re.UNICODE,
        ),
    ),
    (
        "Possible anaphylaxis",
        re.compile(
            r"\b(anaphylaxis|anaphylactic"
            r"|(throat|tongue|lips?|face) (is |are )?(closing|swelling|swollen|tightening)"
            r"|swelling of (my |the )?(throat|tongue|face)"
            r"|gala band ho raha|gala sujj)\b"
            r"|गला\s*बंद|गला\s*सूज",
            re.UNICODE,
        ),
    ),
    (
        "Suicidal ideation or self-harm",
        re.compile(
            r"\b(suicidal|suicide|kill myself|end my life|harm myself"
            r"|khudkushi|atmahatya|jaan dena chahta|marna chahta)\b"
            r"|आत्महत्या|खुदकुशी",
            re.UNICODE,
        ),
    ),
    (
        "Poisoning or overdose",
        re.compile(
            r"\b(overdose|poisoning|swallowed poison|took too many pills"
            r"|zeher|zahar khaya)\b"
            r"|ज़हर|जहर",
            re.UNICODE,
        ),
    ),
    (
        "Seizure activity",
        re.compile(
            r"\b(seizure|convulsion|fitting|having a fit|mirgi|daura pad)\b"
            r"|मिर्गी|दौरा",
            re.UNICODE,
        ),
    ),
)

# Negations that, when they immediately precede a match, indicate the patient is
# *denying* the finding. Deliberately prefix-only: Hindi/Hinglish negation
# frequently trails the noun ("saans nahi aa raha") and is itself the positive
# emergency signal, so trailing negation must never suppress a match.
_NEGATION_PREFIXES = (
    "no",
    "not",
    "never",
    "without",
    "denies",
    "deny",
    "denied",
    "any",
)
_NEGATION_LOOKBACK_CHARS = 14


def _is_negated(normalised_text: str, match_start: int) -> bool:
    """True when a negation word sits immediately before the matched span."""
    window_start = max(0, match_start - _NEGATION_LOOKBACK_CHARS)
    window = normalised_text[window_start:match_start].strip()
    if not window:
        return False
    preceding = window.split(" ")
    if not preceding:
        return False
    return preceding[-1] in _NEGATION_PREFIXES


def detect_red_flags(text: str) -> list[str]:
    """
    Scan free text for emergency presentations.

    Deterministic and LLM-independent: this must keep working when the model is
    rate-limited, unreachable or wrong. Returns de-duplicated clinical labels,
    empty when nothing is found.
    """
    if not text or not text.strip():
        return []

    normalised_text = normalise(text)
    found: list[str] = []

    for label, pattern in _RED_FLAG_PATTERNS:
        for match in pattern.finditer(normalised_text):
            if _is_negated(normalised_text, match.start()):
                continue
            if label not in found:
                found.append(label)
            break

    return found


def urgency_for_red_flags(red_flags: list[str]) -> UrgencyLevel:
    """Any confirmed red flag makes the case critical."""
    return UrgencyLevel.CRITICAL if red_flags else UrgencyLevel.MEDIUM


# --------------------------------------------------------------------------
# 3. Readiness — ask rather than guess
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReadinessVerdict:
    """Outcome of checking whether enough is known to generate a case."""

    is_ready: bool
    overall_confidence: Confidence
    missing_kinds: tuple[EntityKind, ...] = ()
    weak_kinds: tuple[EntityKind, ...] = ()
    forced: bool = False
    reason: str = ""

    @property
    def missing_labels(self) -> list[str]:
        return [k.value for k in self.missing_kinds]


def _best_confidence(session: IntakeSession, kind: EntityKind) -> Confidence:
    """Highest confidence among usable extractions of one kind."""
    candidates = [
        e.confidence.score
        for e in session.entities_of(kind)
        if e.confidence.meets(MIN_ENTITY_CONFIDENCE)
    ]
    return Confidence(max(candidates)) if candidates else Confidence.unknown()


def evaluate_readiness(session: IntakeSession) -> ReadinessVerdict:
    """
    Decide whether the agent may generate a structured case or must ask a
    follow-up question first.

    Ready requires every mandatory field to clear `MIN_ENTITY_CONFIDENCE` and
    the aggregate to clear `MIN_OVERALL_CONFIDENCE`. Once `MAX_FOLLOWUP_ROUNDS`
    is exhausted we proceed regardless, with the shortfall recorded so unknown
    fields land in the case as `Unknown` instead of being invented.
    """
    missing: list[EntityKind] = []
    weak: list[EntityKind] = []
    scores: list[float] = []

    for kind in MANDATORY_ENTITY_KINDS:
        confidence = _best_confidence(session, kind)
        scores.append(confidence.score)
        if confidence.is_unknown:
            missing.append(kind)
        elif not confidence.meets(MIN_ENTITY_CONFIDENCE):
            weak.append(kind)

    overall = Confidence(sum(scores) / len(scores)) if scores else Confidence.unknown()
    exhausted = session.followup_rounds >= MAX_FOLLOWUP_ROUNDS

    if not missing and not weak and overall.meets(MIN_OVERALL_CONFIDENCE):
        return ReadinessVerdict(
            is_ready=True,
            overall_confidence=overall,
            reason="All mandatory fields met the confidence threshold.",
        )

    if exhausted:
        return ReadinessVerdict(
            is_ready=True,
            overall_confidence=overall,
            missing_kinds=tuple(missing),
            weak_kinds=tuple(weak),
            forced=True,
            reason=(
                f"Follow-up limit ({MAX_FOLLOWUP_ROUNDS}) reached; generating case "
                f"with unresolved fields explicitly marked as Unknown."
            ),
        )

    return ReadinessVerdict(
        is_ready=False,
        overall_confidence=overall,
        missing_kinds=tuple(missing),
        weak_kinds=tuple(weak),
        reason="Insufficient confidence in mandatory fields; clarification required.",
    )
