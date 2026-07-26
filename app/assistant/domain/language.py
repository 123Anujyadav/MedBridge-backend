"""
Language detection for patient messages.

Script inspection rather than a model call: it is deterministic, free, and
instant, and the assistant must keep answering in the right language even when
the LLM is unavailable.

The heuristic overlaps slightly with the intake agent's detector, which is
deliberate — that one is a graph node with an LLM tie-break bound to intake's
state, and coupling the two bounded contexts through a private import would be
worse than ~20 lines of shared regex.
"""

from __future__ import annotations

import re

from app.intake.domain.enums import Language

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Romanised-Hindi function words: frequent in Hinglish, rare in clinical English.
_HINGLISH_MARKERS = frozenset(
    {
        "hai", "hain", "haan", "nahi", "nahin", "mujhe", "mera", "meri", "mere",
        "kya", "kyu", "kyun", "aur", "bahut", "thoda", "zyada", "dard", "din",
        "raha", "rahi", "rha", "rhi", "ho", "hota", "hoti", "se", "ka", "ki",
        "ke", "me", "mein", "par", "kar", "karo", "kuch", "abhi", "kal", "aaj",
        "saans", "pet", "sir", "bukhar", "khansi", "chakkar", "kamzori", "ulti",
        "sardi", "jukam", "gala", "pair", "haath", "aankh", "kamar",
    }
)

_WORD_RE = re.compile(r"[a-z]+")


def detect_language(text: str) -> Language:
    """
    Classify a message as English, Hindi, Hinglish, or Mixed.

    Both scripts present -> MIXED. Devanagari only -> HINDI. Latin script with
    two or more Hinglish marker words -> HINGLISH. Otherwise ENGLISH.
    """
    if not text or not text.strip():
        return Language.UNKNOWN

    has_devanagari = bool(_DEVANAGARI_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))

    if has_devanagari and has_latin:
        return Language.MIXED
    if has_devanagari:
        return Language.HINDI

    words = set(_WORD_RE.findall(text.casefold()))
    if len(words & _HINGLISH_MARKERS) >= 2:
        return Language.HINGLISH

    return Language.ENGLISH


def language_instruction(language: Language) -> str:
    """A short directive appended to prompts so the model replies in kind."""
    return {
        Language.ENGLISH: "Reply in English.",
        Language.HINDI: "Reply in Hindi using Devanagari script.",
        Language.HINGLISH: (
            "Reply in Hinglish — romanised Hindi in Latin script. "
            "Do not use Devanagari and do not translate to pure English."
        ),
        Language.MIXED: (
            "The patient mixes Hindi and English. Reply in the same mixed style."
        ),
        Language.UNKNOWN: "Reply in English.",
    }.get(language, "Reply in English.")
