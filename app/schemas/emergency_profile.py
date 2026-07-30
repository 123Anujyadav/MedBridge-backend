"""
Request and response shapes for the patient Emergency Profile.

All of the module's input validation lives here, so there is one place to read
in order to know what the database can be made to hold. Three rules run through
it:

* **Normalise before judging.** Every free-text field is trimmed and its
  internal whitespace collapsed *first*, so `"  Ravi   Kumar "` is accepted as
  `"Ravi Kumar"` rather than rejected for a length that came from spaces.
* **Reject the characters that have no business in a postal address.** Control
  characters, angle brackets and the rest are refused rather than escaped
  downstream, because this data is rendered into a page and, later, read aloud
  or handed to a dispatcher.
* **Say what is wrong.** Messages name the field and the rule, since the person
  reading them is a patient filling in a form, not a developer.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── shared primitives ────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")

_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
"""
Control characters to delete outright.

Deliberately excludes `\\x09`–`\\x0d` — tab, newline, vertical tab, form feed
and carriage return. Those are whitespace, so they are left for `_WHITESPACE`
to collapse into a single space. Deleting them instead would fuse the words on
either side: a patient pasting a two-line address would have `Patna\\nBihar`
stored as `PatnaBihar`, which is not their city and not anywhere.
"""

TEXT_ALLOWED = re.compile(r"^[A-Za-z0-9 .,\-'/&()#]+$")
"""
What a name, street or locality may contain.

Letters, digits, spaces and the punctuation that genuinely occurs in Indian
postal addresses — `No. 12/A`, `D'Souza`, `Gandhi Rd (East)`, `#4`. Everything
else, including `<`, `>`, `{`, `}`, `|`, `\\` and every control character, is
refused.
"""

NAME_ALLOWED = re.compile(r"^[A-Za-z .\-']+$")
"""A person's name — no digits, and no address punctuation."""

PHONE_ALLOWED = re.compile(r"^\+?[0-9]{7,15}$")
"""
E.164-shaped: an optional leading `+` then 7 to 15 digits.

Separators are stripped before this is applied, so `+91 98765 43210` and
`(0)98765-43210` both arrive here as digit strings.
"""

INDIA_PIN = re.compile(r"^[1-9][0-9]{5}$")
GENERIC_POSTCODE = re.compile(r"^[A-Za-z0-9]{3,10}$")


def _clean(value: Optional[str]) -> str:
    """Trim, collapse internal whitespace, and strip control characters."""
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", _CONTROL.sub("", str(value))).strip()


def _clean_phone(value: Optional[str]) -> str:
    """
    Reduce a typed phone number to `+` and digits.

    People paste numbers with spaces, hyphens, brackets and dots. None of that
    changes which telephone rings, so it is removed before the number is
    validated or compared — otherwise `+919876543210` and `+91 98765 43210`
    would be stored as two different numbers and the duplicate check would miss.
    """
    if value is None:
        return ""
    stripped = re.sub(r"[\s\-().]", "", _CONTROL.sub("", str(value)))
    return stripped.strip()


def _require(value: str, field: str) -> str:
    if not value:
        raise ValueError(f"{field} is required.")
    return value


# ── emergency contact ────────────────────────────────────────────────────

class EmergencyContactSchema(BaseModel):
    """Who to call, and on what number."""

    model_config = ConfigDict(str_strip_whitespace=True)

    contact_name: str = Field(max_length=120)
    contact_phone: str = Field(max_length=20)
    contact_relationship: str = Field(max_length=60)
    alternate_phone: Optional[str] = Field(default=None, max_length=20)

    @field_validator("contact_name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = _require(_clean(v), "Emergency contact name")
        if len(v) < 2:
            raise ValueError("Emergency contact name must be at least 2 characters.")
        if len(v) > 120:
            raise ValueError("Emergency contact name must be at most 120 characters.")
        if not NAME_ALLOWED.match(v):
            raise ValueError(
                "Emergency contact name may only contain letters, spaces, "
                "apostrophes, hyphens and full stops."
            )
        return v

    @field_validator("contact_relationship")
    @classmethod
    def _validate_relationship(cls, v: str) -> str:
        v = _require(_clean(v), "Relationship")
        if len(v) > 60:
            raise ValueError("Relationship must be at most 60 characters.")
        if not NAME_ALLOWED.match(v):
            raise ValueError(
                "Relationship may only contain letters, spaces, apostrophes, "
                "hyphens and full stops."
            )
        return v.title()

    @field_validator("contact_phone")
    @classmethod
    def _validate_phone(cls, v: str) -> str:
        v = _require(_clean_phone(v), "Emergency contact number")
        if not PHONE_ALLOWED.match(v):
            raise ValueError(
                "Emergency contact number must be 7 to 15 digits, optionally "
                "prefixed with a country code such as +91."
            )
        return v

    @field_validator("alternate_phone")
    @classmethod
    def _validate_alternate(cls, v: Optional[str]) -> Optional[str]:
        cleaned = _clean_phone(v)
        if not cleaned:
            return None  # genuinely optional; blank means "not provided"
        if not PHONE_ALLOWED.match(cleaned):
            raise ValueError(
                "Alternative contact number must be 7 to 15 digits, optionally "
                "prefixed with a country code such as +91."
            )
        return cleaned

    @model_validator(mode="after")
    def _numbers_must_differ(self) -> "EmergencyContactSchema":
        """
        The alternative number has to be an *alternative*.

        Storing the same number twice looks like redundancy and provides none:
        whoever works down the list in an emergency dials the same unanswered
        phone a second time.
        """
        if self.alternate_phone and self.alternate_phone == self.contact_phone:
            raise ValueError(
                "Alternative contact number must be different from the primary "
                "emergency contact number."
            )
        return self


# ── registered address ───────────────────────────────────────────────────

class EmergencyAddressSchema(BaseModel):
    """Where the patient lives, broken into parts an ambulance can be given."""

    model_config = ConfigDict(str_strip_whitespace=True)

    house_number: str = Field(max_length=60)
    street: str = Field(max_length=150)
    landmark: Optional[str] = Field(default=None, max_length=150)
    locality: str = Field(max_length=120)
    city: str = Field(max_length=100)
    district: str = Field(max_length=100)
    state: str = Field(max_length=100)
    country: str = Field(default="India", max_length=100)
    pincode: str = Field(max_length=12)

    @field_validator("house_number", "street", "locality", "city",
                     "district", "state", "country")
    @classmethod
    def _validate_required_text(cls, v: str, info) -> str:
        label = info.field_name.replace("_", " ").title()
        v = _require(_clean(v), label)
        if not TEXT_ALLOWED.match(v):
            raise ValueError(
                f"{label} contains characters that are not allowed. Use letters, "
                "digits, spaces and . , - ' / & ( ) # only."
            )
        return v

    @field_validator("landmark")
    @classmethod
    def _validate_landmark(cls, v: Optional[str]) -> Optional[str]:
        cleaned = _clean(v)
        if not cleaned:
            return None  # the one optional part of an address
        if not TEXT_ALLOWED.match(cleaned):
            raise ValueError(
                "Landmark contains characters that are not allowed. Use letters, "
                "digits, spaces and . , - ' / & ( ) # only."
            )
        return cleaned

    @field_validator("pincode")
    @classmethod
    def _validate_pincode(cls, v: str) -> str:
        cleaned = re.sub(r"[\s\-]", "", _clean(v)).upper()
        _require(cleaned, "Pincode")
        if not GENERIC_POSTCODE.match(cleaned):
            raise ValueError(
                "Pincode must be 3 to 10 letters or digits."
            )
        return cleaned

    @model_validator(mode="after")
    def _validate_pincode_for_country(self) -> "EmergencyAddressSchema":
        """
        Apply the country's own rule once the country is known.

        A field validator cannot do this — it sees `pincode` without `country`.
        India is checked strictly because it is the deployment's home country
        and a six-digit PIN is unambiguous; anywhere else keeps the general
        rule rather than inventing one per country and rejecting valid input.
        """
        if self.country.strip().lower() in ("india", "in", "bharat"):
            if not INDIA_PIN.match(self.pincode):
                raise ValueError(
                    "An Indian pincode must be exactly 6 digits and cannot "
                    "start with 0."
                )
        return self


# ── coordinates ──────────────────────────────────────────────────────────

class EmergencyLocationUpdate(BaseModel):
    """
    A position captured from the browser.

    Only the two numbers are accepted. The Maps link is derived from them on the
    server — a client-supplied URL is a client-controlled destination, and this
    one gets rendered as a link for somebody to follow in an emergency.
    """

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def _reject_null_island(self) -> "EmergencyLocationUpdate":
        """
        Exactly (0, 0) is in the Gulf of Guinea.

        It is what a failed sensor and an uninitialised variable both produce,
        and it is not where the patient is. Refusing it keeps a plausible-looking
        but useless fix out of the record.
        """
        if self.latitude == 0 and self.longitude == 0:
            raise ValueError(
                "Coordinates (0, 0) are not a valid location. Please retry "
                "capturing your position."
            )
        return self


# ── the profile itself ───────────────────────────────────────────────────

class EmergencyProfileUpsert(BaseModel):
    """
    The full profile as the form submits it.

    Contact and address together, because they are saved together: a profile
    with a contact and no address is not usable by the system this feeds.
    Coordinates are captured separately, by their own endpoint, since they come
    from a browser permission prompt rather than from typing.
    """

    contact: EmergencyContactSchema
    address: EmergencyAddressSchema


class EmergencyProfileResponse(BaseModel):
    """The stored profile, flattened for the client."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    contact_name: str
    contact_phone: str
    contact_relationship: str
    alternate_phone: Optional[str] = None

    house_number: str
    street: str
    landmark: Optional[str] = None
    locality: str
    city: str
    district: str
    state: str
    country: str
    pincode: str

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    maps_url: Optional[str] = None
    location_updated_at: Optional[str] = None

    formatted_address: str
    """The address as one line, assembled once on the server so every screen agrees."""

    @field_validator("id", mode="before")
    @classmethod
    def _stringify_id(cls, v) -> str:
        return str(v)

    @field_validator("location_updated_at", mode="before")
    @classmethod
    def _isoformat(cls, v):
        return v.isoformat() if hasattr(v, "isoformat") else v
