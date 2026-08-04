"""
Prescription safety verification.

A bounded context that reviews an issued prescription and reports what it finds.
It is strictly advisory: nothing in here may write to `prescriptions` or
`medications`. The clinician's order is the record; this context annotates it.

Layout mirrors the existing `intake/` and `assistant/` contexts:

    domain/          value objects and the provider ports
    infrastructure/  RxNorm, openFDA and Groq adapters
    application/     the verification service that composes them
"""
