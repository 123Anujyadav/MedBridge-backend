# AI Medical Case Intake Agent

A stateful, multi-turn clinical intake agent that turns a patient's free-text
symptom description — in English, Hindi, Hinglish or a mix — into a structured
medical case, then routes it to a real specialist.

Built as a new bounded context under `app/intake/`. The pre-existing
single-shot `POST /api/v1/ai/symptom-intake` endpoint is **untouched and still
mounted**, so the current frontend keeps working.

---

## 1. Architecture

Clean Architecture, dependencies pointing inward:

```
presentation   app/api/v1/endpoints/intake.py     thin async controllers
                       │                          (no DB, no AI, no rules)
                       ▼
application    app/intake/application/            use cases + ports (Protocols)
                       │
                       ▼
domain         app/intake/domain/                 entities, policies, safety rules
                       ▲                          pure Python, zero framework imports
                       │
infrastructure app/intake/infrastructure/         Groq, Redis, SQLAlchemy adapters

AI orchestration  app/intake/workflow/            LangGraph — the ONLY package
                                                  that knows an LLM exists
```

Two rules the layout enforces:

* Controllers never touch the database or the model — they call one use case.
* The domain layer has no framework imports, so every safety rule is unit-testable
  with no I/O.

### File map

| Path | Responsibility |
| --- | --- |
| `domain/entities.py` | `IntakeSession` (aggregate root), `MedicalCase`, `ExtractedEntity` |
| `domain/policies.py` | **Safety core** — evidence grounding, red flags, confidence gate |
| `domain/specialties.py` | Closed specialty vocabulary + canonicalisation |
| `domain/value_objects.py` | `Confidence`, `Evidence` (frozen) |
| `application/ports.py` | `LLMPort`, `SessionStorePort`, `CaseRepositoryPort`, `DoctorDirectoryPort`, `IntakeAuditPort`, `IntakeWorkflowPort` |
| `application/use_cases.py` | `StartIntake`, `SubmitAnswer`, `GetSession`, `SelectDoctor` |
| `workflow/graph.py` | LangGraph `StateGraph` assembly + routing functions |
| `workflow/nodes.py` | One method per workflow stage |
| `workflow/prompts.py` | Prompt templates with anti-fabrication constraints |
| `infrastructure/llm_groq.py` | Async Groq client, JSON mode, tier fallback |
| `infrastructure/session_store.py` | Redis session persistence (with in-memory fallback) |
| `infrastructure/repositories.py` | Writes `cases`/`symptoms` + audit tables |
| `infrastructure/doctor_directory.py` | Real specialty-aware doctor lookup |
| `dependencies.py` | Composition root (all DI wiring) |

---

## 2. The workflow

```
START
  ├─ receive_input          sanitise + deterministic red-flag scan
  ├─ detect_language        script heuristics first, model only if ambiguous
  ├─ detect_intent
  │
  ├─[emergency?]──yes──▶ escalate_emergency ──▶ END
  │
  ├─ extract_entities       + evidence grounding (fabrication filter)
  ├─ evaluate_confidence
  │
  ├─[ready?]──no───▶ generate_followup ──────▶ END   (waits for patient)
  │
  └─[ready?]──yes──▶ generate_case ──▶ recommend_specialist ──▶ END
```

Both branch points are pure functions over domain state, so routing is testable
without running the graph or calling a model.

The compiled graph is cached process-wide (`build_intake_graph` is
`lru_cache`d). The request-scoped doctor directory travels in `IntakeState`
rather than being captured at construction — this took per-request workflow
setup from **15.5 ms to 0.0005 ms**.

---

## 3. How medical accuracy is enforced

Three independent mechanisms. None of them relies on the model choosing to
behave.

### 3.1 Evidence grounding — the anti-fabrication control

Every extracted entity must cite a verbatim span of the patient's own words.
`policies.enforce_grounding` then checks that span actually appears in the
transcript:

* exact normalised containment → accepted at full confidence
* ≥70 % token overlap → accepted with **reduced** confidence
* below that → **rejected and discarded**

A model that invents a symptom must also invent a quote, and an invented quote
will not be found. Verified in tests: given a transcript mentioning only
ibuprofen, a fabricated `"penicillin"` allergy with `confidence: 0.99` is
dropped while the real ibuprofen allergy passes through.

Rejections are counted in the API response (`rejected_extraction_count`), logged,
and persisted with `was_accepted=False` so the fabrication rate is **measurable
rather than invisible**.

Agent questions are excluded from the evidence transcript, so the model cannot
launder its own earlier wording into a fake patient statement.

### 3.2 Red flags — deterministic emergency detection

Regex-based, LLM-independent, and multilingual (English + Devanagari Hindi +
romanised Hinglish). Covers acute coronary syndrome, respiratory distress,
altered consciousness, stroke, haemorrhage, anaphylaxis, suicidal ideation,
poisoning and seizures.

* Runs **before** any model call, so a rate-limited or unreachable model cannot
  suppress an escalation. Tested explicitly against a dead LLM.
* On a hit the workflow **short-circuits** — a patient describing crushing chest
  pain is never asked to rate their symptoms before being told to seek care.
* Leading negation is filtered ("no chest pain" → no flag), but **trailing**
  negation is not, because Hindi/Hinglish negation follows the noun and is itself
  the positive signal (`saans nahi aa raha` = "I cannot breathe").
* The model **cannot downgrade** a red-flag urgency; urgency is the max of the
  model's read and the deterministic assessment.

### 3.3 Confidence gate — ask, don't guess

A case is generated only when every mandatory field (symptom, duration,
severity) clears `MIN_ENTITY_CONFIDENCE` (0.55) and the aggregate clears
`MIN_OVERALL_CONFIDENCE` (0.65). Otherwise the agent asks one targeted question.

After `MAX_FOLLOWUP_ROUNDS` (3) the case is generated regardless, with unresolved
fields written as the literal string `"Unknown"` and listed in
`missing_information`. It never fills a gap with a plausible value.

Other guards:

* Unrecognised entity categories are discarded, not coerced into a new bucket.
* Unparseable confidence becomes `0.0`, so it cannot satisfy a mandatory field.
* Specialty text is canonicalised against a closed vocabulary, so a hallucinated
  specialty can never reach the doctor query.
* `differential_considerations` are framed and prompted as conditions for a
  clinician to **rule out**, never as diagnoses, and are capped at 3.
* Patient age is `0` when unparseable rather than a plausible-looking default.

---

## 4. API

All routes require a Bearer token. Session routes require the `patient` role.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/ai/intake/sessions` | Start intake |
| `POST` | `/api/v1/ai/intake/sessions/{id}/turns` | Answer a follow-up |
| `GET` | `/api/v1/ai/intake/sessions/{id}` | Read session state |
| `POST` | `/api/v1/ai/intake/sessions/{id}/select-doctor` | Persist + route the case |
| `GET` | `/api/v1/ai/intake/health` | Dependency health (any authenticated user) |

### Status values

| `status` | Meaning |
| --- | --- |
| `collecting` | Agent asked a question; `pending_question` is set |
| `awaiting_doctor_selection` | Case ready; `recommendations` populated |
| `emergency_escalated` | Red flags found; intake halted, terminal |
| `routed` | Case persisted and assigned, terminal |

### Example

```bash
# 1. Start
curl -X POST localhost:8000/api/v1/ai/intake/sessions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"symptoms":"I have had moderate chest discomfort for 3 days"}'

# 2. Answer a follow-up if status == "collecting"
curl -X POST localhost:8000/api/v1/ai/intake/sessions/$SID/turns \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"answer":"about 3 days, and it is moderate"}'

# 3. Route once status == "awaiting_doctor_selection"
curl -X POST localhost:8000/api/v1/ai/intake/sessions/$SID/select-doctor \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"doctor_id":"<uuid from recommendations[]>"}'
```

### Error mapping

Domain errors extend existing project exceptions, so they route through the
handlers already registered in `app/middleware/exceptions.py`:

| Error | Status |
| --- | --- |
| `SessionNotFoundError` | 404 |
| `InvalidSessionStateError` | 422 |
| `AuthorizationException` (wrong patient / wrong role) | 403 |
| Pydantic validation | 422 |

---

## 5. Persistence

**Reuses** the existing clinical tables — a routed intake produces a normal
`cases` row plus `symptoms` rows, indistinguishable to the doctor portal from
any other case.

**Adds** two audit tables (`app/models/intake.py`):

* `intake_sessions` — durable record of each conversation (transcript, language,
  red flags, case snapshot, routing result)
* `intake_extracted_entities` — per-entity provenance: value, confidence, band,
  verbatim evidence quote, and `was_accepted`

The audit trail is deliberately separate: clinicians read the case, compliance
reads the audit. Audit writes are best-effort — a compliance-log failure is
logged but never costs the patient their consultation, while a failed *clinical*
write propagates.

Specialty routing queries real doctors filtered by specialty, active user
account and availability, ranked by verification → rating → experience. (For
contrast, the legacy `ai_service.process_symptom_intake` path uses
`select(Doctor).limit(1)`.)

> **Migration note:** this project has no Alembic revisions — schema comes from
> `Base.metadata.create_all` in the seed scripts. The two new models are
> registered in `app/db/base.py` and will be created by that path. If Alembic is
> adopted later, they need a revision.

---

## 6. Testing

**Full suite: 195 passed, 0 failed** — 149 new intake tests plus all 46
pre-existing tests still green (no regressions).

```bash
cd Backend
./venv/Scripts/python.exe -m pytest tests/intake/ -q     # this agent  (149)
./venv/Scripts/python.exe -m pytest tests/ -q            # everything  (195)
```

| File | Tests | Scope |
| --- | ---: | --- |
| `test_domain.py` | 70 | Grounding, red flags, readiness, entities, specialties |
| `test_workflow.py` | 27 | Graph behaviour across every required input class |
| `test_use_cases.py` | 18 | Authorisation, lifecycle, failure containment |
| `test_api.py` | 20 | Real HTTP + real JWT + real database |
| `test_llm_adapter.py` | 14 | Groq adapter parsing and tier fallback |

Required scenario coverage:

| Scenario | Where |
| --- | --- |
| English input | `test_workflow.py::TestCompleteEnglishIntake` |
| Hindi (Devanagari) | `TestLanguageHandling::test_hindi_devanagari_is_detected` |
| Hinglish | `TestLanguageHandling::test_hinglish_is_detected_without_model_call` |
| Mixed language | `TestLanguageHandling::test_mixed_script_is_detected` |
| Typographical errors | `TestTypoTolerance` |
| Emergency cases | `TestEmergencyEscalation` (6 tests) + `TestEmergencyOverHttp` |
| Incomplete descriptions | `TestIncompleteDescription` (4 tests) |
| Multiple simultaneous symptoms | `TestMultipleSymptoms` |
| Invalid requests | `test_api.py::TestRequestValidation` |
| Database failures | `TestSelectDoctor::test_database_failure_propagates`, `test_database_failure_does_not_silently_succeed` |
| API/model failures | `TestSafetyAndDegradation`, `test_llm_outage_degrades_without_500` |

No test makes a network call — the LLM is faked at the port boundary
(`ScriptedLLM`/`DeadLLM`) and at the transport boundary (`StubGroqClient`).

---

## 7. Failure behaviour

| Failure | Behaviour |
| --- | --- |
| LLM unreachable / all tiers fail | Adapter returns `{}`; nodes fall back deterministically; response has `degraded: true`; **no 500** |
| LLM returns garbage | Unparseable output discarded; entity dropped rather than guessed |
| Graph raises | Caught in `run_detailed`; session left coherent and retryable |
| Redis down | Existing `ResilientRedisClient` in-memory fallback |
| Session read corrupt | Treated as missing → fresh intake, not a 500 |
| Clinical DB write fails | Propagates; session stays `awaiting_doctor_selection` (retryable) |
| Audit write fails | Logged; clinical case retained |

**Emergency detection is unaffected by every row above** — it is deterministic
and runs before any model call.

---

## 8. Known environment issue

`GROQ_API_KEY` in `Backend/.env` is currently **rejected by Groq with
`401 invalid_api_key`** (well-formed: `gsk_…`, 56 chars).

This is pre-existing and not specific to this agent — the existing
`app.ai_core.ModelManager` fails identically with the same key, which means
`/ai/chat`, `/ai/analyze-report` and the legacy `/ai/symptom-intake` are equally
non-functional in this environment.

Observed live behaviour with the invalid key was exactly the designed
degradation: graceful fallback questions, `degraded: true`, no crashes, no
fabricated data — and **emergency escalation still worked correctly**, because
it never depends on the model.

Replace the key to enable real model output. No code change is required.
