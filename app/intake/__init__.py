"""
AI Medical Case Intake Agent.

A stateful, multi-turn clinical intake workflow built on Clean Architecture:

    presentation (app/api/v1/endpoints/intake.py)
        -> application (use cases, ports)
            -> domain (entities, policies, safety rules)
                <- infrastructure (LLM, Redis, SQLAlchemy adapters)

The `domain` package is pure Python with no framework imports. The `workflow`
package holds all LangGraph orchestration and is the *only* place that knows an
LLM exists. Controllers never touch the database or the LLM directly.
"""
