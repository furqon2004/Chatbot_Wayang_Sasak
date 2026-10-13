# Wayang Local Knowledge Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route natural Wayang role and trait questions into evidence-backed local profile search, with grounded web fallback when local evidence is insufficient.

**Architecture:** Extend the existing role-question parser and local profile evidence verifier. Keep the existing catalog, retrieval, provider, API schema, and chat UI; preserve explicit out-of-domain rejection and source evidence IDs.

**Tech Stack:** Python, pytest, existing FastAPI chat service and profile retrieval.

**Spec:** User attachment `CRITICAL FIX — Restore Wayang Sasak Knowledge Understanding Without Breaking Existing Features`.

## Global Constraints

- Do not read or modify `.env`, secrets, credentials, API keys, tokens, or private keys.
- Do not change data, API contracts, provider implementation, UI, image handling, or session management.
- Answer role or trait claims only from direct, validated local evidence or existing grounded web results.
- Keep irrelevant explicit topics out of the Wayang Sasak retrieval path.

## Review Focus

- Inverted and colloquial role questions must be recognized without making unrelated questions in-domain.
- Trait evidence stored outside descriptions must retain the correct profile field and evidence index.
- Negated, hypothetical, or unsupported profile claims must not identify a character.
- Ambiguous follow-ups must use validated context or ask a clarification.
- Insufficient local evidence must reach the existing web fallback when configured.

### Task 1: Parse natural role questions and resolve domain context

**Files:**
- Modify: `app/role_search.py`
- Modify: `app/chat.py`
- Modify: `app/query_normalizer.py`
- Modify: `app/conversation.py`
- Test: `tests/test_role_search.py`, `tests/test_query_normalizer.py`, `tests/test_conversation_context.py`, `tests/test_chat_api.py`

- [x] Add failing tests for `tokoh utamanya siapa`, `tokoh yang licik siapa`, context-aware role questions, and explicit out-of-domain rejection.
- [x] Run the new tests and confirm they fail on the current parser/early scope gate.
- [x] Extend only the existing role-question grammar; treat explicit `tokoh`/`karakter` role forms as Wayang questions, while retaining explicit non-Wayang rejection.
- [x] Verify ambiguous nameless role questions clarify instead of returning the out-of-domain catalog message.
- [x] Run the role-search tests.

### Task 2: Use all approved local profile fields and preserve evidence provenance

**Files:**
- Modify: `app/role_search.py`
- Test: `tests/test_role_search.py`

- [x] Add failing tests for direct traits in `ciri_ciri`/`filosofi` and keyword metadata, including the exact evidence field/index.
- [x] Run the tests and confirm current evidence matching omits those fields.
- [x] Extend conservative profile evidence matching to approved public fields and return the actual field/index for citations; keep negation and uncertainty checks.
- [x] Run role-search, retrieval, and schema tests.

### Task 3: Evaluate existing web fallback when local role evidence is insufficient

**Files:**
- Modify: `app/chat.py`
- Test: `tests/test_role_search.py`

- [x] Add a failing test that a Wayang role query with insufficient local evidence calls the existing provider search and returns its grounded source.
- [x] Run the test and confirm the current local fallback stops before web search.
- [x] Route only recognized in-domain role requests with insufficient evidence through `_web_answer`; preserve the safe local fallback when web search is unavailable.
- [x] Run relevant chat, web, role, and general RAG tests.

### Task 4: Verify unaffected behavior

**Files:** No additional production changes expected.

- [x] Run syntax compilation for changed Python files.
- [x] Run the full pytest suite and report warnings or failures.
- [x] Confirm the tests cover general overview, direct profile lookup, catalog, image/chat UI, recommendations, and out-of-domain requests.
