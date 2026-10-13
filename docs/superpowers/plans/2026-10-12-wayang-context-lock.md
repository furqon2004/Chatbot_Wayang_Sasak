# Wayang Catalog Context Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the selected catalog character after its first successful answer while preserving explicit text-based character changes and validated follow-ups.

**Architecture:** Add a backend-issued HMAC token plus a TTL lock record keyed by a UUID conversation ID. The backend record is authoritative, so removing or changing a frontend token cannot unlock an existing conversation. Store locks in memory for one process and in the shared protection store when Redis is configured.

**Tech Stack:** Python 3.10+, Pydantic, HMAC-SHA256, FastAPI, shared protection-store interface from the request-protection plan, pytest.

**Spec:** [2026-10-12-wayang-sasak-chatbot-ux-and-queue-design.md](../specs/2026-10-12-wayang-sasak-chatbot-ux-and-queue-design.md)

## Global Constraints

- A lock is created only after a catalog-selected character receives a successful grounded answer.
- Locked catalog selection cannot change in the same conversation; explicit character names in text remain allowed.
- The backend record is authoritative; the UI flag alone never enforces the lock.
- Context locks expire with the existing five-minute ephemeral conversation lifetime.
- Keep conversation evidence validation, retrieval, and prompt-injection defenses unchanged.

## Review Focus

- Missing, malformed, or forged lock token cannot change a stored catalog choice; pin in Task 2.
- Same locked character remains selectable for the existing “explain more” action; pin in Task 2.
- A different explicit character name in text still overrides active conversation subject; pin in Task 3.
- New chat UUID permits a fresh selection; pin in Task 2.
- Unsuccessful/fallback answer does not create a lock; pin in Task 2.

---

### Task 1: Lock token and store primitives

**Files:**
- Create: `app/context_lock.py`
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/test_context_lock.py`

**Interfaces:**
- `ContextLockSigner.issue(conversation_id: UUID, character_id: int) -> str`
- `ContextLockSigner.verify(token: str, conversation_id: UUID, character_id: int) -> bool`
- `ContextLockStore.get(conversation_id: UUID) -> int | None`
- `ContextLockStore.set_if_absent(conversation_id: UUID, character_id: int, ttl_seconds: int) -> bool`
- Memory and shared implementations use the same methods; TTL is 300 seconds.

- [ ] **Step 1: Write failing tests:** `test_context_lock_signature_binds_conversation_and_character` asserts only the exact UUID/character pair verifies; `test_context_lock_store_first_write_and_ttl` asserts first writer wins and entry expires after 300 seconds.
- [ ] **Step 2: Run** `pytest tests/test_context_lock.py -q`; expected: the module is missing.
- [ ] **Step 3: Implement** HMAC-SHA256 tokens using `WAYANG_CONTEXT_SIGNING_KEY`; generate an ephemeral key only in single-process development. Require a stable configured key when shared multi-worker mode is enabled and document an empty development value in `.env.example`.
- [ ] **Step 4: Implement** the memory lock store with bounded entries and expiry; extend the shared-store interface from the request-protection plan for atomic `get`/`set_if_absent`.
- [ ] **Step 5: Run** `pytest tests/test_context_lock.py -q`; expected: signature, first-writer, bound, and expiry tests pass.

### Task 2: Enforce lock in the chat API

**Files:**
- Modify: `app/chat_schemas.py`
- Modify: `app/chat_routes.py`
- Modify: `app/api.py`
- Test: `tests/test_chat_api.py`

**Interfaces:**
- `ChatRequest` gains optional strict `conversation_id: UUID` and `catalog_lock_token: str` fields.
- `ChatResponse` gains `catalog_locked: bool` and `catalog_character_id: int | None`; include `catalog_lock_token` only when a backend lock exists.
- HTTP `409` with a safe detail is returned when a new `character_id` conflicts with the stored catalog selection.

- [ ] **Step 1: Write failing API tests:** `test_first_successful_catalog_answer_locks_selection` asserts response lock fields; `test_catalog_change_rejected_without_or_with_forged_token` asserts HTTP 409; `test_locked_character_and_explain_more_are_allowed` asserts same ID succeeds; `test_fallback_does_not_lock_and_new_conversation_can_select` asserts no lock on failure and a fresh UUID can select a different ID.
- [ ] **Step 2: Run** `pytest tests/test_chat_api.py -q`; expected: response fields and enforcement tests fail.
- [ ] **Step 3: Implement** lifespan wiring for the memory/shared `ContextLockStore`; validate the token and compare any new catalog `character_id` to stored state before calling `ChatService`.
- [ ] **Step 4: After a successful grounded response** whose selected catalog ID appears in the returned context, atomically lock the first selected ID and return the signed token and lock state. Do not lock fallback, error, or text-only requests.
- [ ] **Step 5: Run** `pytest tests/test_chat_api.py tests/test_conversation_context.py -q`; expected: lock behavior and existing context behavior pass.

### Task 3: Preserve conversational overrides and shared behavior

**Files:**
- Modify: `app/chat.py` only if route-level context needs a narrow adjustment
- Modify: `tests/test_conversation_context.py`
- Modify: `tests/test_shared_protection.py`

- [ ] **Step 1: Add regression tests:** `test_explicit_text_name_overrides_locked_catalog_selection` asserts Maktal is answered after Jayangrana was catalog-locked; `test_locked_context_still_resolves_followup` asserts prior subject/evidence remain available; `test_two_instances_share_context_lock` asserts a lock created by one Redis-backed app is visible to the other.
- [ ] **Step 2: Run** `pytest tests/test_conversation_context.py tests/test_shared_protection.py -q`; expected: shared lock test fails until the store is wired.
- [ ] **Step 3: Make only the smallest route/service adjustment** needed to keep explicit-name resolution ahead of prior conversation context while protecting catalog-selection fields.
- [ ] **Step 4: Run** `pytest tests/test_chat_api.py tests/test_conversation_context.py tests/test_shared_protection.py -q`; expected: all lock and follow-up tests pass.
