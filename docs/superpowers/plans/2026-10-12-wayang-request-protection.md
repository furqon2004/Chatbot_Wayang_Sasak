# Wayang Request Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded request admission, retry guidance, idempotency, and a shared coordination option without changing existing API payload results.

**Architecture:** Keep `SecurityMiddleware` as the request boundary. Add a FIFO bounded admission controller and a protection-store interface with an in-memory single-process implementation and an optional Redis implementation for scaled deployments. Scope anonymous session limits alongside the existing peer/IP limit; cache completed responses only for keyed inference/chat POSTs.

**Tech Stack:** Python 3.10+, FastAPI/Starlette ASGI, `asyncio`, `redis` async client (optional at runtime), pytest and HTTPX.

**Spec:** [2026-10-12-wayang-sasak-chatbot-ux-and-queue-design.md](../specs/2026-10-12-wayang-sasak-chatbot-ux-and-queue-design.md)

## Global Constraints

- Keep request body limits before JSON/multipart parsing.
- Bound both active work and waiting work; never allow an unbounded queue.
- Preserve request IDs and safe error bodies; never log payloads or credentials.
- A configured shared store failure must fail closed; do not fall back to local deduplication or limits.
- Keep one-worker development usable without a Redis service.
- Treat the browser session ID as an anonymous rate-limit key, not authentication.
- Never trust arbitrary forwarded IP headers.

## Review Focus

- Same idempotency key and same body returns one cached response; pin in Task 3.
- Same idempotency key with a different body is rejected; pin in Task 3.
- Concurrent duplicate arrives while the first request is running; pin in Task 3.
- A full queue and a timed-out waiter release/retain slots correctly; pin in Task 1.
- Two app instances sharing one store observe common rate, capacity, and idempotency state; pin in Task 4.

---

### Task 1: Bounded FIFO admission

**Files:**
- Modify: `app/security.py`
- Modify: `app/config.py`
- Modify: `app/security_middleware.py`
- Test: `tests/test_security.py`

**Interfaces:**
- Produces `AdmissionResult(admitted: bool, waited_seconds: float, retry_after: int | None)` and `AdmissionController.acquire() -> AdmissionResult`; `release()` always releases one active slot.
- `AdmissionController` accepts the active limit, `queue_max_waiters`, `queue_wait_timeout`, and optional `SharedAdmissionQueue`. It uses local FIFO in one-worker mode and the global FIFO queue in shared mode.
- Adds validated `SecuritySettings.queue_max_waiters` and `queue_wait_timeout` with conservative defaults.

- [ ] **Step 1: Write failing admission tests:** `test_admission_queue_is_fifo` asserts waiter order; `test_admission_rejects_when_waiter_limit_is_full` asserts `admitted is False`; `test_admission_timeout_and_cancellation_release_slots` asserts the next waiter acquires after cancellation/timeout.
- [ ] **Step 2: Run** `pytest tests/test_security.py -q`; expected: the new admission tests fail because `AdmissionController` does not exist.
- [ ] **Step 3: Implement** a bounded async FIFO admission controller and wire it around protected request execution in `SecurityMiddleware`.
- [ ] **Step 4: Run** `pytest tests/test_security.py -q`; expected: admission tests and existing capacity tests pass.

### Task 2: Rate-limit identity and retry headers

**Files:**
- Modify: `app/security.py`
- Modify: `app/security_middleware.py`
- Modify: `app/config.py`
- Test: `tests/test_security.py`

**Interfaces:**
- `MemoryRateLimiter.check(key: str) -> tuple[bool, int]` returns admission and whole seconds until retry; keep `allow(key)` as a compatibility wrapper if existing callers/tests need it. Add `SecuritySettings.session_requests_per_minute` separately from the existing peer limit.
- The middleware consumes a validated `X-Wayang-Session` UUID when present and applies limits to both peer/endpoint and session/endpoint keys.

- [ ] **Step 1: Write failing tests:** `test_rate_limit_retry_after_uses_ceil_seconds` asserts the reset header rounds up; `test_session_limit_is_independent_within_peer_limit` configures a high peer cap and low per-session cap, then asserts a second session remains allowed; `test_invalid_session_id_falls_back_to_peer_limit` asserts malformed IDs are not trusted.
- [ ] **Step 2: Run** `pytest tests/test_security.py -q`; expected: new header/session assertions fail.
- [ ] **Step 3: Implement** retry calculation and the two rate-limit keys; return `429` with `Retry-After` when the fixed window is exhausted, and `503` with `Retry-After` when admission wait expires or is full.
- [ ] **Step 4: Run** `pytest tests/test_security.py -q`; expected: rate, retry header, and existing authentication/security tests pass.

### Task 3: Idempotent ASGI responses

**Files:**
- Create: `app/request_protection.py`
- Modify: `app/security_middleware.py`
- Test: `tests/test_security.py`

**Interfaces:**
- `IdempotencyStore.begin(key: str, fingerprint: str, ttl: int) -> tuple[str, bytes | None]` returns `owner`, `replay`, `in_progress`, or `conflict`.
- `IdempotencyStore.complete(key: str, response: bytes, ttl: int) -> None` stores a bounded response; `abandon(key: str) -> None` clears an owner claim after a pre-response failure.
- Define `ContextLockStore` here as a protocol with `get(conversation_id: UUID) -> int | None` and atomic `set_if_absent(conversation_id: UUID, character_id: int, ttl_seconds: int) -> bool`; the memory and Redis stores implement it.
- Define `SharedAdmissionQueue.acquire(ticket_id: str, active_limit: int, max_waiters: int, timeout_seconds: int) -> AdmissionResult` and `release(lease_token: str) -> None`; this owns the globally bounded FIFO waiter list and active leases.
- Middleware requires a UUID `Idempotency-Key` for `/chat` and `/predict-image`, scopes it by endpoint and session, and fingerprints the already-buffered body.

- [ ] **Step 1: Write failing API tests:** `test_idempotency_replays_completed_response` asserts the handler runs once and response bytes match; `test_idempotency_rejects_key_with_different_body` asserts HTTP 409; `test_idempotency_rejects_concurrent_duplicate` asserts one handler call and a retryable busy response.
- [ ] **Step 2: Run** `pytest tests/test_security.py -q`; expected: duplicate requests currently execute independently.
- [ ] **Step 3: Implement** bounded response capture/replay for successful and safe error responses; exclude the previous `X-Request-ID` on replay so the current request keeps its own correlation ID. Limit cache TTL and response bytes.
- [ ] **Step 4: Run** `pytest tests/test_security.py -q`; expected: same-key replay, conflict, concurrent duplicate, and request-ID assertions pass.

### Task 4: Shared Redis protection store

**Files:**
- Create: `app/redis_protection.py`
- Modify: `app/api.py`
- Modify: `app/config.py`
- Modify: `requirements.txt`
- Test: `tests/test_security.py`
- Test: `tests/test_shared_protection.py`

**Interfaces:**
- `RedisProtectionStore(url: str)` implements `RateLimiter`, the globally bounded FIFO `SharedAdmissionQueue`, and `IdempotencyStore`; scripts are atomic and active leases expire after the maximum request timeout plus a safety margin.
- `RedisProtectionStore` also implements the `ContextLockStore` protocol defined in Task 3.
- Add `WAYANG_REDIS_URL`; absent means the existing one-worker memory mode. A configured but unreachable store yields a safe `503`, never an in-memory fallback.

- [ ] **Step 1: Write failing two-instance tests:** `test_shared_redis_rate_queue_capacity_and_idempotency` uses two store objects over one fake shared command/script backend and asserts common rate counts, FIFO order across instances, one global active cap, bounded global waiters, replay, and conflicting payload result.
- [ ] **Step 2: Run** `pytest tests/test_shared_protection.py -q`; expected: module/interface is missing.
- [ ] **Step 3: Implement** Redis atomic operations for the shared FIFO queue, concurrency leases, rate limits, and idempotency; wire app-lifespan construction/cleanup and add the async Redis client dependency without requiring a running Redis service in single-worker mode.
- [ ] **Step 4: Run** `pytest tests/test_shared_protection.py tests/test_security.py -q`; expected: both independent instances share FIFO, rate, concurrency, idempotency state and all existing middleware tests pass.

### Task 5: Deployment and regression checks

**Files:**
- Modify: `docs/security-hardening.md`
- Modify: `docs/chat.md`
- Modify: `.env.example`
- Test: `tests/test_security.py`

- [ ] **Step 1: Add `test_configured_redis_failure_fails_closed`** asserting missing URL selects memory mode while unreachable configured Redis yields HTTP 503 and never invokes a route handler.
- [ ] **Step 2: Run** `pytest tests/test_security.py -q`; expected: fail for the configured-store failure case until wiring is complete.
- [ ] **Step 3: Document** single-worker defaults, required shared-store configuration for multi-worker/replica deployment, queue/rate defaults, and `Retry-After` semantics.
- [ ] **Step 4: Run** `pytest tests/test_security.py tests/test_api.py tests/test_chat_api.py -q`; expected: pass with no network service required.
