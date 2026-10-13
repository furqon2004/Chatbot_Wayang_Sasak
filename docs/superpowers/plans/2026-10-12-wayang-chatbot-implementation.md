# Wayang Sasak Chatbot Improvement Plan Index

The approved design spans three reviewable subsystems. Implement each plan in
this order; each plan has its own test cycle and reviewable deliverable.

1. [Request protection](2026-10-12-wayang-request-protection.md) — bounded
   admission, rate limits, idempotency, and shared Redis coordination. This
   defines the shared store interface used by the next plan.
2. [Catalog context lock](2026-10-12-wayang-context-lock.md) — backend-issued
   lock state with explicit text override preserved. It consumes the shared
   store from plan 1.
3. [Chat UX](2026-10-12-wayang-chat-ux.md) — welcome, attachments, one guarded
   submit path, and truthful loading/retry feedback. It consumes the API
   contracts from plans 1 and 2.

Finish with the full verification listed in the design spec: relevant pytest
suites, syntax/import checks, and the available UI/API integration tests. Do
not run live provider calls as part of automated verification. Keep all plan,
implementation, and test changes uncommitted until the user explicitly says
they are ready to commit.
