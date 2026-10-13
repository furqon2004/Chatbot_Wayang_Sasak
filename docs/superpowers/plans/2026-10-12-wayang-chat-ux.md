# Wayang Chat UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Wayang Sasak welcome, catalog selection, and request feedback clear while preserving Streamlit history and the existing image/chat endpoints.

**Architecture:** Keep the native Streamlit app and composer. Render welcome suggestions only when history is empty, move catalog discovery into the attachment popover, and route text, selected-character questions, and uploads through one guarded submit event. UI lock and request-limit feedback come from backend response fields/headers.

**Tech Stack:** Python 3.10+, Streamlit/AppTest, HTTPX, existing history component, FastAPI response contract.

**Spec:** [2026-10-12-wayang-sasak-chatbot-ux-and-queue-design.md](../specs/2026-10-12-wayang-sasak-chatbot-ux-and-queue-design.md)

## Global Constraints

- Do not append a greeting or suggestion to chat history until a user sends it.
- Preserve the existing five-minute ephemeral history and image preview/upload validation.
- Send selected-character context only through the existing grounded chat route.
- Keep the text override for explicit character names, even when catalog selection is locked.
- Show no fake progress percentage or unreported retrieval/provider stage.
- A per-tab session ID is for rate limiting only and is not an authenticated identity.

## Review Focus

- First visit, rerun, refresh, and new chat do not duplicate greeting; pin in Task 1.
- Catalog search failure or absent thumbnail leaves image upload usable; pin in Task 2.
- Locked catalog with image upload still open and text override allowed; pin in Task 2.
- Enter and send button cannot dispatch one action twice; pin in Task 3.
- HTTP 429/503 never leave spinner active and show only backend-supplied retry timing; pin in Task 3.

---

### Task 1: Welcome screen and minimal sidebar

**Files:**
- Modify: `streamlit_app.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Add `composer_prefill` for welcome/catalog prompts; preserve `pending_question` auto-send behavior for existing internal follow-up actions.
- Sidebar exposes one primary **Chat Baru** action and no permanent catalog action.

- [ ] **Step 1: Write failing AppTest cases:** `test_welcome_suggestions_prefill_without_sending` asserts suggestions on empty chat and zero API calls; `test_greeting_never_enters_history_on_rerun` asserts no user/assistant greeting entries after rerun; `test_sidebar_has_only_new_chat_action` asserts catalog is absent from sidebar.
- [ ] **Step 2: Run** `pytest tests/test_ui.py -q`; expected: old sidebar and suggestion behavior fail.
- [ ] **Step 3: Implement** Wayang-specific greeting and data-supported suggestions; clicking a suggestion sets `composer_prefill` and the text input value before widget construction, then reruns without sending.
- [ ] **Step 4: Simplify** sidebar to the new-chat action while preserving `clear_history()` behavior.
- [ ] **Step 5: Run** `pytest tests/test_ui.py -q`; expected: welcome, new-chat, and existing history tests pass.

### Task 2: Attachment catalog and locked selection display

**Files:**
- Modify: `streamlit_app.py`
- Modify: `ui_media.py` only if existing thumbnail resolver needs reuse
- Test: `tests/test_ui.py`

**Interfaces:**
- Reuse `request_api("/search", payload={"query": "daftar tokoh", "top_k": 3})` and the existing catalog response; cache the returned list in session state for the active tab.
- Store `selected_character_id`, `catalog_locked`, `catalog_character_id`, `catalog_lock_token`, and `conversation_id` in session state. Map `selected_character_id` to the existing `ChatRequest.character_id`; the backend lock contract is produced by the context-lock plan.
- Catalog button selection sets the selected ID and a useful prefilled question; it does not send until the user submits.

- [ ] **Step 1: Write failing AppTest cases:** `test_catalog_lives_in_attachment_popover` asserts one catalog request and no sidebar list; `test_catalog_selection_sets_composer_without_upload` asserts selected ID/name and zero chat sends; `test_missing_thumbnail_and_catalog_error_keep_upload_available` asserts image upload remains enabled; `test_locked_catalog_is_noninteractive_but_image_and_text_work` asserts locked text container, upload availability, and explicit-name submission; `test_new_chat_clears_selection_and_lock` asserts fresh state.
- [ ] **Step 2: Run** `pytest tests/test_ui.py -q`; expected: catalog is still outside the attachment popover.
- [ ] **Step 3: Implement** catalog discovery in the popover while retaining the existing `file_uploader`, upload guards, and preview. Do not call the search endpoint on every Streamlit rerun.
- [ ] **Step 4: Implement** selected-character display and consume `catalog_locked`/`catalog_character_id`/token returned by the context-lock API. Render locked state as a noninteractive container; keep upload available and allow explicit names in text.
- [ ] **Step 5: Run** `pytest tests/test_ui.py -q`; expected: attachment, selection, lock, image, and new-chat tests pass.

### Task 3: Single guarded submit and truthful processing feedback

**Files:**
- Modify: `streamlit_app.py`
- Modify: `ui_client.py`
- Modify: `ui_history.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- `send_message(text, uploaded=None, *, idempotency_key: str, session_id: str, conversation_id: str | None = None, catalog_lock_token: str | None = None, selected_character_id: int | None = None, awaiting_more=False, seen_ids=(), explain_more=False, conversation_context=None)` attaches stable action/session headers and the optional lock/selection fields.
- `BackendError` exposes safe `status_code: int | None` and `retry_after: int | None`, never raw response bodies.
- Add `client_session_id` and `conversation_id` to ephemeral history state; regenerate conversation ID on new chat, retain session ID for the tab TTL.

- [ ] **Step 1: Write failing tests:** `test_send_message_attaches_action_and_session_keys` asserts both headers; `test_enter_and_send_button_dispatch_one_action` asserts one backend call; `test_session_and_conversation_ids_restore_and_reset_separately` asserts session persists while new chat replaces conversation ID; `test_intentional_identical_sends_get_distinct_keys` asserts different UUIDs; `test_backend_error_exposes_only_retry_after_metadata` asserts status/retry metadata and no body leakage.
- [ ] **Step 2: Run** `pytest tests/test_ui.py -q`; expected: headers, retry metadata, and state assertions fail.
- [ ] **Step 3: Implement** a single guarded submit path. Preserve native text/file constraints and Enter behavior; make Enter invoke the same guarded action once, and create one UUID idempotency key per deliberate send.
- [ ] **Step 4: Attach** `Idempotency-Key` and `X-Wayang-Session`; pass conversation/selection token in `/chat` JSON. Preserve the same key when replaying the same action; do not add automatic retry.
- [ ] **Step 5: Show** a text spinner for understanding/search and an image spinner for image analysis. Map backend `429`/`503` plus `Retry-After` to a safe busy message; never claim live queue position or web stage. Add assertions that web source labels appear only after a web response, while local answers never claim internet search.
- [ ] **Step 6: Run** `pytest tests/test_ui.py -q`; expected: submission, loading/error cleanup, and history tests pass.

### Task 4: End-to-end regression and UI copy

**Files:**
- Modify: `tests/test_ui.py`
- Modify: `docs/chat.md`
- Modify: `README.md` only if run instructions need a UI behavior note

- [ ] **Step 1: Add `test_full_chat_flow_preserves_catalog_lock_and_image_upload`** from empty welcome through catalog selection, grounded response, locked indicator, explicit text override, image upload, and new chat; assert one send per action and no greeting in history.
- [ ] **Step 2: Run** `pytest tests/test_ui.py tests/test_chat_api.py tests/test_conversation_context.py tests/test_chat_quality.py -q`; expected: welcome, grounding, casual/typo, follow-up, and lock integration tests pass.
- [ ] **Step 3: Document** new attachment selection and the truthful limits of synchronous status feedback.
- [ ] **Step 4: Run** `pytest tests/test_ui.py tests/test_chat_api.py tests/test_conversation_context.py tests/test_chat_quality.py -q`; expected: pass without real provider/API calls.
