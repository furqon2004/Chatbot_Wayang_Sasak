# Security hardening — Stage 3

Branch: `feat/stage3-secure-chat`, baseline `084f27e`. Baseline before changes:
**165 passed, 3 warnings**. The source model/data/Chroma files are unavailable
locally and were not modified. No real API key or `.env` was read. No commit/push.

## Findings and decisions

| Severity | Baseline finding | Change / reason |
|---|---|---|
| High, conditional on public deployment | No authentication, admission control or request rate limit | Configurable bearer auth; production fails closed without a token; per-peer rate limit; bounded concurrency/body reader |
| High | Weak separation of cultural requests and injected/unsupported instructions | Deterministic chat intent decision before retrieval/Gemini; supported clauses only for mixed requests |
| High | Evidence IDs were checked, but text/source integrity and poisoned excerpts were not fully checked | Verify exact text against catalog profile, official ID/label/name, score, bounds, duplicates and request-specific ID allowlist |
| Medium | Arbitrary JSON input can be echoed by validation errors | Generic 422 plus generated request ID; no input values or traceback |
| Medium | Unbounded raw request parsing and aggregate upload work | ASGI body/deadline limit before FastAPI parses multipart, admission control, explicit pixel limit |
| Medium | No cooldown, input byte budget or explicit disabling of SDK automatic tool use | Per-client breaker/capacity/deadline, bounded retry+jitter, 8000-byte prompt, no tools and no function calling |
| Medium | No deployment Host/CORS policy | Production host allowlist, exact HTTPS origins, public docs opt-in, minimum production health |
| Low | Unstructured security events / attacker request IDs | JSON metadata event stream, generated IDs, allowlisted fields, nested redaction and control stripping |
| Critical | None demonstrated locally | Tool execution would make some risks critical; there is no such capability |

Full per-threat likelihood, impact, controls and residual risk are in
[security-threat-model.md](security-threat-model.md). Detection patterns are
only one layer; they are not treated as a complete prompt-injection defense.

## File inventory

| Files changed/added | Purpose |
|---|---|
| `app/api.py` | Opt-in existing chat router, production health/docs, middleware, generic validation errors, bounded image format/pixels/cleanup |
| `app/config.py`, `.env.example` | Explicit Stage 3 activation and deployment security settings without secrets |
| `app/query_normalizer.py` | Separate inspection normalization and deterministic intent classification; existing retrieval normalization retained |
| `app/chat.py`, `app/chat_schemas.py` | Per-request source integrity, bounded evidence/output, comparison checks and scope notice |
| `app/gemini.py` | Structured prompt, no tools, capacity/deadline/jitter/breaker and safety-rejection handling |
| `app/chat_routes.py`, `app/search_routes.py` | Safe structured mode/status event metadata |
| `app/security.py`, `app/security_middleware.py` | Injected/resettable state, bounded request admission, auth, headers, request IDs and redaction |
| `app/semantic.py`, `app/retrieval_resources.py` | Conflicting identity rejection and strict runtime artifact collection/count/metric validation |
| `tests/conftest.py`, `tests/test_security.py` | Environment/network isolation and 96 adversarial tests; existing test cases unchanged |
| `requirements.txt`, `requirements-dev.txt`, `requirements-security.txt` | Persist tested Pillow/pytest ranges and keep security tools outside production dependencies |
| `tests/test_security_scan.py` | Ten regression tests for explicit version-specific VEX acceptance, new findings and incomplete reports |
| `scripts/security_scan.py`, `.github/workflows/security.yml`, `.gitignore` | Source-only secret scan, CI checks, optional advisory audit, ignored artifacts/logs/tools |
| `docs/security-threat-model.md`, `docs/security-hardening.md`, `docs/constraints-win-py310.txt` | Threat register, implementation/triage/residual risks and local dependency version inventory |

## Compatibility and activation

`create_app()` remains the Stage 2 application by default. Set
`WAYANG_ENABLE_CHAT=true` to register the existing `/chat` router and use the
existing Gemini adapter. An injected `gemini_loader` still works for tests.
No parallel chat implementation, new SDK or `google-adk` was added.
With chat disabled, Settings does not read `GEMINI_API_KEY`. With chat enabled,
the backend reads the server environment; missing key is allowed and selects
deterministic retrieval/catalog replies with `llm_used=false`.

`/predict-image` and `/search` keep their successful payloads. Threshold remains
`0.5590755322575569`, hybrid weights remain 55/45, and the ordered catalog remains
exactly IDs 0–19. Model/classifier/temperature code was not changed.
Development `/health` retains the baseline response. Production `/health`
returns only `{"status":"ok"}` when ready. Production auth, host validation,
rate/admission limits and generic 422 errors intentionally restrict requests;
clients must handle 401/413/422/429/503. Production OpenAPI/docs are disabled
unless explicitly configured. No proxy-forwarded header is trusted by this code.

Chat adds optional `scope_notice` and an 8000-character answer bound. History
is deliberately unsupported: any `history` field, even an empty list, is rejected.
The new intent policy applies to `/chat`; Stage 2 ranking/query semantics are
preserved. General overview requests use the scoped retrieval query “Siapa tokoh
utama Wayang Sasak?” to obtain an evidence-backed introduction, not an invented
complete cultural history. If that retrieval fails its existing threshold,
chat falls back. This routing must be reviewed with real Colab evidence.

## Grounding and output boundary

1. Validate input bounds, domain/intent and mixed clauses deterministically.
2. Retrieve at most top-3 and check scores, official source identity, exact
   profile excerpt, known fields, unsafe text and duplicate evidence IDs.
3. Missing/invalid evidence uses catalog fallback without an LLM request.
4. Send only `untrusted_query` and bounded `untrusted_evidence` JSON. The system
   instruction uses the official SDK's separate `system_instruction` parameter.
5. Gemini may return only up to six evidence IDs, never a free-form factual
   answer. Every returned ID must belong to this request; duplicates/extra
   fields/malformed JSON/safety rejections fail validation.
6. The backend renders exact source excerpts and lists only sources used.
   Comparisons require evidence for every named character before and after ID
   selection; an absent Bali comparison falls back. Catalogs are backend-only.

`llm_used=true` means valid evidence selection actually succeeded. Vendor errors,
timeout, invalid output, cooldown or missing credentials yield deterministic
safe evidence/catalog replies and `llm_used=false`. Blocked malicious requests
retain legacy `llm_status=invalid_response` for compatibility, without calling
Gemini. Empty evidence/selection yields `insufficient_evidence` catalog fallback.
Retrieval unavailable retains the existing generic 503 contract.

There is no eval/exec, template rendering, URL fetching, shell, SQL, filesystem
or tool execution path. Chroma document bodies are not requested for prompts:
evidence comes from the validated public JSON profile. Poisoned profile text is
rejected for chat without editing the JSON or changing the Stage 2 serializer.

The frontend is not present here. It **must** display text using escaped/text
APIs; never `unsafe_allow_html=True`, `mark_safe`, `innerHTML`, or executable
Markdown links for model output. JSON and response headers alone do not prove
safe frontend rendering. No arbitrary URL in an answer should be auto-fetched.

## Resource and cost envelope

| Resource | Bound |
|---|---|
| Message / history | 1000 characters / history prohibited |
| Search top_k / chat top_k | 1–5 / 3 |
| JSON request body | 16 KiB before parsing |
| Upload body / file | configured file limit + 64 KiB multipart overhead / default 10 MiB |
| Body receive deadline | 15 seconds, injectable settings |
| Pixel count | 16 million default; also respects Pillow bomb limits |
| Evidence context | 12 entries, each ≤500 characters, JSON ≤6000 characters |
| SDK input | ≤8000 UTF-8 bytes plus fixed system instruction |
| SDK output / attempts | 1024 tokens per attempt / 1–3, default 2 |
| Gemini timeout | ≤60 seconds configurable; default 15 seconds total including retry |
| Backoff | 0.5, 1.0 seconds base with ≤0.25 second jitter, bounded by deadline |
| App concurrency / LLM concurrency | default 2 / 1 per adapter |
| Rate limit | default 60 requests/minute per transport peer and endpoint |
| Limiter memory | at most 10,000 live keys; deny new keys if full |
| Circuit breaker | 3 terminal failures or immediate quota/auth error → 60-second cooldown |
| Chat answer / raw SDK text accepted | 8000 / 8192 characters |

This bounds token exposure per request, not a guaranteed invoice amount. With
three attempts the maximum configured generated output is 3072 tokens; retries
may each be billable. Approximate cost envelope is attempts × (bounded input
tokens × provider input price + 1024 × output price). Provider token accounting,
cancellation billing and pricing must be verified before real activation.
Configure provider/project quotas and budget alerts; a central monetary budget
enforcer is deferred. No current model price is assumed in this implementation.

Rate limiter and breaker are per instance, injectable/resettable and use locks;
capacity releases in `finally`, including cancellation. Fake SDK clients have
deterministic jitter by default; production clients use randomized jitter. Tests
also inject nonzero jitter explicitly. Never retry auth/quota/schema/safety
errors; SDK retries are disabled so retry layers cannot multiply.

In-memory state is **not sufficient for multiple workers/replicas**. Inject a
shared implementation of `RateLimiter` (e.g. Redis atomic windows), use gateway
identity quotas and shared cost/breaker policy before scaling. A shared service
token is not per-user authentication. OAuth/JWT verified through an IdP is the
recommended real-user deployment boundary; no unverified JWT parser was added.

## Deployment configuration

The application does not automatically open `.env`; configure its server process
environment or secret manager. `.env.example` contains blank credential values.

Development defaults: `WAYANG_ENV=development`, auth disabled with blank service
token, chat opt-in, no allowed CORS origins by default, existing localhost tests
and Stage 2 health contract preserved.

Production requires `WAYANG_ENV=production`, a service token of at least 32
characters delivered by a secret manager, and exact `WAYANG_ALLOWED_HOSTS`.
Use cryptographically generated tokens; length validation alone cannot prove
entropy. Set `WAYANG_CORS_ORIGINS` only to explicit HTTPS frontend origins (no
wildcards). `WAYANG_PUBLIC_DOCS=false` is the safe default. Values for rate,
concurrency and pixel caps are in `.env.example`. Never put real tokens in that
file, the browser, fixture, notebook output or documentation.

Terminate HTTPS at a trusted reverse proxy. Enforce connection/header/body
limits, slow-client timeouts, global rate limits and worker memory/CPU limits
there. Configure Uvicorn proxy trust explicitly; do not accept arbitrary client
forwarded IP headers. With proxy headers disabled all users behind one proxy
share its IP quota; use verified identity quotas for larger deployments.
Security headers: nosniff, DENY framing, frame-ancestors none, no-referrer,
no-store. CSP is a framing control here, not a claim that arbitrary HTML is safe.

Use a non-root runtime identity, read-only mounts for original data/model,
restricted egress, and writable bounded temporary storage. The existing Chroma
snapshot approach is preserved: original source is read, PersistentClient opens
only a temporary copy, and smoke compares source hashes. This is **not** a native
read-only Chroma client; the temporary copy can be migrated/written by the
library. Runtime environment loading validates the `wayang_sasak` collection,
48 documents and cosine metric. Small explicitly injected test resources retain
their baseline contract. No mutation endpoint or embedding rebuild was added.

## Secrets, logs and incident response

`wayang.security` emits JSON metadata: generated request ID, endpoint/method,
status, duration, mode, safe category, `llm_used`, limit/retry/circuit events.
No full query, Authorization, cookie, prompt, raw vendor response, upload,
embedding or path is logged by that stream. Nested sensitive field names are
redacted case-insensitively; controls/newlines are stripped. Baseline fixed
operational log messages remain. Configure proxy/SDK logs with equivalent rules.

Restrict log access to operators, default retention 7 days unless an approved
policy requires longer, encrypt transport/storage, and audit access. Do not use
logs as conversation storage. Retention and external log configuration are
deployment work, not enforced by this repository.

If a credential was exposed: revoke it at the provider, issue a new credential,
update only the server secret manager, restart affected instances, review quota
and access metadata, and remove leaked material from logs/history through the
authorized incident process. Do not print or partially display a key to verify
rotation. No real credential was inspected for this task.

## Validation and commands

Original hardening results: **261 passed (165 unchanged baseline + 96 security tests)**.
After persisting tested dependency remediation: **271 passed** (the same 261 plus
10 VEX-policy regression tests). `compileall`, `pip check`, secret scan and
`git diff --check` pass. Three full-suite warnings are the two existing TestClient
deprecations and the intentionally triggered Pillow bomb warning. The explicit
pixel check rejects that image with 413; a promoted warning/error is also caught.
Network guards were active for the entire final suite. Artifact-path verification
is based on Git scope and mock source-hash tests; no claim is made that absent
Colab artifacts were hashed locally.

Final embedding-loader hardening: **292 passed**, including 21 new mock tests
for development/production policies, local directories, request isolation and
safe cache-miss startup failure. Existing network-denial fixtures remain active.
Compileall, pip check, source secret scan and diff checks pass; Bandit still
reports two Medium B615 in the unchanged vision loader and zero High findings.

### Advisory and static scan results

Remediation history (intermediate counts reported from the user's tested upgrade;
final environment and full audit verified again in this pass):

| Stage | Advisory entries | Packages / unique IDs |
|---|---|---|
| Initial audit | 64 | 5 packages / 35 unique IDs |
| After upgrading pip | 50 | pip advisories removed |
| After Pillow 12.3.0 | 15 | Pillow advisories removed |
| After pytest 9.1.1 | 13 | 2 packages / 9 unique IDs |

Final installed versions: Pillow **12.3.0**, pytest **9.1.1**, pip **26.2.1**,
Bandit **1.9.4**, pip-audit **2.10.1**, ChromaDB **1.5.9**, Transformers **4.57.6**.
Pillow/pytest/pip have no advisory entries in the final scan. Remaining entries:
ChromaDB **5 / 4 unique**, Transformers **8 / 5 unique**. Duplicate upstream
entries are retained in raw reports but not counted as independent risks.

`requirements.txt` now uses `pillow>=12.3,<13`; `requirements-dev.txt` uses
`pytest>=9.0.3,<10`. Security tools live in `requirements-security.txt`, which
includes development requirements; production installs only `requirements.txt`.
The local constraints update only Pillow/pytest/pip and add direct Bandit/pip-audit
pins. Other pins are unchanged. Security-tool transitive packages are not a new
fully hash-locked environment. No package installation or retrieval/model change
was performed in this persistence pass; it records the already-upgraded environment.

The raw `python -m pip_audit` still exits **1** because vulnerabilities remain.
This is not a zero-vulnerability result. The VEX policy below permits only the
reviewed package/version/ID combinations, preserving raw findings and failing
on new advisories, changed affected versions, skipped packages or malformed reports.

### Dependency VEX

Reviewed 2026-09-23 against application source and the installed versions below.
Severity is the upstream CVSS rating from each linked OSV/PyPA advisory, not a
down-rating based on local reachability. Reachability is a source-review inference,
not proof about a deployed host or arbitrary future models. `accepted` means a
conditional technical exception for the current architecture; `deferred` means
upgrade work is explicitly postponed pending compatibility validation. Neither
means the vulnerable package has been fixed or authorizes exposing a new path.

| Advisory / package and version | Upstream severity | Affected functionality | Reachability in this project | Existing mitigation | Decision | Residual risk | Condition making it reachable |
|---|---|---|---|---|---|---|---|
| [PYSEC-2026-311](https://osv.dev/vulnerability/PYSEC-2026-311) — ChromaDB 1.5.9 | Critical 9.3 (CVSS v4) | Pre-auth code injection via HTTP collection creation and attacker model repository | No Chroma HTTP server; only local PersistentClient | Fixed `wayang_sasak`, `get_collection(..., embedding_function=None)`, no mutation endpoint | accepted, conditional | Library remains affected; compromised local runtime/config is outside this boundary | Exposing a Chroma server/create endpoint or accepting a user embedding function/model repository |
| [PYSEC-2026-3813](https://osv.dev/vulnerability/PYSEC-2026-3813) — ChromaDB 1.5.9 | High 8.8 (CVSS v3/v4) | Missing tenant authorization permits cross-tenant reads/writes/deletes | No HTTP multi-tenant API or user-selected collection | Single local allowlisted collection; no create/update/delete routes | accepted, conditional | Local process/filesystem access remains privileged | Adding Chroma HTTP access, remote tenants or user-controlled collection operations |
| [PYSEC-2026-3814](https://osv.dev/vulnerability/PYSEC-2026-3814) — ChromaDB 1.5.9 | Critical 9.4 (CVSS v4) | Collection update loads attacker repository with remote code | No update-collection API or model configuration from requests | Logical read-only retrieval; no embedding function mutation | accepted, conditional | Future administrative integrations can reopen RCE path | Exposing UPDATE_COLLECTION or accepting remote-code embedding configuration |
| [PYSEC-2026-3815](https://osv.dev/vulnerability/PYSEC-2026-3815) — ChromaDB 1.5.9 | High 8.8 (CVSS v3/v4) | SimpleRBAC permissions not scoped to tenant/database/collection | SimpleRBAC HTTP authorization provider not used | Local single-collection client; backend auth is separate | accepted, conditional | No claim of safe Chroma multi-tenant isolation | Starting a shared Chroma server with SimpleRBAC or relying on it for tenant isolation |
| [PYSEC-2025-217](https://osv.dev/vulnerability/PYSEC-2025-217) — Transformers 4.57.6 | High 7.8 (CVSS v3) | Unsafe X-CLIP checkpoint conversion/deserialization | X-CLIP conversion absent from app and scripts | Existing local CLIP loader; classifier uses `torch.load(weights_only=True)` | deferred | Malicious checkpoints remain dangerous outside approved loader | Adding X-CLIP conversion or opening untrusted checkpoint files |
| [PYSEC-2026-2288](https://osv.dev/vulnerability/PYSEC-2026-2288) — Transformers 4.57.6 | High 7.8 (CVSS v3) | Trainer `_load_rng_state()` unsafe torch loading on older Torch | No Trainer/resume-training path; installed Torch 2.14.0 is above the advisory's <2.6 precondition | Inference-only path and explicit `weights_only=True` for classifier | deferred | Training utilities introduced later could change preconditions | Using Trainer checkpoint resume, particularly with Torch below 2.6 and untrusted RNG state |
| [PYSEC-2026-2289](https://osv.dev/vulnerability/PYSEC-2026-2289) — Transformers 4.57.6 | High 7.8 (CVSS v3; prose describes critical RCE) | AutoModelForCausalLM config redirects attention kernels to attacker repository | No AutoModelForCausalLM or user model repository input | Fixed CLIP architecture/local model path; fixed sentence embedder identifier | deferred | Local/cache model provenance remains important; no universal remote-code safety claim | Adding causal model loading or processing attacker-controlled model config/kernel repository |
| [PYSEC-2026-2290](https://osv.dev/vulnerability/PYSEC-2026-2290) — Transformers 4.57.6 | Critical 9.6 (CVSS v3) | LightGlue nested config overrides remote-code trust | No LightGlue path in application; retained because audit flags this installed version despite narrower description | Local CLIP classes and fixed non-LightGlue sentence model; no user-selected model | deferred | Advisory range/description differences require upstream review; cache compromise remains possible | Introducing LightGlue/AutoModel over attacker-controlled LightGlue config, even with remote-code flag disabled |
| [PYSEC-2026-3929](https://osv.dev/vulnerability/PYSEC-2026-3929) — Transformers 4.57.6 | High 7.1 (CVSS v3) | `save_pretrained()` chat-template keys allow path traversal writes | No `save_pretrained()` or model/tokenizer export path in app/scripts | Serving performs no model serialization; model IDs are not accepted from users | deferred | New export or notebook workflows are outside this review | Downloading a malicious tokenizer/processor and calling `save_pretrained()` |

ChromaDB stays at 1.5.9 in this pass. **Do not expose Chroma HTTP/server.** There
are no collection create/update/delete endpoints; only `wayang_sasak` is allowed.
Retain logically read-only retrieval. PersistentClient opens a temporary snapshot;
the library may migrate that copy, so this is not a claim of native read-only
storage. Original source hash protection remains. Re-review as soon as a fixed
version is published and whenever server/mutation/tenant functionality changes.

Transformers stays at 4.57.6, without a 5.x upgrade. Source review of `app/` and
`scripts/` found no Trainer, AutoModelForCausalLM, LightGlue, X-CLIP conversion,
`save_pretrained`, enabled `trust_remote_code` or user-controlled model repository.
`resources.py` retains local model paths, `local_files_only=True`, and classifier
`torch.load(weights_only=True)`. SentenceTransformer now always receives
`trust_remote_code=False`; production enforces `local_files_only=True` even if
`WAYANG_EMBEDDING_LOCAL_ONLY=false`. Development/Colab retain bootstrap downloads
of the unchanged fixed identifier unless local-only is explicitly enabled.
`WAYANG_EMBEDDING_REVISION` is optional (blank means the library default).
Operators should pin a verified commit matching the original Chroma embeddings
before public deployment; no revision is invented here. Model identity, cache
directory and loader options cannot be supplied by API requests. Cache placement
remains controlled by the deployment/library environment, not a request field.
An existing application-owned local model directory is resolved and loaded with
local-only enabled; an explicit missing local path fails without a Hub retry.
There is still only one SentenceTransformer constructor in the existing loader.
Production cache miss/invalid resources stop startup with a generic
`ResourceValidationError`; logs record only a fixed error category, with no
vendor exception/path/token. Development retains its degraded-startup behavior.
Pre-download the matching model/revision into the runtime account's cache in
Colab before enabling production. Protect cache and local model directories with
restricted permissions and verified fingerprints; local-only does not establish
artifact integrity or eliminate the existing advisory findings. VEX decisions
remain unchanged. A Transformers major upgrade
requires original vision-model, SentenceTransformer, ranking and Colab validation.

Re-review these exceptions on upstream fixes, model/config provenance changes,
new public endpoints or dependency-version changes; deployment owners still need
to approve their production risk. No global ignores or blanket suppression are used.

Original full Bandit scan had 30 alerts (2 Medium, 28 Low, 0 High). The required
remediation rerun, `python -m bandit -r app scripts -ll`, exits **1** and displays
only **2 Medium B615, 0 High**; no suppression was added. Manual triage:

- Two B615 alerts point to baseline `resources.py` calls with
  `local_files_only=True`; they are not remote downloads in that path. Integrity
  of local artifacts and the SentenceTransformer cache still needs deployment
  provenance controls.
- One B105 is the literal spelling correction `saksak`, not a credential.
- 26 B101 alerts are assertions in smoke scripts. Run smoke without `python -O`;
  these scripts are verification tools, not production authorization controls.
- One B110 is the existing best-effort Colab secret-loader branch in the chat
  smoke script. It was not executed or modified; real-secret smoke remains out
  of scope for this local run.

Raw reports are retained only in ignored `security-scan-results/`. The audit
virtualenv is also ignored. Neither report contents nor caches are staged.

Run with the project interpreter (`.venv/Scripts/python.exe` on this workspace):

```text
python -m pytest -q
python -m pytest tests/test_security.py -q
python -m compileall app tests scripts evaluation main.py
python -m pip check
python -m pip_audit --format json --output security-scan-results/dependencies-remediated.json
python -m bandit -r app scripts -ll
python scripts/security_scan.py
python scripts/security_scan.py --audit-json security-scan-results/dependencies-remediated.json
git diff --check
```

All unit tests replace the environment mapping without inspecting credentials.
Socket connect/connect_ex, DNS, create_connection and datagram send are blocked.
Only the interpreter's internal socketpair wakeup is allowed for Windows asyncio;
HTTP tests use in-process ASGI/TestClient, SDK tests use mock transports.
The scanner reads an explicit source allowlist and `.env.example`, never `.env`,
model/data files or Git history, and reports only file/category, not matched text.
It detects common patterns; absence of matches is not proof of absence of secrets.

CI runs test, secret scan, compile and dependency consistency checks without
project credentials. Online advisory audit is optional via workflow_dispatch.
Security tools can be installed in an isolated audit environment or the development
venv using `requirements-security.txt`; they are absent from runtime requirements.
This rerun uses the upgraded project `.venv`. Raw scanner output remains ignored
in `security-scan-results/`; source, VEX documentation and constraint files remain
visible to Git. CI inspects the complete JSON report with the version-specific VEX
checker rather than hiding IDs via `--ignore-vuln`. Advisory audit remains optional
via workflow_dispatch; when requested, unknown/new advisories fail the job.
`constraints-win-py310.txt` records exact installed versions for local reproduction
with `pip install -r requirements-dev.txt -c docs/constraints-win-py310.txt`; it is
not a secure release lock, not hash-pinned, and not portable to Colab/Linux.
Install security tools with `-r requirements-security.txt -c docs/constraints-win-py310.txt`
to use the same direct tool pins; no unrelated constraint pins were changed.
Package ranges and CI action major tags still need platform-specific lockfiles
and immutable action SHAs for a hardened release.

## Control status and residual risk

| Status | Controls |
|---|---|
| Implemented | Backend intent routing, supported-clause extraction, ID-only Gemini output, exact evidence/source checks, no tools, no output execution, input/context/answer bounds, deadline/retry/cooldown, per-instance capacity/rate limit, optional auth, production fail-closed configuration, headers/host/CORS, safe validation errors, source-only scan, mock adversarial tests |
| Partially implemented | Injection detection (heuristic); data authenticity (identity/shape, not provenance); cost control (token envelope, not monetary budget); supply-chain reproducibility (local version constraints); log policies (application stream only); distributed rate/breaker consistency (interface only); frontend encoding (documented, no frontend code) |
| Deferred | Redis/shared quotas, OAuth/IdP integration, provider hard monetary budget, artifact signing/provenance, OS/container resource isolation, immutable cross-platform lockfile and action SHAs, production gateway configuration, live Colab/Gemini validation |

| Residual risk | Priority / required follow-up |
|---|---|
| Unknown prompt obfuscations or subtle poisoned facts | High: maintain adversarial corpus, review upstream data; no tools and exact sources limit impact |
| Native image/model/dependency vulnerabilities | High: advisory triage, isolated runtime, tested dependency remediation before public exposure |
| Multi-worker quota bypass / IP rotation / NAT unfairness | High if exposed: shared limiter and verified identity at gateway |
| Expensive native work cannot be forcibly stopped safely in a thread | High if exposed: process/container limits; admission slot held while shielded thread work completes |
| Provider cost after cancellation and retry | Medium: quotas, spend alerts and central budget enforcement |
| UI unsafe rendering or automatic URL handling | High if present: verify actual frontend; text-only rendering policy |
| Strict source filtering may reject valid text; general overview is narrow | Medium: real Colab regression and cultural review, without reducing safety silently |
| No independent holdout for Stage 2 threshold | Medium: preserve calibration and add separate holdout rather than retune here |

Before deployment, run the existing 15-case retrieval smoke and hash checks on
original Colab artifacts, rerun image reference/temperature regression, validate
strict metadata/48/cosine startup, and test `/chat` first with fake Gemini and
real retrieval. Real Gemini activation, keys, quota/cost and provider safety are
not proven locally and require a separate controlled validation. Review the
actual frontend, TLS proxy, distributed limits and dependency findings first.


### Intentional artifact baseline update ? 25 September 2026

The current strict/production count is 48, centrally defined by
`app.config.EXPECTED_CHROMA_DOCUMENT_COUNT`. The project intentionally updated
Drive JSON and Chroma with `gunungan__filosofi`, using project data/informants;
see [provenance and reported Colab results](retrieval.md). This changes the
expected count, not metadata, cosine, snapshot isolation, or security controls.
Counts 47 and 49 remain rejected. Previous hashes describe previous artifacts;
within-run source fingerprint checks remain mandatory. Count checks alone do
not authenticate artifact contents. Local tests use synthetic fixtures only.
