> Catatan migrasi OpenAI: uraian provider/konfigurasi lama di bawah merupakan
> catatan historis. Implementasi aktif memakai OpenAI untuk narasi dan web;
> lihat [konfigurasi chat](chat.md). Hasil audit lama bukan validasi migrasi terbaru.

# Threat model — secure chat

Baseline: `feat/stage3-secure-chat`, `084f27e`; 165 tests passed before changes.
Assessment covers code and synthetic resources, not an exposed deployment. No
real secret, model, dataset, JSON database, or Chroma source was inspected.

## Assets and boundaries

Assets: Gemini/service credentials; API quota and cost; system instruction;
official JSON and 20-character catalog; Chroma, embeddings and source metadata;
model/classifier weights, dataset and training artifacts; CPU/GPU/RAM/threads/
storage; logs; API responses; deployment configuration.

Boundaries: browser/Streamlit → reverse proxy → FastAPI; upload → Pillow → CLIP;
user query → deterministic intent router → retrieval; Chroma → selected evidence
→ Gemini; Gemini → backend validation → frontend; secret manager → backend;
runtime → operator logs; dependency distribution → runtime.

The LLM has no tools or execution authority. A source ID is not proof that a
claim is true: answers must be exact backend evidence, with IDs validated against
the current request. The official JSON remains an upstream trust dependency.

## Threat register

Likelihood/impact are qualitative, conditional on exposure. Severity reflects
the unhardened baseline; residual risk assumes the documented controls.

| Threat | Asset / attack path | Likelihood | Impact | Severity | Existing controls | Required controls | Residual risk |
|---|---|---|---|---|---|---|---|
| Direct injection | User overrides system instructions | High | False answer | High | Strict evidence ID output | Intent routing, source validation, exact extracts | Novel phrasing can evade detection |
| Indirect injection | Poisoned evidence to Gemini | Medium | False answer | High | Chroma document bodies not used in prompts | Screen trusted-profile extracts, reject unsafe source text | Subtle factual poisoning needs human review |
| Prompt leakage | User asks for system instruction | Medium | Internal design disclosure | Medium | No free-text LLM answer | Guard and strict output allowlist | System instruction is not a secret boundary |
| Key exfiltration | User/LLM requests environment | Medium | Account compromise | High | No tools; key absent from context | No raw SDK output/errors; source-only replies | Compromised host still exposes credentials |
| Credential guessing | Repeated bearer attempts at public API | Conditional high | Quota/access abuse | High | None | Constant-time auth, rate limit, proxy controls | Shared demo token has no user identity |
| Error/log exfiltration | Validation input/vendor error echoed | Medium | Sensitive text disclosed | High | Generic vendor errors | Generic validation errors; metadata-only logs | Proxy/vendor logging outside app scope |
| Malicious HTML/JS | LLM/profile text rendered by UI | Medium | Browser script execution | High | JSON API; no frontend here | Reject executable evidence, exact text; UI encoding policy | Must verify actual frontend rendering |
| Excessive agency | LLM calls tools or executes output | Low currently | Data destruction | Critical conditional | No execution path | Explicitly disable SDK tools/function calling | Future features must preserve boundary |
| RAG/vector poisoning | Offline database alteration | Medium | Wrong retrieval/answer | High | Catalog mapping; read-source snapshot | Count/metric checks, per-request source allowlist | Fingerprints detect change, not truth |
| Path traversal | Upload filename interpreted as path | Low | File overwrite | High conditional | Filename never used; upload not persisted | Regression test traversal name | Framework temporary storage still requires permissions |
| Upload abuse | Fake MIME, corrupt/unsupported format | High | Parser crash/DoS | Medium | MIME + Pillow format allowlist | Bounded raw body before multipart parsing | Native codec bugs remain |
| Decompression bomb | Tiny compressed file expands | High | RAM exhaustion | High | Pillow limit and RGB conversion | Explicit pixel cap; warning/error handling | Decode/inference needs OS isolation |
| Flooding | Many requests or IP rotation | High if public | Availability loss | High | No app limiter | Bounded per-peer buckets, concurrency, proxy quotas | In-memory state is per worker |
| Resource exhaustion | Expensive CPU/GPU work | High if public | Availability loss | High | Threadpool; byte limit | Admission control held until work completes | Threads cannot forcibly kill native work |
| Oversized prompt | Huge JSON body/message | High | RAM/token consumption | Medium | 1000-character Pydantic bound | Raw body limit before parsing | Reverse proxy must bound headers/connections |
| Oversized history | Repeated history input | Medium | Token/PI growth | Medium | History forbidden | Preserve rejection incl. empty history | No conversational memory supported |
| Excessive top_k | Large retrieval fanout | Medium | Compute/context growth | Medium | Strict 1–5 | Preserve tests; chat fixed top-3 | Direct Python callers are trusted code |
| Excessive output | Vendor huge/free-form response | Medium | RAM/cost/disclosure | Medium | 1024 output tokens; 8192-char read bound | Answer/schema/evidence limits | SDK may allocate before validation |
| Supply chain | Vulnerable package or model download | Medium | Runtime compromise | High | Package ranges; local vision loader | Optional audit, version inventory, controlled upgrade | No signed artifact provenance locally |
| Permissive CORS | Hostile browser origin | Conditional medium | Cross-origin API abuse | Medium | No CORS middleware | Explicit origin allowlist; no credentials wildcard | CORS is not authentication |
| Host header abuse | Forged Host/proxy headers | Conditional medium | Routing/URL abuse | Medium | No URL construction | Production host allowlist; ignore untrusted XFF | Proxy trust must be configured outside app |
| Verbose traceback | Unexpected app exception | Medium | Internal details | Medium | Route generic errors | Outer generic handler/request ID; debug off | Startup operator errors require restricted logs |
| Unauthorized access | Public endpoint without auth | High if exposed | Resource/cost theft | High | Development-only assumptions | Production startup requires token; dev opt-out | OAuth/IdP recommended for real users |
| SSRF | Future URL fetch feature | Not currently reachable | Internal network access | High conditional | No URL-fetch API | Keep URLs unexecuted; future egress allowlist | Reassess before adding fetch tools |
| Metadata manipulation | Wrong/conflicting class labels | Medium | Wrong source attribution | High | Mapping validated at load | Reject conflicting identity labels; prompt source checks | Authenticity of upstream catalog not proven |
| Invented source ID | LLM returns nonexistent/unused ID | High | Hallucinated attribution | High | Per-request ID allowlist | Bounds, duplicate check, exact evidence verification | Source truth still relies on data provenance |
| Unicode/homoglyph | Obfuscated instruction tokens | High | Guard bypass | Medium | NFKC query normalization | Separate inspection normalization; preserve original cultural text | Cannot enumerate all obfuscations |
| Encoded injection | Base64/percent/escaped commands | Medium | Guard bypass | Medium | No decode/execute output | Bounded inspection and conservative rejection | Detection imperfect; no-tools/schema remain essential |
| Log injection | CR/LF in query/header/request ID | High | Forged log entries | Medium | Query not intentionally logged | Generated request ID, fixed fields, control stripping | External access logs need same policy |
| Retry storm | Multiple SDK/app retry layers | Medium | Quota/availability loss | High | SDK attempts=1; bounded retry | Deadline, jitter, no quota retries, cooldown | Remote request may be billed after cancellation |
| Quota exhaustion | Repeated valid chat requests | High if public | Service interruption | High | Catalog avoids LLM | Rate/admission limits; token budget | Provider quota must be enforced externally |
| Denial of wallet | Valid but expensive generation | High if public | Financial loss | High | Small context and output | Max attempts/tokens; documented cost envelope | Monetary cap depends on provider pricing/budget |
| State races | Shared limiter/breaker updated concurrently | Medium | Limit bypass/deadlock | Medium | No prior global guard state | Per-instance locks/injection/reset; cancellation tests | Multi-process shared backend deferred |
| Cache poisoning | Future shared response cache | Not currently reachable | Cross-user incorrect data | Medium conditional | No cache | Keep cache absent; scope/identity keys if added | Reassess before implementation |

No confirmed Critical exploitation was established locally. Critical tool-abuse
impact is conditional on adding an execution capability that does not exist.

## Embedding loader boundary

Production forces SentenceTransformer local-only and disables remote code; a
missing or invalid cached model fails startup with a safe ResourceValidationError.
Development/Colab may bootstrap the unchanged fixed identifier from the Hub,
unless WAYANG_EMBEDDING_LOCAL_ONLY=true. WAYANG_EMBEDDING_REVISION is operator-only;
a commit must be verified against the original Chroma embeddings before pinning.
Requests cannot select model identity, revision, trust flags or cache directory.
Existing application-owned local directories load locally without Hub fallback.
Colab must pre-download/cache the matching model before production mode.
Cache/local model permissions and verified fingerprints remain necessary against
tampering; local-only does not fix vulnerable dependencies. Existing VEX decisions
for ChromaDB/Transformers remain accepted/deferred as documented.

## Design references

- [OWASP prompt injection prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)
- [OWASP RAG security](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html)
- [Starlette middleware](https://www.starlette.io/middleware/)
- [Official google-genai SDK](https://github.com/googleapis/python-genai)
