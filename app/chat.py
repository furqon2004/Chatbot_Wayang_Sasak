from __future__ import annotations

import asyncio
import json
import logging
import time
import math
import re
import random

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .chat_schemas import ChatResponse, ChatSource, EvidenceSelection, ConversationContext, WebSource
from .provider import ProviderFailure
from .schemas import CatalogSearch
from .query_normalizer import classify_intent, unsafe_evidence, normalize_query, general_knowledge_intent, local_overview_intent
from .semantic import RetrievalUnavailable

LOGGER = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 6000
CATALOG_SCOPE = ("Aplikasi ini menyediakan profil rinci dan pengenalan gambar untuk 20 tokoh Wayang Sasak. "
                 "Angka 20 adalah cakupan katalog aplikasi, bukan jumlah seluruh tokoh dalam tradisi Wayang Sasak.")


def overview_context(context, topic):
    """Require direct statements, never infer cultural history from a character.

    Conservatively retain complete, directly relevant sentences from verified
    descriptions. Keyword matches (including Menak Senduk) are not evidence.
    """
    rows = []
    for row in context:
        if row["evidence_id"].split(":")[2] not in {"deskripsi", "deskripsi_lengkap"}:
            continue
        sentences = re.findall(r"(?:^|(?<=[.!?])\s+)([^.!?]+[.!?])(?=\s|$)", display_excerpt(row["text"]))
        for sentence in sentences:
            value = normalize_query(sentence)
            definition = re.match(r"^wayang sasak (?:adalah|merupakan) (?:seni|kesenian|tradisi|pertunjukan|wayang)\b", value)
            relation = re.match(r"^wayang sasak (?:menggunakan|mengangkat|bersumber pada|bersumber dari) (?:cerita|kisah|serat|epos) menak\b", value)
            if (topic == "overview" and definition) or (topic == "menak" and relation):
                rows.append({**row, "text": sentence.strip()})
                break
    return rows


def grounding_fallback(status="insufficient_evidence"):
    limitation = {
        "not_configured": "Pencarian web belum tersedia saat ini.",
        "insufficient_evidence": "Sumber yang tersedia belum cukup untuk menjawabnya dengan yakin.",
    }.get(status, "Pencarian web belum dapat diverifikasi saat ini.")
    return ChatResponse(mode="conversation", llm_status=status,
        answer="Aku belum menemukan sumber terverifikasi yang cukup untuk menjawab pertanyaan Wayang Sasak itu. "
               f"{limitation} Aku tidak ingin mengarang; coba tanyakan hal yang lebih spesifik tentang tokoh, "
               "atau coba lagi nanti.",
        sources=[], conversation_context=ConversationContext(domain="wayang_sasak"))


def overview_fallback(status):
    return ChatResponse(
        mode="web_answer",
        answer=("Wayang Sasak adalah seni pertunjukan wayang kulit yang tumbuh di Lombok. Dalang "
                "menggerakkan tokoh-tokohnya sambil menguasai gending, tembang, dan dialog yang mengikuti cerita. "
                "Lakonnya bersumber dari Serat Menak; dalam tradisi ini, ‘menak’ juga dijelaskan sebagai orang "
                "yang menang melawan hawa nafsu. Tokoh-tokohnya dipandang sebagai lambang yang dapat dikaji, "
                "jadi pertunjukannya membawa pesan selain hiburan. [1]"),
        sources=[], llm_status=status,
        web_sources=[WebSource(
            title="Yayasan Pedalangan Wayang Sasak — Multi Talenta, Kini Bertahan Hidup dari Jual Wayang",
            url="https://wayangsasak.org/baca/berita/26/Multi-Talenta,-Kini-Bertahan-Hidup-dari-Jual-Wayang",
        )],
    )


def evidence_context(results, catalog=None) -> list[dict]:
    context = []
    size = 2  # JSON array brackets
    for hit in results:
        if (not all(math.isfinite(getattr(hit, key)) and 0 <= getattr(hit, key) <= 1
                    for key in ("score", "semantic_score", "lexical_score"))
                or unsafe_evidence(hit.nama) or len(hit.nama) > 120):
            continue
        for evidence_id, text in zip(hit.evidence_ids, hit.evidence):
            if len(context) >= 12 or not text.strip() or len(text) > 500 or unsafe_evidence(text):
                continue
            if any(row["evidence_id"] == evidence_id for row in context):
                continue
            if catalog is not None:
                try:
                    prefix, class_id, field, index = evidence_id.split(":")
                    if field not in {"deskripsi", "deskripsi_lengkap", "filosofi", "ciri_ciri", "kata_kunci"}:
                        continue
                    raw = catalog.profiles[hit.id][field]
                    values = raw if isinstance(raw, list) else [raw]
                    original = values[int(index)]
                    if (prefix != "profile" or int(class_id) != hit.id or int(index) < 0
                            or hit.label != catalog.labels[hit.id] or hit.nama != catalog.items[hit.id].nama
                            or not isinstance(original, str) or unsafe_evidence(original)
                            or original[:500] != text):
                        continue
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
            row = {"evidence_id": evidence_id, "id": hit.id, "label": hit.label,
                   "nama": hit.nama, "text": text}
            row_size = len(json.dumps(row, ensure_ascii=False)) + (2 if context else 0)
            if size + row_size > MAX_CONTEXT_CHARS:
                continue
            size += row_size
            context.append(row)
    return context


def catalog_reply(result: CatalogSearch, status="not_needed") -> ChatResponse:
    # Tidak ada teks Provider pada mode katalog: nama tidak bisa ditambah/diganti.
    names = ", ".join(item.nama for item in result.catalog)
    if len(names) > 3000 or any(unsafe_evidence(item.nama) for item in result.catalog):
        raise ValueError("Katalog tidak aman")
    return ChatResponse(
        mode=result.mode, answer=f"{result.message}\n\n{names}.",
        sources=[ChatSource()], catalog=result.catalog,
        suggestions=result.suggestions, llm_status=status,
    )


def display_excerpt(text: str) -> str:
    """Render capped evidence without a possibly incomplete final word."""
    if len(text) < 500:
        return text.strip()
    clipped = text[:500]
    endings = list(re.finditer(r'[.!?](?:["\u201d\u2019]*)?(?=\s|$)', clipped))
    if endings:
        return clipped[:endings[-1].end()].strip()
    # At the evidence cap we cannot know whether the last token is complete.
    boundaries = list(re.finditer(r'\s+', clipped))
    return clipped[:boundaries[-1].start()].strip() if boundaries else ""


def rendered_evidence(rows: list[dict]) -> list[tuple[dict, str]]:
    """Keep the longest contained quote per character, with stable tie ordering."""
    candidates = [(index, row, display_excerpt(row["text"])) for index, row in enumerate(rows)]
    kept = []
    for index, row, text in sorted(candidates, key=lambda item: (
            -len(" ".join(item[2].split()).casefold()), item[0])):
        normalized = " ".join(text.split()).casefold()
        if not normalized:
            continue
        if any(row["id"] == other["id"] and re.search(
                r'(?<!\w)' + re.escape(normalized) + r'(?!\w)',
                " ".join(quote.split()).casefold()) for _, other, quote in kept):
            continue
        kept.append((index, row, text))
    return [(row, text) for _, row, text in sorted(kept, key=lambda item: item[0])]


def evidence_reply(rows: list[dict], status: str) -> ChatResponse:
    sources: dict[int, ChatSource] = {}
    paragraphs = []
    for row in rows:
        source = sources.setdefault(row["id"], ChatSource(id=row["id"], label=row["label"]))
        if row["evidence_id"] not in source.evidence_ids:
            source.evidence_ids.append(row["evidence_id"])
    for _, text in rendered_evidence(rows):
        paragraphs.append(text)
    return ChatResponse(
        mode="answer", answer="\n\n".join(paragraphs),
        sources=list(sources.values()), llm_used=status == "used", llm_status=status,
        suggestions=[f'Apa ciri-ciri {rows[0]["nama"]}?'],
    )


class ChatService:
    def __init__(self, retrieval, provider=None, timeout_seconds: float = 15, web_timeout_seconds: float = 90, clock=time.monotonic):
        from .rag_cache import RagCache
        self.rag_cache = RagCache(clock)
        self.retrieval = retrieval
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.web_timeout_seconds = web_timeout_seconds

    async def chat(self, message: str, *, awaiting_more=False, seen_ids=(), character_id=None, explain_more=False,
                   conversation_context=None) -> ChatResponse:
        from .conversation import named_ids, question_intent, reference_question, validated_context
        from .provider import REQUEST_METER
        catalog = self.retrieval.catalog
        original = message
        intent = question_intent(message)
        def clarify(text="Tokoh mana yang Anda maksud? Sebutkan satu tokoh atau kedua nama untuk perbandingan."):
            return ChatResponse(mode="conversation", answer=text, sources=[])
        # Inspect the original message before resolving any reference.
        if unsafe_evidence(message):
            return catalog_reply(catalog.response(message, "", True))
        if re.fullmatch(r"(?:bagaimana|gimana) cara (?:mengenali|ngenalin) tokoh(?:nya)?", normalize_query(message)):
            message += " dalam Wayang Sasak"
        # Inspect the original request before old context or UI identity can broaden it.
        from .role_search import role_question
        initial = classify_intent(message, catalog)
        role = role_question(message)
        contextual_role = role and (initial.domain_relevant or conversation_context is not None)
        comparison_requested = bool(re.search(r"\b(?:bandingkan|perbedaan|beda(?:nya)?|compare|versus|vs)\b", normalize_query(message)))
        contextual_comparison = (conversation_context is not None and comparison_requested
                                 and len(named_ids(message, catalog)) == 1)
        local_ack = normalize_query(message) in {
            "ya", "iya", "iya dong", "ya mau", "mau", "boleh", "boleh dong", "oke", "ok",
            "lanjut", "yuk", "tentu", "ada yang lain", "ada yang lainnya", "yang lain", "lainnya",
            "tidak", "nggak", "enggak", "tidak dulu", "nggak dulu", "cukup", "nanti"}
        vague_followup = bool(conversation_context is not None and re.fullmatch(
            r"(?:yang tadi|yg tadi|yang sebelumnya)(?: bagaimana| gimana)?", normalize_query(message)))
        if initial.intent == "unsupported" and not (reference_question(message) or contextual_role
                                                     or contextual_comparison or vague_followup
                                                     or (awaiting_more and local_ack)):
            reply = catalog_reply(catalog.response(message, "", True))
            reply.answer = ("Maaf, aku khusus membantu membahas Wayang Sasak. "
                            "Kamu bisa mengenal tokoh-tokohnya melalui katalog berikut! 🎭\n\n" + reply.answer)
            return reply
        scope_notice = initial.scope_notice
        if initial.intent == "mixed":
            message = original = initial.query
            intent = question_intent(message)
        try:
            previous, prior_evidence = validated_context(conversation_context, catalog)
        except (ValueError, KeyError, IndexError, TypeError):
            return clarify("Konteks percakapan tidak cocok dengan evidence lokal. Sebutkan kembali tokoh yang dimaksud.")
        if vague_followup:
            if previous.subject_ids:
                names = " dan ".join(catalog.items[i].nama for i in previous.subject_ids)
                return clarify(f"Bagian mana dari penjelasan {names} yang ingin kamu tahu—ciri-ciri, filosofi, atau cerita?")
            if previous.domain == "wayang_sasak":
                return clarify("Bagian mana tentang Wayang Sasak yang ingin kamu tanyakan?")
        explicit = named_ids(message, catalog)
        from .role_search import role_question, identify_role
        if (not explicit and initial.intent in {"allowed", "mixed"}
                and re.search(r"\b(?:paling kuat|terkuat|paling sakti|tersakti)\b", normalize_query(message))):
            reply = await identify_role(self.retrieval, "tokoh utama")
            if reply.character is not None:
                reply.opening = None
                reply.answer = (f"Jika yang Anda maksud tokoh utama, namanya {reply.character.nama}. [1] "
                    "Untuk menyebutnya tokoh paling kuat, saya perlu sumber yang membandingkan "
                    "kekuatannya dengan tokoh lain.")
                reply.follow_up = "Apakah Anda ingin mengetahui peran tokoh utama tersebut?"
            else:
                reply.answer = ("Tokoh utama belum tentu tokoh paling kuat. Saya belum memiliki sumber "
                    "yang cukup untuk membandingkan kekuatan para tokoh. "
                    "Anda ingin mengenal peran atau sifat tokoh tertentu?")
            reply.scope_notice = scope_notice
            return reply
        role = role_question(message) if not explicit else None
        if role:
            phrase, explicit_domain = role
            domain_known = (explicit_domain or previous.domain == "wayang_sasak" or bool(previous.subject_ids)
                            or bool(re.match(r"^(?:tokoh|karakter)\b", normalize_query(message))))
            inspection = message if explicit_domain else normalize_query(message) + " Wayang Sasak"
            decision = classify_intent(inspection, catalog)
            if decision.intent == "allowed":
                if not domain_known:
                    return clarify("Apakah yang Anda maksud tokoh dalam Wayang Sasak? Sebutkan konteks atau perannya.")
                reply = await identify_role(self.retrieval, phrase)
                if reply.llm_status == "insufficient_evidence":
                    reply = await self._web_answer(phrase + " dalam Wayang Sasak")
                    reply.conversation_context = ConversationContext(domain="wayang_sasak")
                return reply
        subjects = explicit or ([character_id] if character_id is not None else [])
        if intent == "comparison" and len(explicit) == 1 and previous.subject_ids:
            subjects = list(dict.fromkeys([*previous.subject_ids, *explicit]))
            if len(subjects) == 2:
                message = "Bandingkan " + " dan ".join(catalog.items[i].nama for i in subjects)
        if len(subjects) > 2:
            return clarify()
        if explicit and character_id is not None and (character_id not in explicit or len(explicit) == 2):
            character_id = None  # Latest explicit names take precedence over stale UI identity.
            explain_more = False
        if subjects and intent in {"philosophy", "traits"} and not explain_more:
            character_id = None  # A field question is not a profile-button request.
        if reference_question(message):
            subjects = subjects or previous.subject_ids
            query = normalize_query(message)
            if intent == "comparison" and len(subjects) < 2:
                return clarify("Tokoh mana yang ingin kamu bandingkan dengan tokoh tadi?")
            if not subjects and previous.domain == "wayang_sasak":
                if re.search(r"\b(?:sejarah|cerita|asal usul)", query):
                    message = "Sejarah Wayang Sasak"
                elif "mengenali" in query:
                    message = "Bagaimana cara mengenali tokoh Wayang Sasak?"
                else:
                    return clarify()
                character_id = None
                explain_more = False
            else:
                plural = bool(re.search(r"\b(?:mereka|keduanya)\b", query))
                if not subjects or (len(subjects) == 2 and not plural):
                    return clarify()
                names = " dan ".join(catalog.items[i].nama for i in subjects)
                prefix = {"philosophy": "Apa filosofi ", "traits": "Apa ciri-ciri ",
                          "comparison": "Bandingkan ", "simplify": "Jelaskan lebih sederhana "}.get(intent, "Jelaskan ")
                message = prefix + names
                if len(subjects) == 2 and intent != "comparison":
                    message = "Bandingkan " + message
                character_id = None
                explain_more = False
        if intent == "simplify" and (subjects or previous.subject_ids) and getattr(self.provider, "rag_provider", None) != "openai":
            return ChatResponse(mode="conversation", sources=[],
                answer="Saya belum dapat menyederhanakan atau memparafrasakan penjelasan. Saya dapat menampilkan kutipan lokal tentang deskripsi, filosofi, atau ciri tokoh.",
                conversation_context=previous if not explicit else ConversationContext(subject_ids=explicit, intent="simplify"))
        from .narrative import answer_style
        resolved = {"subject_ids": subjects, "intent": intent, "style": answer_style(original),
                    "evidence_intent": previous.intent if reference_question(original) else intent,
                    "previous_evidence": prior_evidence if not explicit else []}
        meter = {"count": 0}
        token = REQUEST_METER.set(meter)
        try:
            reply = await self._chat(message, awaiting_more=awaiting_more, seen_ids=seen_ids,
                character_id=character_id, explain_more=explain_more, original_message=original,
                resolved_context=resolved)
            reply.provider_requests = meter["count"]
            if scope_notice:
                reply.scope_notice = scope_notice
        finally:
            REQUEST_METER.reset(token)
        if reply.mode == "answer":
            ids = sorted({source.id for source in reply.sources if source.id is not None})
            if 0 < len(ids) <= 2 and not local_overview_intent(message):
                reply.subjects = [catalog.items[i] for i in ids]
                reply.conversation_context = ConversationContext(subject_ids=ids, intent=intent, domain="wayang_sasak",
                    evidence_ids=list(dict.fromkeys(e for source in reply.sources for e in source.evidence_ids))[:12])
                if len(ids) == 2:
                    reply.character = None
        elif reply.mode == "fallback_catalog" and subjects:
            reply.conversation_context = ConversationContext(subject_ids=subjects, intent=intent, domain="wayang_sasak")
        if reply.conversation_context is None and (local_overview_intent(message)
                or (general_knowledge_intent(message) and "wayang sasak" in normalize_query(message))
                or reply.mode == "web_answer" or reply.mode == "catalog"):
            reply.conversation_context = ConversationContext(domain="wayang_sasak")
        return reply

    async def _chat(self, message: str, *, awaiting_more=False, seen_ids=(), character_id=None, explain_more=False,
                    original_message=None, resolved_context=None) -> ChatResponse:
        normalized = normalize_query(message).strip(" .!?")
        if awaiting_more and normalized in {"ya", "iya", "iya dong", "ya mau", "mau", "boleh", "boleh dong", "oke", "ok", "lanjut", "yuk", "tentu", "ada yang lain", "ada yang lainnya", "yang lain", "lainnya"}:
            candidates = [item for item in self.retrieval.catalog.items if item.id not in seen_ids]
            if len(candidates) < 3:
                candidates += [item for item in self.retrieval.catalog.items if item.id in seen_ids]
            choices = random.SystemRandom().sample(candidates, min(3, len(candidates)))
            return ChatResponse(mode="suggestions", answer="Mari kenali tokoh lainnya. Pilih salah satu dari tiga tokoh berikut:",
                                sources=[], choices=choices)
        if awaiting_more and normalized in {"tidak", "nggak", "enggak", "tidak dulu", "nggak dulu", "cukup", "nanti"}:
            return ChatResponse(mode="conversation", answer="Baik, kapan pun Anda ingin melanjutkan, tulis pertanyaan atau kirim gambar wayang di sini.", sources=[])
        decision = classify_intent(message, self.retrieval.catalog)
        if decision.intent in {"unsupported", "malicious"}:
            return catalog_reply(self.retrieval.catalog.response(message, "", True))
        if character_id is not None:
            if explain_more:
                return await self._answer(message, character_id=character_id, original_message=original_message, resolved_context=resolved_context)
            return self.character_profile(character_id)
        query = normalize_query(decision.query)
        count_question = bool(re.search(r"\b(?:jumlah|berapa|banyak|total)\b", query) and
                              re.search(r"\b(?:tokoh|karakter|wayang)\b", query))
        local_count = re.fullmatch(
            r"(?:ada )?(?:berapa(?: jumlah| banyak)?|jumlah|total) (?:tokoh|karakter)"
            r"(?: wayang sasak)?(?: (?:di|dalam) (?:sistem|aplikasi|katalog|database)(?: ini)?)?(?: yang ada)?", query)
        if local_count and re.search(r"\b(?:sistem|aplikasi|katalog|database)\b", query):
            return ChatResponse(mode="conversation", answer=CATALOG_SCOPE, sources=[ChatSource()],
                                follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")
        story_question = bool(re.search(r"\b(?:sejarah|cerita|kisah|lakon|asal usul)\b", query))
        named_character = any(re.search(r"(?<!\w)" + re.escape(normalize_query(name)) + r"(?!\w)", query)
                              for item in self.retrieval.catalog.items for name in (item.nama, item.label))
        topic = local_overview_intent(decision.query)
        family_question = bool(re.search(
            r"\b(?:istri|isteri|suami|pasangan|menikah|pernikahan|ayah|ibu|anak|orang tua|saudara|keturunan)(?:nya)?\b", query))
        if named_character and family_question:
            reply = await self._web_answer(decision.query + " dalam konteks Wayang Sasak")
            if decision.scope_notice:
                reply.scope_notice = decision.scope_notice
            return reply
        if not named_character and topic:
            reply = await self._answer(decision.query, original_message=original_message or message, overview_topic=topic, resolved_context=resolved_context)
            if decision.scope_notice:
                reply.scope_notice = decision.scope_notice
            return reply
        if not named_character and (count_question or re.search(r"\binternet\b", query)):
            reply = await self._web_answer(decision.query)
            if decision.scope_notice:
                reply.scope_notice = decision.scope_notice
            return reply
        if not named_character and (general_knowledge_intent(query) or story_question
                                    or (decision.domain_relevant and decision.semantic_intent != "CATALOG_REQUEST")):
            reply = await self._answer(decision.query, original_message=original_message or message,
                                       overview_topic="general", resolved_context=resolved_context)
            if decision.scope_notice:
                reply.scope_notice = decision.scope_notice
            return reply
        reply = await self._answer(decision.query, decision.comparison_ids, original_message=original_message or message, resolved_context=resolved_context)
        reply.scope_notice = decision.scope_notice
        return reply

    def character_profile(self, character_id):
        catalog = self.retrieval.catalog
        item = catalog.items[character_id]
        profile = catalog.profiles[character_id]
        description = "deskripsi_lengkap" if profile.get("deskripsi_lengkap") else "deskripsi"
        paragraphs, evidence_ids = [], []
        for field in (description, "filosofi", "ciri_ciri"):
            text = profile.get(field)
            if isinstance(text, str) and text.strip() and not unsafe_evidence(text):
                if text.strip() not in paragraphs:
                    paragraphs.append(text.strip())
                evidence_ids.append(f"profile:{character_id}:{field}:0")
        if not paragraphs:
            return catalog_reply(catalog.response(item.nama, "", True))
        return ChatResponse(mode="answer", answer="\n\n".join(paragraphs), character=item,
            sources=[ChatSource(id=item.id, label=item.label, evidence_ids=evidence_ids)],
            llm_status="not_needed", follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")

    async def _web_answer(self, message):
        decision = classify_intent(message, self.retrieval.catalog)
        if decision.intent not in {"allowed", "mixed"}:
            return catalog_reply(self.retrieval.catalog.response(message, "", True))
        message = decision.query
        if (re.search(r"\b(?:serat|cerita) menak\b", normalize_query(message))
                and "wayang sasak" not in normalize_query(message)):
            message += " dalam konteks Wayang Sasak"
        status = "not_configured"
        if self.provider is not None and callable(getattr(self.provider, "search", None)):
            try:
                answer, sources, suggestions_html = await asyncio.wait_for(self.provider.search(message), timeout=self.web_timeout_seconds)
                if not sources:
                    raise ProviderFailure("insufficient_evidence")
                return ChatResponse(mode="web_answer", answer=answer, sources=[], web_sources=sources,
                                    search_suggestions_html=suggestions_html, llm_used=True, llm_status="used",
                                    scope_notice=decision.scope_notice or CATALOG_SCOPE,
                                    follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")
            except ProviderFailure as exc:
                status = exc.category
            except asyncio.TimeoutError:
                status = "timeout"
            except Exception:
                status = "internal"
        LOGGER.warning("Web chat failed: category=%s", status)
        topic = local_overview_intent(message)
        if topic == "overview":
            return overview_fallback(status)
        if topic == "menak":
            return grounding_fallback(status)
        return ChatResponse(mode="fallback_catalog",
            answer="Informasi internet belum dapat diverifikasi saat ini. Basis pengetahuan Wayang AI memuat 20 tokoh Wayang Sasak yang dapat Anda jelajahi.",
            sources=[ChatSource()], catalog=self.retrieval.catalog.items,
            llm_status=status, follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")

    async def _answer(self, message: str, comparison_ids=frozenset(), *, character_id=None, original_message=None, overview_topic=None, resolved_context=None) -> ChatResponse:
        retrieval_query = "Apa itu Wayang Sasak" if overview_topic == "overview" else message
        narrative = getattr(self.provider, "rag_provider", None) == "openai"
        resolved = resolved_context or {}
        def requested_local_profile():
            from .conversation import named_ids
            ids = named_ids(original_message or message, self.retrieval.catalog)
            if len(ids) != 1 or comparison_ids or overview_topic:
                return None
            item = self.retrieval.catalog.items[ids[0]]
            text = normalize_query(original_message or message)
            for name in sorted((item.nama, item.label), key=len, reverse=True):
                text = re.sub(r"(?<!\w)" + re.escape(normalize_query(name)) + r"(?!\w)", "SUBJEK", text)
            if re.fullmatch(
                r"(?:(?:aku|saya) (?:pengen|ingin|mau) (?:tahu|tau|mengenal)(?: lebih banyak)?(?: tentang)?|"
                r"(?:tolong |coba )?(?:jelaskan|ceritakan|kenalkan)(?: saya| aku)?(?: tentang)?|"
                r"siapa(?:kah)?|profil(?: tokoh)?)(?: tokoh)? SUBJEK(?: dong| ya)?", text):
                return self.character_profile(ids[0])
            return None
        if narrative and resolved.get("style") in {"simplify", "summary", "detail"} and resolved.get("subject_ids"):
            names = " dan ".join(self.retrieval.catalog.items[i].nama for i in resolved["subject_ids"])
            prefix = {"philosophy": "Apa filosofi ", "traits": "Apa ciri-ciri "}.get(resolved.get("evidence_intent"), "Jelaskan ")
            retrieval_query = ("Bandingkan " if len(resolved["subject_ids"]) == 2 else "") + prefix + names
        if character_id is not None:
            from .conversation import question_intent
            prefix = {"philosophy": "Apa filosofi ", "traits": "Apa ciri-ciri "}.get(question_intent(message), "Jelaskan ")
            retrieval_query = prefix + self.retrieval.catalog.items[character_id].nama
        try:
            result = await run_in_threadpool(self.retrieval.search, retrieval_query, 3)
        except RetrievalUnavailable:
            local = requested_local_profile()
            if local is not None:
                return local
            if overview_topic:
                return await self._web_answer(message)
            raise
        if isinstance(result, CatalogSearch):
            local = requested_local_profile()
            if local is not None:
                return local
            if overview_topic == "general":
                return await self._web_answer(message)
            if overview_topic:
                return await self._web_answer(message)
            if narrative:
                from .narrative import narrative_fallback
                return narrative_fallback(self.retrieval.catalog)
            return catalog_reply(result)
        hits = [hit for hit in result.results if math.isfinite(hit.score) and hit.score >= self.retrieval.threshold]
        if comparison_ids:
            # Interleave before applying global budgets so neither subject is starved.
            groups = [evidence_context([hit], self.retrieval.catalog) for hit in hits if hit.id in comparison_ids]
            context = [rows[i] for i in range(3) for rows in groups if i < len(rows)]
            budget, kept = 2, []
            for row in context:
                size = len(json.dumps(row, ensure_ascii=False)) + 2
                if len(kept) < 12 and budget + size <= MAX_CONTEXT_CHARS:
                    kept.append(row)
                    budget += size
            context = kept
        else:
            context = evidence_context(hits, self.retrieval.catalog)
        from .conversation import question_intent
        field = {"philosophy": "filosofi", "traits": "ciri_ciri"}.get(question_intent(message))
        if field:
            context = [row for row in context if row["evidence_id"].split(":")[2] == field]
        if character_id is not None:
            context = [row for row in context if row["id"] == character_id]
        if narrative and resolved.get("subject_ids"):
            context = [row for row in context if row["id"] in resolved["subject_ids"]]
        if narrative and resolved.get("style") in {"simplify", "summary"} and resolved.get("previous_evidence"):
            context = []
            for prior in resolved["previous_evidence"]:
                identity = int(prior["evidence_id"].split(":")[1])
                if identity in resolved.get("subject_ids", []):
                    item = self.retrieval.catalog.items[identity]
                    excerpt = display_excerpt(prior["text"][:500])
                    if excerpt:
                        context.append({"evidence_id": prior["evidence_id"], "id": identity,
                            "label": item.label, "nama": item.nama, "text": excerpt})
        if overview_topic in {"overview", "menak"}:
            context = overview_context(context, overview_topic)
            if not context:
                return await self._web_answer(message)
        if comparison_ids and not comparison_ids <= {row["id"] for row in context}:
            context = []
        if not context:
            local = requested_local_profile()
            if local is not None:
                return local
            if overview_topic:
                return await self._web_answer(message)
            if narrative:
                from .narrative import narrative_fallback
                return narrative_fallback(self.retrieval.catalog)
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True),
                                 "insufficient_evidence" if field or comparison_ids else "not_needed")
        cache_key = self.rag_cache.key(original_message or message, message, context,
            character_id, comparison_ids, self.provider, resolved_context)
        cached = self.rag_cache.get(cache_key)
        if cached is not None:
            cached.cache_hit = True
            cached.provider_requests = 0
            return cached
        if narrative:
            from .narrative import render_narrative, narrative_fallback, answer_style
            def failure_reply(status):
                fallback = narrative_fallback(self.retrieval.catalog, status)
                identities = resolved.get("subject_ids", [])
                if (len(identities) == 1 and not field
                        and status not in {"insufficient_evidence", "not_configured"}):
                    local = self.character_profile(identities[0])
                    if local.mode == "answer":
                        local.answer = fallback.answer + "\n\nBerikut informasi profil lokal:\n\n" + local.answer
                        local.llm_status = status
                        return local
                return fallback
            try:
                narrative_context = {**resolved, "style": answer_style(original_message or message)}
                raw = await asyncio.wait_for(self.provider.narrate(message, context, narrative_context), self.timeout_seconds)
                reply = render_narrative(raw, context, self.retrieval.catalog,
                    resolved.get("subject_ids", []), comparison=bool(comparison_ids))
                if overview_topic:
                    reply.character = None
                if reply.mode == "answer":
                    self.rag_cache.put(cache_key, reply)
                elif overview_topic:
                    return await self._web_answer(message)
                return reply
            except asyncio.TimeoutError:
                if overview_topic:
                    return await self._web_answer(message)
                return failure_reply("timeout")
            except ProviderFailure as exc:
                if overview_topic:
                    return await self._web_answer(message)
                return failure_reply(exc.category)
            except (ValueError, TypeError):
                if overview_topic:
                    return await self._web_answer(message)
                return failure_reply("invalid_response")
            except Exception:
                if overview_topic:
                    return await self._web_answer(message)
                return failure_reply("internal")
        status = "not_configured"
        follow_up = "Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?"
        selected = context[:3]
        if comparison_ids and not comparison_ids <= {row["id"] for row in selected}:
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True))
        if self.provider is not None:
            started = time.perf_counter()
            try:
                raw = await asyncio.wait_for(self.provider.select(message, context), timeout=self.timeout_seconds)
                if not isinstance(raw, str) or len(raw) > 8192:
                    raise ValueError("Respons malformed")
                selection = EvidenceSelection.model_validate_json(raw)
                by_id = {row["evidence_id"]: row for row in context}
                if len(selection.evidence_ids) != len(set(selection.evidence_ids)) or any(
                    evidence_id not in by_id for evidence_id in selection.evidence_ids
                ):
                    raise ValueError("ID evidence tidak valid")
                if not selection.evidence_ids:
                    status = "insufficient_evidence"
                    if overview_topic:
                        return await self._web_answer(message)
                    return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True), "insufficient_evidence")
                selected = [by_id[evidence_id] for evidence_id in selection.evidence_ids]
                follow_up = selection.follow_up
                if comparison_ids and not comparison_ids <= {row["id"] for row in selected}:
                    return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True), "insufficient_evidence")
                status = "used"
            except asyncio.TimeoutError:
                status = "timeout"
            except ProviderFailure as exc:
                status = exc.category
            except (ValidationError, ValueError):
                status = "invalid_response"
            except Exception:
                status = "internal"
            finally:
                LOGGER.info("Chat Provider selesai: category=%s duration=%.3f", status, time.perf_counter() - started)
        if status not in {"used", "not_configured"}:
            if overview_topic:
                return await self._web_answer(message)
            # Provider failure must never turn an unverified profile into a specific answer.
            return ChatResponse(mode="fallback_catalog", answer="Penjelasan belum dapat diverifikasi karena penyusun jawaban gagal. Silakan coba lagi atau pilih profil tokoh.",
                sources=[], catalog=self.retrieval.catalog.items, llm_status=status)
        if not rendered_evidence(selected):
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True),
                                 "not_needed" if self.provider is None else "insufficient_evidence")
        reply = evidence_reply(selected, status)
        if not overview_topic and not comparison_ids:
            reply.character = self.retrieval.catalog.items[selected[0]["id"]]
        reply.follow_up = follow_up
        reply.suggestions = []
        if status == "used":
            self.rag_cache.put(cache_key, reply)
        return reply
