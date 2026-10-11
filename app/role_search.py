"""Conservative identification from explicit profile assertions, never scores alone."""
import math
import re

from starlette.concurrency import run_in_threadpool

from .chat_schemas import ChatResponse, ChatSource, ConversationContext
from .query_normalizer import normalize_query, unsafe_evidence
from .semantic import RetrievalUnavailable


def role_question(message):
    query = normalize_query(message)
    if re.search(r"\b(?:internet|google|web|terbaru)\b", query):
        return None
    match = re.fullmatch(r"(?:siapa(?:kah)?(?: tokoh)?|tokoh mana)(?: yang)? (.+)", query)
    if not match:
        return None
    phrase = match[1]
    explicit_domain = "wayang sasak" in phrase or "pewayangan sasak" in phrase
    phrase = re.sub(r"(?:dalam |di |pada )?(?:pewayangan sasak|wayang sasak)$", "", phrase).strip()
    phrase = re.sub(r"^(?:berperan sebagai|menjadi|bersifat|dikenal sebagai) ", "", phrase)
    if query.startswith(("siapa tokoh ", "siapakah tokoh ")) and phrase.startswith(("utama", "utamanya")):
        phrase = "tokoh " + phrase
    phrase = re.sub(r"nya$", "", phrase)
    if not phrase or len(phrase) > 120 or phrase in {"saja", "ini", "itu", "dia", "mereka"}:
        return None
    return phrase, explicit_domain


def supporting_quote(profile, name, phrase, names):
    """Keep only complete assertions whose grammatical subject is this profile.

    This deliberately rejects negations, hypotheticals, and references to other
    named characters before the matching phrase. Unrecognized wording falls back.
    """
    for field in ("deskripsi", "deskripsi_lengkap"):
        raw = profile.get(field)
        if not isinstance(raw, str) or unsafe_evidence(raw):
            continue
        for sentence in re.findall(r"(?:^|(?<=[.!?])\s+)([^.!?]+[.])(?=\s|$)", raw):
            value = normalize_query(sentence)
            prefix = normalize_query(name)
            if not re.match(re.escape(prefix) + r" (?:merupakan|adalah|dikenal|memiliki|memakai|membawa|berperan|menjadi|bersifat)\b", value):
                continue
            if re.search(r"\b(?:bukan|tidak|belum|tanpa|mungkin|konon|jika|seandainya|diduga)\b", value):
                continue
            body = value[len(prefix):].strip()
            match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", body)
            if not match or len(sentence.strip()) > 500:
                continue
            before = body[:match.start()].strip()
            # Only a direct predicate can establish identity. A matching phrase
            # inside an object (e.g. "pendamping tokoh utama") proves no such role.
            if not re.fullmatch(
                r"(?:(?:merupakan|adalah|menjadi)(?: seorang| sosok| tokoh)?(?: yang)? ?)?"
                r"(?:(?:dikenal|digambarkan)(?: sebagai| dengan sifat)? ?|"
                r"berperan sebagai ?|bersifat ?|memiliki sifat ?)?"
                r"(?:(?:seorang |sosok )?tokoh yang)?", before):
                continue
            if any(re.search(r"(?<!\w)" + re.escape(normalize_query(other)) + r"(?!\w)", before) for other in names):
                continue
            return field, sentence.strip()
    return None


async def identify_role(retrieval, phrase):
    catalog = retrieval.catalog
    def fallback():
        return ChatResponse(mode="fallback_catalog", answer="Evidence deskripsi lokal belum cukup untuk menetapkan tokoh dengan peran atau sifat tersebut.",
            sources=[], catalog=catalog.items, llm_status="insufficient_evidence",
            conversation_context=ConversationContext(domain="wayang_sasak"))
    try:
        # rank returns the full catalog: do not infer uniqueness from a truncated top 3.
        hits = await run_in_threadpool(retrieval.rank, normalize_query(phrase + " Wayang Sasak"), 5)
    except RetrievalUnavailable:
        return fallback()
    matches = []
    for hit in hits:
        if (not all(math.isfinite(getattr(hit, key)) and 0 <= getattr(hit, key) <= 1
                    for key in ("score", "semantic_score", "lexical_score")) or hit.score < retrieval.threshold):
            continue
        if hit.id not in catalog.profiles or hit.nama != catalog.items[hit.id].nama or hit.label != catalog.labels[hit.id]:
            continue
        item = catalog.items[hit.id]
        if unsafe_evidence(item.nama) or len(item.nama) > 120:
            continue
        proof = supporting_quote(catalog.profiles[item.id], item.nama, phrase, [i.nama for i in catalog.items])
        if proof and item.id not in {i.id for i, _, _ in matches}:
            field, quote = proof
            matches.append((item, quote, ChatSource(id=item.id, label=item.label,
                evidence_ids=[f"profile:{item.id}:{field}:0"])))
    if not matches:
        return fallback()
    if len(matches) > 1:
        shown = matches[:3]
        return ChatResponse(mode="suggestions", answer=(f"Ada {len(matches)} tokoh dengan evidence yang cocok. "
            "Pilih tokoh yang Anda maksud. Berikut maksimal tiga pilihan:\n\n" +
            "\n\n".join(f"{item.nama}: {quote}" for item, quote, _ in shown)),
            choices=[item for item, _, _ in shown], sources=[source for _, _, source in shown],
            conversation_context=ConversationContext(domain="wayang_sasak"))
    item, quote, source = matches[0]
    opening = f"Berdasarkan deskripsi lokal, tokoh dengan peran atau sifat {phrase} adalah {item.nama}."
    return ChatResponse(mode="answer", opening=opening, answer=opening + "\n\n" + quote,
        character=item, subjects=[item], sources=[source],
        conversation_context=ConversationContext(subject_ids=[item.id], evidence_ids=source.evidence_ids, domain="wayang_sasak"),
        follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")
