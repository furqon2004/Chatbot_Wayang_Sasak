"""Bounded references, resolved only against the server's catalog."""
import re

from .chat_schemas import ConversationContext
from .query_normalizer import normalize_query, unsafe_evidence


def named_ids(message, catalog):
    query = normalize_query(message)
    return [item.id for item in catalog.items if any(
        re.search(r"(?<!\w)" + re.escape(normalize_query(name)) + r"(?!\w)", query)
        for name in (item.nama, item.label))]


def question_intent(message):
    query = normalize_query(message)
    if "lebih sederhana" in query:
        return "simplify"
    if re.search(r"\bfilosofi(?:nya)?\b", query):
        return "philosophy"
    if re.search(r"\bciri(?:nya)?\b", query):
        return "traits"
    if re.search(r"\b(?:bandingkan|perbedaan|beda|compare|versus|vs)\b", query):
        return "comparison"
    return "profile"


def reference_question(message):
    return bool(re.fullmatch(
        r"(?:apa|jelaskan|bagaimana)(?: itu)? (?:filosofinya|cirinya|filosofi|ciri ciri|ciri)"
        r"(?: tokoh ini| tokoh tersebut| dia| mereka| keduanya)?|"
        r"(?:(?:coba|tolong|bisa|boleh) )?jelaskan(?: dong)? (?:lebih sederhana|lebih lanjut|lebih detail|lebih rinci|lebih terperinci|secara rinci|secara terperinci|secara ringkas)(?: lagi)?(?: dong| ya)?|ringkas(?: penjelasan(?:nya)?)?|bandingkan (?:mereka|keduanya)",
        normalize_query(message)))


def validated_context(value, catalog):
    if value is None:
        return ConversationContext(), []
    context = ConversationContext.model_validate(value)
    texts = []
    for evidence_id in context.evidence_ids:
        _, identity, field, index = evidence_id.split(":")
        identity, index = int(identity), int(index)
        if identity not in context.subject_ids:
            raise ValueError("Evidence di luar subjek")
        raw = catalog.profiles[identity][field]
        values = raw if isinstance(raw, list) else [raw]
        text = values[index]
        if not isinstance(text, str) or not text.strip() or unsafe_evidence(text):
            raise ValueError("Evidence tidak valid")
        texts.append({"evidence_id": evidence_id, "text": text})
    if context.evidence_ids and set(context.subject_ids) != {int(e.split(":")[1]) for e in context.evidence_ids}:
        raise ValueError("Evidence tidak mencakup subjek")
    return context, texts
