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
    if re.search(r"\b(?:ciri(?:nya)?|sifat(?:nya)?)\b", query):
        return "traits"
    if re.search(r"\b(?:bandingkan|perbedaan|beda(?:nya)?|compare|versus|vs)\b", query):
        return "comparison"
    return "profile"


def reference_question(message):
    query = normalize_query(message)
    return bool(re.fullmatch(
        r"(?:apa|jelaskan|bagaimana)(?: itu)? (?:filosofinya|cirinya|filosofi|ciri ciri|ciri)"
        r"(?: tokoh ini| tokoh tersebut| dia| mereka| keduanya)?|"
        r"(?:(?:coba|tolong|bisa|boleh) )?jelaskan(?: dong)? (?:lebih sederhana|lebih lanjut|lebih detail|lebih rinci|lebih terperinci|secara rinci|secara terperinci|secara ringkas)(?: lagi)?(?: dong| ya)?|ringkas(?: penjelasan(?:nya)?)?|bandingkan (?:mereka|keduanya)",
        query) or re.fullmatch(
        r"(?:(?:kalau|kalo|terus|lalu|yang tadi) )?(?:apa )?(?:sifat|ciri(?: ciri)?)(?:nya| dia| tokoh ini| tokoh tadi)?"
        r"(?: bagaimana| gimana| seperti apa)?", query) or re.fullmatch(
        r"(?:bagaimana|gimana) cara (?:mengenali|ngenalin) (?:tokoh(?:nya)?|tokoh wayang sasak)", query) or re.fullmatch(
        r"(?:sejarah|cerita|asal usul)(?:nya)(?: bagaimana| gimana)?", query) or re.fullmatch(
        r"(?:filosofi(?:nya)?|ciri(?: ciri)?(?:nya)?|sifat(?:nya)?) (?:apa|bagaimana|gimana)", query) or re.fullmatch(
        r"asal(?:nya)? dari mana|(?:cerita|sejarah)(?:nya)? (?:bagaimana|gimana)|ceritakan(?: dong)?", query) or re.fullmatch(
        r"(?:beda|bedanya) (?:sama|dengan) (?:tokoh )?(?:tadi|sebelumnya) (?:apa)", query))


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
