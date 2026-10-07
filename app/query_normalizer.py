import re
import unicodedata
import html
from dataclasses import dataclass
from urllib.parse import unquote

STOPWORDS = frozenset("apa apakah siapa bagaimana jelaskan tolong coba berikan saya tentang dan atau dari yang ini itu dengan untuk adalah di ke tokoh karakter wayang sasak fungsi filosofi ciri cirinya deskripsi".split())

# Whole-token aliases only: preserve stored names and avoid arbitrary corrections.
QUERY_ALIASES = {
    "jlskn": "jelaskan", "jlsin": "jelaskan", "sy": "saya",
    "ttg": "tentang", "tntg": "tentang", "wyng": "wayang",
    "wyg": "wayang", "saksak": "sasak", "nurswan": "nursiwan",
}


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", query).casefold()
    text = "".join(char if char.isalnum() else " " for char in text)
    return " ".join(QUERY_ALIASES.get(token, token) for token in text.split())


def terms(query: str) -> set[str]:
    return {word for word in normalize_query(query).split() if word not in STOPWORDS}


def catalog_intent(normalized: str) -> bool:
    return bool(re.search(
        r"\b(?:contoh (?:tokoh|karakter)|daftar (?:tokoh|karakter)|siapa saja|"
        r"pilihan (?:tokoh|karakter)|(?:tokoh|karakter)(?: wayang)?(?: sasak)? apa saja)\b",
        normalized,
    ))


def unsupported_instruction(normalized: str) -> bool:
    return bool(re.search(
        r"(?:abaikan|ignore).*(?:aturan|instruksi|instruction|rules)|"
        r"(?:buat|ciptakan|karang|invent|create).*?(?:tokoh|karakter|character).*?(?:baru|new)|"
        r"(?:invent|create).*?new.*?character", normalized,
    ))


@dataclass(frozen=True)
class IntentDecision:
    domain_relevant: bool
    intent: str
    query: str = ""
    scope_notice: str | None = None
    comparison_ids: frozenset[int] = frozenset()


def inspection_text(value: str) -> str:
    """Inspection only: do not rewrite stored cultural names or public text."""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = html.unescape(unquote(unquote(text)))
    text = text.translate(str.maketrans({"а": "a", "е": "e", "і": "i", "о": "o",
                                        "р": "p", "с": "c", "х": "x", "ѕ": "s", "у": "y"}))
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def unsafe_evidence(value: str) -> bool:
    text = inspection_text(value)
    normalized = normalize_query(text)
    return bool(unsupported_instruction(normalized) or re.search(
        r"<[^>]+>|```|https?://|javascript:|data:|\\u[0-9a-f]{4}|"
        r"\b(?:base64|rot13|jailbreak|terminal|powershell|credential|credentials|password)\b|"
        r"\b(?:api[ _-]?key|gemini_api_key|system prompt|system instruction|prompt sistem)\b|"
        r"\b(?:system|developer|assistant)\s*:|"
        r"\b(?:abaikan|ignore|override|bypass|pretend|berpura|roleplay|role[- ]play)\b|"
        r"\b(?:hapus|delete|drop|ubah|read|baca|write|tulis|execute|jalankan)\b.*"
        r"\b(?:file|database|chromadb|env|source|code|kode|model|dataset|perintah)\b|"
        r"AIza[\w-]{35}|sk-[\w-]{20,}|gh[pousr]_[\w]{20,}|"
        r"/content/|[a-z]:[\\/]|\b(?:authorization|secret|cookie)\s*[:=]",
        text, re.I))


def classify_intent(query: str, catalog) -> IntentDecision:
    text = inspection_text(query)
    normalized = normalize_query(text)
    names = [normalize_query(item.nama) for item in catalog.items]
    names += [normalize_query(item.label) for item in catalog.items]
    def domain(part):
        value = normalize_query(part)
        return bool(re.search(r"\b(?:wayang|tokoh|karakter|filosofi|ciri|jayagrana|gununggan)\b", value)
                    or general_knowledge_intent(value)
                    or any(re.search(r"\b" + re.escape(name) + r"\b", value) for name in names))
    relevant = domain(query)
    if unsafe_evidence(query):
        return IntentDecision(relevant, "malicious")
    comparison = frozenset()
    if re.search(r"\b(?:bandingkan|perbedaan|beda|compare|comparison|versus|vs)\b", normalized):
        comparison = frozenset(item.id for item in catalog.items
            if normalize_query(item.nama) in normalized or normalize_query(item.label) in normalized)
        if len(comparison) < 2 or re.search(r"\b(?:bali|jawa|sunda)\b", normalized):
            return IntentDecision(relevant, "unsupported")
    clauses = re.split(r"[.;!?\n,]|\b(?:tetapi|tapi|sebelumnya|kemudian|lalu|but|before|then)\b", text)
    accepted, rejected = [], []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        inspected_clause = clause
        if (re.search(r"\b(?:berapa|jumlah|total|banyak)\b", clause)
                and re.search(r"\b(?:tokoh|karakter|wayang)\b", clause)
                and not re.search(r"\b(?:buat|buatkan|create|bangun)\b", clause)):
            inspected_clause = re.sub(r"\b(?:aplikasi|application)\b", "", clause)
        if (not domain(clause) or re.search(
            r"\b(?:website|aplikasi|application|kode|code|html|script|python|sql|shell|javascript|"
            r"source|presiden|bitcoin|program)\b", inspected_clause)):
            rejected.append(clause)
        else:
            accepted.append(clause)
    if not accepted:
        return IntentDecision(relevant, "unsupported")
    if rejected:
        return IntentDecision(relevant, "mixed", ". ".join(accepted),
            "Instruksi di luar domain diabaikan; jawaban hanya berdasarkan evidence Wayang Sasak.", comparison)
    return IntentDecision(relevant, "allowed", query, comparison_ids=comparison)


def general_knowledge_intent(query: str) -> bool:
    value = normalize_query(query)
    standalone = {"sejarah", "sejarahnya", "jalan cerita", "alur cerita", "ceritanya", "asal usul"}
    if value in standalone:
        return True
    cultural = bool(re.search(r"\b(?:wayang|sasak|menak|selaparang|lombok)\b", value))
    topics = bool(re.search(r"\b(?:sejarah|asal|usul|cerita|ceritakan|ceritain|bercerita|kisah|lakon|epos|serat|babad|dalang|pertunjukan|jumlah|berapa|total|banyak|internet)\b", value))
    overview = bool(re.search(
        r"\b(?:apa itu|apa yang dimaksud dengan|jelaskan|penjelasan|informasi|mengenal)"
        r"(?: dong| saya| aku| kepada saya| kepada aku)?(?: tentang| mengenai)? wayang sasak\b",
        value,
    ))
    return cultural and (topics or overview)
