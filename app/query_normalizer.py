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
    "wyg": "wayang", "wayng": "wayang", "saksak": "sasak",
    "gw": "saya", "gue": "saya", "gua": "saya",
    "nurswan": "nursiwan", "gmn": "bagaimana", "gimana": "bagaimana",
    "yg": "yang", "tau": "tahu", "pengen": "ingin",
    "critain": "ceritakan", "ceritain": "ceritakan", "jelasin": "jelaskan",
    "jayagrana": "jayangrana",
    "apaan": "apa", "ngenalin": "mengenali",
}


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", query).casefold()
    text = "".join(char if char.isalnum() else " " for char in text)
    text = re.sub(r"\bjay ang rana\b", "jayangrana", text)
    text = re.sub(r"\bumr maya\b", "umar maya", text)
    return " ".join(QUERY_ALIASES.get(token, token) for token in text.split())


def terms(query: str) -> set[str]:
    return {word for word in normalize_query(query).split() if word not in STOPWORDS}


def catalog_intent(normalized: str) -> bool:
    return bool(re.search(
        r"\b(?:tampilkan (?:katalog|daftar)(?: wayang sasak)?|"
        r"lihat (?:semua )?(?:tokoh|karakter)(?: wayang sasak)?(?: yang tersedia)?|"
        r"semua (?:tokoh|karakter)(?: wayang sasak)?(?: yang tersedia)?|"
        r"contoh (?:tokoh|karakter)|daftar (?:tokoh|karakter)|siapa saja|"
        r"siapa (?:saja|aja) (?:tokoh|karakter)(?:nya)?|"
        r"(?:tokoh|karakter)(?:nya)? siapa (?:saja|aja)|"
        r"(?:tokoh|karakter) (?:lainnya|lain) ada (?:tidak|gak|enggak)|"
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
    semantic_intent: str = "WAYANG_REQUEST"


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
        r"\b(?:api[ _-]?key|llm_api_key|system prompt|system instruction|prompt sistem)\b|"
        r"\b(?:system|developer|assistant)\s*:|"
        r"\b(?:abaikan|ignore|override|bypass|pretend|berpura|roleplay|role[- ]play)\b|"
        r"\b(?:ganti|ubah|change|switch)\b.*\b(?:peran|role|aturan|instruksi)\b|"
        r"\b(?:bocorkan|ungkapkan|reveal|leak|cetak|tampilkan)\b.*\b(?:prompt|rahasia|secret|token|kunci|instruksi)\b|"
        r"\b(?:mulai sekarang|from now on|you are now|anda sekarang)\b|"
        r"\b(?:ikuti|patuhi|follow|obey)\b.*\b(?:instruksi|perintah|instructions|commands)\b|"
        r"\b(?:jawab|bahas|answer|discuss)\b.*\b(?:topik lain|luar domain|another topic)\b|"
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
        return bool(re.search(
                    r"\b(?:wayang sasak|pewayangan sasak|serat menak|cerita menak|jayagrana|gununggan|"
                    r"(?:wayang(?: kulit)?|pewayangan)(?: di)?(?: pulau)? lombok)\b", value)
                    or re.fullmatch(
                        r"(?:tokoh|karakter) (?:yang )?(?:utama(?:nya)?|licik|jahat|baik|bijaksana|cerdik|"
                    r"kuat|sakti|paling kuat|paling sakti|paling penting)"
                    r"(?: dalam (?:cerita|kisah) (?:ini|tersebut|tadi))? siapa(?: namanya)?", value)
                    or re.fullmatch(
                        r"siapa(?:kah)? (?:tokoh|karakter)(?: yang)? (?:utama(?:nya)?|licik|jahat|baik|"
                        r"bijaksana|cerdik|kuat|sakti|paling kuat|paling sakti|paling penting)(?: namanya)?", value)
                    or re.fullmatch(r"(?:berapa|jumlah|berapa jumlah|total) (?:tokoh|karakter)(?: di| dalam)? (?:sistem|aplikasi|katalog|database)", value)
                    or re.search(r"\b(?:mengenali|identifikasi)\b.*\btokoh", value)
                    or catalog_intent(value)
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
        related_name_question = bool(accepted and re.fullmatch(
            r"(?:(?:kalau|jika) (?:punya|ada) )?(?:siapa|siapakah) namanya",
            normalize_query(clause)))
        if (not (domain(clause) or related_name_question) or re.search(
            r"\b(?:website|aplikasi|application|kode|code|html|script|python|sql|shell|javascript|"
            r"source|presiden|bitcoin|program|pemrograman|coding|politik|pemilu|partai|"
            r"resep|masakan|cuaca|bola|saham|kripto|matematika|kalkulus|obat|diagnosis|"
            r"covid|excel|email|surat|algoritma|komputer|game|puisi|pantun|"
            r"hitung|calculate|solve|terjemahkan|translate|investasi|kampanye|election|"
            r"recipe|weather|football|elections|president|programming)\b", inspected_clause)):
            rejected.append(clause)
        else:
            accepted.append(clause)
    if not accepted:
        return IntentDecision(relevant, "unsupported")
    if rejected:
        return IntentDecision(relevant, "mixed", ". ".join(accepted),
            "Aku khusus membantu membahas Wayang Sasak, jadi kita fokus ke Wayang Sasak, ya! 🎭", comparison,
            "MIXED")
    semantic_intent = (
        "CATALOG_REQUEST" if catalog_intent(normalized) else
        "GENERAL_WAYANG_EXPLANATION" if local_overview_intent(query) == "overview" else
        "WAYANG_REQUEST"
    )
    return IntentDecision(relevant, "allowed", query, comparison_ids=comparison,
                          semantic_intent=semantic_intent)


def general_knowledge_intent(query: str) -> bool:
    value = normalize_query(query)
    if local_overview_intent(query) == "overview":
        return True
    standalone = {"sejarah", "sejarahnya", "jalan cerita", "alur cerita", "ceritanya", "asal usul"}
    if value in standalone:
        return True
    cultural = bool(re.search(r"\b(?:wayang|sasak|menak|selaparang|lombok)\b", value)
                    or re.search(r"\b(?:mengenali|identifikasi)\b.*\btokoh", value))
    topics = bool(re.search(r"\b(?:sejarah|asal|usul|cerita|ceritakan|bercerita|kisah|lakon|epos|serat|babad|dalang|pertunjukan|pengertian|apaan|mengenali|jumlah|berapa|total|banyak|internet)\b", value))
    overview = bool(re.search(
        r"\b(?:apa itu|apa yang dimaksud dengan|jelaskan|penjelasan|informasi|mengenal|ingin tahu)"
        r"(?: dong| saya| aku| kepada saya| kepada aku)?(?: tentang| mengenai| soal)? wayang sasak\b",
        value,
    ) or re.search(r"\bwayang sasak apaan(?: sih| dong)?\b", value))
    return cultural and (topics or overview)


def local_overview_intent(query: str) -> str | None:
    """Only bounded overview requests; internet and richer history stay grounded."""
    value = normalize_query(query)
    if catalog_intent(value):
        return None
    if re.fullmatch(r"(?:pewayangan sasak|wayang sasak)(?: adalah| itu)?", value):
        return "overview"
    if "wayang sasak" in value and re.search(
        r"\b(?:dari (?:daerah |wilayah )?mana|dari mana asal|di (?:daerah |wilayah )?mana)\b", value
    ):
        return "overview"
    action = (
        r"(?:(?:(?:coba|tolong) )?(?:apa(?: sih?)? itu|apa yang dimaksud(?: dengan)?|"
        r"jelaskan(?: (?:saya|aku))?|ceritakan(?: (?:saya|aku))?|penjelasan|berikan saya informasi|explain)|"
        r"(?:aku|saya) (?:ingin|mau) (?:tahu|mengenal))"
    )
    if re.search(action + r"(?: lebih banyak)?(?: tentang| mengenai| soal)? wayang sasak(?: ya| dong| sih)?", value):
        return "overview"
    if re.fullmatch(r"(?:(?:sebenarnya|jadi) )?wayang sasak (?:(?:itu )?apa(?: sih)?|apa)", value):
        return "overview"
    if re.fullmatch(r"(?:apa hubungan|apa kaitan|jelaskan hubungan) wayang sasak dengan (?:cerita |serat )?menak", value):
        return "menak"
    return None
