import re
import unicodedata

STOPWORDS = frozenset("apa apakah siapa bagaimana jelaskan tolong coba berikan saya tentang dan atau dari yang ini itu dengan untuk adalah di ke tokoh karakter wayang sasak fungsi filosofi ciri cirinya deskripsi".split())


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", query).casefold()
    text = "".join(char if char.isalnum() else " " for char in text)
    return " ".join("sasak" if token == "saksak" else token for token in text.split())


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
