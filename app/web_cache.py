"""Bounded process-local cache for validated public web answers; never persisted."""
from collections import OrderedDict
from copy import deepcopy
import re

from .chat_schemas import WebSource
from .query_normalizer import inspection_text, normalize_query, unsafe_evidence


def public_query(query):
    text = inspection_text(query)
    return not unsafe_evidence(query) and not re.search(
        r"@|\d{4,}|\b(?:token|rahasia|sandi|pin|nik|ktp|passport|paspor|"
        r"rekening|telepon|ponsel|whatsapp|alamat|email|e-mail|pribadi|"
        r"nama saya|nama lengkap|tanggal lahir|kartu kredit|diagnosis)\b", text)


class PublicWebCache:
    def __init__(self, settings, clock):
        self.ttl = settings.wayang_web_cache_ttl_seconds
        self.limit = settings.wayang_web_cache_max_entries
        self.clock = clock
        self.entries = OrderedDict()

    def _expire(self):
        now = self.clock()
        for key, (expires, _) in list(self.entries.items()):
            if expires <= now:
                del self.entries[key]

    def get(self, query):
        self._expire()
        if not public_query(query):
            return None
        key = normalize_query(query)
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return deepcopy(self.entries[key][1])

    def put(self, query, answer):
        self._expire()
        if not public_query(query):
            return
        text, sources, _ = answer
        if not text or not sources:
            return
        for source in sources:
            WebSource.model_validate(source.model_dump())
        key = normalize_query(query)
        self.entries[key] = (self.clock() + self.ttl, deepcopy(answer))
        self.entries.move_to_end(key)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)
