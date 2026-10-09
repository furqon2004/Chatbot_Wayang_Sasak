"""Process-local validated RAG answers; no persistence or sensitive query keys."""
from collections import OrderedDict
import hashlib
import json

from .web_cache import public_query
from .query_normalizer import normalize_query

PROMPT_VERSION = "evidence-selection-v1"


class RagCache:
    def __init__(self, clock):
        self.clock = clock
        self.entries = OrderedDict()

    def key(self, original, query, context, character_id, comparison_ids, gemini):
        from .gemini import SYSTEM_INSTRUCTION
        from .chat_schemas import EvidenceSelection
        if gemini is None or not public_query(original) or not public_query(query):
            return None
        if any(not public_query(row["text"]) or not public_query(row["nama"]) for row in context):
            return None
        model = getattr(getattr(gemini, "settings", None), "gemini_model", None)
        payload = [normalize_query(query), context, character_id, sorted(comparison_ids),
                   model, PROMPT_VERSION, SYSTEM_INSTRUCTION, EvidenceSelection.model_json_schema()]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _expire(self):
        for key, (deadline, _) in list(self.entries.items()):
            if deadline <= self.clock():
                del self.entries[key]

    def get(self, key):
        self._expire()
        if key is None or key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key][1].model_copy(deep=True)

    def put(self, key, reply):
        self._expire()
        if key is None or reply.mode != "answer" or reply.llm_status != "used":
            return
        self.entries[key] = (self.clock() + 900, reply.model_copy(deep=True))
        self.entries.move_to_end(key)
        while len(self.entries) > 32:
            self.entries.popitem(last=False)
