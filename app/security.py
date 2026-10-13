"""Per-instance admission controls and allowlisted security events."""
from collections import OrderedDict
import asyncio
import copy
from contextvars import ContextVar
import hashlib
import json
import logging
import re
import threading
import time
from typing import Protocol

REQUEST_ID = ContextVar("request_id", default="")
SECRET_FIELD = re.compile(r"key|token|authorization|cookie|secret|credential|password", re.I)
SECRET_VALUE = re.compile(r"AIza[\w-]{35}|sk-[\w-]{20,}|gh[pousr]_[\w]{20,}")


def redact(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if SECRET_FIELD.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        clean = "".join(c if c.isprintable() else " " for c in value)
        return SECRET_VALUE.sub("[REDACTED]", clean)[:500]
    return value


def event(**fields):
    allowed = {"endpoint", "method", "status", "latency_ms", "mode", "category", "llm_used"}
    record = {key: value for key, value in fields.items() if key in allowed}
    record["request_id"] = REQUEST_ID.get()
    logging.getLogger("wayang.security").info(json.dumps(redact(record)))


class RateLimiter(Protocol):
    async def allow(self, key: str) -> bool: ...


class MemoryRateLimiter:
    """Fixed window, bounded keys, thread-safe. Use a shared implementation for workers."""
    def __init__(self, limit=60, clock=time.monotonic, max_keys=10000):
        self.limit, self.clock, self.max_keys = limit, clock, max_keys
        self._lock = threading.Lock()
        self._buckets = OrderedDict()

    def reset(self):
        with self._lock:
            self._buckets.clear()

    async def allow(self, key):
        with self._lock:
            now = self.clock()
            while self._buckets and next(iter(self._buckets.values()))[0] <= now - 60:
                self._buckets.popitem(last=False)
            if key not in self._buckets:
                if len(self._buckets) >= self.max_keys:
                    return False
                self._buckets[key] = [now, 0]
            bucket = self._buckets[key]
            if bucket[1] >= self.limit:
                return False
            bucket[1] += 1
            return True


class Capacity:
    """Nonblocking admission; cancellation cannot strand queued waiters."""
    def __init__(self, limit=2):
        self._slots = threading.BoundedSemaphore(limit)

    def acquire(self):
        return self._slots.acquire(blocking=False)

    def release(self):
        self._slots.release()


class IdempotencyConflict(ValueError):
    """A message ID was reused for a different payload."""


class IdempotencyUnavailable(RuntimeError):
    """The bounded store has no room for another in-flight action."""


class IdempotencyStore:
    """Bounded per-process result cache for retried synchronous chat actions."""
    def __init__(self, ttl=300, max_entries=512, clock=time.monotonic):
        self.ttl, self.max_entries, self.clock = ttl, max_entries, clock
        self._entries = OrderedDict()
        self._lock = asyncio.Lock()

    async def run(self, message_id, payload, operation):
        fingerprint = hashlib.sha256(json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        owner = False
        async with self._lock:
            now = self.clock()
            for key, entry in list(self._entries.items()):
                if entry["future"].done() and entry["created"] <= now - self.ttl:
                    self._entries.pop(key)
            entry = self._entries.get(message_id)
            if entry and entry["fingerprint"] != fingerprint:
                raise IdempotencyConflict
            if entry:
                self._entries.move_to_end(message_id)
                future = entry["future"]
            else:
                while len(self._entries) >= self.max_entries:
                    removable = next((key for key, value in self._entries.items()
                                      if value["future"].done()), None)
                    if removable is None:
                        raise IdempotencyUnavailable
                    self._entries.pop(removable)
                future = asyncio.get_running_loop().create_future()
                self._entries[message_id] = {"fingerprint": fingerprint, "future": future, "created": now}
                owner = True
        if not owner:
            status, result = await asyncio.shield(future)
            if status == "error":
                raise result
            return copy.deepcopy(result)
        try:
            result = await operation()
            cached = copy.deepcopy(result)
        except BaseException as exc:
            async with self._lock:
                self._entries.pop(message_id, None)
                future.set_result(("error", exc))
            raise
        async with self._lock:
            future.set_result(("ok", cached))
        return result


class CircuitBreaker:
    def __init__(self, threshold=3, cooldown=60, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self._lock = threading.Lock()
        self.failures = 0
        self.until = 0.0

    def reset(self):
        with self._lock:
            self.failures, self.until = 0, 0.0

    def available(self):
        with self._lock:
            return self.clock() >= self.until

    def failed(self, immediate=False):
        with self._lock:
            self.failures = min(self.threshold, self.failures + 1)
            if immediate or self.failures >= self.threshold:
                self.until = self.clock() + self.cooldown

    def succeeded(self):
        with self._lock:
            # A concurrent older success must not clear an open breaker.
            if self.clock() >= self.until:
                self.failures = 0
