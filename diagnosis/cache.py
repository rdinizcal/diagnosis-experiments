"""Verdict cache for GA candidate evaluation.

The cache is verdict-preserving: keys are SHA-256 hashes of the exact property
expression string, and values are the first observable GA verdict for that
expression plus its first solve time. A cache hit therefore skips redundant
solver work without changing candidate classification or fitness assignment.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional


def expression_key(expression: str) -> str:
    """Return the stable SHA-256 cache key for a property expression."""
    return hashlib.sha256(expression.encode("utf-8")).hexdigest()


class VerdictCache:
    """In-memory and SQLite-backed cache for property expression verdicts."""

    def __init__(self, db_path: str | Path, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.db_path = Path(db_path)
        self._memory: dict[str, tuple[str, float]] = {}
        self.hits = 0
        self.misses = 0
        self._conn: Optional[sqlite3.Connection] = None

        if not self.enabled:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS verdict_cache (
                key TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                verdict TEXT NOT NULL,
                solve_seconds REAL NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        for key, verdict, solve_seconds in self._conn.execute(
            "SELECT key, verdict, solve_seconds FROM verdict_cache"
        ):
            self._memory[str(key)] = (str(verdict), float(solve_seconds))

    def get(self, expression: str) -> tuple[str, float] | None:
        """Return a cached ``(verdict, solve_seconds)`` pair, if present."""
        if not self.enabled:
            return None

        key = expression_key(expression)
        cached = self._memory.get(key)
        if cached is None and self._conn is not None:
            row = self._conn.execute(
                "SELECT verdict, solve_seconds FROM verdict_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                cached = (str(row[0]), float(row[1]))
                self._memory[key] = cached

        if cached is None:
            self.misses += 1
            return None

        self.hits += 1
        return cached

    def put(self, expression: str, verdict: str, solve_seconds: float) -> None:
        """Store a verdict if caching is enabled."""
        if not self.enabled:
            return

        key = expression_key(expression)
        value = (verdict, float(solve_seconds))
        self._memory[key] = value
        if self._conn is not None:
            self._conn.execute(
                "INSERT OR IGNORE INTO verdict_cache "
                "(key, expression, verdict, solve_seconds, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, expression, verdict, value[1], time.time()),
            )
            self._conn.commit()

    def purge_expressions(self, expressions: "Iterable[str]") -> int:
        """Delete cache entries for the given expressions; return the count removed.

        Used by the Sprint 7 quantization validator to evict a class's collapsed
        entries after a same-class verdict disagreement, so the disabled position
        no longer serves stale canonicalized verdicts.
        """
        removed = 0
        for expression in expressions:
            key = expression_key(expression)
            if self._memory.pop(key, None) is not None:
                removed += 1
            if self._conn is not None:
                self._conn.execute("DELETE FROM verdict_cache WHERE key = ?", (key,))
        if self._conn is not None:
            self._conn.commit()
        return removed

    def stats(self) -> dict[str, int]:
        """Return report counters for enabled cache runs."""
        distinct = len(self._memory)
        if self._conn is not None:
            row = self._conn.execute("SELECT COUNT(*) FROM verdict_cache").fetchone()
            if row is not None:
                distinct = int(row[0])
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_distinct": distinct,
        }

    def close(self) -> None:
        """Close the SQLite connection, if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
