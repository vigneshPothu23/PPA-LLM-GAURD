"""Append-only, hash-chained audit events per organization.

hash = SHA-256(prev_hash + canonical JSON of the event). `verify` recomputes the chain and reports the first
broken sequence number. This detects edits, deletions and re-ordering after the fact; it does not stop
someone with direct database write access from rewriting the whole chain.
Writes are serialised per process (lock) and per database (row lock on audit_head where supported).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Connection

from .db import audit_events, audit_head, now

GENESIS = "0" * 64
_lock = threading.Lock()


def _canon(org_id: str, seq: int, created_at: datetime, actor: str, action: str, target: str, details_json: str,
           prev_hash: str) -> str:
    if created_at.tzinfo is not None:  # PostgreSQL returns aware datetimes in the session time zone
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    return json.dumps({"org_id": org_id, "seq": seq, "created_at": created_at.isoformat(),
                       "actor": actor, "action": action, "target": target, "details": details_json,
                       "prev_hash": prev_hash}, sort_keys=True, separators=(",", ":"))


def append(conn: Connection, org_id: str, actor: str, action: str, target: str, details: dict | None = None) -> int:
    """Must be called inside the caller's transaction so the event commits (or rolls back) with the change."""
    details_json = json.dumps(details or {}, sort_keys=True, default=str)
    with _lock:
        q = select(audit_head).where(audit_head.c.org_id == org_id)
        if conn.dialect.name == "postgresql":
            q = q.with_for_update()
        head = conn.execute(q).mappings().first()
        if head is None:
            conn.execute(insert(audit_head).values(org_id=org_id, seq=0, last_hash=GENESIS))
            seq, prev = 0, GENESIS
            if conn.dialect.name == "postgresql":
                conn.execute(select(audit_head).where(audit_head.c.org_id == org_id).with_for_update())
        else:
            seq, prev = head["seq"], head["last_hash"]
        seq += 1
        ts = now()
        h = hashlib.sha256((prev + _canon(org_id, seq, ts, actor, action, target, details_json, prev)).encode()).hexdigest()
        conn.execute(insert(audit_events).values(org_id=org_id, seq=seq, created_at=ts, actor=actor, action=action,
                                                 target=target, details_json=details_json, prev_hash=prev, hash=h))
        conn.execute(update(audit_head).where(audit_head.c.org_id == org_id).values(seq=seq, last_hash=h))
    return seq


def verify(conn: Connection, org_id: str) -> dict:
    rows = conn.execute(select(audit_events).where(audit_events.c.org_id == org_id)
                        .order_by(audit_events.c.seq)).mappings().all()
    prev, expected_seq = GENESIS, 1
    for r in rows:
        if r["seq"] != expected_seq:
            return {"ok": False, "checked": expected_seq - 1, "broken_at": expected_seq, "reason": "missing or re-ordered event"}
        if r["prev_hash"] != prev:
            return {"ok": False, "checked": expected_seq - 1, "broken_at": r["seq"], "reason": "previous-hash link broken"}
        h = hashlib.sha256((prev + _canon(r["org_id"], r["seq"], r["created_at"], r["actor"], r["action"], r["target"],
                                          r["details_json"], r["prev_hash"])).encode()).hexdigest()
        if h != r["hash"]:
            return {"ok": False, "checked": expected_seq - 1, "broken_at": r["seq"], "reason": "event content changed"}
        prev, expected_seq = h, expected_seq + 1
    head = conn.execute(select(audit_head).where(audit_head.c.org_id == org_id)).mappings().first()
    if head and (head["seq"] != len(rows) or head["last_hash"] != prev):
        return {"ok": False, "checked": len(rows), "broken_at": len(rows), "reason": "events after the last one were deleted"}
    return {"ok": True, "checked": len(rows), "broken_at": None, "reason": None}
