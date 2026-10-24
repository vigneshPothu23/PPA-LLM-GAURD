"""Storage: one SQLAlchemy Core schema for SQLite (local / ephemeral) and PostgreSQL (persistent, e.g. Neon Free).

Every table except `users` carries `org_id`. Tenant-owned references are composite (org_id, id) foreign keys,
so a row can never point at another organization's application or scan.
Raw prompts are NOT stored. `retained_inputs` holds redacted text only, only when an application enables
retention, and only until `expires_at`.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKeyConstraint, Integer, MetaData, String, Table, Text,
                        UniqueConstraint, create_engine, event)
from sqlalchemy.engine import Engine

meta = MetaData()


def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def new_id() -> str:
    return uuid.uuid4().hex


organizations = Table(
    "organizations", meta,
    Column("id", String(32), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("industry", String(40), nullable=False, default="general"),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

users = Table(
    "users", meta,
    Column("id", String(32), primary_key=True),
    Column("email", String(320), nullable=False, unique=True),
    Column("display_name", String(200), nullable=False),
    Column("status", String(16), nullable=False, default="active"),  # active | disabled
    Column("created_at", DateTime(timezone=True), nullable=False),
)

memberships = Table(
    "memberships", meta,
    Column("user_id", String(32), nullable=False),
    Column("org_id", String(32), nullable=False),
    Column("role", String(32), nullable=False),
    Column("status", String(16), nullable=False, default="active"),  # active | disabled (per organization)
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("user_id", "org_id", name="uq_membership"),
    ForeignKeyConstraint(["user_id"], ["users.id"]),
    ForeignKeyConstraint(["org_id"], ["organizations.id"]),
)

applications = Table(
    "applications", meta,
    Column("id", String(32), primary_key=True),
    Column("org_id", String(32), nullable=False),
    Column("name", String(120), nullable=False),
    Column("purpose", String(40), nullable=False),          # coding | email | documents | general
    Column("owner_user_id", String(32), nullable=False),
    Column("system_prompt", Text, nullable=False),          # Engine 1 trusted instructions (contains {user_input})
    Column("task_topic", String(300), nullable=False),
    Column("profile", String(40), nullable=False),          # industry policy profile it was created from
    Column("warn_threshold", Float, nullable=False),
    Column("block_threshold", Float, nullable=False),
    Column("warn_action", String(20), nullable=False),      # hold | allow_with_flag
    Column("forgery_action", String(8), nullable=False),    # WARN | BLOCK
    Column("hidden_chars_action", String(8), nullable=False),
    Column("truncated_action", String(8), nullable=False),
    Column("max_chars", Integer, nullable=False),
    Column("retention_days", Integer, nullable=False),      # 0 = keep no text at all
    Column("rate_limit_per_min", Integer, nullable=False),
    Column("status", String(16), nullable=False),           # active | paused
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("org_id", "id", name="uq_app_org"),
    UniqueConstraint("org_id", "name", name="uq_app_name"),
    ForeignKeyConstraint(["org_id"], ["organizations.id"]),
)

app_members = Table(
    "app_members", meta,
    Column("org_id", String(32), nullable=False),
    Column("app_id", String(32), nullable=False),
    Column("user_id", String(32), nullable=False),
    UniqueConstraint("app_id", "user_id", name="uq_app_member"),
    ForeignKeyConstraint(["org_id", "app_id"], ["applications.org_id", "applications.id"]),
    ForeignKeyConstraint(["user_id", "org_id"], ["memberships.user_id", "memberships.org_id"]),
)

api_keys = Table(
    "api_keys", meta,
    Column("id", String(32), primary_key=True),
    Column("org_id", String(32), nullable=False),
    Column("app_id", String(32), nullable=False),
    Column("prefix", String(16), nullable=False),
    Column("key_hash", String(64), nullable=False, unique=True),
    Column("created_by", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("revoked_at", DateTime(timezone=True)),
    Column("last_used_at", DateTime(timezone=True)),
    ForeignKeyConstraint(["org_id", "app_id"], ["applications.org_id", "applications.id"]),
)

scans = Table(
    "scans", meta,
    Column("id", String(32), primary_key=True),
    Column("org_id", String(32), nullable=False),
    Column("app_id", String(32), nullable=False),
    Column("user_id", String(32)),                          # null for API-key calls
    Column("api_key_id", String(32)),
    Column("channel", String(16), nullable=False),          # console | sdk
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("input_sha256", String(64), nullable=False),
    Column("input_chars", Integer, nullable=False),
    Column("n_tokens", Integer),
    Column("truncated", Boolean),
    Column("score", Float),
    Column("detector_band", String(8)),
    Column("decision", String(8), nullable=False),          # ALLOW | WARN | BLOCK
    Column("outcome", String(32), nullable=False),          # protected_prompt_returned | held_for_review | blocked
    Column("reasons_json", Text, nullable=False),
    Column("policy_json", Text, nullable=False),
    Column("model_id", String(120)),
    Column("engine1_applied", Boolean, nullable=False),
    Column("review_id", String(32)),
    Column("latency_ms", Float, nullable=False),
    UniqueConstraint("org_id", "id", name="uq_scan_org"),
    ForeignKeyConstraint(["org_id", "app_id"], ["applications.org_id", "applications.id"]),
)

retained_inputs = Table(
    "retained_inputs", meta,
    Column("scan_id", String(32), primary_key=True),
    Column("org_id", String(32), nullable=False),
    Column("redacted_text", Text, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(["org_id", "scan_id"], ["scans.org_id", "scans.id"]),
)

reviews = Table(
    "reviews", meta,
    Column("id", String(32), primary_key=True),
    Column("org_id", String(32), nullable=False),
    Column("app_id", String(32), nullable=False),
    Column("scan_id", String(32), nullable=False),
    Column("requester_id", String(32)),
    Column("input_sha256", String(64), nullable=False),
    Column("status", String(20), nullable=False),           # open | confirmed_injection | false_positive | released | release_used
    Column("reviewer_id", String(32)),
    Column("note", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("decided_at", DateTime(timezone=True)),
    Column("release_expires_at", DateTime(timezone=True)),
    Column("release_used_at", DateTime(timezone=True)),
    ForeignKeyConstraint(["org_id", "scan_id"], ["scans.org_id", "scans.id"]),
)

audit_events = Table(
    "audit_events", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("org_id", String(32), nullable=False),
    Column("seq", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("actor", String(320), nullable=False),           # email, "api-key:<prefix>", or "system"
    Column("action", String(60), nullable=False),
    Column("target", String(200), nullable=False),
    Column("details_json", Text, nullable=False),
    Column("prev_hash", String(64), nullable=False),
    Column("hash", String(64), nullable=False),
    UniqueConstraint("org_id", "seq", name="uq_audit_seq"),
)

audit_head = Table(
    "audit_head", meta,
    Column("org_id", String(32), primary_key=True),
    Column("seq", Integer, nullable=False),
    Column("last_hash", String(64), nullable=False),
)


def make_engine(url: str | None = None) -> Engine:
    """DATABASE_URL (env or Streamlit secret) -> PostgreSQL; otherwise a local SQLite file."""
    url = url or os.environ.get("PPA_DATABASE_URL") or "sqlite:///ppa_guard.db"
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    kw = {"pool_pre_ping": True, "future": True}
    sqlite = url.startswith("sqlite")
    if sqlite:
        kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if url in ("sqlite://", "sqlite:///:memory:"):
            from sqlalchemy.pool import StaticPool  # one shared in-memory database (single-threaded use only)
            kw["poolclass"] = StaticPool
    eng = create_engine(url, **kw)
    if sqlite:
        @event.listens_for(eng, "connect")
        def _sqlite_setup(dbapi_conn, _):
            dbapi_conn.isolation_level = None             # let SQLAlchemy control transactions (below)
            dbapi_conn.execute("PRAGMA foreign_keys=ON")  # SQLite enforces foreign keys only when asked
            dbapi_conn.execute("PRAGMA busy_timeout=30000")

        @event.listens_for(eng, "begin")
        def _sqlite_begin(conn):
            # BEGIN IMMEDIATE takes the write lock up front, so read-then-write sequences (audit chain head,
            # single-use releases, rate limits) are serialised across threads and processes.
            conn.exec_driver_sql("BEGIN IMMEDIATE")
    meta.create_all(eng)
    return eng


def backend_name(engine: Engine) -> str:
    return engine.dialect.name
