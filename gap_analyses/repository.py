"""
repository.py — Model repository for validated PolicyStatement instances.

This is the persistence layer that sits AFTER the extraction + validation
pipeline and BEFORE the gap analysis layer:

    LLM extraction
        ↓
    Pydantic validation (models.py)
        ↓
    ModelRepository.store()         ← this file
        ↓
    GapAnalyser queries (gap_analysis.py)

Schema design
-------------
The `statements` table stores each validated PolicyStatement as:
  - Full JSON (for complete reconstruction)
  - Denormalised scalar/array columns (for fast GROUP BY queries without
    parsing JSON in every row)

The denormalised columns are derived at store time from the validated JSON
and are the only columns the gap analysis queries touch. This means gap
analysis queries are pure SQL + Python set operations — no JSON parsing
at query time.

Denormalised columns (all stored as JSON arrays):
  legal_basis_types    — e.g. ["Consent", "LegitimateInterest"]
  right_types          — e.g. ["Access", "Erasure"]
  constraint_types     — e.g. ["Retention", "Security"]
  purpose_categories   — e.g. ["Marketing", "ServiceProvision"]
  processing_actions   — e.g. ["Collect", "Use"]
  sensitivity_levels   — e.g. ["Low", "SpecialCategory"]
  transfer_mechanisms  — e.g. ["StandardContractualClauses"]

Boolean flags (INTEGER 0/1):
  has_retention         — retentionPolicies is non-empty
  has_transfer          — dataTransfers is non-empty
  has_consent_withdrawal — consentWithdrawal is non-empty
  has_dpia              — any processingActivity.riskAssessmentReference not null
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_STATEMENTS = """
CREATE TABLE IF NOT EXISTS statements (
    statement_id          TEXT PRIMARY KEY,
    law                   TEXT NOT NULL,
    article_ref           TEXT NOT NULL,
    source_clause         TEXT NOT NULL DEFAULT '',
    statement_json        TEXT NOT NULL,

    -- Denormalised arrays (JSON)
    legal_basis_types     TEXT NOT NULL DEFAULT '[]',
    right_types           TEXT NOT NULL DEFAULT '[]',
    constraint_types      TEXT NOT NULL DEFAULT '[]',
    purpose_categories    TEXT NOT NULL DEFAULT '[]',
    processing_actions    TEXT NOT NULL DEFAULT '[]',
    sensitivity_levels    TEXT NOT NULL DEFAULT '[]',
    transfer_mechanisms   TEXT NOT NULL DEFAULT '[]',

    -- Boolean flags
    has_retention          INTEGER NOT NULL DEFAULT 0,
    has_transfer           INTEGER NOT NULL DEFAULT 0,
    has_consent_withdrawal INTEGER NOT NULL DEFAULT 0,
    has_dpia               INTEGER NOT NULL DEFAULT 0,

    stored_at             TEXT NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_stmt_law ON statements(law);",
    "CREATE INDEX IF NOT EXISTS idx_stmt_article ON statements(article_ref);",
]


# ── Denormalisation helpers ───────────────────────────────────────────────────

def _extract_denorm(stmt: dict[str, Any]) -> dict[str, Any]:
    """
    Derive all denormalised columns from a validated PolicyStatement dict.
    This is called once at store time so queries never parse JSON.
    """

    # Legal basis types
    lb = stmt.get("legalBasis") or stmt.get("legal_basis") or {}
    lb_type = lb.get("type", "")
    legal_basis_types = [lb_type] if lb_type else []

    # Right types
    rights = stmt.get("rightImpacted") or stmt.get("rights_impacted") or []
    right_types = list({r.get("type", "") for r in rights if r.get("type")})

    # Constraint types
    constraints = stmt.get("constraints") or []
    constraint_types = list({c.get("type", "") for c in constraints if c.get("type")})

    # Purpose categories
    purposes = stmt.get("purposes") or []
    purpose_categories = list({p.get("category", "") for p in purposes if p.get("category")})

    # Processing actions + sensitivity levels
    pa = stmt.get("processingActivity") or stmt.get("processing_activity") or {}
    processing_actions = [pa.get("action", "")] if pa.get("action") else []
    data_processed = pa.get("dataProcessed") or pa.get("data_processed") or []
    sensitivity_levels = list({d.get("sensitivity", "") for d in data_processed if d.get("sensitivity")})

    # Transfer mechanisms
    transfers = stmt.get("dataTransfers") or stmt.get("data_transfers") or []
    transfer_mechanisms = list({t.get("mechanism", "") for t in transfers if t.get("mechanism")})

    # Boolean flags
    has_retention = 1 if (stmt.get("retentionPolicies") or stmt.get("retention_policies")) else 0
    has_transfer  = 1 if transfers else 0
    has_cw        = 1 if (stmt.get("consentWithdrawal") or stmt.get("consent_withdrawal")) else 0
    has_dpia      = 1 if pa.get("riskAssessmentReference") or pa.get("risk_assessment_reference") else 0

    return {
        "legal_basis_types":     json.dumps(sorted(legal_basis_types)),
        "right_types":           json.dumps(sorted(right_types)),
        "constraint_types":      json.dumps(sorted(constraint_types)),
        "purpose_categories":    json.dumps(sorted(purpose_categories)),
        "processing_actions":    json.dumps(sorted(processing_actions)),
        "sensitivity_levels":    json.dumps(sorted(sensitivity_levels)),
        "transfer_mechanisms":   json.dumps(sorted(transfer_mechanisms)),
        "has_retention":          has_retention,
        "has_transfer":           has_transfer,
        "has_consent_withdrawal": has_cw,
        "has_dpia":               has_dpia,
    }


# ── ModelRepository ───────────────────────────────────────────────────────────

class ModelRepository:
    """
    SQLite-backed repository for validated PolicyStatement instances.

    Usage
    -----
        repo = ModelRepository("data/model_repo.db")
        repo.store("GDPR", "Art.6", validated_statement_dict)
        laws = repo.laws()
        statements = repo.statements_for_law("GDPR")
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        log.info(f"ModelRepository opened: {self.db_path}")

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_CREATE_STATEMENTS)
        for idx in _CREATE_INDEXES:
            cur.execute(idx)
        self._conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        law: str,
        article_ref: str,
        statement: dict[str, Any],
        replace: bool = True,
    ) -> str:
        """
        Store a validated PolicyStatement dict.

        Parameters
        ----------
        law          : canonical law name, e.g. "GDPR"
        article_ref  : article reference, e.g. "Art.6(1)(a)"
        statement    : validated PolicyStatement as a plain dict
                       (output of PolicyStatementModel.model_dump() or raw LLM JSON)
        replace      : if True, replace existing entry with same statement_id

        Returns
        -------
        statement_id stored.
        """
        law = law.upper()
        stmt_id = statement.get("statementId") or statement.get("statement_id") or ""
        if not stmt_id:
            import hashlib
            raw = f"{law}|{article_ref}|{json.dumps(statement)[:64]}"
            stmt_id = hashlib.sha1(raw.encode()).hexdigest()[:16]

        source_clause = (
            statement.get("source_clause") or
            statement.get("sourceClause") or
            article_ref
        )

        denorm = _extract_denorm(statement)
        now = datetime.now(timezone.utc).isoformat()

        sql = (
            "INSERT OR REPLACE INTO statements "
            "(statement_id, law, article_ref, source_clause, statement_json, "
            " legal_basis_types, right_types, constraint_types, purpose_categories, "
            " processing_actions, sensitivity_levels, transfer_mechanisms, "
            " has_retention, has_transfer, has_consent_withdrawal, has_dpia, stored_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            if replace else
            "INSERT OR IGNORE INTO statements "
            "(statement_id, law, article_ref, source_clause, statement_json, "
            " legal_basis_types, right_types, constraint_types, purpose_categories, "
            " processing_actions, sensitivity_levels, transfer_mechanisms, "
            " has_retention, has_transfer, has_consent_withdrawal, has_dpia, stored_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )

        self._conn.execute(sql, (
            stmt_id, law, article_ref, source_clause,
            json.dumps(statement),
            denorm["legal_basis_types"],
            denorm["right_types"],
            denorm["constraint_types"],
            denorm["purpose_categories"],
            denorm["processing_actions"],
            denorm["sensitivity_levels"],
            denorm["transfer_mechanisms"],
            denorm["has_retention"],
            denorm["has_transfer"],
            denorm["has_consent_withdrawal"],
            denorm["has_dpia"],
            now,
        ))
        self._conn.commit()
        log.debug(f"Stored statement {stmt_id} ({law} / {article_ref})")
        return stmt_id

    def store_many(
        self,
        law: str,
        statements: list[tuple[str, dict[str, Any]]],
    ) -> int:
        """
        Bulk store. statements is a list of (article_ref, statement_dict).
        Returns number stored.
        """
        count = 0
        for article_ref, stmt in statements:
            self.store(law, article_ref, stmt)
            count += 1
        log.info(f"Stored {count} statements for {law}")
        return count

    # ── Read ──────────────────────────────────────────────────────────────────

    def laws(self) -> list[str]:
        """All distinct law names in the repository."""
        cur = self._conn.execute(
            "SELECT DISTINCT law FROM statements ORDER BY law"
        )
        return [row[0] for row in cur.fetchall()]

    def statements_for_law(self, law: str) -> list[dict[str, Any]]:
        """All raw statement dicts for a law, in article order."""
        cur = self._conn.execute(
            "SELECT statement_json FROM statements WHERE law=? ORDER BY article_ref",
            (law.upper(),),
        )
        return [json.loads(row[0]) for row in cur.fetchall()]

    def stats(self) -> dict[str, Any]:
        """Summary of what is in the repository."""
        cur = self._conn.execute(
            "SELECT law, COUNT(*) as n FROM statements GROUP BY law ORDER BY law"
        )
        by_law = {row["law"]: row["n"] for row in cur.fetchall()}
        total = sum(by_law.values())
        return {"total": total, "by_law": by_law}

    # ── Raw column access (used by GapAnalyser) ───────────────────────────────

    def distinct_values(self, law: str, column: str) -> set[str]:
        """
        Return the union of all values in a JSON-array column for a given law.
        e.g. distinct_values("GDPR", "legal_basis_types") →
             {"Consent", "Contract", "LegitimateInterest", ...}
        """
        cur = self._conn.execute(
            f"SELECT {column} FROM statements WHERE law=?",
            (law.upper(),),
        )
        result: set[str] = set()
        for row in cur.fetchall():
            values = json.loads(row[0])
            result.update(v for v in values if v)
        return result

    def flag_count(self, law: str, flag_column: str) -> int:
        """Count of statements where a boolean flag is 1 for a given law."""
        cur = self._conn.execute(
            f"SELECT COUNT(*) FROM statements WHERE law=? AND {flag_column}=1",
            (law.upper(),),
        )
        return cur.fetchone()[0]

    def statement_count(self, law: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM statements WHERE law=?",
            (law.upper(),),
        )
        return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
