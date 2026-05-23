from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .schema import CustomerProfile


class CustomerMemory:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS customer_profiles (
                        customer_id TEXT PRIMARY KEY,
                        country TEXT,
                        budget TEXT,
                        interests TEXT,
                        negotiated_before INTEGER,
                        next_action TEXT
                    )
                    """
                )

    def load(self, customer_id: str) -> CustomerProfile:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT customer_id, country, budget, interests, negotiated_before, next_action FROM customer_profiles WHERE customer_id=?",
                (customer_id,),
            ).fetchone()
        if not row:
            return CustomerProfile(customer_id=customer_id)
        return CustomerProfile(
            customer_id=row[0],
            country=row[1] or "unknown",
            budget=row[2] or "unknown",
            interests=json.loads(row[3] or "[]"),
            negotiated_before=bool(row[4]),
            next_action=row[5] or "Ask one concrete follow-up question.",
        )

    def save(self, profile: CustomerProfile) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO customer_profiles(customer_id, country, budget, interests, negotiated_before, next_action)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(customer_id) DO UPDATE SET
                        country=excluded.country,
                        budget=excluded.budget,
                        interests=excluded.interests,
                        negotiated_before=excluded.negotiated_before,
                        next_action=excluded.next_action
                    """,
                    (
                        profile.customer_id,
                        profile.country,
                        profile.budget,
                        json.dumps(profile.interests, ensure_ascii=False),
                        int(profile.negotiated_before),
                        profile.next_action,
                    ),
                )
