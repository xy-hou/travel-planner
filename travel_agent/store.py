"""Persistent trip state, observations, and audit trail. SQLite is the memory store."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import AgentReport, Trip


class StateStore:
    def __init__(self, filename: str):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(filename)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS state(
            trip_id TEXT PRIMARY KEY,
            request_json TEXT NOT NULL,
            evidence_signature TEXT NOT NULL,
            latest_report_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS audit(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id TEXT NOT NULL,
            event TEXT NOT NULL,
            details TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )""")
        self.conn.commit()

    def latest_signature(self, trip_id: str) -> str | None:
        row = self.conn.execute("SELECT evidence_signature FROM state WHERE trip_id = ?", (trip_id,)).fetchone()
        return row[0] if row else None

    def record(self, report: AgentReport, signature: str) -> None:
        trip = report.trip
        self.conn.execute("""INSERT INTO state VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(trip_id) DO UPDATE SET request_json=excluded.request_json,
             evidence_signature=excluded.evidence_signature,
             latest_report_json=excluded.latest_report_json, updated_at=excluded.updated_at""",
             (trip.trip_id, json.dumps({"origin": trip.origin, "destination": trip.destination,
               "deadline": trip.deadline.isoformat(), "budget_usd": trip.budget_usd}),
              signature, json.dumps(report.to_dict()), datetime.now(timezone.utc).isoformat()))
        self.audit(trip.trip_id, "evaluation", {"risk": report.risk_level,
            "candidate_count": len(report.candidates), "booking": report.booking_result,
            "replan": report.replan_triggered})
        self.conn.commit()

    def audit(self, trip_id: str, event: str, details: dict) -> None:
        self.conn.execute("INSERT INTO audit(trip_id,event,details,timestamp) VALUES (?,?,?,?)",
                          (trip_id, event, json.dumps(details, default=str),
                           datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
