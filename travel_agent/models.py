from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("All timestamps must include a timezone offset (e.g. -05:00 or Z)")
    return dt.astimezone(timezone.utc)


@dataclass
class Trip:
    trip_id: str
    origin: str
    destination: str
    deadline: datetime
    budget_usd: float
    flight_number: str = ""
    auto_booking_authorized: bool = False
    auto_booking_limit_usd: float = 0.0
    preferences: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.origin = self.origin.upper()
        self.destination = self.destination.upper()
        self.deadline = parse_dt(self.deadline)
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if self.auto_booking_limit_usd < 0:
            raise ValueError("auto_booking_limit_usd cannot be negative")


@dataclass
class Evidence:
    kind: str
    place: str
    summary: str
    source: str
    source_url: str
    observed_at: datetime
    retrieved_at: datetime
    expires_at: datetime
    official: bool
    severity: str = "INFO"
    simulated: bool = False

    def fresh(self, at: datetime | None = None) -> bool:
        at = at or utcnow()
        return self.observed_at <= at + timedelta(minutes=5) and at <= self.expires_at

    def identity(self) -> tuple:
        return self.kind, self.place, self.summary, self.source

    def to_dict(self) -> dict:
        result = asdict(self)
        for k in ("observed_at", "retrieved_at", "expires_at"):
            result[k] = getattr(self, k).isoformat()
        result["fresh"] = self.fresh()
        return result


@dataclass(frozen=True)
class Leg:
    code: str
    origin: str
    destination: str
    mode: str
    depart: datetime
    arrive: datetime
    price_usd: float
    provider: str
    availability_verified: bool
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.arrive <= self.depart or self.price_usd < 0:
            raise ValueError("Invalid leg time or price")

    def to_dict(self) -> dict:
        item = asdict(self)
        item["depart"] = self.depart.isoformat()
        item["arrive"] = self.arrive.isoformat()
        return item


@dataclass
class Candidate:
    legs: list[Leg]
    total_usd: float
    arrival: datetime
    score: float = 0.0
    status: str = "UNVALIDATED"
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "legs": [leg.to_dict() for leg in self.legs],
            "total_usd": round(self.total_usd, 2),
            "arrival": self.arrival.isoformat(),
            "score": round(self.score, 2),
            "status": self.status,
            "issues": self.issues,
        }


@dataclass
class AgentReport:
    trip: Trip
    mode: str
    collected_at: datetime
    evidence: list[Evidence]
    retrieved_evidence: list[Evidence]
    risk_level: str
    risk_reasons: list[str]
    candidates: list[Candidate]
    decisions: list[str]
    escalations: list[str]
    booking_result: str
    replan_triggered: bool
    state_changed: bool

    def to_dict(self) -> dict:
        return {
            "trip_id": self.trip.trip_id,
            "mode": self.mode,
            "collected_at": self.collected_at.isoformat(),
            "origin": self.trip.origin,
            "destination": self.trip.destination,
            "deadline": self.trip.deadline.isoformat(),
            "budget_usd": self.trip.budget_usd,
            "risk_level": self.risk_level,
            "risk_reasons": self.risk_reasons,
            "evidence": [x.to_dict() for x in self.evidence],
            "retrieved_evidence": [x.to_dict() for x in self.retrieved_evidence],
            "candidates": [x.to_dict() for x in self.candidates],
            "decisions": self.decisions,
            "escalations": self.escalations,
            "booking_result": self.booking_result,
            "replan_triggered": self.replan_triggered,
            "state_changed": self.state_changed,
        }
