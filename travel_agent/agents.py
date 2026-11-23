"""Five logical roles + ReAct-like observe/reason/act/observe cycles.

These are deterministic specialists, intentionally runnable without LLM keys.
Use an LLM as a *supplement* for explanation or candidate hypothesis generation,
never for asserting live seat availability or approving purchases.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from .feeds import DemoFeeds, LiveFeeds
from .models import AgentReport, Candidate, Evidence, Leg, Trip, utcnow
from .planner import BeamSearchPlanner, OfferProvider
from .retrieval import EvidenceRetriever
from .store import StateStore


class Source(Protocol):
    def collect(self, trip: Trip) -> list[Evidence]: ...


class RetrievalAgent:
    """ACT: retrieve authoritative observations; OBSERVE: rank relevant evidence."""
    def __init__(self, source: Source, retriever: EvidenceRetriever):
        self.source = source
        self.retriever = retriever

    def execute(self, trip: Trip) -> tuple[list[Evidence], list[Evidence]]:
        evidence = self.source.collect(trip)
        query = (f"{trip.origin} {trip.destination} {trip.flight_number} "
                 "flight cancellation airport closure ground stop storm warning roads")
        relevant = self.retriever.search(query, evidence, k=5)
        # Safety-critical alerts must NEVER be filtered out simply because of a low
        # semantic-similarity score. Retrieval ranks supporting context only.
        return evidence, relevant


class RiskAgent:
    """REASON: separate confirmed operational restrictions from speculative risk."""
    def assess(self, trip: Trip, evidence: list[Evidence]) -> tuple[str, list[str]]:
        fresh = [x for x in evidence if x.fresh()]
        if not fresh:
            return "UNKNOWN", ["No fresh verified observations; no safety conclusion possible"]
        reasons = []
        high = False
        for item in fresh:
            if item.kind == "FLIGHT_CANCELLED":
                high = True
                reasons.append("Confirmed in synthetic fixture: current flight canceled" if item.simulated
                               else "Flight-specific cancellation confirmed by provider")
            elif item.kind in {"FAA_RESTRICTION", "GROUND_CLOSURE"}:
                high = True
                reasons.append(f"{item.place}: {item.summary}")
            elif item.kind in {"STORM_WARNING", "NWS_ALERT"}:
                reasons.append(f"Weather-related concern: {item.place}: {item.summary[:160]}")
            elif item.kind == "SOURCE_UNAVAILABLE":
                reasons.append(f"Critical source unavailable: {item.source}")
        if high:
            return "HIGH", reasons
        if any(x.kind in {"STORM_WARNING", "NWS_ALERT"} for x in fresh):
            return "ELEVATED", reasons
        if any(x.kind == "SOURCE_UNAVAILABLE" for x in fresh):
            return "UNKNOWN", reasons or ["Source outage"]
        return "UNDETERMINED", ["No verified disruption identified by sampled feeds; not proof of safety"]


class PlanningAgent:
    """Bounded ToT branch expansion by a deterministic beam-search controller."""
    def __init__(self, provider: OfferProvider):
        self.provider = provider
        self.search = BeamSearchPlanner(branch_factor=4, beam_width=3, depth_limit=3)

    def execute(self, trip: Trip, at: datetime) -> list[Candidate]:
        return self.search.plan(trip, self.provider.offers(trip, at), at)


class ValidationAndActionAgent:
    """Deterministic hard checks. Demo bookings are NEVER real purchases."""
    def validate(self, trip: Trip, routes: list[Candidate],
                 evidence: list[Evidence], mode: str) -> list[Candidate]:
        fresh = [x for x in evidence if x.fresh()]
        road_closures = {x.place for x in fresh if x.kind == "GROUND_CLOSURE"}
        # FAA operational restrictions may not equal universal airport shutdown.
        # They trigger manual verification, not unsupported hard 'unsafe' conclusions.
        restricted = {x.place for x in fresh if x.kind == "FAA_RESTRICTION"}
        serious_weather = {x.place for x in fresh if x.kind in {"NWS_ALERT", "STORM_WARNING"}
                           and x.severity in {"HIGH", "SEVERE", "EXTREME"}}
        validated = []
        for route in routes:
            issues = []
            if route.total_usd > trip.budget_usd:
                issues.append("OVER_BUDGET")
            if route.arrival > trip.deadline:
                issues.append("MISSES_DEADLINE")
            if any(l.mode == "ground" and f"{l.origin}->{l.destination}" in road_closures
                   for l in route.legs):
                issues.append("ROAD_CLOSED")
            for leg in route.legs:
                if leg.mode == "flight" and (leg.origin in restricted or leg.destination in restricted):
                    issues.append("AIRPORT_RESTRICTION_REVERIFY")
                if (leg.origin in serious_weather or leg.destination in serious_weather or
                    ("NYC" in serious_weather and leg.destination in {"EWR", "JFK", "LGA"})):
                    issues.append("WEATHER_ALERT_REVIEW")
                if not leg.availability_verified:
                    issues.append("TICKET_NOT_VERIFIED")
            if any(l.synthetic for l in route.legs):
                issues.append("SIMULATION_ONLY")
            if issues and any(s in issues for s in ("OVER_BUDGET", "MISSES_DEADLINE", "ROAD_CLOSED")):
                route.status = "REJECTED"
            elif mode == "demo":
                route.status = "SIMULATED_OPTION"
            elif issues:
                route.status = "NEEDS_VERIFICATION"
            else:
                route.status = "PROVIDER_VERIFIED"
            route.issues = sorted(set(issues))
            if route.status != "REJECTED":
                validated.append(route)
        return validated[:3]

    def act(self, trip: Trip, candidates: list[Candidate], mode: str,
            simulate_auto_booking: bool) -> tuple[str, list[str]]:
        if not candidates:
            return "NOT_ATTEMPTED", ["No validated ticket offers; human should check official booking providers"]
        if not trip.auto_booking_authorized:
            return "NOT_ATTEMPTED", ["Booking requires explicit user authorization"]
        best = candidates[0]
        if best.total_usd > trip.auto_booking_limit_usd or best.total_usd > trip.budget_usd:
            return "BLOCKED_BY_BUDGET", ["Offer is above the auto-booking spending limit"]
        if any(x in best.issues for x in ["AIRPORT_RESTRICTION_REVERIFY", "TICKET_NOT_VERIFIED", "WEATHER_ALERT_REVIEW"]):
            return "BLOCKED_BY_VERIFICATION", ["Operational or ticket data requires verification"]
        if mode != "demo" or not simulate_auto_booking:
            return "DISABLED", ["Real payment is deliberately not implemented in this prototype"]
        if not all(x.synthetic for x in best.legs):
            return "DISABLED", ["Cannot simulate booking unless every segment is a fixture"]
        return "SIMULATED_BOOKING_ONLY", ["Synthetic booking recorded in demo; no payment or ticket issued"]


class CoordinatorAgent:
    """REASON chooses next actions; OBSERVE updates state; REASON decides to replan."""
    def __init__(self, source: Source, provider: OfferProvider,
                 store: StateStore, mode: str = "demo", semantic: bool = False):
        self.mode = mode
        self.store = store
        self.retrieval = RetrievalAgent(source, EvidenceRetriever(semantic))
        self.risk = RiskAgent()
        self.planning = PlanningAgent(provider)
        self.guard = ValidationAndActionAgent()

    @staticmethod
    def signature(items: list[Evidence], trip: Trip) -> str:
        material = {
            "events": sorted(list(x.identity()) for x in items if x.kind != "METAR"),
            "constraints": [trip.origin, trip.destination, trip.deadline.isoformat(),
                            trip.budget_usd, trip.flight_number, trip.auto_booking_authorized,
                            trip.auto_booking_limit_usd, trip.preferences],
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def run(self, trip: Trip, simulate_auto_booking: bool = False) -> AgentReport:
        if trip.deadline <= utcnow():
            raise ValueError("Trip deadline must be in the future")
        # ReAct Thought: identify missing flight, weather and restriction evidence.
        # Action: query read-only external sources.
        raw, selected = self.retrieval.execute(trip)
        # Observation: collect full set plus semantically retrieved relevant subset.
        level, reasons = self.risk.assess(trip, raw)
        sig = self.signature(raw, trip)
        previous = self.store.latest_signature(trip.trip_id)
        changed = previous != sig
        canceled = any(x.kind == "FLIGHT_CANCELLED" and x.fresh() for x in raw)
        # If the same risk environment is observed again, don't create alert spam.
        # The first run shows a plan in the simulated cancellation scenario.
        replan = (changed and (canceled or level in {"HIGH", "ELEVATED"}))
        decisions: list[str] = []
        escalations: list[str] = []
        booking = "NOT_ATTEMPTED"
        candidates: list[Candidate] = []
        if replan:
            decisions.append("Risk state changed; initiate bounded ToT alternative planning")
            candidates = self.planning.execute(trip, utcnow())
            candidates = self.guard.validate(trip, candidates, raw, self.mode)
            booking, escalations = self.guard.act(trip, candidates, self.mode, simulate_auto_booking)
            if not candidates:
                decisions.append("No verified offers available. Do not invent prices or seats.")
            else:
                decisions.append(f"Generated {len(candidates)} validated or clearly labeled synthetic route options")
        elif level == "UNKNOWN":
            decisions.append("Travel risk unknown because key evidence is unavailable; do NOT assume the original plan is safe or valid")
            escalations.append("Manually verify flight, airport, and weather status before taking action")
        elif not changed:
            decisions.append("No material change; continue monitoring without redundant replanning")
        else:
            decisions.append("No confirmed disruption in sampled feeds; keep monitoring (flight status is not independently verified)")
        if self.mode == "live":
            decisions.append("Public feeds give airport/weather status, NOT passenger flight status or seat availability")
            if any(x.kind == "SOURCE_UNAVAILABLE" for x in raw):
                escalations.append("At least one source is unavailable; corroborate before relying on safety-critical advice")
        report = AgentReport(trip, self.mode, utcnow(), raw, selected, level, reasons,
                             candidates, decisions, escalations, booking, replan, changed)
        self.store.record(report, sig)
        return report
