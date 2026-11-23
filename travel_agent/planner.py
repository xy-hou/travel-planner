"""Bounded Tree-of-Thoughts-inspired beam search over verified provider legs.

Each 'thought' is a partial itinerary; each edge adds a leg. This is a
transparent search algorithm, not an LLM's hidden chain of thought.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from .models import Candidate, Leg, Trip, utcnow


class OfferProvider(Protocol):
    def offers(self, trip: Trip, at: datetime) -> list[Leg]: ...


class NoTicketProvider:
    """Public government feeds do not supply flight offers or passenger statuses."""
    def offers(self, trip: Trip, at: datetime) -> list[Leg]:
        return []


class SyntheticOfferProvider:
    """Demo offers only: fictional prices, schedules, and availability."""
    def offers(self, trip: Trip, at: datetime) -> list[Leg]:
        if trip.origin != "MIA" or trip.destination != "NYC":
            return []
        # Round to the nearest minute so repeat demo runs remain readable.
        t = at.replace(second=0, microsecond=0)
        def leg(name, a, b, mode, d, ar, price):
            return Leg(name, a, b, mode, t+timedelta(hours=d),
                       t+timedelta(hours=ar), price, "SYNTHETIC",
                       True, True)
        return [
            leg("DEMO-FLL-TRANSFER", "MIA", "FLL", "ground", 0.25, 1.75, 45),
            leg("DEMO-ATL", "MIA", "ATL", "flight", 2, 4.25, 180),
            leg("DEMO-PHL", "MIA", "PHL", "flight", 4, 7.5, 300),
            leg("DEMO-DRIVE", "MIA", "NYC", "ground", 1, 29, 290),
            leg("DEMO-FLL-PHL", "FLL", "PHL", "flight", 4, 7, 225),
            leg("DEMO-ATL-NYC", "ATL", "NYC", "flight", 7, 10, 270),
            leg("DEMO-PHL-RAIL", "PHL", "NYC", "rail", 9, 11, 65),
        ]


@dataclass
class PartialRoute:
    legs: tuple[Leg, ...]
    place: str
    finish: datetime
    total: float


class BeamSearchPlanner:
    def __init__(self, branch_factor: int = 4, beam_width: int = 3,
                 depth_limit: int = 3):
        if min(branch_factor, beam_width, depth_limit) < 1:
            raise ValueError("Beam parameters must be >= 1")
        self.branch_factor = branch_factor
        self.beam_width = beam_width
        self.depth_limit = depth_limit

    @staticmethod
    def _rank(node: PartialRoute, trip: Trip, at: datetime) -> float:
        # Heuristic, not a calibrated probability; lower scores rank higher.
        travel_hours = (node.finish - at).total_seconds() / 3600
        arrival_bonus = -7 if node.place == trip.destination else 0
        return node.total / 60 + travel_hours * 0.22 + 0.6 * len(node.legs) + arrival_bonus

    def plan(self, trip: Trip, offers: list[Leg], at: datetime | None = None) -> list[Candidate]:
        at = at or utcnow()
        beam = [PartialRoute((), trip.origin, at, 0.0)]
        completed: dict[tuple[str, ...], Candidate] = {}
        for _ in range(self.depth_limit):
            next_beam: list[PartialRoute] = []
            for state in beam:
                outgoing = sorted(
                    (leg for leg in offers if leg.origin == state.place and
                     leg.depart >= state.finish + timedelta(minutes=15) and
                     leg.arrive <= trip.deadline and
                     leg.destination not in [x.origin for x in state.legs] and
                     state.total + leg.price_usd <= trip.budget_usd),
                    key=lambda l: (l.arrive, l.price_usd))[:self.branch_factor]
                for leg in outgoing:
                    new = PartialRoute(state.legs + (leg,), leg.destination,
                                       leg.arrive, state.total + leg.price_usd)
                    if new.place == trip.destination:
                        code = tuple(x.code for x in new.legs)
                        completed[code] = Candidate(list(new.legs), new.total, new.finish,
                                                     score=self._rank(new, trip, at))
                    else:
                        next_beam.append(new)
            beam = sorted(next_beam, key=lambda n: self._rank(n, trip, at))[:self.beam_width]
            if not beam:
                break
        return sorted(completed.values(), key=lambda c: c.score)[:8]
