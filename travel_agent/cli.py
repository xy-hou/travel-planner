from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import time
from uuid import uuid4

from .agents import CoordinatorAgent
from .feeds import DemoFeeds, LiveFeeds
from .models import Trip, utcnow
from .planner import NoTicketProvider, SyntheticOfferProvider
from .store import StateStore


def render(report) -> None:
    print("\n" + "="*76)
    print(f"Travel Risk Research Agent | {report.mode.upper()} | Trip {report.trip.trip_id}")
    print("="*76)
    print(f"{report.trip.origin} → {report.trip.destination} | Deadline UTC: {report.trip.deadline:%Y-%m-%d %H:%M}")
    print(f"Risk classification: {report.risk_level}; State changed: {report.state_changed}")
    for message in report.risk_reasons[:6]:
        print(f"  Risk evidence: {message}")
    print("Actions:")
    for decision in report.decisions:
        print(f"  • {decision}")
    print(f"Evidence collected: {len(report.evidence)}; retrieval top results: {len(report.retrieved_evidence)}")
    for e in report.evidence[:12]:
        print(f"  [{e.source}] {e.place} | {e.kind}: {e.summary[:125]} | Fresh: {e.fresh()}")
        print(f"    {e.source_url}")
    if report.candidates:
        print("\nAlternative itineraries (SYNTHETIC prices/times, NOT bookable):")
        for idx, c in enumerate(report.candidates, 1):
            print(f"  {idx}. ${c.total_usd:.2f}, arrival {c.arrival:%Y-%m-%d %H:%M} UTC, {c.status}")
            for l in c.legs:
                print(f"     {l.origin} → {l.destination} [{l.mode}] {l.depart:%m/%d %H:%M} → {l.arrive:%m/%d %H:%M} UTC")
            if c.issues:
                print(f"     Issues: {', '.join(c.issues)}")
    print("Booking:", report.booking_result)
    for item in report.escalations:
        print("Human review:", item)
    if report.mode == "live":
        print("NOTICE: Airport-level alerts are NOT flight-specific status; no real ticket inventory or booking is available.")
    else:
        print("NOTICE: Entire demo is fictional. No live airline availability, payment, or booking.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Research MVP: 5-role travel risk agent")
    parser.add_argument("mode", choices=["demo", "live"])
    parser.add_argument("--db", default="travel_agent_state.sqlite3")
    parser.add_argument("--trip-id", default="")
    parser.add_argument("--origin", default="MIA")
    parser.add_argument("--destination", default="NYC")
    parser.add_argument("--flight", default="DEMO123")
    parser.add_argument("--deadline", default="", help="ISO8601 with timezone, e.g. 2026-12-01T20:00:00-05:00")
    parser.add_argument("--budget", type=float, default=650.0)
    parser.add_argument("--semantic", action="store_true", help="Use true sentence embeddings (optional heavy dependency)")
    parser.add_argument("--news", action="store_true", help="Also query GDELT for unverified news")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON reports")
    parser.add_argument("--authorize-auto-booking", action="store_true", help="For synthetic demo only; never charges")
    parser.add_argument("--auto-book-max", type=float, default=0.0)
    parser.add_argument("--simulate-auto-book", action="store_true", help="Simulate a fake booking; no payments")
    parser.add_argument("--steps", default="1,2,3,4", help="Demo steps: 1,2,3,4 (or 2)")
    parser.add_argument("--cycles", type=int, default=1, help="Live checks, max 100; default 1")
    parser.add_argument("--interval", type=int, default=600, help="Polling seconds, min 60")
    args = parser.parse_args(argv)
    if args.mode == "live" and (args.simulate_auto_book or args.authorize_auto_booking):
        parser.error("Live mode cannot perform purchases. Remove auto-booking flags.")
    if args.mode == "live" and not args.deadline:
        parser.error("Live mode requires --deadline with an explicit timezone offset")
    if not 1 <= args.cycles <= 100 or args.interval < 60:
        parser.error("cycles must be 1..100 and interval >= 60 seconds")
    if args.authorize_auto_booking and args.auto_book_max <= 0:
        parser.error("--authorize-auto-booking requires --auto-book-max > 0")
    if args.simulate_auto_book and not args.authorize_auto_booking:
        parser.error("Simulation requires --authorize-auto-booking and --auto-book-max")
    deadline = args.deadline or (utcnow() + timedelta(hours=48)).isoformat()
    trip = Trip(trip_id=args.trip_id or ("demo-" + uuid4().hex[:8]),
                origin=args.origin, destination=args.destination,
                deadline=deadline, budget_usd=args.budget,
                flight_number=args.flight,
                auto_booking_authorized=args.authorize_auto_booking,
                auto_booking_limit_usd=args.auto_book_max)
    store = StateStore(args.db)
    try:
        if args.mode == "demo":
            try:
                steps = [int(n) for n in args.steps.split(",")]
                if not steps or any(n not in (1, 2, 3, 4) for n in steps):
                    raise ValueError
            except ValueError:
                parser.error("--steps should be comma-separated from 1,2,3,4")
            for step in steps:
                agent = CoordinatorAgent(DemoFeeds(step), SyntheticOfferProvider(),
                                         store, mode="demo", semantic=args.semantic)
                report = agent.run(trip, simulate_auto_booking=args.simulate_auto_book)
                print(json.dumps(report.to_dict(), indent=2) if args.json else
                      f"\n--- SIMULATED SCENARIO STEP {step} ---")
                if not args.json:
                    render(report)
        else:
            agent = CoordinatorAgent(LiveFeeds(include_news=args.news), NoTicketProvider(),
                                     store, mode="live", semantic=args.semantic)
            for cycle in range(args.cycles):
                report = agent.run(trip)
                if args.json:
                    print(json.dumps(report.to_dict(), indent=2))
                else:
                    render(report)
                if cycle + 1 < args.cycles:
                    time.sleep(args.interval)
    finally:
        store.close()


if __name__ == "__main__":
    main()
