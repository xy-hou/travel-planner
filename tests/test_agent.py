from __future__ import annotations

from datetime import timedelta

import pytest

from travel_agent.agents import CoordinatorAgent, ValidationAndActionAgent
from travel_agent.feeds import DemoFeeds, parse_faa_xml
from travel_agent.models import Trip, utcnow
from travel_agent.planner import BeamSearchPlanner, SyntheticOfferProvider
from travel_agent.store import StateStore


def trip(budget=700, auto=False, limit=0):
    return Trip("test-trip", "MIA", "NYC", utcnow()+timedelta(hours=48),
                budget, "TEST123", auto, limit)


def agent(tmp_path, step, store=None):
    store = store or StateStore(str(tmp_path/"state.sqlite3"))
    return CoordinatorAgent(DemoFeeds(step), SyntheticOfferProvider(), store), store


def test_cancelled_flight_triggers_replanning(tmp_path):
    a, s = agent(tmp_path, 2)
    r = a.run(trip())
    assert r.replan_triggered
    assert r.risk_level == "HIGH"
    assert len(r.candidates) >= 2
    assert all(c.status == "SIMULATED_OPTION" for c in r.candidates)
    assert r.booking_result == "NOT_ATTEMPTED"
    s.close()


def test_repeated_state_does_not_replan(tmp_path):
    a, s = agent(tmp_path, 2)
    t = trip()
    assert a.run(t).replan_triggered
    assert not a.run(t).replan_triggered
    s.close()


def test_new_road_closure_changes_output(tmp_path):
    t = trip()
    a, s = agent(tmp_path, 2)
    before = a.run(t)
    b, _ = agent(tmp_path, 3, s)
    after = b.run(t)
    assert after.state_changed
    assert after.replan_triggered
    assert not any(any(l.code == "DEMO-DRIVE" for l in c.legs) for c in after.candidates)
    s.close()


def test_no_unauthorized_booking(tmp_path):
    a, s = agent(tmp_path, 2)
    r = a.run(trip(auto=False), simulate_auto_booking=True)
    assert r.booking_result == "NOT_ATTEMPTED"
    s.close()


def test_explicit_demo_booking_never_charges(tmp_path):
    a, s = agent(tmp_path, 4)
    r = a.run(trip(auto=True, limit=650), simulate_auto_booking=True)
    assert r.booking_result == "SIMULATED_BOOKING_ONLY"
    assert "no payment" in " ".join(r.escalations).lower()
    s.close()


def test_budget_limit_blocks_demo_booking(tmp_path):
    a, s = agent(tmp_path, 4)
    r = a.run(trip(auto=True, limit=1), simulate_auto_booking=True)
    assert r.booking_result == "BLOCKED_BY_BUDGET"
    s.close()


def test_deadline_prunes_routes():
    t = trip()
    now = utcnow()
    t.deadline = now + timedelta(hours=3)
    result = BeamSearchPlanner().plan(t, SyntheticOfferProvider().offers(t, now), now)
    assert not result


def test_no_flight_inventory_is_not_invented(tmp_path):
    from travel_agent.planner import NoTicketProvider
    assert NoTicketProvider().offers(trip(), utcnow()) == []


def test_faa_xml_parsing():
    sample = '''<AIRPORT_STATUS_INFORMATION><Delay_type><Name>Ground Stops</Name>
      <Ground_Stop_List><Ground_Stop><ARPT>EWR</ARPT><Reason>Snow</Reason></Ground_Stop>
      </Ground_Stop_List></Delay_type></AIRPORT_STATUS_INFORMATION>'''
    ev = parse_faa_xml(sample, ["EWR", "JFK"])
    assert len(ev) == 1 and ev[0].place == "EWR" and "Ground Stops" in ev[0].summary


def test_datetime_must_include_timezone():
    with pytest.raises(ValueError):
        Trip("t", "MIA", "NYC", "2026-12-01T12:00:00", 100)


def test_weather_risk_blocks_even_authorized_booking(tmp_path):
    a, s = agent(tmp_path, 2)
    r = a.run(trip(auto=True, limit=650), simulate_auto_booking=True)
    assert r.booking_result == "BLOCKED_BY_VERIFICATION"
    s.close()


def test_changed_budget_triggers_replanning(tmp_path):
    a, s = agent(tmp_path, 2)
    first = trip(budget=700)
    assert a.run(first).replan_triggered
    altered = trip(budget=550)
    assert a.run(altered).replan_triggered
    s.close()


def test_live_missing_sources_fail_closed(tmp_path):
    from travel_agent.models import Evidence
    from travel_agent.planner import NoTicketProvider
    class OfflineSource:
        def collect(self, request):
            from travel_agent.feeds import make_evidence
            return [make_evidence("SOURCE_UNAVAILABLE", "MIA", "network unreachable",
                                  "NWS", "https://api.weather.gov", False, "UNKNOWN")]
    store = StateStore(str(tmp_path/"live.sqlite3"))
    coordinator = CoordinatorAgent(OfflineSource(), NoTicketProvider(), store, mode="live")
    report = coordinator.run(trip())
    assert report.risk_level == "UNKNOWN"
    assert report.candidates == []
    assert "do NOT assume" in " ".join(report.decisions)
    assert report.escalations
    store.close()
