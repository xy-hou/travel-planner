"""External observations. Public feeds are READ ONLY and never supply ticket offers.

A flight's cancellation must come from a flight-specific provider; an FAA airport
restriction is not equivalent to cancellation of a given passenger's flight.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import re
import xml.etree.ElementTree as ET
from typing import Iterable

import requests

from .models import Evidence, Trip, parse_dt, utcnow

FAA_URL = "https://nasstatus.faa.gov/api/airport-status-information"
NWS_URL = "https://api.weather.gov/alerts/active"
AWC_URL = "https://aviationweather.gov/api/data/metar"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Fixed airport geometry for the scope of this research MVP, not an airport catalog.
AIRPORTS = {
    "MIA": ("KMIA", 25.795, -80.290),
    "FLL": ("KFLL", 26.074, -80.151),
    "PBI": ("KPBI", 26.683, -80.096),
    "EWR": ("KEWR", 40.689, -74.175),
    "JFK": ("KJFK", 40.641, -73.778),
    "LGA": ("KLGA", 40.777, -73.873),
    "PHL": ("KPHL", 39.873, -75.243),
    "ATL": ("KATL", 33.640, -84.427),
    "MCO": ("KMCO", 28.431, -81.309),
}


def norm_code(place: str) -> str:
    return place.strip().upper()


def related_airports(trip: Trip) -> list[str]:
    airports = [trip.origin, trip.destination]
    if trip.destination == "NYC":
        airports += ["EWR", "JFK", "LGA", "PHL"]
    if trip.origin == "MIA":
        airports += ["FLL", "PBI"]
    return list(dict.fromkeys(a for a in airports if a in AIRPORTS))


def make_evidence(kind: str, place: str, summary: str, source: str,
                  url: str, official: bool, severity: str = "INFO",
                  source_time: datetime | None = None, ttl_minutes: int = 30,
                  simulated: bool = False) -> Evidence:
    now = utcnow()
    obs = source_time or now
    if obs > now + timedelta(minutes=5):
        obs = now
    return Evidence(kind, place, summary, source, url, obs, now,
                    min(now + timedelta(minutes=ttl_minutes), obs + timedelta(minutes=ttl_minutes)),
                    official, severity, simulated)


def _time(value: object) -> datetime | None:
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if not value:
            return None
        s = str(value).strip()
        if re.fullmatch(r"\d{12,14}", s):
            dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S" if len(s) == 14 else "%Y%m%d%H%M")
            return dt.replace(tzinfo=timezone.utc)
        return parse_dt(s)
    except (ValueError, OverflowError, TypeError):
        return None


def _get(url: str, params: dict | None = None, accept: str = "application/json") -> requests.Response:
    resp = requests.get(url, params=params, timeout=9, headers={
        "User-Agent": __import__("os").environ.get("TRAVEL_AGENT_USER_AGENT", "TravelRiskResearchAgent/0.1 (research prototype)"),
        "Accept": accept,
    })
    resp.raise_for_status()
    return resp


def parse_faa_xml(xml_text: str, airports: Iterable[str], retrieved_at: datetime | None = None) -> list[Evidence]:
    """Parse FAA's type-grouped XML. Handles <ARPT> and <Airport> code tags.

    Only emitted matches are evidence of a reported event; absence of a match
    doesn't imply the airport or every airline flight is operating normally.
    """
    root = ET.fromstring(xml_text)
    expected = {x.upper() for x in airports}
    out: list[Evidence] = []
    delay_types = root.findall(".//Delay_type")
    if not delay_types:
        delay_types = root.findall(".//Delay_Type")
    now = retrieved_at or utcnow()
    for group in delay_types:
        event_type = (group.findtext("Name") or "FAA operational event").strip()
        for parent in group.iter():
            for item in parent:
                tag = item.tag.split("}")[-1].lower()
                if tag not in {"arpt", "airport", "airport_code", "iata"}:
                    continue
                airport = (item.text or "").strip().upper()
                if airport not in expected:
                    continue
                details = " ".join((x.text or "").strip() for x in parent if x.text and x is not item)
                details = re.sub(r"\s+", " ", details).strip()[:270]
                message = f"{event_type}: {details}" if details else event_type
                out.append(make_evidence("FAA_RESTRICTION", airport, message, "FAA NAS Status",
                                         FAA_URL, True, "HIGH", source_time=now, ttl_minutes=15))
    # De-duplicate airport+program combinations.
    return list({x.identity(): x for x in out}.values())


class LiveFeeds:
    def __init__(self, include_news: bool = False):
        self.include_news = include_news

    @staticmethod
    def _error(source: str, url: str, place: str, exception: Exception) -> Evidence:
        return make_evidence("SOURCE_UNAVAILABLE", place,
                             f"{source} could not be read ({type(exception).__name__}); do not assume normal operation",
                             source, url, False, "UNKNOWN", ttl_minutes=5)

    def _faa(self, airports: list[str]) -> list[Evidence]:
        try:
            response = _get(FAA_URL, accept="application/xml, text/xml")
            results = parse_faa_xml(response.text, airports)
            if not results:
                return [make_evidence("FAA_NO_MATCHING_EVENTS", ",".join(airports),
                          "No matching airport-wide event in retrieved FAA snapshot; not a flight-status guarantee",
                          "FAA NAS Status", FAA_URL, True, ttl_minutes=15)]
            return results
        except (requests.RequestException, ET.ParseError) as exc:
            return [self._error("FAA NAS Status", FAA_URL, ",".join(airports), exc)]

    def _nws(self, airport: str) -> list[Evidence]:
        _, lat, lon = AIRPORTS[airport]
        try:
            response = _get(NWS_URL, {"point": f"{lat},{lon}"}, "application/geo+json")
            features = response.json().get("features", [])
            output = []
            for f in features[:8]:
                p = f.get("properties") or {}
                severity = str(p.get("severity") or "Unknown").upper()
                issued = _time(p.get("sent")) or utcnow()
                output.append(make_evidence("NWS_ALERT", airport,
                     f"{p.get('event', 'Weather alert')}: {p.get('headline') or p.get('description', '')[:160]}",
                     "National Weather Service", str(p.get("@id") or f.get("id") or NWS_URL),
                     True, severity, issued, ttl_minutes=240))
                # NWS active alert may have been issued many hours ago, while still
                # in effect. Trust the active feed's time of retrieval and the
                # published expiry, not a fixed time since initial issuance.
                official_expiry = _time(p.get("expires"))
                output[-1].expires_at = min(utcnow() + timedelta(minutes=15),
                                            official_expiry or utcnow() + timedelta(minutes=15))
            return output or [make_evidence("NWS_NO_ACTIVE_ALERT", airport,
                 "No active NWS alerts at this point in the latest response (not a safety guarantee)",
                 "National Weather Service", NWS_URL, True, ttl_minutes=10)]
        except (requests.RequestException, ValueError) as exc:
            return [self._error("National Weather Service", NWS_URL, airport, exc)]

    def _metar(self, airport: str) -> list[Evidence]:
        code = AIRPORTS[airport][0]
        try:
            response = _get(AWC_URL, {"ids": code, "format": "json"})
            if response.status_code == 204:
                return [make_evidence("METAR_MISSING", airport, "No recent METAR response",
                                      "NOAA Aviation Weather", AWC_URL, True, "UNKNOWN", ttl_minutes=10)]
            results = response.json()
            if not isinstance(results, list) or not results:
                raise ValueError("METAR response contains no observations")
            metar = results[0]
            issued = _time(metar.get("obsTime")) or _time(metar.get("reportTime"))
            raw = str(metar.get("rawOb") or metar.get("raw_text") or "")[:250]
            return [make_evidence("METAR", airport, raw or "METAR received without raw observation",
                                  "NOAA Aviation Weather", AWC_URL, True, "INFO",
                                  issued, ttl_minutes=90)]
        except (requests.RequestException, ValueError, TypeError) as exc:
            return [self._error("NOAA Aviation Weather", AWC_URL, airport, exc)]

    def _news(self, trip: Trip) -> list[Evidence]:
        # News is supplemental, never authoritative for flight cancellations.
        location = "New York" if trip.destination == "NYC" else trip.destination
        try:
            response = _get(GDELT_URL, {"query": f'"{location}" (airport OR storm OR train OR delay)',
                        "mode": "artlist", "format": "json", "maxrecords": 5, "timespan": "24h"})
            output = []
            for article in response.json().get("articles", [])[:5]:
                published = _time(article.get("seendate")) or utcnow()
                output.append(make_evidence("NEWS", location,
                     str(article.get("title", "Untitled news story"))[:220],
                     str(article.get("domain", "GDELT indexed article")),
                     str(article.get("url") or GDELT_URL), False, "UNVERIFIED",
                     published, ttl_minutes=1440))
            return output
        except (requests.RequestException, ValueError, TypeError) as exc:
            return [self._error("GDELT news", GDELT_URL, location, exc)]

    def collect(self, trip: Trip) -> list[Evidence]:
        airports = related_airports(trip)
        funcs = [lambda: self._faa(airports)]
        # Keep API usage bounded; check the requested endpoints and key NYC alternates.
        weather_airports = list(dict.fromkeys(
            [a for a in (trip.origin, "EWR" if trip.destination == "NYC" else trip.destination,
                         "PHL" if trip.destination == "NYC" else None) if a in AIRPORTS]))
        for a in weather_airports:
            funcs.append(lambda a=a: self._nws(a))
        for a in weather_airports[:2]:
            funcs.append(lambda a=a: self._metar(a))
        if self.include_news:
            funcs.append(lambda: self._news(trip))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda f: f(), funcs))
        return [item for batch in results for item in batch]


class DemoFeeds:
    """Deliberately fictional observations, never real operational facts."""
    def __init__(self, step: int = 2):
        if step not in (1, 2, 3, 4):
            raise ValueError("step must be 1, 2, 3, or 4")
        self.step = step

    def collect(self, trip: Trip) -> list[Evidence]:
        def e(kind: str, place: str, summary: str, severity: str = "INFO") -> Evidence:
            return make_evidence(kind, place, summary, "SIMULATED demo feed",
                      "https://example.invalid/synthetic-fixture", False, severity,
                      ttl_minutes=60, simulated=True)
        data = ([e("STORM_CLEARED", "NYC",
                   "Simulated weather monitoring reports the prior warning has expired")]
                if self.step == 4 else
                [e("STORM_WARNING", "NYC",
                   "Synthetic winter storm warning near New York airports", "HIGH")])
        if self.step == 1:
            data.append(e("FLIGHT_STATUS", trip.flight_number or "DEMO123",
                          "Flight status: SCHEDULED (synthetic)"))
        else:
            data.append(e("FLIGHT_CANCELLED", trip.flight_number or "DEMO123",
                          "Flight status: CANCELLED (synthetic)", "HIGH"))
        if self.step == 3:
            data.append(e("GROUND_CLOSURE", "MIA->NYC",
                          "Synthetic severe road closure on the direct ground route", "HIGH"))
        return data
