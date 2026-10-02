"""Live-data nodes. Code only, no LLM: every number the bot can ever quote is computed here."""
from datetime import datetime, timedelta

import requests
from langchain_core.messages import AIMessage

from nodes.state import WINDOWS, State

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10

HOURLY = ["temperature_2m", "apparent_temperature", "precipitation", "precipitation_probability",
          "wind_speed_10m", "wind_gusts_10m", "uv_index", "weather_code"]

# Window -> (days from today, first hour, last hour), local time. None = the current hour.
WINDOW_HOURS = {
    "now": (0, None, None),
    "this_morning": (0, 6, 11),
    "this_afternoon": (0, 12, 16),
    "this_evening": (0, 17, 20),
    "tonight": (0, 21, 23),
    "today": (0, None, 23),
    "tomorrow": (1, 6, 21),
}
assert set(WINDOW_HOURS) == set(WINDOWS)

# Every metric summarise() produces; SOP conditions may only use these.
METRICS = ["temp_max_c", "temp_min_c", "feels_like_max_c", "feels_like_min_c", "precip_total_mm", "precip_prob_max_pct", "wind_max_kmh",
           "gust_max_kmh", "uv_max", "thunderstorm", "rain_past_24h_mm", "rain_next_24h_mm",
           "rain_48h_mm", "rain_hours_48h"]


def _get(url, **params):
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def geocode(state: State):
    """'Bhopal, Madhya Pradesh' -> search 'Bhopal', prefer the result whose state/country matches the hint, else first."""
    name, _, hint = state["location"].partition(",")
    name, hint = name.strip(), hint.strip().lower()
    try:
        results = _get(GEOCODE_URL, name=name, count=10, language="en", format="json").get("results") or []
    except (requests.RequestException, ValueError) as e:
        return {"error": f"the location service is unreachable ({type(e).__name__})"}
    if not results:
        return {"error": f"I couldn't find a place called '{name}'"}
    best = next((r for r in results
                 if hint and hint in f"{r.get('admin1', '')} {r.get('country', '')}".lower()), results[0])
    label = ", ".join(filter(None, [best["name"], best.get("admin1"), best.get("country")]))
    return {"place": {"name": label, "lat": best["latitude"], "lon": best["longitude"]}}


def summarise(data: dict, window: str) -> dict:
    """Aggregate hourly forecast into the metrics SOPs are checked against. Raises ValueError on missing data."""
    hourly = data["hourly"]
    now = datetime.fromisoformat(data["current"]["time"]).replace(minute=0)
    times = [datetime.fromisoformat(t) for t in hourly["time"]]
    days, first, last = WINDOW_HOURS[window]
    date = now.date() + timedelta(days=days)
    first = now.hour if first is None else first
    last = now.hour if last is None else last

    def rows(pred):
        return [i for i, t in enumerate(times) if pred(t)]

    def col(name, idx):
        vals = [hourly[name][i] for i in idx if hourly[name][i] is not None]
        if not vals:
            raise ValueError(f"no {name} data for {window}")
        return vals

    win = rows(lambda t: t.date() == date and first <= t.hour <= last)
    past24 = rows(lambda t: now - timedelta(hours=24) <= t < now)
    next24 = rows(lambda t: now <= t < now + timedelta(hours=24))
    # Rounded once, here: the number the user sees is exactly the number the SOP thresholds were checked against.
    return {
        "hours": f"{date} {first:02d}:00-{last:02d}:59",
        "metrics": {
            "temp_max_c": round(max(col("temperature_2m", win))),
            "temp_min_c": round(min(col("temperature_2m", win))),
            "feels_like_max_c": round(max(col("apparent_temperature", win))),
            "feels_like_min_c": round(min(col("apparent_temperature", win))),
            "precip_total_mm": round(sum(col("precipitation", win)), 1),
            "precip_prob_max_pct": round(max(col("precipitation_probability", win))),
            "wind_max_kmh": round(max(col("wind_speed_10m", win))),
            "gust_max_kmh": round(max(col("wind_gusts_10m", win))),
            "uv_max": round(max(col("uv_index", win))),
            "thunderstorm": any(c >= 95 for c in col("weather_code", win)),  # WMO codes 95-99
            # Window-independent: catches a rain *system*, not just this window's rain.
            "rain_past_24h_mm": round(sum(col("precipitation", past24)), 1),
            "rain_next_24h_mm": round(sum(col("precipitation", next24)), 1),
            # Persistence: a low-pressure system rains for many hours, often never "heavy" in any one hour or day
            # (Bhopal 3-4 Sep 2026: rain in 38 of 48 h, 32.9 mm, no hour above 3.2 mm).
            "rain_48h_mm": round(sum(col("precipitation", past24 + next24)), 1),
            "rain_hours_48h": sum(r >= 0.1 for r in col("precipitation", past24 + next24)),
        },
    }


def fetch_weather(state: State):
    p = state["place"]
    try:
        data = _get(FORECAST_URL, latitude=p["lat"], longitude=p["lon"], hourly=",".join(HOURLY),
                    current="temperature_2m", timezone="auto", past_days=1, forecast_days=2)
        summary = summarise(data, state["window"])
    except requests.RequestException as e:
        return {"error": f"the weather service is unreachable ({type(e).__name__})"}
    except (ValueError, KeyError, TypeError) as e:
        return {"error": f"the weather service returned incomplete data ({e})"}
    return {"weather": {"place": p["name"], "window": state["window"],
                        "observed_at": data["current"]["time"], **summary}}


def weather_failed(state: State):
    # Fixed text, no LLM, no numbers: geocode and forecast failures both land here.
    reason = f"I couldn't get live weather for {state['location']} ({state['error']}), so no policy was checked"
    return {"last_decision": {"sops": [], "reason": reason}, "messages": [AIMessage(
        f"Sorry, I couldn't get live weather for **{state['location']}**: {state['error']}. "
        "I won't guess at conditions without real data, so I can't give advice right now. "
        "Please try again shortly, or name a nearby larger city.")]}


if __name__ == "__main__":
    # Offline check of window slicing on synthetic data: 2 days of hours, value = hour of day.
    start = datetime(2026, 10, 1)
    hours = [start + timedelta(hours=i) for i in range(72)]
    fake = {"current": {"time": "2026-10-02T13:30"},
            "hourly": {"time": [t.isoformat() for t in hours],
                       **{f: [float(t.hour) for t in hours] for f in HOURLY},
                       "precipitation": [1.0] * 72}}
    s = summarise(fake, "this_evening")
    assert s["hours"] == "2026-10-02 17:00-20:59" and s["metrics"]["temp_max_c"] == 20 and s["metrics"]["temp_min_c"] == 17
    assert s["metrics"]["precip_total_mm"] == 4.0                     # 4 hours x 1 mm
    assert s["metrics"]["rain_past_24h_mm"] == s["metrics"]["rain_next_24h_mm"] == 24.0
    assert (s["metrics"]["rain_48h_mm"], s["metrics"]["rain_hours_48h"]) == (48.0, 48)
    assert list(s["metrics"]) == METRICS
    assert summarise(fake, "now")["metrics"]["temp_max_c"] == 13      # current hour only
    assert summarise(fake, "tomorrow")["hours"].startswith("2026-10-03")
    assert summarise(fake, "today")["metrics"]["temp_min_c"] == 13     # rest of today, not from midnight
    fake["hourly"]["uv_index"] = [None] * 72
    try:
        summarise(fake, "today")
        raise AssertionError("missing data must raise")
    except ValueError:
        pass
    print("summarise ok")
