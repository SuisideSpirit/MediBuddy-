"""Routing tests.

Router only (no LLM):  python tests/test_graph.py --offline
Plus live chat (Groq): python tests/test_graph.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console + LLM unicode
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import nullcontext
from unittest.mock import patch

import requests
from langchain_core.messages import HumanMessage

from graph import after_fetch, after_geocode, after_match, after_understand, build
from nodes import weather

# Router: pure function, no LLM needed.
assert after_understand({"explain": True, "on_topic": True, "location": None}) == "explain"
assert after_understand({"on_topic": False, "location": "Bhopal"}) == "no_guidance"
assert after_understand({"on_topic": True, "location": None}) == "ask_location"
assert after_understand({"on_topic": True}) == "ask_location"
assert after_understand({"on_topic": True, "location": "Bhopal"}) == "geocode"
assert after_geocode({"error": "not found"}) == "weather_failed"
assert after_geocode({"error": None}) == "fetch_weather"
assert after_fetch({"error": "down"}) == "weather_failed"
assert after_fetch({"error": None}) == "match_sops"
assert after_match({"sops": []}) == "no_guidance"
assert after_match({"sops": [{"id": "X-01"}]}) == "compose"
assert after_match({"sops": [{"id": "RAIN-SYS-01"}], "situational": True}) == "override"
assert after_understand({"error": "llm down", "on_topic": False}) == "intake_failed"
print("router ok")

if "--offline" in sys.argv:
    sys.exit()

real_get = weather.requests.get


def outage(*a, **k):  # whole Open-Meteo down
    raise requests.ConnectionError("simulated outage")


def forecast_outage(url, *a, **k):  # geocoding works, forecast API down
    if url == weather.FORECAST_URL:
        raise requests.ConnectionError("simulated outage")
    return real_get(url, *a, **k)


# Live: one session (thread_id). Each turn: (message, node the graph must end on, fake requests.get or None).
TURNS = [
    ("can my grandpa go for his walk tomorrow morning?", "ask_location", None),
    ("Bhopal", "compose", None),  # memory: grandpa/walk/tomorrow carried over; real weather fetched
    ("what about in Xqzvbtown?", "weather_failed", None),  # geocode finds nothing
    ("ok what about Bhopal this evening?", "weather_failed", outage),  # Open-Meteo unreachable
    ("and tomorrow?", "weather_failed", forecast_outage),  # city resolves, forecast fails
    ("what's the capital of France?", "no_guidance", None),
    ("what about tomorrow then?", "compose", None),  # memory survived the off-topic turn
    ("why did you say that?", "explain", None),  # replays last decision, no new fetch
]

app = build()
cfg = {"configurable": {"thread_id": "test"}}
fails = 0
for msg, want, down in TURNS:
    with patch("nodes.weather.requests.get", down) if down else nullcontext():
        path = [node for step in app.stream({"messages": [HumanMessage(msg)]}, cfg, stream_mode="updates") for node in step]
    state = app.get_state(cfg).values
    ok = path[-1] == want
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {msg!r}\n      path: {' -> '.join(path)}  (want {want})")
    print(f"      memory: { {k: state.get(k) for k in ('activity', 'audience', 'location', 'window')} }")
    print(f"      bot: {state['messages'][-1].content}")

print(f"\n{len(TURNS) - fails}/{len(TURNS)} passed")
sys.exit(1 if fails else 0)
