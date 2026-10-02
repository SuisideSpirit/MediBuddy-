"""Offline checks: no LLM key, no network. Runs each module's built-in self-check plus the graph routers."""
import runpy

import pytest

from graph import after_fetch, after_geocode, after_match, after_understand, build

MODULES = ["policy", "nodes.match_sops", "nodes.weather", "nodes.understand", "nodes.compose", "nodes.explain",
           "nodes.override"]


@pytest.mark.parametrize("module", MODULES)
def test_self_check(module):
    runpy.run_module(module, run_name="__main__")  # each module asserts its own behaviour


def test_routers():
    assert after_understand({"error": "llm down"}) == "intake_failed"
    assert after_understand({"explain": True, "on_topic": True}) == "explain"
    assert after_understand({"on_topic": False, "location": "Bhopal"}) == "no_guidance"
    assert after_understand({"on_topic": True, "location": None}) == "ask_location"
    assert after_understand({"on_topic": True, "location": "Bhopal"}) == "geocode"
    assert after_geocode({"error": "not found"}) == "weather_failed"
    assert after_geocode({"error": None}) == "fetch_weather"
    assert after_fetch({"error": "down"}) == "weather_failed"
    assert after_fetch({"error": None}) == "match_sops"
    assert after_match({"sops": []}) == "no_guidance"
    assert after_match({"sops": [{"id": "X-01"}], "situational": False}) == "compose"
    assert after_match({"sops": [{"id": "RAIN-SYS-01"}], "situational": True}) == "override"


def test_graph_has_every_branch():
    nodes = set(build().get_graph().nodes)
    assert {"intake_failed", "explain", "no_guidance", "ask_location", "weather_failed", "override", "compose"} <= nodes
