"""The LangGraph agent. All control flow lives here; nodes live in nodes/.

understand -> ("why?")      explain (replays last_decision: SOPs + the exact condition/value that triggered them)
           -> (off-topic)   no_guidance
           -> (no location) ask_location
           -> geocode -> (not found / down) weather_failed
                      -> fetch_weather -> (down / incomplete) weather_failed
                                       -> match_sops -> (no SOP applies) no_guidance
                                                     -> compose (LLM words it, code verifies, else template)
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes.ask_location import ask_location
from nodes.compose import compose
from nodes.explain import explain
from nodes.match_sops import match_sops
from nodes.no_guidance import no_guidance
from nodes.state import State
from nodes.understand import understand
from nodes.weather import fetch_weather, geocode, weather_failed


def after_understand(state: State) -> str:
    if state.get("explain"):
        return "explain"
    if not state["on_topic"]:
        return "no_guidance"
    if not state.get("location"):
        return "ask_location"
    return "geocode"


def after_geocode(state: State) -> str:
    return "weather_failed" if state.get("error") else "fetch_weather"


def after_fetch(state: State) -> str:
    return "weather_failed" if state.get("error") else "match_sops"


def after_match(state: State) -> str:
    return "compose" if state["sops"] else "no_guidance"


def build():
    g = StateGraph(State)
    for node in (understand, explain, ask_location, geocode, fetch_weather, weather_failed, match_sops, no_guidance, compose):
        g.add_node(node)
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", after_understand, ["explain", "no_guidance", "ask_location", "geocode"])
    g.add_conditional_edges("geocode", after_geocode, ["weather_failed", "fetch_weather"])
    g.add_conditional_edges("fetch_weather", after_fetch, ["weather_failed", "match_sops"])
    g.add_conditional_edges("match_sops", after_match, ["no_guidance", "compose"])
    for node in ("explain", "ask_location", "weather_failed", "no_guidance", "compose"):
        g.add_edge(node, END)
    return g.compile(checkpointer=MemorySaver())  # MemorySaver = per-session memory, keyed by thread_id
