from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

# Time windows the bot understands. fetch_weather maps each one to forecast hours.
WINDOWS = ["now", "this_morning", "this_afternoon", "this_evening", "tonight", "today", "tomorrow"]


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    # Session memory: the checkpointer keeps these across turns, so follow-ups reuse them.
    activity: str | None
    audience: str | None
    location: str | None
    window: str | None
    # What the bot last decided and why. Written by every answering node, read by `explain`, never reset.
    last_decision: dict | None  # {activity, sops (with basis), weather} or {sops: [], reason}
    # Per-turn results, reset by understand.
    on_topic: bool
    explain: bool         # user asked "why did you say that?" -> explain node
    place: dict | None    # geocode: {name, lat, lon}
    weather: dict | None  # fetch_weather: {place, window, hours, observed_at, metrics}
    error: str | None     # understand -> intake_failed; geocode/fetch_weather -> weather_failed
    sops: list            # match_sops: rendered SOPs, highest severity first
    grounded: bool | None # compose: True if the LLM's wording passed verify(), False if the template was used
    situational: bool     # match_sops: a situational SOP (weather system) matched -> override branch
    banner: str | None    # override: code-written lead line naming the weather system, prepended by compose
