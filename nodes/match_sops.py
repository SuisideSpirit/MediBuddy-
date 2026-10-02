"""Decides which SOPs apply. Deterministic code over the numbers, except fuzzy SOPs, where the LLM picks one of the
SOP's own verdicts from numbers only (it never sees the user's text).

Self-check:  python -m nodes.match_sops
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from nodes.state import State
from nodes.understand import structured
from policy import SEVERITY, describe, holds, load, parse


class Verdict(BaseModel):
    verdict: str


def judge(sop, metrics):
    j = sop["judgement"]
    out = structured(Verdict, [
        SystemMessage(f"Apply this written criterion to the weather numbers.\nCriterion: {j['criteria']}\n"
                      f"Reply with exactly one verdict from: {list(j['verdicts'])}."),
        HumanMessage(json.dumps({k: metrics[k] for k in j["uses"]}))])
    return pick(out.verdict, j["verdicts"])


def pick(raw, verdicts):
    """Fail safe: an answer that isn't one of the SOP's verdicts gets the most cautious verdict."""
    raw = raw.strip().lower()
    return raw if raw in verdicts else max(verdicts, key=lambda v: SEVERITY.index(verdicts[v]["severity"]))


def applies(sop, activity, audience):
    a = sop["applies_to"]
    return (a["activities"] == "any" or activity in a["activities"]) and \
        ("audiences" not in a or audience in a["audiences"])


def hit(s, severity, guidance, basis, title=None):
    return {"id": s["id"], "title": title or s["title"], "severity": severity, "guidance": guidance,
            "basis": basis, "situational": s.get("situational", False)}


def match(activity, audience, metrics, ctx, judge=judge):
    """Returns rendered SOPs: situational first, then highest severity. Conflict policy is in sops/README.md."""
    sops = [s for s in load()["sops"] if applies(s, activity, audience)]
    fill = {**metrics, **ctx}
    hits = []
    for s in sops:
        if "judgement" in s:
            verdicts = s["judgement"]["verdicts"]
            inputs = ", ".join(f"{k} was {metrics[k]}" for k in s["judgement"]["uses"])
            try:
                verdict = judge(s, metrics)
                why = f"the written criteria, applied to {inputs}, gave the verdict '{verdict}'"
            except Exception:  # LLM down: never skip the SOP or guess a nice verdict - take the most cautious one
                verdict = pick("", verdicts)
                why = f"the judgement service was unavailable, so the most cautious verdict '{verdict}' was applied"
            v = verdicts[verdict]
            hits.append(hit(s, v["severity"], v["guidance"], [why], f"{s['title']}: {verdict}"))
        elif not s.get("default"):
            held = [line for line in s["when"] if holds(line, metrics)]
            if held:
                hits.append(hit(s, s["severity"], s["guidance"], [describe(line, metrics) for line in held]))
    if not hits:
        # A default (all-clear) only applies inside its `when` envelope. Extreme weather that no SOP covers
        # must end in "no guidance", never in a silent all-clear (found live: a walk in Antarctica got CLEAR-01).
        for s in sops:
            held = [line for line in s.get("when", []) if holds(line, metrics)]
            if s.get("default") and (held or not s.get("when")):
                hits.append(hit(s, s["severity"], s["guidance"],
                                ["no other policy was triggered"] + [describe(line, metrics) for line in held]))
    if any(h["severity"] in ("high", "critical") for h in hits):
        hits = [h for h in hits if h["severity"] != "info"]
    # Situational (weather system) first, then severity; stable sort keeps file order within a tie.
    hits.sort(key=lambda h: (not h["situational"], -SEVERITY.index(h["severity"])))
    by_id = {s["id"]: s for s in sops}
    for h in hits:  # numbers the policy says the reply MUST state (compose.verify enforces it)
        h["must_quote"] = [metrics[k] for k in by_id[h["id"]].get("must_quote", [])]
        h["guidance"] = " ".join(h["guidance"].format_map(fill).split())
    return hits


def match_sops(state: State):
    w = state["weather"]
    ctx = {"place": w["place"], "hours": w["hours"], "activity": state["activity"].replace("_", " ")}
    sops = match(state["activity"], state["audience"], w["metrics"], ctx)
    return {"sops": sops, "situational": any(s["situational"] for s in sops)}


if __name__ == "__main__":
    calm = dict(temp_max_c=28, temp_min_c=20, feels_like_max_c=30, feels_like_min_c=22, precip_total_mm=0, precip_prob_max_pct=5,
                wind_max_kmh=8, gust_max_kmh=15, uv_max=5, thunderstorm=False, rain_past_24h_mm=0, rain_next_24h_mm=0,
                rain_48h_mm=0, rain_hours_48h=0)
    ctx = {"place": "Testville", "hours": "2026-10-02 17:00-20:59", "activity": "x"}
    ids = lambda act, m, aud="self", j=lambda s, m: "good": [h["id"] for h in match(act, aud, {**calm, **m}, ctx, j)]

    assert ids("cycling", {}) == ["CLEAR-01"]                                   # nothing triggered -> default
    assert ids("other", {}) == []                                               # no policy for scuba -> no_guidance
    assert ids("other", {"thunderstorm": True}) == ["STORM-01"]                 # "any" covers unlisted activities
    assert ids("cycling", {"gust_max_kmh": 45, "uv_max": 9}) == ["WIND-RIDE-01", "UV-01"]  # ranked by severity
    assert ids("travel", {"gust_max_kmh": 45}) == ["CLEAR-01"]                  # wind SOP is riders-only
    # sustained rain: neither day >= 64.5 alone, the system still triggers, and leads
    assert ids("picnic", {"rain_past_24h_mm": 40, "rain_next_24h_mm": 38})[0] == "RAIN-SYS-01"
    assert "PICNIC-01" not in ids("picnic", {"rain_past_24h_mm": 70})           # "good" picnic dropped under critical
    assert ids("picnic", {"feels_like_max_c": 36}, "children") == ["HEAT-VUL-01"]  # "good" contradicts a heat warning
    poor = lambda s, m: "poor"
    assert ids("picnic", {"feels_like_max_c": 36}, "children", poor) == ["HEAT-VUL-01", "PICNIC-01"]
    # situational outranks even an equal-severity non-situational SOP (extreme heat is critical too)
    assert ids("running", {"feels_like_max_c": 47, "thunderstorm": True})[:2] == ["STORM-01", "HEAT-EXT-01"]
    # extremes: Antarctica walk must never be an all-clear
    assert ids("running", {"feels_like_min_c": -60, "feels_like_max_c": -55})[0] == "COLD-EXT-01"
    assert ids("running", {"feels_like_min_c": -5}) == ["COLD-01"]
    assert ids("travel", {"feels_like_max_c": 47})[0] == "HEAT-EXT-01"
    # safety net: weather outside the all-clear envelope with no SOP for it -> [] (no_guidance), not CLEAR-01
    assert ids("travel", {"feels_like_max_c": 42}) == []        # hot, but no heat SOP for adult travel
    assert ids("hiking", {"precip_total_mm": 15}) == []          # wet hike, no hiking-rain SOP
    assert ids("travel", {"gust_max_kmh": 70}) == []             # wind SOPs don't cover travel
    # persistent rain system: Bhopal 3-4 Sep 2026 numbers trigger it and it leads; a short downpour does not
    bhopal = {"rain_past_24h_mm": 8.3, "rain_next_24h_mm": 24.6, "rain_48h_mm": 32.9, "rain_hours_48h": 38}
    assert ids("cycling", bhopal)[0] == "RAIN-SYS-01"
    top = match("cycling", "self", {**calm, **bhopal}, ctx, None)[0]
    assert top["must_quote"] == [38, 32.9] and top["situational"]
    assert "RAIN-SYS-01" not in ids("cycling", {"rain_48h_mm": 20, "rain_hours_48h": 2, "precip_total_mm": 20})
    assert "RAIN-SYS-01" not in ids("cycling", {"rain_48h_mm": 5, "rain_hours_48h": 40})     # all-day drizzle
    # climbing: wet rock SOP fires on recent rain even if the window is dry; calm dry day -> all-clear
    assert ids("climbing", {"rain_past_24h_mm": 8}) == ["CLIMB-WET-01"]
    assert ids("climbing", {}) == ["CLEAR-01"]
    # basis records the exact condition + real value, for `explain`
    first = match("cycling", "self", {**calm, "gust_max_kmh": 48}, ctx, None)[0]
    assert first["basis"] == ["gust_max_kmh was 48 (rule: >= 40)"], first["basis"]
    # numbers in advice come from metrics, not the file
    assert "45 km/h" in match("cycling", "self", {**calm, "gust_max_kmh": 45}, ctx, None)[0]["guidance"]
    # bad verdict from the model -> most cautious one; judge LLM down -> most cautious one, not a crash
    vs = {"good": {"severity": "info"}, "poor": {"severity": "moderate"}}
    assert pick(" Good ", vs) == "good" and pick("lovely!", vs) == "poor"

    def down(s, m):
        raise ConnectionError
    assert match("picnic", "self", calm, ctx, down)[0]["title"] == "Picnic suitability: poor"
    try:
        parse("gust_max_kph >= 40")
        raise AssertionError
    except ValueError:
        pass
    print("match ok")
