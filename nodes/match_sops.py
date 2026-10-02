"""Decides which SOPs apply. Deterministic code over the numbers, except fuzzy SOPs, where the LLM picks one of the
SOP's own verdicts from numbers only (it never sees the user's text).

Validate sops.yaml:  .venv/Scripts/python -m nodes.match_sops
"""
import json
import operator
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from nodes.state import State
from nodes.understand import structured, vocabulary
from nodes.weather import METRICS

SEVERITY = ["info", "low", "moderate", "high", "critical"]
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq, "!=": operator.ne}
COND = re.compile(r"^\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*(\S+)\s*$")


def parse(line: str):
    """'gust_max_kmh >= 40 and thunderstorm == true' -> [(metric, op, value), ...]. Raises on anything else."""
    out = []
    for part in line.split(" and "):
        m = COND.match(part)
        if not m or m[1] not in METRICS:
            raise ValueError(f"bad condition {part!r} (metrics: {', '.join(METRICS)})")
        raw = m[3].lower()
        out.append((m[1], OPS[m[2]], {"true": True, "false": False}[raw] if raw in ("true", "false") else float(raw)))
    return out


def describe(line, metrics):
    """'gust_max_kmh >= 40' -> 'gust_max_kmh was 48 (rule: >= 40)'. Used by `explain` to show what triggered a SOP."""
    return " and ".join(f"{m[1]} was {metrics[m[1]]} (rule: {m[2]} {m[3]})"
                        for m in (COND.match(part) for part in line.split(" and ")))


def holds(line, metrics):
    return all(op(metrics[k], v) for k, op, v in parse(line))


def load():
    """Read + validate sops.yaml on every call (hot reload). A broken SOP fails loudly instead of being skipped."""
    doc = vocabulary()
    sample = {**{k: 1 for k in METRICS}, "place": "", "hours": "", "activity": ""}
    seen = set()
    for s in doc["sops"]:
        sid = s.get("id", "?")
        try:
            assert sid not in seen, "duplicate id"
            seen.add(sid)
            acts = s["applies_to"]["activities"]
            assert acts == "any" or set(acts) <= set(doc["activities"]), f"unknown activity in {acts}"
            assert set(s["applies_to"].get("audiences", [])) <= set(doc["audiences"]), "unknown audience"
            if "judgement" in s:
                j = s["judgement"]
                assert set(j["uses"]) <= set(METRICS), f"unknown metric in uses {j['uses']}"
                for v in j["verdicts"].values():
                    assert v["severity"] in SEVERITY, f"bad severity {v['severity']}"
                    v["guidance"].format_map(sample)
            else:
                assert s["severity"] in SEVERITY, f"bad severity {s['severity']}"
                assert s.get("default") or s["when"], "needs `when`, `judgement` or `default: true`"
                for k in s.get("must_quote", []):
                    assert k in METRICS and k != "thunderstorm", f"must_quote: unknown/non-numeric metric {k}"
                    assert f"{{{k}}}" in s["guidance"], f"must_quote {k} isn't in the guidance text"
                for line in s.get("when", []):
                    parse(line)
                s["guidance"].format_map(sample)
        except (AssertionError, KeyError, ValueError, TypeError) as e:
            raise ValueError(f"sops.yaml: SOP {sid} is invalid: {e!r}") from e
    return doc


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


def match(activity, audience, metrics, ctx, judge=judge):
    """Returns rendered SOPs, highest severity first. Conflict policy is documented in sops.yaml."""
    sops = [s for s in load()["sops"] if applies(s, activity, audience)]
    fill = {**metrics, **ctx}
    hits = []
    for s in sops:
        if "judgement" in s:
            verdict = judge(s, metrics)
            v = s["judgement"]["verdicts"][verdict]
            inputs = ", ".join(f"{k} was {metrics[k]}" for k in s["judgement"]["uses"])
            hits.append({"id": s["id"], "title": f"{s['title']}: {verdict}", "severity": v["severity"],
                         "guidance": v["guidance"],
                         "basis": [f"the written criteria, applied to {inputs}, gave the verdict '{verdict}'"]})
        elif not s.get("default"):
            held = [line for line in s["when"] if holds(line, metrics)]
            if held:
                hits.append({"id": s["id"], "title": s["title"], "severity": s["severity"], "guidance": s["guidance"],
                             "basis": [describe(line, metrics) for line in held]})
    if not hits:
        # A default (all-clear) only applies inside its `when` envelope. Extreme weather that no SOP covers
        # must end in "no guidance", never in a silent all-clear (found live: a walk in Antarctica got CLEAR-01).
        for s in sops:
            held = [line for line in s.get("when", []) if holds(line, metrics)]
            if s.get("default") and (held or not s.get("when")):
                hits.append({"id": s["id"], "title": s["title"], "severity": s["severity"], "guidance": s["guidance"],
                             "basis": ["no other policy was triggered"] + [describe(line, metrics) for line in held]})
    if any(h["severity"] in ("high", "critical") for h in hits):
        hits = [h for h in hits if h["severity"] != "info"]
    hits.sort(key=lambda h: -SEVERITY.index(h["severity"]))  # stable: file order within a severity
    by_id = {s["id"]: s for s in sops}
    for h in hits:  # numbers the policy says the reply MUST state (compose.verify enforces it)
        h["must_quote"] = [metrics[k] for k in by_id[h["id"]].get("must_quote", [])]
    for h in hits:
        h["guidance"] = " ".join(h["guidance"].format_map(fill).split())
    return hits


def match_sops(state: State):
    w = state["weather"]
    ctx = {"place": w["place"], "hours": w["hours"], "activity": state["activity"].replace("_", " ")}
    return {"sops": match(state["activity"], state["audience"], w["metrics"], ctx)}


if __name__ == "__main__":
    doc = load()
    print(f"sops.yaml valid: {len(doc['sops'])} SOPs")

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
    assert match("cycling", "self", {**calm, **bhopal}, ctx, None)[0]["must_quote"] == [38, 32.9]
    assert "RAIN-SYS-01" not in ids("cycling", {"rain_48h_mm": 20, "rain_hours_48h": 2, "precip_total_mm": 20})
    assert "RAIN-SYS-01" not in ids("cycling", {"rain_48h_mm": 5, "rain_hours_48h": 40})     # all-day drizzle
    # climbing: wet rock SOP fires on recent rain even if the window is dry; calm dry day -> all-clear
    assert ids("climbing", {"rain_past_24h_mm": 8}) == ["CLIMB-WET-01"]
    assert ids("climbing", {}) == ["CLEAR-01"]
    # basis records the exact condition + real value, for `explain`
    hit = match("cycling", "self", {**calm, "gust_max_kmh": 48}, ctx, None)[0]
    assert hit["basis"] == ["gust_max_kmh was 48 (rule: >= 40)"], hit["basis"]
    # numbers in advice come from metrics, not the file
    assert "45 km/h" in match("cycling", "self", {**calm, "gust_max_kmh": 45}, ctx, None)[0]["guidance"]
    # bad verdict from the model -> most cautious one
    vs = {"good": {"severity": "info"}, "poor": {"severity": "moderate"}}
    assert pick(" Good ", vs) == "good" and pick("lovely!", vs) == "poor"
    # validation catches typos
    try:
        parse("gust_max_kph >= 40")
        raise AssertionError
    except ValueError:
        pass
    print("match ok")
