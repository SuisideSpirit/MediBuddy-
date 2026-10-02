"""Final node. The LLM only re-words the matched SOPs; code checks the wording and falls back to a template.
The citation footer is added by code, so every reply is traceable even if the LLM misbehaves."""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nodes.state import State
from nodes.understand import llm

ID_RE = re.compile(r"\b[A-Z]+(?:-[A-Z]+)*-\d+\b")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers(text):
    return {float(n) for n in NUM_RE.findall(ID_RE.sub("", text))}


def normalise(text):
    """gpt-oss sometimes cites as 【WIND‑RIDE‑01】 (full-width brackets, U+2011 non-breaking hyphens). Same text to a
    reader, but the citation/number checks wouldn't see it (found in evals: correct replies rejected)."""
    return text.translate(str.maketrans({"【": "[", "】": "]", "‐": "-", "‑": "-"}))


def verify(reply, context, sop_ids, required=()):
    """Reply may only use numbers exactly as they appear in the facts/policies (no rounding: 7.5 -> 8 would cross
    the UV threshold), must cite every matched SOP, and must not cite any SOP we didn't match."""
    allowed = numbers(context)
    cited = set(ID_RE.findall(reply))
    # `required`: each SOP's must_quote values. Leading with "[RAIN-SYS-01] be careful" without its numbers fails.
    return numbers(reply) <= allowed and cited == set(sop_ids) and {float(v) for v in required} <= numbers(reply)


def template(sops):
    """Deterministic reply: used when the LLM fails or its wording fails verify()."""
    return "\n\n".join(f"**[{s['id']}] {s['title']}** ({s['severity']}): {s['guidance']}" for s in sops)


def compose(state: State):
    sops, w = state["sops"], state["weather"]
    # The writer never sees the user's raw text, only what `understand` extracted into closed keys.
    # Prompt injection in the message has no path to the reply (blocked by design, not by asking nicely).
    request = {"activity": state["activity"].replace("_", " "), "for": state["audience"],
               "place": w["place"], "time": w["window"].replace("_", " ")}
    policies = [{k: s[k] for k in ("id", "title", "severity", "guidance", "must_quote")} for s in sops]
    facts = {"place": w["place"], "hours": w["hours"], **w["metrics"]}
    context = f"FACTS (live data):\n{json.dumps(facts, indent=1)}\n\nPOLICIES (highest severity first):\n{json.dumps(policies, indent=1)}"
    system = (
        "You are the voice of a weather-safety advisory service. You only re-phrase the policy guidance you are "
        "given; you never add advice, tips or opinions of your own.\n"
        "- Open with a one-line direct answer to the REQUEST based on the first (highest-severity) policy. "
        "Never call anything 'safe' or 'unsafe' unless the policy text says so.\n"
        "- Every number in a policy's must_quote list must appear in your reply.\n"
        "- Cover every policy in the order given and cite its id in square brackets, e.g. [ABC-01]. Cite no other ids.\n"
        "- Quote numbers only from FACTS or POLICIES, copied exactly (no rounding), with units. Copy the hours "
        "exactly as written. Never mention any other number.\n"
        "- Do not add your own conclusions or reassurance (e.g. 'no precautions needed', 'you'll be fine'); "
        "if the policy doesn't say it, don't say it.\n"
        "- 3 to 6 sentences, warm and plain."
    )
    try:
        reply = llm().invoke([SystemMessage(system), HumanMessage(f"REQUEST: {json.dumps(request)}\n\n{context}")]).content
        reply = normalise(reply)
        grounded = verify(reply, context, [s["id"] for s in sops], [v for s in sops for v in s["must_quote"]])
    except Exception:  # LLM down/timeout: the template still gives correct, cited advice
        reply, grounded = "", False
    if not grounded:
        reply = template(sops)
    if state.get("banner"):  # override branch: the weather system is named first, by code
        reply = f"{state['banner']}\n\n{reply}"
    cites = " · ".join(f"{s['id']} ({s['severity']})" for s in sops)
    # The LLM's one judgement call (how it read the question), shown so a misreading is visible to the user.
    who = "yourself" if state["audience"] == "self" else state["audience"]
    read_as = f"{state['activity'].replace('_', ' ')} · for {who} · {w['place']} · {w['window'].replace('_', ' ')}"
    footer = (f"\n\n---\n*Interpreted as: {read_as}*  \n"
              f"*Policy applied: {cites}*  \n"
              f"*Data: Open-Meteo, {w['place']}, {w['hours']}, as of {w['observed_at']}*")
    decision = {"activity": state["activity"], "sops": sops, "weather": w}
    return {"messages": [AIMessage(reply + footer)], "grounded": grounded, "last_decision": decision}


if __name__ == "__main__":
    ctx = 'FACTS {"gust_max_kmh": 45.3, "hours": "2026-10-02 17:00-20:59"} POLICIES [WIND-RIDE-01]'
    assert verify("Gusts up to 45.3 km/h [WIND-RIDE-01] between 17:00-20:59.", ctx, ["WIND-RIDE-01"])
    assert not verify("Gusts up to 45 km/h [WIND-RIDE-01].", ctx, ["WIND-RIDE-01"])         # rounded = not exact
    assert not verify("Gusts up to 60 km/h [WIND-RIDE-01].", ctx, ["WIND-RIDE-01"])        # invented number
    assert not verify("Gusts up to 45 km/h.", ctx, ["WIND-RIDE-01"])                        # missing citation
    assert not verify("Gusts 45 km/h [WIND-RIDE-01] [SOP-99].", ctx, ["WIND-RIDE-01"])      # fake policy
    assert not verify("Be careful [WIND-RIDE-01].", ctx, ["WIND-RIDE-01"], [45.3])          # must_quote missing
    assert verify("Gusts 45.3 km/h [WIND-RIDE-01].", ctx, ["WIND-RIDE-01"], [45.3])
    assert verify(normalise("Gusts 45.3 km/h 【WIND‑RIDE‑01】"), ctx, ["WIND-RIDE-01"])
    print("verify ok")
