"""Answers "why did you say that?" from the saved last decision. Code only, no LLM, no new weather fetch:
it replays the evidence for the answer already given, so it can never contradict it."""
from langchain_core.messages import AIMessage

from nodes.state import State


def explain(state: State):
    d = state.get("last_decision")
    if not d:
        text = "I haven't given any advice in this chat yet, so there's nothing to explain. Ask me about an outdoor plan."
    elif not d["sops"]:
        text = f"My last answer didn't apply any policy because {d['reason']}."
    else:
        w = d["weather"]
        lines = [f"My last answer was about **{d['activity'].replace('_', ' ')}** in **{w['place']}** for "
                 f"{w['hours']} (Open-Meteo data as of {w['observed_at']}). It applied these policies:"]
        for s in d["sops"]:
            lines.append(f"\n**[{s['id']}] {s['title']}** ({s['severity']}): triggered because "
                         + "; ".join(s["basis"]) + ".")
        text = "\n".join(lines)
    return {"messages": [AIMessage(text)]}


if __name__ == "__main__":
    msg = lambda st: explain(st)["messages"][0].content
    assert "haven't given any advice" in msg({})
    assert "outside what any" in msg({"last_decision": {"sops": [], "reason": "the weather was outside what any of our policies covers"}})
    d = {"activity": "cycling", "weather": {"place": "Bhopal", "hours": "2026-10-02 17:00-20:59", "observed_at": "x"},
         "sops": [{"id": "WIND-RIDE-01", "title": "Strong gusts", "severity": "high",
                   "basis": ["gust_max_kmh was 48 (rule: >= 40)"]}]}
    out = msg({"last_decision": d})
    assert "[WIND-RIDE-01]" in out and "gust_max_kmh was 48 (rule: >= 40)" in out, out
    print("explain ok")
