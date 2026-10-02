from langchain_core.messages import AIMessage

from nodes.state import State
from nodes.understand import vocabulary


def no_guidance(state: State):
    # Fixed text, no LLM: an honest "no policy" is always better than invented advice.
    acts = ", ".join(a.replace("_", " ") for a in vocabulary()["activities"])
    w = state.get("weather")
    if not state.get("on_topic"):
        text = "I can only help with weather safety for outdoor plans, so I can't answer that one."
        reason = "the question wasn't about weather safety for an outdoor plan"
    elif state.get("activity") == "other" or not w:
        text = "I'm sorry, we don't have a written policy that covers that activity, so I'd rather not guess."
        reason = "none of our written policies covers that activity"
    else:  # known activity, but weather outside every SOP incl. the all-clear envelope
        m = w["metrics"]
        readings = (f"feels like {m['feels_like_min_c']} to {m['feels_like_max_c']}°C, gusts up to {m['gust_max_kmh']} km/h, "
                    f"{m['precip_total_mm']} mm rain")
        reason = f"the weather in {w['place']} ({readings}) was outside what any of our policies covers"
        return {"last_decision": {"sops": [], "reason": reason}, "messages": [AIMessage(
            f"Conditions in {w['place']} during {w['hours']} are outside what our written policies cover "
            f"({readings}), so I can't give advice either way. Please treat this as unusual "
            "weather and check official local warnings before going out.")]}
    return {"last_decision": {"sops": [], "reason": reason},
            "messages": [AIMessage(f"{text} I can advise on: {acts}. Just tell me the activity, the city and when.")]}
