from nodes.state import State


def override(state: State):
    """A situational SOP matched (a weather SYSTEM such as an active rain system or thunderstorm). Code, not the LLM,
    writes the lead line naming it, so it is guaranteed to come before any activity-specific advice."""
    lead = [s for s in state["sops"] if s["situational"]]
    names = " and ".join(f"{s['title']} [{s['id']}]" for s in lead)
    place = state["weather"]["place"]
    return {"banner": f"⚠️ **{names}** in {place}. This applies to any outdoor plan, whatever the activity."}


if __name__ == "__main__":
    st = {"weather": {"place": "Bhopal"}, "sops": [
        {"id": "RAIN-SYS-01", "title": "Active heavy-rain system", "situational": True},
        {"id": "WIND-RIDE-01", "title": "Strong gusts", "situational": False}]}
    b = override(st)["banner"]
    assert "RAIN-SYS-01" in b and "WIND-RIDE-01" not in b and "Bhopal" in b, b
    print("override ok")
