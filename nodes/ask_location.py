from langchain_core.messages import AIMessage

from nodes.state import State


def ask_location(state: State):
    # Fixed text, no LLM. Activity/window stay in memory, so the user's next message can be just a city.
    return {"messages": [AIMessage("Which city or town are you in? I need a location to check the live weather "
                                   "before I can give any advice.")]}
