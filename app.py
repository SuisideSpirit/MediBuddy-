"""Chat frontend.  Run:  .venv/Scripts/streamlit run app.py"""
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import build

st.set_page_config(page_title="Weather Safety Advisor", page_icon="🌦️")


@st.cache_resource
def agent():
    return build()


@st.cache_resource
def sessions():
    # thread_id -> title. Messages + memory live in the graph's checkpointer under the same thread_id.
    # ponytail: in-memory, lost on server restart (brief says that's fine); SqliteSaver if persistence is ever needed.
    return {}


def cfg(tid):
    return {"configurable": {"thread_id": tid}}


def messages(tid):
    return agent().get_state(cfg(tid)).values.get("messages", [])


def new_chat():
    # Reuse the current chat if it's still empty, like ChatGPT.
    tid = st.session_state.get("thread")
    if tid in sessions() and not messages(tid):
        return
    tid = str(uuid.uuid4())
    sessions()[tid] = "New chat"
    st.session_state.thread = tid


def graph_dot():
    g = agent().get_graph()
    edges = [f'"{e.source}" -> "{e.target}"{" [style=dashed]" if e.conditional else ""}' for e in g.edges]
    return "digraph { node [shape=box, style=rounded, fontname=Helvetica]; " + "; ".join(edges) + " }"


if st.session_state.get("thread") not in sessions():
    new_chat()
tid = st.session_state.thread

# ---- sidebar: chats ----
with st.sidebar:
    if st.button("➕ New chat", width="stretch"):
        new_chat()
        st.rerun()
    st.caption("Chats")
    for t, title in reversed(list(sessions().items())):
        if st.button(title, key=t, width="stretch", type="primary" if t == tid else "secondary"):
            st.session_state.thread = t
            st.rerun()

# ---- main: conversation ----
st.title("🌦️ Weather Safety Advisor")
st.caption("Advice comes only from our written policies (SOPs) and live Open-Meteo data. Every answer cites its policy.")

for m in messages(tid):
    st.chat_message("user" if isinstance(m, HumanMessage) else "assistant").markdown(m.content)

if q := st.chat_input("e.g. Is it safe to cycle to work in Bhopal this evening?"):
    if sessions()[tid] == "New chat":
        sessions()[tid] = q[:40] + ("…" if len(q) > 40 else "")
    st.chat_message("user").markdown(q)
    with st.chat_message("assistant"), st.spinner("Checking live weather and policies..."):
        try:
            agent().invoke({"messages": [HumanMessage(q)]}, cfg(tid))
        except Exception as e:  # e.g. LLM provider down/over quota in `understand`: fail honestly, never a half-answer
            why = ("the AI service's usage limit is reached for now" if type(e).__name__ == "RateLimitError"
                   else "something went wrong on my side")
            reply = f"Sorry, I can't answer right now: {why}. I won't guess at advice. Please try again in a few minutes."
            # Save the reply so the failure stays visible after the rerun below (it used to vanish).
            # LangGraph already checkpointed the user's message before the failing node ran.
            agent().update_state(cfg(tid), {"messages": [AIMessage(reply)]})
    st.rerun()  # refresh the sidebar title + memory panel

# ---- sidebar: what the bot knows + how it works ----
with st.sidebar:
    st.divider()
    s = agent().get_state(cfg(tid)).values
    with st.expander("Session memory"):
        st.json({k: s.get(k) for k in ("activity", "audience", "location", "window")})
    if d := s.get("last_decision"):
        with st.expander("Last decision", expanded=True):
            st.json({"sops": {x["id"]: x["basis"] for x in d["sops"]}, "reason": d.get("reason"),
                     "llm_wording_verified": s.get("grounded")} if d["sops"] else {"sops": [], "reason": d["reason"]})
    with st.expander("Agent graph"):
        st.graphviz_chart(graph_dot())
        st.caption("Dashed = conditional branch")
