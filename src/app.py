import os
import time
import streamlit as st
from llm.router import route_intent_stream
from agent_logic import run_agent, cleanup_artifacts, get_generated_plots

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_SRC_DIR, "agent_backend.log")


def show_plot(plot_path: str):
    if os.path.exists(plot_path):
        st.image(plot_path)


def build_conversation_context(messages: list) -> str:
    filtered = [
        m for m in messages if m["role"] in ["user", "assistant"] and "content" in m
    ]
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in filtered])


st.set_page_config(page_title="OpenAeroStruct Agent", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_run" not in st.session_state:
    st.session_state["stop_run"] = False
if "relaxation_prompt" not in st.session_state:
    st.session_state["relaxation_prompt"] = None
if "last_successful_code" not in st.session_state:
    st.session_state["last_successful_code"] = ""

st.sidebar.title("Configuration")
provider = st.sidebar.selectbox("Provider", ["Gemini API", "Ollama"])
model_name = st.sidebar.selectbox(
    "Model Selection", ["gemini-flash-lite-latest", "gemini-2.0-flash"]
)
max_retries = st.sidebar.slider("Max retries", min_value=1, max_value=6, value=3)

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state["last_successful_code"] = ""
    cleanup_artifacts()
    st.rerun()

st.title("OpenAeroStruct — LLM Agent")


def _make_ui_callback(stream_state: dict, no_converge_store: dict):
    def cb(event: str, data: dict):
        if st.session_state.get("stop_run", False):
            return
        attempt = data.get("attempt", 0)
        if event == "attempt_start":
            st.markdown(f"**Attempt {data['attempt']} of {data['max_retries']}**")
            stream_state[attempt] = {"placeholder": st.empty(), "text": ""}
        elif event == "code_chunk":
            state = stream_state.get(attempt, {})
            state["text"] = state.get("text", "") + data["chunk"]
            state.get("placeholder", st.empty()).markdown(
                f"```python\n{state['text']}\n```"
            )
        elif event == "code_ready":
            state = stream_state.get(attempt, {})
            if "placeholder" in state:
                state["placeholder"].empty()
            st.info(f"**Coder's Reasoning:** {data['reasoning']}")
            with st.expander(f"Generated Code (Attempt {attempt})", expanded=False):
                st.code(data["code"], language="python")
        elif event == "exec_success":
            st.success("✅ Execution completed successfully.")
            if data.get("db_summary"):
                st.markdown(data["db_summary"])
            if data.get("plots"):
                for p in data["plots"]:
                    show_plot(p)
        elif event == "exec_error":
            st.error("❌ Python error occurred.")
            st.code(data["stderr_tail"], language="text")
        elif event == "no_converge":
            st.warning("⚠️ Optimiser did not converge — checking physics constraints...")
            if data.get("db_summary"):
                st.markdown(data["db_summary"])
            if data.get("plots"):
                for p in data["plots"]:
                    show_plot(p)
        elif event == "no_converge_final":
            no_converge_store.update(data)

    return cb


def _handle_agent_result(
    result, no_converge_data: dict, context: str, blueprints: list, routing_json: dict
):
    if result.final_code and (result.success or result.converged == "no"):
        st.session_state["last_successful_code"] = result.final_code

    if result.success:
        content = "### ✅ Optimization Complete\nThe script ran successfully. Results and plots are shown above."
        st.markdown(content)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "code": result.final_code,
                "summary": result.final_summary,
                "plots": result.plots,
                "routing_json": routing_json,
            }
        )
    elif no_converge_data:
        content = "### ⚠️ Optimization did not converge"
        st.warning(content)
        # Store results so they persist during replay
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "code": result.final_code,
                "summary": result.final_summary,
                "plots": result.plots,
                "routing_json": routing_json,
            }
        )
        # Add special Relaxation message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "is_relaxation_ui": True,
                "status": "pending",
                "suggestion": no_converge_data.get("suggestion", ""),
                "user_prompt": context,
                "blueprints": blueprints,
                "error_logs": no_converge_data.get("error_logs", []),
            }
        )
        st.rerun()
    else:
        st.error("### ❌ Agent failed")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Failed to complete task.",
                "code": result.final_code,
                "routing_json": routing_json,
            }
        )


# ---------------------------------------------------------------------------
# REPLAY HISTORY
# ---------------------------------------------------------------------------
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("is_relaxation_ui"):
            with st.container(border=True):
                st.warning("### ⚠️ Setup Issue - Relaxation Required")
                st.markdown(message["suggestion"])
                if message["status"] == "pending":
                    user_fix = st.text_area(
                        "Describe your own fix (or leave empty to use suggestion):",
                        key=f"fix_{i}",
                        height=100,
                    )
                    if st.button("✅ Apply and retry", key=f"btn_{i}"):
                        applied = (
                            user_fix.strip()
                            if user_fix.strip()
                            else message["suggestion"]
                        )
                        message["status"] = "applied"
                        message["applied_fix"] = applied
                        st.session_state["relaxation_prompt"] = (
                            message["user_prompt"] + "\n\nRETRY: " + applied
                        )
                        st.session_state["relaxation_blueprints"] = message[
                            "blueprints"
                        ]
                        st.session_state["relaxation_error_logs"] = message[
                            "error_logs"
                        ]
                        st.rerun()
                else:
                    st.info(f"**Applied Fix:** {message.get('applied_fix')}")
        else:
            st.markdown(message["content"])
            if "routing_json" in message:
                with st.expander("Show Routing Logic (JSON)"):
                    st.json(message["routing_json"])
            if "code" in message:
                with st.expander("Show Generated Code"):
                    st.code(message["code"], language="python")
            if "summary" in message and message["summary"]:
                st.markdown(message["summary"])
            if "plots" in message and message["plots"]:
                for p in message["plots"]:
                    show_plot(p)

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Enter your design request...")
_rel_p = st.session_state.pop("relaxation_prompt", None)
_rel_b = st.session_state.pop("relaxation_blueprints", None)
_rel_e = st.session_state.pop("relaxation_error_logs", None)

if _rel_p:
    with st.chat_message("assistant"):
        with st.expander("Relaxation Retry — Agent Thought Process", expanded=True):
            _rs, _rn = {}, {}
            result = run_agent(
                user_prompt=_rel_p,
                blueprints=_rel_b or ["aero_opt.py"],
                model_name=model_name,
                provider=provider,
                max_retries=max_retries,
                stream=True,
                callback=_make_ui_callback(_rs, _rn),
                prior_error_logs=_rel_e,
                prior_code=st.session_state.get("last_successful_code", ""),
            )
        prev_routing = next(
            (
                m["routing_json"]
                for m in reversed(st.session_state.messages)
                if "routing_json" in m
            ),
            {},
        )
        _handle_agent_result(result, _rn, _rel_p, _rel_b or [], prev_routing)

if user_prompt:
    cleanup_artifacts()
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    context = build_conversation_context(st.session_state.messages)
    with st.chat_message("assistant"):
        with st.expander("Agent Thought Process", expanded=True):
            st.write("### 1. Intent Routing")
            route_container = st.container()
            router_streamed, routing_data = "", {}
            router_placeholder = route_container.empty()
            for chunk in route_intent_stream(context, model_name, provider):
                if isinstance(chunk, dict):
                    routing_data = chunk
                else:
                    router_streamed += chunk
                    router_placeholder.markdown(f"```\n{router_streamed}\n```")
            router_placeholder.empty()
            blueprints = routing_data.get("blueprints", ["aero_opt.py"])
            route_container.info(
                f"**Selected Blueprint(s):** `{', '.join(blueprints)}`  \n**Reason:** {routing_data.get('reason')}"
            )
            time.sleep(0.2)
            st.write("### 2. Iterative Generation & Execution")
            _ss, _nc = {}, {}
            result = run_agent(
                user_prompt=context,
                blueprints=blueprints,
                model_name=model_name,
                provider=provider,
                max_retries=max_retries,
                stream=True,
                callback=_make_ui_callback(_ss, _nc),
                prior_code=st.session_state.get("last_successful_code", ""),
            )
        _handle_agent_result(result, _nc, context, blueprints, routing_data)
