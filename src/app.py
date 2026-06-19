import os
import re
import shutil
import uuid
import streamlit as st

from llm.router import route_intent_stream
from llm.config import is_ollama_provider
from agent_logic import (
    run_agent,
    cleanup_artifacts,
    get_generated_plots,
)

# ---------------------------------------------------------------------------
# Resolve absolute paths relative to this file
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_LOG_FILE = os.path.join(_SRC_DIR, "agent_backend.log")
_PLOT_ARCHIVE_DIR = os.path.join(_PROJECT_DIR, ".plot_history")

if "current_routing_data" not in st.session_state:
    st.session_state["current_routing_data"] = {}
if "active_attempts" not in st.session_state:
    st.session_state["active_attempts"] = []

# ---------------------------------------------------------------------------
# Design variable reference card
# ---------------------------------------------------------------------------
_DV_REFERENCE = """
**Available design variables you can specify:**

| Category | Variable | Description |
|---|---|---|
| Flight | `alpha` | Angle of attack [deg] |
| Geometry | `twist_cp` | Spanwise twist B-spline control points [deg] |
| Geometry | `chord_cp` | Chord scaling B-spline control points |
| Geometry | `taper` | Taper ratio (tip/root chord) |
| Geometry | `sweep` | Leading-edge sweep angle [deg] |
| Geometry | `dihedral` | Dihedral angle [deg] |
| Geometry | `xshear_cp` | Generalised sweep (x-shear B-spline CPs) [m] |
| Geometry | `zshear_cp` | Generalised dihedral (z-shear B-spline CPs) [m] |
| Struct (tube) | `thickness_cp` | Tube wall thickness B-spline CPs [m] |
| Struct (tube) | `radius_cp` | Tube outer radius B-spline CPs [m] |
| Struct (wingbox) | `spar_thickness_cp` | Spar wall thickness CPs [m] |
| Struct (wingbox) | `skin_thickness_cp` | Skin thickness CPs [m] |
| Struct (wingbox) | `t_over_c_cp` | Thickness-to-chord ratio CPs |
| Aerostructural | `fuel_mass` | Fuel mass [kg] (wingbox fuel loop) |
| Aerostructural | `alpha_maneuver` | Maneuver AoA [deg] (wingbox 2-point) |

**Example well-formed requests:**
- *"Minimize drag on a rect wing with span=12m, root_chord=2m, at Mach 0.5. DVs: alpha and twist_cp. Constraint: CL=0.5."*
- *"Aerostructural tube optimization on a CRM wing at Mach 0.84, minimize fuelburn. DVs: twist_cp, thickness_cp. Constraints: failure≤0, L=W."*
"""

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def show_plot(plot_path: str):
    """Display a plot image inline."""
    if os.path.exists(plot_path):
        st.image(plot_path)


def _snapshot_plot_files(plot_paths: list[str], scope: str) -> list[str]:
    """
    Copy generated plots to a stable archive path so older chat messages
    keep their original figures even after later runs overwrite filenames.
    """
    if not plot_paths:
        return []

    os.makedirs(_PLOT_ARCHIVE_DIR, exist_ok=True)
    snapshot_dir = os.path.join(
        _PLOT_ARCHIVE_DIR, f"{scope}_{uuid.uuid4().hex[:10]}"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    snap_paths = []
    for idx, plot_path in enumerate(plot_paths, start=1):
        if not os.path.exists(plot_path):
            continue
        name = os.path.basename(plot_path)
        base, ext = os.path.splitext(name)
        snap_name = f"{idx:02d}_{base}{ext}"
        dst = os.path.join(snapshot_dir, snap_name)
        shutil.copy2(plot_path, dst)
        snap_paths.append(dst)

    return snap_paths


def _clear_plot_history():
    """Remove archived plot snapshots created for prior chat messages."""
    if os.path.exists(_PLOT_ARCHIVE_DIR):
        shutil.rmtree(_PLOT_ARCHIVE_DIR, ignore_errors=True)


def show_vagueness_card(missing_info: str):
    """Render a structured clarification card in the chat."""
    st.warning("#### The agent needs more information before it can run.")
    st.markdown(f"**What's missing:**\n\n{missing_info}")
    with st.expander(
        "💡 What can I specify? (design variable reference)", expanded=False
    ):
        st.markdown(_DV_REFERENCE)
    st.info("Please reply with the missing details and the agent will proceed.")


def build_conversation_context(messages: list) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _get_ollama_model_names() -> tuple[list[str], str | None]:
    """
    Return installed Ollama model names plus an optional warning message.
    """
    try:
        import ollama

        response = ollama.list()
        raw_models = response.get("models", [])
        names = []
        for model in raw_models:
            if isinstance(model, dict):
                name = model.get("model") or model.get("name")
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(name)

        names = sorted(dict.fromkeys(names))
        if names:
            return names, None
        return [], "Ollama is reachable, but no local models are installed yet."
    except Exception as exc:
        return [], f"Could not load Ollama models: {exc}"


# ---------------------------------------------------------------------------
# Streamlit app setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="OpenAeroStruct Agent", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_run" not in st.session_state:
    st.session_state["stop_run"] = False
if "pending_relaxation" not in st.session_state:
    st.session_state["pending_relaxation"] = None
if "relaxation_prompt" not in st.session_state:
    st.session_state["relaxation_prompt"] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
provider = st.sidebar.selectbox("Provider", ["Gemini API", "Ollama"])

if is_ollama_provider(provider):
    ollama_models, ollama_warning = _get_ollama_model_names()
    if ollama_models:
        model_name = st.sidebar.selectbox("Ollama Model", ollama_models)
    else:
        model_name = st.sidebar.text_input(
            "Ollama Model",
            value="gemini-2.0-flash",
            help="Enter the exact local Ollama model name to use.",
        )
        if ollama_warning:
            st.sidebar.warning(ollama_warning)
else:
    model_name = st.sidebar.selectbox(
        "Gemini Model",
        [
            "gemini-flash-lite-latest",
            "gemini-2.0-flash",
        ],
    )

max_retries = st.sidebar.slider("Max retries", min_value=1, max_value=6, value=3)

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state["stop_run"] = True
    st.session_state["pending_relaxation"] = None
    st.session_state["active_attempts"] = []
    cleanup_artifacts()
    _clear_plot_history()
    # Truncate/clear backend log
    if os.path.exists(_LOG_FILE):
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    st.rerun()

st.title("OpenAeroStruct — LLM Agent")

# ---------------------------------------------------------------------------
# Shared agent UI helpers
# ---------------------------------------------------------------------------


def _make_ui_callback(stream_state: dict, no_converge_store: dict):
    """Return a callback that routes agent events to Streamlit UI elements."""

    def cb(event: str, data: dict):
        if st.session_state.get("stop_run", False):
            return
        attempt = data.get("attempt", 0)

        if event == "attempt_start":
            st.markdown(f"**Attempt {data['attempt']} of {data['max_retries']}**")
            stream_state[attempt] = {
                "placeholder": st.empty(),
                "reasoning_placeholder": st.empty(),
                "code_placeholder": st.empty(),
                "text": "",
            }
            # Record start of attempt for state preservation
            attempt_info = {
                "attempt": attempt,
                "max_retries": data["max_retries"],
                "reasoning": "",
                "code": "",
                "status": "running",
                "logs": "",
                "db_summary": "",
                "plots": [],
            }
            st.session_state["active_attempts"].append(attempt_info)

        elif event == "code_chunk":
            state = stream_state.get(attempt, {})
            state["text"] = state.get("text", "") + data["chunk"]
            txt = state["text"]

            # Robust live extraction supporting both XML and legacy headers
            r_match = re.search(r"<reasoning>(.*?)(?:</reasoning>|$)", txt, re.S | re.I)
            c_match = re.search(r"<code>(.*?)(?:</code>|$)", txt, re.S | re.I)

            if not r_match and not c_match and "REASONING:" not in txt:
                state.get("placeholder", st.empty()).markdown(f"```text\n{txt}\n```")
            else:
                if "placeholder" in state and state["placeholder"]:
                    state["placeholder"].empty()

                if "reasoning_placeholder" not in state:
                    state["reasoning_placeholder"] = st.empty()
                if "code_placeholder" not in state:
                    state["code_placeholder"] = st.empty()

                if r_match:
                    content = r_match.group(1).strip()
                    if content:
                        state["reasoning_placeholder"].info(f"**Thinking:** {content}")
                elif "REASONING:" in txt:
                    legacy_text = (
                        txt.split("##### REASONING ENDS #####")[0]
                        .replace("REASONING:", "")
                        .strip()
                    )
                    if legacy_text:
                        state["reasoning_placeholder"].info(
                            f"**Thinking:** {legacy_text}"
                        )

                if c_match:
                    code_content = re.sub(
                        r"^```python\s*|^```\s*",
                        "",
                        c_match.group(1).strip(),
                        flags=re.M,
                    )
                    state["code_placeholder"].markdown(
                        f"```python\n{code_content}\n```"
                    )
                elif "##### REASONING ENDS #####" in txt:
                    legacy_code = txt.split("##### REASONING ENDS #####")[-1].strip()
                    legacy_code = re.sub(
                        r"^```python\s*|^```\s*", "", legacy_code, flags=re.M
                    )
                    state["code_placeholder"].markdown(f"```python\n{legacy_code}\n```")

        elif event == "code_ready":
            state = stream_state.get(attempt, {})
            for key in ["placeholder", "reasoning_placeholder", "code_placeholder"]:
                if key in state and state[key]:
                    state[key].empty()

            reasoning_val = data.get("reasoning", "No reasoning found.")
            st.info(f"**Coder's Reasoning:** {reasoning_val}")
            with st.expander(f"Generated Code (Attempt {attempt})", expanded=False):
                st.code(data["code"], language="python")

            # Log Code and Reasoning to current attempt state
            if st.session_state["active_attempts"]:
                st.session_state["active_attempts"][-1]["reasoning"] = reasoning_val
                st.session_state["active_attempts"][-1]["code"] = data["code"]

        elif event == "generation_error":
            st.error("❌ Code generation failed before execution.")
            st.code(data["message"], language="text")

            if st.session_state["active_attempts"]:
                st.session_state["active_attempts"][-1]["status"] = "generation_error"
                st.session_state["active_attempts"][-1]["logs"] = data["message"]

        elif event == "safety_blocked":
            violation_text = "\n".join(f"- {v}" for v in data["violations"])
            st.error(
                f"**Safety check failed — script blocked.**\n\n"
                f"Dangerous patterns detected:\n{violation_text}"
            )
            if st.session_state["active_attempts"]:
                st.session_state["active_attempts"][-1]["status"] = "blocked"
                st.session_state["active_attempts"][-1]["logs"] = violation_text

        elif event == "exec_success":
            st.success("✅ Execution completed successfully.")
            db_sum = data.get("db_summary", "")
            if db_sum and db_sum != "No optimization database found.":
                st.markdown(db_sum)

            if data.get("plots"):
                st.write("#### Results")
                for plot_path in data["plots"]:
                    show_plot(plot_path)
            else:
                st.info("No plots were generated by this run.")

            # Log success to current attempt state
            if st.session_state["active_attempts"]:
                st.session_state["active_attempts"][-1]["status"] = "success"
                st.session_state["active_attempts"][-1]["db_summary"] = db_sum
                st.session_state["active_attempts"][-1]["plots"] = (
                    _snapshot_plot_files(
                        data.get("plots", []), f"attempt_{attempt}_success"
                    )
                )

        elif event == "exec_error":
            st.error("❌ Python error occurred.")
            st.code(data["stderr_tail"], language="text")

            # Log error to current attempt state
            if st.session_state["active_attempts"]:
                st.session_state["active_attempts"][-1]["status"] = "error"
                st.session_state["active_attempts"][-1]["logs"] = data["stderr_tail"]

        elif event == "no_converge":
            st.warning("⚠️ Optimiser did not converge — stopping (setup issue).")
            db_sum = data.get("db_summary", "")
            if db_sum and db_sum != "No optimization database found.":
                st.markdown(db_sum)
            if data.get("plots"):
                for plot_path in data["plots"]:
                    show_plot(plot_path)

            # Log non-convergence to current attempt state
            if st.session_state["active_attempts"]:
                st.session_state["active_attempts"][-1]["status"] = "no_converge"
                st.session_state["active_attempts"][-1]["db_summary"] = db_sum
                st.session_state["active_attempts"][-1]["plots"] = (
                    _snapshot_plot_files(
                        data.get("plots", []), f"attempt_{attempt}_no_converge"
                    )
                )

        elif event == "no_converge_final":
            no_converge_store.update(data)

    return cb


def _handle_agent_result(
    result,
    no_converge_data: dict,
    conversation_context: str,
    blueprints: list,
    routing_json: dict,
):
    """Render the final status card and save the assistant message to history."""
    attempts_history = list(st.session_state.get("active_attempts", []))
    archived_plots = _snapshot_plot_files(get_generated_plots(), "message")

    if result.success:
        is_opt = "run_driver()" in result.final_code
        task_word = "Optimization" if is_opt else "Analysis"
        content = (
            f"### ✅ {task_word} Complete\n"
            "The script ran successfully. Results and plots are shown above."
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "code": result.final_code,
                "summary": result.final_summary,
                "plots": archived_plots,
                "routing_json": routing_json,
                "attempts_history": attempts_history,
            }
        )
        st.rerun()

    elif no_converge_data:
        content = (
            "### ⚠️ Optimization did not converge\n"
            "The script ran without errors but the optimiser could not find a solution."
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "code": result.final_code,
                "summary": result.final_summary,
                "plots": archived_plots,
                "routing_json": routing_json,
                "attempts_history": attempts_history,
            }
        )
        st.session_state["pending_relaxation"] = {
            "suggestion": no_converge_data.get(
                "suggestion", "No suggestion available."
            ),
            "user_prompt": conversation_context,
            "blueprints": blueprints,
            "error_logs": no_converge_data.get("error_logs", []),
            "prior_code": result.final_code,
        }
        st.rerun()

    else:
        content = (
            f"### ❌ Agent could not complete the task\n"
            f"Failed after {result.attempts} attempt(s)."
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "code": result.final_code,
                "summary": result.final_summary,
                "plots": archived_plots,
                "routing_json": routing_json,
                "attempts_history": attempts_history,
            }
        )
        st.rerun()


# ---------------------------------------------------------------------------
# Replay conversation history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Reconstruct the "Agent Thought Process" expander (kept open by default on replay)
        if "attempts_history" in message:
            with st.expander("Agent Thought Process", expanded=True):
                # 1. Routing Replay
                st.write("### 1. Intent Routing")
                blueprints = message.get("routing_json", {}).get("blueprints", [])
                reason = message.get("routing_json", {}).get(
                    "reason", "No reason provided"
                )
                st.info(
                    f"**Selected Blueprint(s):** `{', '.join(blueprints)}`\n\n**Reason:** {reason}"
                )

                # Nested Routing JSON block inside the Intent Routing section
                if "routing_json" in message and message["routing_json"]:
                    with st.expander("Show Routing Logic (JSON)", expanded=False):
                        st.json(message["routing_json"])

                # 2. Generation Replay
                st.write("### 2. Iterative Generation & Execution")
                for att in message["attempts_history"]:
                    st.markdown(f"**Attempt {att['attempt']} of {att['max_retries']}**")
                    if att.get("reasoning"):
                        st.info(f"**Coder's Reasoning:** {att['reasoning']}")

                    # Only show code inside the attempts history loop if the attempt was a failure
                    # (this prevents duplicating the final successful code shown at the bottom)
                    if att.get("code") and att["status"] != "success":
                        with st.expander(
                            f"Generated Code (Attempt {att['attempt']})", expanded=False
                        ):
                            st.code(att["code"], language="python")

                    if att["status"] == "success":
                        st.success("✅ Execution completed successfully.")
                        # Database summary and plots for the successful run are shown once at the bottom
                    elif att["status"] == "error":
                        st.error("❌ Python error occurred.")
                        st.code(att["logs"], language="text")
                    elif att["status"] == "generation_error":
                        st.error("❌ Code generation failed before execution.")
                        st.code(att["logs"], language="text")
                    elif att["status"] == "no_converge":
                        st.warning("⚠️ Optimiser did not converge.")
                        if (
                            att.get("db_summary")
                            and att["db_summary"] != "No optimization database found."
                        ):
                            st.markdown(att["db_summary"])
                        if att.get("plots"):
                            for plot_path in att["plots"]:
                                show_plot(plot_path)

        if "code" in message:
            with st.expander("Show Generated Code", expanded=False):
                st.code(message["code"], language="python")
        if (
            message.get("summary")
            and message["summary"] != "No optimization database found."
        ):
            st.markdown(message["summary"])
        if message.get("plots"):
            for plot_path in message["plots"]:
                show_plot(plot_path)

# ---------------------------------------------------------------------------
# Pending relaxation approval card
# ---------------------------------------------------------------------------
if st.session_state["pending_relaxation"]:
    pr = st.session_state["pending_relaxation"]
    with st.container(border=True):
        st.warning("### ⚠️ Optimization did not converge — this is a setup issue")
        st.markdown(pr["suggestion"])
        st.markdown("**Or describe your own fix:**")
        user_override = st.text_area(
            label="Custom instructions (optional)",
            placeholder="e.g. Change thickness_cp bounds to 0.001–0.3 m...",
            key="relaxation_override",
            height=90,
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Apply and retry", key="approve_relaxation"):
                if user_override.strip():
                    extra = (
                        "Apply the following user-specified changes and retry:\n"
                        + user_override.strip()
                    )
                else:
                    extra = "Apply these relaxations and retry:\n" + pr["suggestion"]
                st.session_state["relaxation_prompt"] = (
                    pr["user_prompt"] + "\n\n" + extra
                )
                st.session_state["relaxation_blueprints"] = pr["blueprints"]
                st.session_state["relaxation_error_logs"] = pr.get("error_logs", [])
                st.session_state["relaxation_prior_code"] = pr.get("prior_code", "")
                st.session_state["pending_relaxation"] = None
                st.rerun()
        with col2:
            if st.button("❌ Dismiss", key="dismiss_relaxation"):
                st.session_state["pending_relaxation"] = None
                st.rerun()

# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Enter your design request...")

_relaxation_prompt = st.session_state.pop("relaxation_prompt", None)
_relaxation_blueprints = st.session_state.pop("relaxation_blueprints", None)
_relaxation_error_logs = st.session_state.pop("relaxation_error_logs", None)

if _relaxation_prompt:
    cleanup_artifacts()
    st.session_state["stop_run"] = False
    st.session_state["active_attempts"] = []  # Clear for retry

    with st.chat_message("assistant"):
        thought_expander = st.expander(
            "Relaxation Retry — Agent Thought Process", expanded=True
        )
        with thought_expander:
            st.info("Retrying with relaxed problem setup (approved by user).")
            st.write("### Iterative Generation & Execution")
            _rs: dict[int, dict] = {}
            _rn: dict = {}
            result = run_agent(
                user_prompt=_relaxation_prompt,
                blueprints=_relaxation_blueprints,
                model_name=model_name,
                provider=provider,
                max_retries=max_retries,
                stream=True,
                callback=_make_ui_callback(_rs, _rn),
                prior_error_logs=_relaxation_error_logs,
                prior_code=st.session_state.pop("relaxation_prior_code", ""),
            )
        # Pull latest available routing to satisfy 5-param signature
        prev_routing = next(
            (
                m["routing_json"]
                for m in reversed(st.session_state.messages)
                if "routing_json" in m
            ),
            {},
        )
        _handle_agent_result(
            result, _rn, _relaxation_prompt, _relaxation_blueprints or [], prev_routing
        )

if user_prompt:
    cleanup_artifacts()
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    st.session_state["stop_run"] = False
    st.session_state["active_attempts"] = []  # Reset active history for the new run
    conversation_context = build_conversation_context(st.session_state.messages)

    with st.chat_message("assistant"):
        thought_expander = st.expander("Agent Thought Process", expanded=True)
        with thought_expander:
            st.write("### 1. Intent Routing")
            router_placeholder = st.empty()
            router_streamed = ""
            routing_data = {}

            for chunk in route_intent_stream(
                conversation_context, model_name=model_name, provider=provider
            ):
                if isinstance(chunk, dict):
                    routing_data = chunk
                else:
                    router_streamed += chunk
                    clean_router = re.sub(r"</?routing>", "", router_streamed).strip()
                    router_placeholder.markdown(f"```\n{clean_router}\n```")

            router_placeholder.empty()
            st.session_state["current_routing_data"] = routing_data

            blueprints = routing_data.get("blueprints", ["aero_opt.py"])
            reason = routing_data.get("reason", "No reason provided")

            if routing_data.get("is_vague", False):
                missing_info = routing_data.get("missing_info", "Details missing.")
                show_vagueness_card(missing_info)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"#### Clarification needed\n\n{missing_info}",
                    }
                )
                st.stop()

            st.info(
                f"**Selected Blueprint(s):** `{', '.join(blueprints)}`\n\n**Reason:** {reason}"
            )

            # Show active Routing JSON expander within the Intent Routing section
            with st.expander("Show Routing Logic (JSON)", expanded=False):
                st.json(routing_data)

            st.write("### 2. Iterative Generation & Execution")
            _ss: dict[int, dict] = {}
            _nc: dict = {}
            result = run_agent(
                user_prompt=conversation_context,
                blueprints=blueprints,
                model_name=model_name,
                provider=provider,
                max_retries=max_retries,
                stream=True,
                callback=_make_ui_callback(_ss, _nc),
            )

        _handle_agent_result(
            result, _nc, conversation_context, blueprints, routing_data
        )
