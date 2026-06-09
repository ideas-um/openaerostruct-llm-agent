import os
import streamlit as st

from llm.router import route_intent_stream
from agent_logic import (
    run_agent,
    cleanup_artifacts,
    get_generated_plots,
)

# ---------------------------------------------------------------------------
# Resolve absolute paths relative to this file
# ---------------------------------------------------------------------------
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_SRC_DIR, "agent_backend.log")

# ---------------------------------------------------------------------------
# Design variable reference card (shown when router flags is_vague)
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
- *"Compute CL/CD polars for a CRM wing at Mach 0.3–0.8, alpha -5 to 15 deg."*
- *"Aerostructural tube optimization on a CRM wing at Mach 0.84, minimize fuelburn. DVs: twist_cp, thickness_cp. Constraints: failure≤0, L=W."*
"""


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_plot(plot_path: str):
    """Display a plot file inline. PDFs get a download button; PNGs render inline."""
    if plot_path.lower().endswith(".pdf"):
        st.download_button(
            label=f"📥 {os.path.basename(plot_path)}",
            data=open(plot_path, "rb").read(),
            file_name=os.path.basename(plot_path),
            mime="application/pdf",
        )
    else:
        st.image(plot_path)


def show_vagueness_card(missing_info: str):
    """Render a structured clarification card in the chat."""
    st.warning("#### The agent needs more information before it can run.")
    st.markdown(f"**What's missing:**\n\n{missing_info}")
    with st.expander("💡 What can I specify? (design variable reference)", expanded=False):
        st.markdown(_DV_REFERENCE)
    st.info("Please reply with the missing details and the agent will proceed.")


def build_conversation_context(messages: list) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit app setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="OpenAeroStruct Agent", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_run" not in st.session_state:
    st.session_state["stop_run"] = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
provider = st.sidebar.selectbox("Provider", ["Gemini API", "Ollama"])
model_name = st.sidebar.selectbox(
    "Model Selection",
    ["gemini-flash-lite-latest", "gemma-4-26b-a4b-it", "gemini-3-flash-preview", "gemini-2.5-flash", "llama3.2:latest"],
)
st.sidebar.markdown("Requires Ollama running locally or `GEMINI_API_KEY`.")

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state["stop_run"] = True
    cleanup_artifacts()
    st.rerun()

st.title("OpenAeroStruct — LLM Agent")

# ---------------------------------------------------------------------------
# Replay conversation history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "code" in message:
            with st.expander("Show Generated Code", expanded=False):
                st.code(message["code"], language="python")
        if "summary" in message:
            st.markdown(message["summary"])
        if "plots" in message:
            for plot_path in message["plots"]:
                if os.path.exists(plot_path):
                    show_plot(plot_path)

# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Enter your design request...")

if user_prompt:
    cleanup_artifacts()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    st.session_state["stop_run"] = False

    conversation_context = build_conversation_context(st.session_state.messages)

    with st.chat_message("assistant"):
        thought_expander = st.expander("Agent Thought Process", expanded=True)

        with thought_expander:
            # ------------------------------------------------------------------
            # Step 1 — Intent routing (streamed)
            # ------------------------------------------------------------------
            st.write("### 1. Intent Routing")
            router_placeholder = st.empty()
            router_streamed = ""
            routing_data = {}

            for chunk in route_intent_stream(
                conversation_context,
                model_name=model_name,
                provider=provider,
            ):
                if isinstance(chunk, dict):
                    routing_data = chunk
                else:
                    router_streamed += chunk
                    router_placeholder.markdown(f"```\n{router_streamed}\n```")

            router_placeholder.empty()

            blueprints = routing_data.get("blueprints", ["aero_opt.py"])
            reason     = routing_data.get("reason", "No reason provided")

            is_vague = routing_data.get("is_vague", False)
            if is_vague:
                missing_info = routing_data.get(
                    "missing_info",
                    "Please provide more detail about your design request."
                )
                show_vagueness_card(missing_info)
                clarification_msg = (
                    f"#### Clarification needed\n\n{missing_info}\n\n"
                    "_Please reply with the missing details._"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clarification_msg,
                })
                st.stop()

            st.info(f"**Selected Blueprint(s):** `{', '.join(blueprints)}`\n\n**Reason:** {reason}")

            # ------------------------------------------------------------------
            # Step 2 — Iterative code generation and execution
            # ------------------------------------------------------------------
            st.write("### 2. Iterative Generation & Execution")

            # Per-attempt stream state (keyed by attempt number)
            _stream_state: dict[int, dict] = {}

            def ui_callback(event: str, data: dict):
                """Translate agent_logic events into Streamlit UI updates."""
                if st.session_state.get("stop_run", False):
                    return

                attempt = data.get("attempt", 0)

                if event == "attempt_start":
                    st.markdown(f"**Attempt {data['attempt']} of {data['max_retries']}**")
                    _stream_state[attempt] = {"placeholder": st.empty(), "text": ""}

                elif event == "code_chunk":
                    state = _stream_state.get(attempt, {})
                    state["text"] = state.get("text", "") + data["chunk"]
                    state.get("placeholder", st.empty()).markdown(
                        f"```\n{state['text']}\n```"
                    )

                elif event == "code_ready":
                    state = _stream_state.get(attempt, {})
                    if "placeholder" in state:
                        state["placeholder"].empty()
                    st.info(f"**Coder's Reasoning:** {data['reasoning']}")
                    with st.expander(f"Generated Code (Attempt {attempt})", expanded=False):
                        st.code(data["code"], language="python")

                elif event == "safety_blocked":
                    violation_text = "\n".join(f"- {v}" for v in data["violations"])
                    st.error(
                        f"**Safety check failed — script blocked.**\n\n"
                        f"The following dangerous patterns were detected:\n{violation_text}"
                    )

                elif event == "exec_success":
                    st.success("✅ Execution completed successfully.")
                    st.markdown(data["db_summary"])
                    if data["plots"]:
                        st.write("#### Results")
                        for plot_path in data["plots"]:
                            show_plot(plot_path)
                    else:
                        st.info("No plots were generated by this run.")

                elif event == "exec_error":
                    st.error("❌ Python error occurred.")
                    st.code(data["stderr_tail"], language="text")

                elif event == "no_converge":
                    st.warning("Script ran but the optimiser did not converge. Retrying with feedback.")

            result = run_agent(
                user_prompt=conversation_context,
                blueprints=blueprints,
                model_name=model_name,
                provider=provider,
                max_retries=3,
                stream=True,
                callback=ui_callback,
            )

        # ------------------------------------------------------------------
        # Final status (outside the thought expander)
        # ------------------------------------------------------------------
        if result.success:
            is_optimization = "run_driver()" in result.final_code
            task_word = "Optimization" if is_optimization else "Analysis"
            assistant_content = (
                f"### ✅ {task_word} Complete\n"
                f"The script ran successfully. Results and plots are shown above."
            )
            st.markdown(assistant_content)
        else:
            assistant_content = (
                "### ❌ Agent could not complete the task\n"
                f"Failed after {result.attempts} attempt(s). "
                "Check the error output above — you may need to rephrase your request "
                "or provide additional constraints."
            )
            st.error(assistant_content)

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_content,
            "code": result.final_code,
            "summary": result.final_summary,
            "plots": get_generated_plots(),
        })