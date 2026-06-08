import base64
import streamlit as st
import streamlit.components.v1 as components
import os
import re
import shutil

from llm.router import route_intent_stream
from llm.coder import generate_code_stream
from tools.executor import execute_run
from tools.db_reader import summarize_optimization

# ---------------------------------------------------------------------------
# Resolve absolute paths relative to this file so the app works regardless
# of the working directory it's launched from.
# ---------------------------------------------------------------------------
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_OUT_DIR     = os.path.join(_PROJECT_DIR, "openaerostruct_out")
_PLOTS_DIR   = os.path.join(_OUT_DIR, "agent_plots")
_GEN_RUN_DIR = os.path.join(_OUT_DIR, "generated_run_out")
_LOG_FILE    = os.path.join(_SRC_DIR, "agent_backend.log")
_GEN_SCRIPT  = os.path.join(_SRC_DIR, "generated_run.py")

# ---------------------------------------------------------------------------
# Safety: strip ANSI colour codes and block prompt-injection attempts.
# ---------------------------------------------------------------------------
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")
_INJECTION_PATTERN = re.compile(
    r"(?i)("
    r"ignore\s+(previous|all|prior)\s+(instructions?|prompts?|context)"
    r"|act\s+as\s+(a|an)\s+"
    r"|forget\s+(everything|all)"
    r"|you\s+are\s+now\s"
    r"|disregard\s+(all|previous|prior)"
    r"|new\s+instruction"
    r"|\bsystem\s*:"
    r")"
)


def sanitize_feedback(text: str, max_chars: int = 1000) -> str:
    text = _ANSI_ESCAPE.sub("", text)
    lines = [line for line in text.splitlines() if not _INJECTION_PATTERN.search(line)]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cleanup_artifacts():
    """Delete plots and output files left over from the previous run."""
    for p in [_OUT_DIR, _PLOTS_DIR, _GEN_RUN_DIR]:
        if os.path.exists(p):
            for filename in os.listdir(p):
                file_path = os.path.join(p, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(p, exist_ok=True)


def get_generated_plots():
    """Return a sorted list of image/PDF files produced by the last run."""
    plots = []
    if os.path.exists(_PLOTS_DIR):
        for f in os.listdir(_PLOTS_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                plots.append(os.path.join(_PLOTS_DIR, f))
    return sorted(plots)


# ---------------------------------------------------------------------------
# Pre-execution safety scan
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Destructive shell command (rm/rmdir/del)",
     re.compile(r"""(?x)
         os\.system\s*\(.*\brm\b
         | subprocess\.\w+\s*\(.*\brm\b
         | shutil\.rmtree
         | os\.remove\s*\(
         | os\.unlink\s*\(
         | os\.rmdir\s*\(
     """, re.IGNORECASE)),

    ("Network access (socket/requests/urllib/httpx)",
     re.compile(r"""(?x)
         import\s+socket
         | from\s+socket\s+import
         | import\s+requests
         | from\s+requests\s+import
         | import\s+urllib
         | from\s+urllib\s+import
         | import\s+httpx
         | import\s+aiohttp
         | urllib\.request
         | requests\.(get|post|put|delete|patch|head|session)
         | socket\.connect
         | socket\.bind
     """, re.IGNORECASE)),

    ("Arbitrary subprocess / shell execution",
     re.compile(r"""(?x)
         subprocess\.(run|call|Popen|check_output|check_call)\s*\(
         | os\.system\s*\(
         | os\.popen\s*\(
         | commands\.getoutput
     """, re.IGNORECASE)),

    ("File write outside allowed output paths",
     re.compile(r"""(?x)
         open\s*\(\s*['"]/(?!.*openaerostruct_out)
         | open\s*\(\s*['"]\.\.
         | open\s*\(\s*['"]~
     """, re.IGNORECASE)),

    ("Dynamic code execution (eval/exec/__import__)",
     re.compile(r"""(?x)
         \beval\s*\(
         | \bexec\s*\(
         | \b__import__\s*\(
         | \bcompile\s*\(.*exec
     """, re.IGNORECASE)),

    ("Environment / credential access",
     re.compile(r"""(?x)
         os\.environ\s*\[
         | os\.getenv\s*\(
         | keyring\.
     """, re.IGNORECASE)),
]


def check_script_safety(code: str) -> list[str]:
    violations = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for label, pattern in _DANGEROUS_PATTERNS:
            if pattern.search(line):
                violations.append(f"Line {lineno} [{label}]: `{stripped}`")
    return violations


# ---------------------------------------------------------------------------
# Conversation history helper
# ---------------------------------------------------------------------------

def build_conversation_context(messages: list) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vagueness help card — shown when the router flags is_vague: true
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


def show_plot(plot_path: str):
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
                # Store a clean summary in history (no DV table — keeps history compact)
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
            max_retries = 3
            attempt = 0
            success = False
            feedback = "Initial generation"
            final_code = ""
            final_summary = ""

            while attempt < max_retries:
                if st.session_state.get("stop_run", False):
                    st.warning("Agent stopped by user.")
                    break

                st.markdown(f"**Attempt {attempt + 1} of {max_retries}**")

                code_stream_placeholder = st.empty()
                streamed_text = ""
                code, reasoning = "", ""

                for chunk in generate_code_stream(
                    conversation_context,
                    blueprints,
                    feedback,
                    model_name=model_name,
                    provider=provider,
                ):
                    if isinstance(chunk, tuple):
                        code, reasoning = chunk
                    else:
                        streamed_text += chunk
                        code_stream_placeholder.markdown(f"```\n{streamed_text}\n```")

                code_stream_placeholder.empty()

                final_code = code
                st.info(f"**Coder's Reasoning:** {reasoning}")

                with open(_GEN_SCRIPT, "w", encoding="utf-8") as f:
                    f.write(code)

                with st.expander(f"Generated Code (Attempt {attempt + 1})", expanded=False):
                    st.code(code, language="python")

                # Safety scan
                violations = check_script_safety(code)
                if violations:
                    violation_text = "\n".join(f"- {v}" for v in violations)
                    st.error(
                        f"**Safety check failed — script blocked.**\n\n"
                        f"The following dangerous patterns were detected:\n{violation_text}"
                    )
                    feedback = (
                        f"Your previous script was blocked by the safety checker. "
                        f"Do NOT include any of the following:\n{violation_text}\n"
                        f"Rewrite the script without these patterns."
                    )
                    attempt += 1
                    continue

                with st.spinner("Executing OpenAeroStruct..."):
                    result = execute_run(_GEN_SCRIPT, timeout=120)

                if result.exit_code == 0:
                    st.success("✅ Execution completed successfully.")

                    possible_paths = [
                        os.path.join(_GEN_RUN_DIR, "aero.db"),
                        os.path.join(_GEN_RUN_DIR, "aerostruct.db"),
                        os.path.join(_GEN_RUN_DIR, "struct.db"),
                        os.path.join(_OUT_DIR, "aero.db"),
                        os.path.join(_OUT_DIR, "aerostruct.db"),
                        os.path.join(_OUT_DIR, "struct.db"),
                    ]
                    db_summary = "No optimization database found."
                    for path in possible_paths:
                        if os.path.exists(path):
                            summary = summarize_optimization(path)
                            if not summary.startswith("Error"):
                                db_summary = summary
                                break

                    st.markdown(db_summary)
                    final_summary = db_summary

                    plots = get_generated_plots()
                    if plots:
                        st.write("#### Results")
                        for plot_path in plots:
                            show_plot(plot_path)
                    else:
                        st.info("No plots were generated by this run.")

                    is_optimization = "run_driver()" in code
                    converged = any(
                        m in result.stdout
                        for m in ["Optimization terminated successfully", "Optimization Complete"]
                    )

                    if is_optimization:
                        if converged:
                            st.write("✅ Optimization converged.")
                            success = True
                            break
                        else:
                            st.warning("Script ran but the optimiser did not converge. Retrying with feedback.")
                            feedback = (
                                f"Optimization failed to converge. Results so far:\n{db_summary}\n\n"
                                f"Stdout tail:\n{sanitize_feedback(result.stdout, max_chars=400)}"
                            )
                    else:
                        st.write("✅ Analysis completed.")
                        success = True
                        break
                else:
                    st.error("❌ Python error occurred.")
                    st.code(result.stderr[-500:], language="text")
                    feedback = f"Python Execution Error:\n{sanitize_feedback(result.stderr, max_chars=1000)}"

                attempt += 1

        # ------------------------------------------------------------------
        # Final status message (outside the thought expander)
        # ------------------------------------------------------------------
        if success:
            is_optimization = "run_driver()" in final_code
            task_word = "Optimization" if is_optimization else "Analysis"
            assistant_content = (
                f"### ✅ {task_word} Complete\n"
                f"The script ran successfully. Results and plots are shown above."
            )
            st.markdown(assistant_content)
        else:
            assistant_content = (
                "### ❌ Agent could not complete the task\n"
                f"Failed after {max_retries} attempt(s). "
                "Check the error output above — you may need to rephrase your request "
                "or provide additional constraints."
            )
            st.error(assistant_content)

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_content,
            "code": final_code,
            "summary": final_summary,
            "plots": get_generated_plots(),
        })