import streamlit as st
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
#
# When the generated script runs, its stdout/stderr is fed back to the LLM
# as "feedback" for the next attempt. A malicious print statement inside
# that script could craft output like "ignore previous instructions: …" and
# trick the LLM into doing something unintended. The patterns below catch
# the most common forms of that attack and silently drop those lines before
# they reach the model.
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
    """
    Clean up process output before handing it back to the LLM.

    Strips terminal colour codes and removes any line that looks like a
    prompt-injection attempt. Also truncates to max_chars so we don't
    flood the model's context with a giant stack trace.
    """
    text = _ANSI_ESCAPE.sub("", text)
    lines = [line for line in text.splitlines() if not _INJECTION_PATTERN.search(line)]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# Helpers for cleaning up result files between runs and finding any plots
# that a script produced.
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
    """Return a sorted list of image files produced by the last run."""
    plots = []
    if os.path.exists(_PLOTS_DIR):
        for f in os.listdir(_PLOTS_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                plots.append(os.path.join(_PLOTS_DIR, f))
    return sorted(plots)


# ---------------------------------------------------------------------------
# Pre-execution safety scan
#
# Before we run any LLM-generated script we check its source code for
# patterns that have no legitimate place in an OpenAeroStruct simulation.
# If anything suspicious is found we abort and show the user exactly which
# line triggered the block.
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
    """
    Scan code line-by-line against _DANGEROUS_PATTERNS.
    Returns a list of violation strings (empty = safe).
    """
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
#
# Flattens the message history into a plain transcript so the router and
# coder understand follow-up messages in context. Code blobs are excluded
# by default to keep the prompt compact.
# ---------------------------------------------------------------------------

def build_conversation_context(messages: list) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit app setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="OpenAeroStruct Agent V3", layout="wide")

# Streamlit re-runs the entire script on every interaction, so we keep the
# chat history and stop-flag in session_state so they survive across re-runs.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_run" not in st.session_state:
    st.session_state["stop_run"] = False

# ---------------------------------------------------------------------------
# Sidebar — model selection and controls
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

st.title("OpenAeroStruct-LLM-Agent")

# ---------------------------------------------------------------------------
# Replay the conversation so it's visible when the page loads or re-runs.
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
                    st.image(plot_path)

# ---------------------------------------------------------------------------
# Main chat loop — handle a new message from the user.
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Enter your design request...")

if user_prompt:
    cleanup_artifacts()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with open(_LOG_FILE, "w") as f:
        f.write("")

    st.session_state["stop_run"] = False

    # Build the full transcript so the router and coder understand follow-ups
    # ("now increase the span to 15m") in the context of prior turns.
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

            blueprints = routing_data.get("blueprints", ["aero_rect_or_CRM.py"])
            reason = routing_data.get("reason", "No reason provided")

            is_vague = routing_data.get("is_vague", False)
            if is_vague:
                clarification_msg = routing_data.get(
                    "missing_info", "Could you give me more detail about what you'd like to do?"
                )
                st.warning(f"### Clarification Needed\n{clarification_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"### Clarification Needed\n{clarification_msg}",
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

                # Create the placeholder here, after the attempt label, so the
                # live stream appears directly below "Attempt X of X".
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

                # Clear the live stream; final code appears in the expander below.
                code_stream_placeholder.empty()

                final_code = code
                st.info(f"**Coder's Reasoning:** {reasoning}")

                with open(_GEN_SCRIPT, "w") as f:
                    f.write(code)

                with st.expander(f"Generated Code (Attempt {attempt + 1})", expanded=False):
                    st.code(code, language="python")

                # Safety scan — block the script if dangerous patterns are found.
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
                    st.success("Execution completed successfully.")

                    # -------------------------------------------------------
                    # Read the database — check both generated_run_out and
                    # the parent output dir as a fallback.
                    # -------------------------------------------------------
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
                            # summarize_optimization returns an error string
                            # starting with "Error" when it can't read the file.
                            if not summary.startswith("Error"):
                                db_summary = summary
                                break

                    st.markdown(db_summary)
                    final_summary = db_summary

                    plots = get_generated_plots()
                    for plot in plots:
                        st.image(plot)

                    if not plots:
                        st.info("No plots were generated by this run.")

                    # ----------------------------------------------------------
                    # Step 3 — Decide whether we're done.
                    # Optimisation runs need the solver to converge; plain
                    # analysis runs just need a clean exit code.
                    # ----------------------------------------------------------
                    is_optimization = "run_driver()" in code
                    converged = any(
                        m in result.stdout
                        for m in ["Optimization terminated successfully", "Optimization Complete"]
                    )

                    if is_optimization:
                        if converged:
                            st.write("Optimization Converged!")
                            success = True
                            break
                        else:
                            st.warning("Script ran but the optimiser did not converge. Retrying with feedback.")
                            feedback = (
                                f"Optimization failed to converge. Results so far:\n{db_summary}\n\n"
                                f"Stdout tail:\n{sanitize_feedback(result.stdout, max_chars=400)}"
                            )
                    else:
                        st.write("Analysis Completed!")
                        success = True
                        break
                else:
                    st.error("Python Error Occurred")
                    st.code(result.stderr[-500:], language="text")
                    feedback = f"Python Execution Error:\n{sanitize_feedback(result.stderr, max_chars=1000)}"

                attempt += 1

        if success:
            assistant_content = "### Optimization Successful\nI have generated and executed the requested design optimization."
            st.markdown(assistant_content)
        else:
            assistant_content = "### Optimization Failed\nThe agent was unable to converge on a working script within the retry limit."
            st.error(assistant_content)

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_content,
            "code": final_code,
            "summary": final_summary,
            "plots": get_generated_plots(),
        })