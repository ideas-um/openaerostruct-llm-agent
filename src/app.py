import streamlit as st
import os
import shutil

from llm.router import route_intent
from llm.coder import generate_code
from tools.executor import execute_run
from tools.db_reader import summarize_optimization

st.set_page_config(page_title="OpenAeroStruct Agent V3", layout="wide")

# Helper to clean up old results
def cleanup_artifacts():
    paths = [
        os.path.join("src", "openaerostruct_out", "agent_plots"),
        os.path.join("src", "openaerostruct_out", "generated_run_out"),
        os.path.join("openaerostruct_out", "agent_plots"),
        os.path.join("openaerostruct_out", "generated_run_out")
    ]
    for p in paths:
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

# Helper to find plots
def get_generated_plots():
    plot_dir = os.path.join("src", "openaerostruct_out", "agent_plots")
    plots = []
    if os.path.exists(plot_dir):
        for f in os.listdir(plot_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                plots.append(os.path.join(plot_dir, f))
    return sorted(plots)

# Persistent State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'stop_run' not in st.session_state:
    st.session_state['stop_run'] = False

st.sidebar.title("Configuration")
provider = st.sidebar.selectbox("Provider", ["Gemini API", "Ollama"])
model_name = st.sidebar.selectbox("Model Selection", ["gemma-4-26b-a4b-it", "gemini-3-flash-preview",  "gemini-2.5-flash", "llama3.2:latest"])
st.sidebar.markdown("Requires Ollama running locally or `GEMINI_API_KEY`.")

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state['stop_run'] = True
    cleanup_artifacts()
    st.rerun()

st.title("OpenAeroStruct-LLM-Agent")

# Display previous messages
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
                # Re-check existence as cleanup might have happened
                if os.path.exists(plot_path):
                    st.image(plot_path)

# Handle new user input
user_prompt = st.chat_input("Enter your design request...")

if user_prompt:
    # 0. CLEANUP
    cleanup_artifacts()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Clear backend log for new run
    log_path = os.path.join("src", "agent_backend.log")
    with open(log_path, "w") as f:
        f.write("")

    st.session_state['stop_run'] = False
    
    with st.chat_message("assistant"):
        thought_expander = st.expander("Agent Thought Process", expanded=True)
        
        with thought_expander:
            st.write("### 1. Intent Routing")
            routing_data = route_intent(user_prompt, model_name=model_name, provider=provider)
            blueprints = routing_data.get("blueprints", ["aero_rect_or_CRM.py"])
            reason = routing_data.get("reason", "No reason provided")
            
            is_vague = routing_data.get("is_vague", False)
            if is_vague:
                msg = f"### Clarification Needed\n{routing_data.get('missing_info', 'I need more details.')}"
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.stop()

            st.info(f"**Selected Blueprint(s):** `{', '.join(blueprints)}`\n\n**Reason:** {reason}")
            
            st.write("### 2. Iterative Generation & Execution")
            max_retries = 3
            attempt = 0
            success = False
            feedback = "Initial generation"
            final_code = ""
            final_summary = ""
            
            while attempt < max_retries:
                if st.session_state.get('stop_run', False):
                    st.warning("Agent stopped by user.")
                    break
                
                st.markdown(f"**Attempt {attempt + 1} of {max_retries}**")
                
                with st.spinner("Generating code..."):
                    code, reasoning = generate_code(user_prompt, blueprints, feedback, model_name=model_name, provider=provider)
                
                final_code = code
                st.info(f"**Coder's Reasoning:** {reasoning}")

                run_script_path = os.path.join("src", "generated_run.py")
                with open(run_script_path, "w") as f:
                    f.write(code)
                    
                with st.expander(f"Generated Code (Attempt {attempt + 1})", expanded=False):
                    st.code(code, language='python')
                
                with st.spinner("Executing OpenAeroStruct..."):
                    result = execute_run(run_script_path, timeout=120)
                
                if result.exit_code == 0:
                    st.success("Execution Completed Successfully")
                    
                    possible_paths = [
                        os.path.join("src", "openaerostruct_out", "aero.db"),
                        os.path.join("src", "openaerostruct_out", "aerostruct.db"),
                        os.path.join("src", "openaerostruct_out", "generated_run_out", "aero.db")
                    ]
                    db_summary = "Error: No database found."
                    for path in possible_paths:
                        summary = summarize_optimization(path)
                        if "File does not exist" not in summary:
                            db_summary = summary
                            break
                    
                    st.markdown(db_summary)
                    final_summary = db_summary

                    # Display any plots generated during execution
                    current_plots = get_generated_plots()
                    for plot in current_plots:
                        st.image(plot)

                    # 3. Success Detection Logic
                    is_optimization = "run_driver()" in code
                    converged = any(m in result.stdout for m in ["Optimization terminated successfully", "Optimization Complete"])
                    
                    if is_optimization:
                        if converged:
                            st.write("Optimization Converged!")
                            success = True
                            break
                        else:
                            st.warning("Executed but optimization did not finish or converge properly.")
                            feedback = f"Optimization failed to converge. Results so far:\n{db_summary}\n\nStdout tail:\n{result.stdout[-400:]}"
                    else:
                        # For analysis, exit_code 0 is enough
                        st.write("Analysis Completed!")
                        success = True
                        break
                else:
                    st.error("Python Error Occurred")
                    st.code(result.stderr[-500:], language='text')
                    feedback = f"Python Execution Error:\n{result.stderr[-1000:]}"
                
                attempt += 1

        if success:
            assistant_content = "### Optimization Successful\nI have generated and executed the requested design optimization."
            st.markdown(assistant_content)
        else:
            assistant_content = "### Optimization Failed\nThe agent was unable to converge on a working script within the retry limit."
            st.error(assistant_content)

        # Store the interaction in history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": assistant_content,
            "code": final_code,
            "summary": final_summary,
            "plots": get_generated_plots()
        })
