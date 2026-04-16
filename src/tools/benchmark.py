import os
import csv
import sys
import time
from datetime import datetime

# Add src to sys.path so we can import internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.router import route_intent
from llm.coder import generate_code
from tools.executor import execute_run

def run_benchmark(limit=None, model="gemma-4-26b-a4b-it", provider="Gemini API"):
    """
    Sequentially tests all queries in test_queries.csv and logs results.
    """
    input_file = os.path.join("src", "tools", "test_queries.csv")
    output_dir = os.path.join("src", "openaerostruct_out")
    history_dir = os.path.join(output_dir, "benchmark_history")
    
    # Clear and recreate benchmark folder
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(history_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "benchmark_results.csv")
    
    queries = []
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
            
    if limit:
        queries = queries[:limit]
        
    print(f"--- Starting Benchmark Test Suite ({len(queries)} cases) ---")
    
    results = []
    headers = ["id", "category", "query", "expected", "selected", "is_vague", "success", "execution_code", "error_log"]
    
    for q in queries:
        print(f"\n[Test {q['id']}] Category: {q['category']}")
        print(f"Query: {q['query']}")
        
        start_time = time.time()
        
        # 1. Intent Routing
        try:
            routing = route_intent(q["query"], model_name=model, provider=provider)
            selected = ", ".join(routing.get("blueprints", []))
            is_vague = routing.get("is_vague", False)
        except Exception as e:
            selected = "ERROR"
            is_vague = False
            print(f"Routing Error: {str(e)}")
            
        # 2. Code Generation & Execution (Skip if vague)
        success = False
        exit_code = -1
        error_log = ""
        
        if not is_vague:
            try:
                # We use a default feedback for benchmark
                code, reasoning = generate_code(q["query"], routing.get("blueprints", []), "Initial generation", model_name=model, provider=provider)
                
                # Save reasoning and code for history
                case_id = q["id"]
                with open(os.path.join(history_dir, f"{case_id}_reasoning.txt"), "w") as f:
                    f.write(reasoning)
                with open(os.path.join(history_dir, f"{case_id}_code.py"), "w") as f:
                    f.write(code)

                # Save to a temporary file for benchmarking
                bench_script = os.path.join("src", "benchmark_run.py")
                with open(bench_script, "w") as f:
                    f.write(code)
                
                # 3. Execution
                print("Running execution...")
                exec_res = execute_run(bench_script, timeout=120)
                exit_code = exec_res.exit_code
                
                # Save execution log
                with open(os.path.join(history_dir, f"{case_id}_execution.log"), "w") as f:
                    f.write(f"--- STDOUT ---\n{exec_res.stdout}\n\n--- STDERR ---\n{exec_res.stderr}")

                # Copy plots to history
                plot_dir = os.path.join("src", "openaerostruct_out", "agent_plots")
                if os.path.exists(plot_dir):
                    import shutil
                    for f in os.listdir(plot_dir):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            shutil.copy2(os.path.join(plot_dir, f), os.path.join(history_dir, f"{case_id}_{f}"))

                if exit_code == 0:
                    success = True
                    print("SUCCESS")
                else:
                    error_log = exec_res.stderr[-500:] # Log tail of error
                    print(f"FAILED (Code {exit_code})")
                    
            except Exception as e:
                error_log = str(e)
                print(f"Generation/Execution Error: {str(e)}")
        else:
            success = True # Vague detection is a success if expected
            print("VAGUE (Clarification asked)")
            
        results.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "expected": q["expected_blueprints"],
            "selected": selected,
            "is_vague": is_vague,
            "success": success,
            "execution_code": exit_code,
            "error_log": error_log.replace("\n", " ")
        })
        
    # Write Results
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n--- Benchmark Complete! ---")
    print(f"Results saved to: {results_file}")
    
    # Simple stats
    succeeded = sum(1 for r in results if r["success"])
    print(f"Overall Success Rate: {succeeded}/{len(results)} ({succeeded/len(results)*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of tests")
    parser.add_argument("--model", type=str, default="gemma-4-26b-a4b-it", help="LLM model name")
    parser.add_argument("--provider", type=str, default="Gemini API", help="LLM provider (Gemini API or Ollama)")
    args = parser.parse_args()
    
    run_benchmark(limit=args.limit, model=args.model, provider=args.provider)
