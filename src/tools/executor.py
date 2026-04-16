import subprocess
import sys
import os

class ExecutionResult:
    def __init__(self, exit_code, stdout, stderr):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

def execute_run(script_path=os.path.join("src", "generated_run.py"), timeout=120):
    """
    Executes the generated OpenAeroStruct script.
    Captures stdout and stderr securely.
    """
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return ExecutionResult(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode('utf-8') if e.stdout else ""
        stderr = e.stderr.decode('utf-8') if e.stderr else ""
        stderr += f"\nTimeoutError: Script execution exceeded {timeout} seconds."
        return ExecutionResult(-1, stdout, stderr)
    except Exception as e:
        return ExecutionResult(-1, "", f"ExecutionException: {str(e)}")

if __name__ == '__main__':
    script = 'generated_run.py'
    if len(sys.argv) > 1:
        script = sys.argv[1]
    res = execute_run(script)
    print(f"Exit Code: {res.exit_code}")
    print(f"Stdout:\n{res.stdout[:500]}..." if res.stdout else "Stdout: None")
    print(f"Stderr:\n{res.stderr[:500]}..." if res.stderr else "Stderr: None")
