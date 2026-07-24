import subprocess
import sys
import os
import ast
import shutil
import tempfile
import uuid
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# __file__-relative base paths
# ---------------------------------------------------------------------------
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_TOOLS_DIR)
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_DEFAULT_SCRIPT = os.path.join(_SRC_DIR, "generated_run.py")
_DOCKER_IMAGE_ENV = "OAS_SANDBOX_IMAGE"
_DOCKER_BACKEND_ENV = "OAS_EXECUTION_BACKEND"
_HOST_PYTHON_ENV = "OAS_HOST_PYTHON"
_DEFAULT_DOCKER_IMAGE = "openaerostruct-sandbox:latest"
_DOCKER_STAGE_DIR_ENV = "OAS_DOCKER_STAGE_DIR"
_DOCKER_SECCOMP_ENV = "OAS_DOCKER_SECCOMP_PROFILE"
_DEFAULT_DOCKER_STAGE_DIR = os.path.join(_PROJECT_DIR, ".docker_stage")
_DEFAULT_DOCKER_SECCOMP = os.path.join(
    _PROJECT_DIR,
    "docker",
    "seccomp-openaerostruct.json",
)


def _load_env_file() -> None:
    """Load a local .env file so execution backend settings work cross-platform."""
    for env_candidate in [
        os.path.join(_PROJECT_DIR, ".env"),
        os.path.join(_SRC_DIR, ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]:
        if os.path.exists(env_candidate):
            load_dotenv(env_candidate, override=True, encoding="utf-8")
            break


_load_env_file()

# ---------------------------------------------------------------------------
# Static-analysis whitelist
# ---------------------------------------------------------------------------
# Only these top-level module names are permitted in LLM-generated scripts.
ALLOWED_TOP_LEVEL_IMPORTS: frozenset = frozenset(
    {
        # Numerics / science
        "numpy",
        "scipy",
        # Simulation frameworks
        "openaerostruct",
        "openmdao",
        # Plotting
        "matplotlib",
        # Data handling
        "pandas",
        # Safe standard-library modules for computation scripts
        "os",
        "sys",
        "warnings",
        "math",
        "functools",
        "itertools",
        "collections",
        "typing",
        "pathlib",
        "re",
    }
)

# AST-level call names that must never appear, even if the module is allowed.
_DANGEROUS_CALL_NAMES: frozenset = frozenset(
    {
        # Built-in code execution
        "eval",
        "exec",
        "compile",
        "__import__",
        "breakpoint",
        # os-module process helpers
        "system",
        "popen",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        # subprocess / shell helpers
        "call",
        "Popen",
        "check_call",
        "check_output",
    }
)


def validate_generated_code(code: str) -> tuple:
    """
    Statically analyse LLM-generated code with the AST before execution.

    Checks:
    - Every top-level imported module must be in ALLOWED_TOP_LEVEL_IMPORTS.
    - No call to a function or method in _DANGEROUS_CALL_NAMES.

    Returns (is_safe: bool, reason: str).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error in generated code: {exc}"

    for node in ast.walk(tree):
        # ── Import checks ─────────────────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_TOP_LEVEL_IMPORTS:
                    return False, f"Disallowed import: '{alias.name}'"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in ALLOWED_TOP_LEVEL_IMPORTS:
                    return False, f"Disallowed 'from' import: '{node.module}'"

        # ── Dangerous call checks ──────────────────────────────────────────
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_CALL_NAMES:
                return False, f"Disallowed call: '{func.id}()'"
            if isinstance(func, ast.Attribute) and func.attr in _DANGEROUS_CALL_NAMES:
                return False, f"Disallowed method call: '.{func.attr}()'"

    return True, "OK"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ExecutionResult:
    def __init__(self, exit_code, stdout, stderr):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def _decode_timeout_output(value) -> str:
    if not value:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _detect_docker_backend() -> tuple[bool, str]:
    """Return whether Docker is usable and a short diagnostic string."""
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return False, "docker CLI not found"
    except Exception as exc:
        return False, f"Docker probe failed: {exc}"

    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "unknown Docker error"
        return False, f"Docker unavailable: {msg}"

    return True, proc.stdout.strip() or "Docker available"


def _docker_image_name() -> str:
    return os.getenv(_DOCKER_IMAGE_ENV, _DEFAULT_DOCKER_IMAGE)


def _docker_seccomp_profile() -> str:
    return os.getenv(_DOCKER_SECCOMP_ENV, _DEFAULT_DOCKER_SECCOMP)


def _docker_image_ready(image: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return False, f"Unable to inspect Docker image '{image}': {exc}"

    if proc.returncode != 0:
        return (
            False,
            f"Docker image '{image}' is not available locally. Build it first.",
        )
    return True, "Docker image is ready"


def _resolve_execution_backend() -> tuple[str, str]:
    """
    Resolve execution backend from env vars.

    `host`   → always use direct subprocess execution.
    `docker` → require Docker and the configured sandbox image.
    `auto`   → prefer Docker when the image is ready; otherwise fall back to host.
    """
    requested = os.getenv(_DOCKER_BACKEND_ENV, "auto").strip().lower()
    if requested not in {"auto", "host", "docker"}:
        requested = "auto"

    if requested == "host":
        return "host", "Host execution requested"

    docker_ok, docker_msg = _detect_docker_backend()
    if not docker_ok:
        if requested == "docker":
            return "error", docker_msg
        return "host", f"{docker_msg}; falling back to host execution"

    image = _docker_image_name()
    image_ok, image_msg = _docker_image_ready(image)
    if not image_ok:
        if requested == "docker":
            return "error", image_msg
        return "host", f"{image_msg}; falling back to host execution"

    seccomp_profile = _docker_seccomp_profile()
    if not os.path.isfile(seccomp_profile):
        msg = f"Docker seccomp profile '{seccomp_profile}' is missing."
        if requested == "docker":
            return "error", msg
        return "host", f"{msg}; falling back to host execution"

    return "docker", f"Using Docker sandbox image '{image}'"


def _host_output_dir_for(script_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    project_dir = os.path.dirname(script_dir)
    return os.path.join(project_dir, "openaerostruct_out")


def _copy_tree_contents(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _docker_stage_root() -> str:
    """
    Return a host temp directory that is more likely to be bind-mountable by Docker.

    By default, stage under the project directory so Docker Desktop can mount it
    without extra file-sharing configuration on macOS. Users can override this
    with `OAS_DOCKER_STAGE_DIR` if they prefer another already-shared location.
    """
    configured = os.getenv(_DOCKER_STAGE_DIR_ENV, "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured

    os.makedirs(_DEFAULT_DOCKER_STAGE_DIR, exist_ok=True)
    return _DEFAULT_DOCKER_STAGE_DIR


def _stage_container_workspace(script_path: str) -> tuple[str, str]:
    """
    Stage the generated script into an isolated temporary workspace.

    Layout inside the temp workspace mirrors the project:
    - <tmp>/src/<script>
    - <tmp>/openaerostruct_out/
    """
    tmp_root = tempfile.mkdtemp(
        prefix="oas-sandbox-",
        dir=_docker_stage_root(),
    )
    staged_src_dir = os.path.join(tmp_root, "src")
    staged_out_dir = os.path.join(tmp_root, "openaerostruct_out")
    staged_src_run_out_dir = os.path.join(staged_src_dir, "generated_run_out")
    os.makedirs(staged_src_dir, exist_ok=True)
    os.makedirs(staged_out_dir, exist_ok=True)
    os.makedirs(staged_src_run_out_dir, exist_ok=True)

    staged_script_path = os.path.join(staged_src_dir, os.path.basename(script_path))
    shutil.copy2(script_path, staged_script_path)
    return tmp_root, staged_script_path


def _host_python_executable() -> str:
    requested = os.getenv(_HOST_PYTHON_ENV)
    if requested:
        return requested
    return sys.executable


def _execute_on_host(script_path: str, timeout: int) -> ExecutionResult:
    """Run the validated script directly on the host Python interpreter."""
    try:
        python_exe = _host_python_executable()
        proc = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_timeout_output(exc.stdout)
        stderr = _decode_timeout_output(exc.stderr)
        stderr += f"\nTimeoutError: Script execution exceeded {timeout} seconds."
        return ExecutionResult(-1, stdout, stderr)
    except Exception as exc:
        return ExecutionResult(-1, "", f"ExecutionException: {str(exc)}")


def _execute_in_docker(script_path: str, timeout: int) -> ExecutionResult:
    """
    Run the validated script inside an isolated Docker container.

    The container receives only a temporary workspace containing the generated
    script and a writable output directory. After the run, the output tree is
    copied back into the host project's `openaerostruct_out/` directory.
    """
    image = _docker_image_name()
    host_output_dir = _host_output_dir_for(script_path)
    staged_root, staged_script_path = _stage_container_workspace(script_path)
    staged_script_name = os.path.basename(staged_script_path)
    staged_out_dir = os.path.join(staged_root, "openaerostruct_out")
    staged_src_run_out_dir = os.path.join(staged_root, "src", "generated_run_out")
    seccomp_profile = os.path.abspath(_docker_seccomp_profile())
    container_name = f"oas-sandbox-{uuid.uuid4().hex[:12]}"

    try:
        uid = os.getuid() if hasattr(os, "getuid") else None
        gid = os.getgid() if hasattr(os, "getgid") else None

        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--security-opt",
            f"seccomp={seccomp_profile}",
            "--pids-limit",
            "256",
            "--memory",
            "4g",
            "--cpus",
            "2",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "--tmpfs",
            "/var/tmp:rw,noexec,nosuid,size=256m",
            "--tmpfs",
            "/run:rw,nosuid,nodev,size=16m",
            "--tmpfs",
            "/dev/shm:rw,nosuid,nodev,size=256m",
            "--mount",
            (
                "type=bind,"
                f"source={staged_script_path},"
                f"target=/workspace/src/{staged_script_name},"
                "readonly"
            ),
            "--mount",
            (
                "type=bind,"
                f"source={staged_src_run_out_dir},"
                "target=/workspace/src/generated_run_out"
            ),
            "--mount",
            (f"type=bind,source={staged_out_dir},target=/workspace/openaerostruct_out"),
            "-w",
            "/workspace/src",
        ]
        if uid is not None and gid is not None:
            cmd.extend(["--user", f"{uid}:{gid}"])

        cmd.extend(
            [image, "python", f"/workspace/src/{staged_script_name}"]
        )

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if os.path.exists(staged_out_dir):
            _copy_tree_contents(staged_out_dir, host_output_dir)

        stderr = proc.stderr
        if stderr:
            stderr = f"[docker sandbox]\n{stderr}"
        else:
            stderr = "[docker sandbox]"
        return ExecutionResult(proc.returncode, proc.stdout, stderr)

    except subprocess.TimeoutExpired as exc:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        stdout = _decode_timeout_output(exc.stdout)
        stderr = _decode_timeout_output(exc.stderr)
        stderr += (
            f"\nTimeoutError: Docker sandbox execution exceeded {timeout} seconds."
        )
        return ExecutionResult(-1, stdout, stderr)
    except Exception as exc:
        return ExecutionResult(-1, "", f"DockerSandboxException: {str(exc)}")
    finally:
        shutil.rmtree(staged_root, ignore_errors=True)


def execute_run(script_path=_DEFAULT_SCRIPT, timeout=120):
    """
    Validate and execute the generated OpenAeroStruct script.

    Validation (static AST scan) runs before execution.  If the code contains
    disallowed imports or dangerous calls the script is rejected immediately
    and no subprocess is spawned.

    Docker execution adds the sandbox boundary when configured; host execution
    relies on the static AST scan.
    """
    # ── Read generated code ────────────────────────────────────────────────
    try:
        with open(script_path, "r") as f:
            code = f.read()
    except OSError as exc:
        return ExecutionResult(-1, "", f"Cannot read script '{script_path}': {exc}")

    # ── Static analysis ────────────────────────────────────────────────────
    is_safe, reason = validate_generated_code(code)
    if not is_safe:
        return ExecutionResult(
            -1,
            "",
            f"SecurityError: Generated code was rejected by static analysis – {reason}",
        )

    backend, backend_msg = _resolve_execution_backend()
    if backend == "error":
        return ExecutionResult(-1, "", f"SandboxConfigurationError: {backend_msg}")

    if backend == "docker":
        res = _execute_in_docker(script_path, timeout)
        if backend_msg:
            res.stderr = f"{backend_msg}\n{res.stderr}".strip()
        return res

    res = _execute_on_host(script_path, timeout)
    if backend_msg:
        res.stderr = f"{backend_msg}\n{res.stderr}".strip()
    return res


if __name__ == "__main__":
    script = _DEFAULT_SCRIPT
    if len(sys.argv) > 1:
        script = sys.argv[1]
    res = execute_run(script)
    print(f"Exit Code: {res.exit_code}")
    print(f"Stdout:\n{res.stdout[:500]}..." if res.stdout else "Stdout: None")
    print(f"Stderr:\n{res.stderr[:500]}..." if res.stderr else "Stderr: None")
