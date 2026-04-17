import subprocess
import sys
import os
import ast
import ctypes
import platform
import struct

# ---------------------------------------------------------------------------
# __file__-relative base paths
# ---------------------------------------------------------------------------
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_TOOLS_DIR)
_DEFAULT_SCRIPT = os.path.join(_SRC_DIR, "generated_run.py")

# ---------------------------------------------------------------------------
# Static-analysis whitelist
# ---------------------------------------------------------------------------
# Only these top-level module names are permitted in LLM-generated scripts.
ALLOWED_TOP_LEVEL_IMPORTS: frozenset = frozenset({
    # Numerics / science
    "numpy", "scipy",
    # Simulation frameworks
    "openaerostruct", "openmdao",
    # Plotting
    "matplotlib", "plotly", "niceplots",
    # Data handling
    "pandas",
    # Safe standard-library modules for computation scripts
    "os", "sys", "warnings", "math", "functools", "itertools",
    "collections", "typing", "pathlib",
})

# AST-level call names that must never appear, even if the module is allowed.
_DANGEROUS_CALL_NAMES: frozenset = frozenset({
    # Built-in code execution
    "eval", "exec", "compile", "__import__", "breakpoint",
    # os-module process helpers
    "system", "popen", "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    # subprocess / shell helpers
    "call", "Popen", "check_call", "check_output",
})


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
# Optional seccomp BPF sandbox (Linux / x86_64 only)
# ---------------------------------------------------------------------------
# Deny-list: network I/O + process-spawning syscalls.
# Everything else is allowed (Python / NumPy / SciPy need many syscalls).
_BLOCKED_SYSCALLS_X86_64: tuple = (
    41,   # socket
    42,   # connect
    43,   # accept
    44,   # sendto
    45,   # recvfrom
    46,   # sendmsg
    47,   # recvmsg
    48,   # shutdown
    49,   # bind
    50,   # listen
    56,   # clone
    57,   # fork
    58,   # vfork
    59,   # execve
    322,  # execveat
)

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP      = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_ALLOW   = 0x7FFF0000
_SECCOMP_RET_ERRNO   = 0x00050001  # ERRNO(EPERM) – returns clear error instead of KILL


def _build_seccomp_preexec():
    """
    Build and return a preexec_fn that applies a seccomp BPF deny-list in the
    child process before exec.  Linux / x86_64 only; returns None on any other
    platform or on any setup error so the caller degrades gracefully.

    BPF program layout (n = len(_BLOCKED_SYSCALLS_X86_64)):
        [0]       LD W ABS 0          – load syscall number
        [1..n]    JEQ <nr>, jt, 0     – if match jump to ERRNO, else fall through
        [n+1]     RET ALLOW           – default path
        [n+2]     RET ERRNO(EPERM)    – blocked path
    """
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return None

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None

    blocked = _BLOCKED_SYSCALLS_X86_64

    def _bpf_stmt(code: int, k: int) -> bytes:
        return struct.pack("<HBBI", code, 0, 0, k)

    def _bpf_jump(code: int, k: int, jt: int, jf: int) -> bytes:
        return struct.pack("<HBBI", code, jt, jf, k)

    BPF_LD  = 0x00
    BPF_JMP = 0x05
    BPF_RET = 0x06
    BPF_W   = 0x00
    BPF_ABS = 0x20
    BPF_JEQ = 0x10
    BPF_K   = 0x00

    n = len(blocked)
    instructions = [_bpf_stmt(BPF_LD | BPF_W | BPF_ABS, 0)]
    for i, nr in enumerate(blocked):
        # jt = forward jump distance to the ERRNO instruction at index n+2
        # Next instruction after this JEQ is at i+2; ERRNO is at n+2.
        jt = n - i
        instructions.append(_bpf_jump(BPF_JMP | BPF_JEQ | BPF_K, nr, jt, 0))
    instructions.append(_bpf_stmt(BPF_RET | BPF_K, _SECCOMP_RET_ALLOW))
    instructions.append(_bpf_stmt(BPF_RET | BPF_K, _SECCOMP_RET_ERRNO))

    filter_bytes = b"".join(instructions)
    # Keep the buffer alive for the lifetime of this module so the child
    # process (forked copy of this process) can safely reference it.
    filter_buf = ctypes.create_string_buffer(filter_bytes)
    filter_ptr = ctypes.cast(filter_buf, ctypes.c_void_p).value

    class _SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = _SockFprog(len=len(instructions), filter=filter_ptr)

    def _apply():
        try:
            libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
            libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(fprog))
        except Exception:
            pass  # fail open – static scan is the primary guard

    return _apply


_SECCOMP_PREEXEC = _build_seccomp_preexec()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ExecutionResult:
    def __init__(self, exit_code, stdout, stderr):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def execute_run(script_path=_DEFAULT_SCRIPT, timeout=120):
    """
    Validate and execute the generated OpenAeroStruct script.

    Validation (static AST scan) runs before execution.  If the code contains
    disallowed imports or dangerous calls the script is rejected immediately
    and no subprocess is spawned.

    On Linux/x86_64 an optional seccomp BPF filter is applied in the child
    process to block network and process-spawning syscalls as an additional
    defence-in-depth layer.
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
            -1, "",
            f"SecurityError: Generated code was rejected by static analysis – {reason}"
        )

    # ── Subprocess execution ───────────────────────────────────────────────
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_SECCOMP_PREEXEC,  # None → no-op; non-None → apply filter
        )
        return ExecutionResult(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8") if e.stdout else ""
        stderr = e.stderr.decode("utf-8") if e.stderr else ""
        stderr += f"\nTimeoutError: Script execution exceeded {timeout} seconds."
        return ExecutionResult(-1, stdout, stderr)
    except Exception as e:
        return ExecutionResult(-1, "", f"ExecutionException: {str(e)}")


if __name__ == "__main__":
    script = _DEFAULT_SCRIPT
    if len(sys.argv) > 1:
        script = sys.argv[1]
    res = execute_run(script)
    print(f"Exit Code: {res.exit_code}")
    print(f"Stdout:\n{res.stdout[:500]}..." if res.stdout else "Stdout: None")
    print(f"Stderr:\n{res.stderr[:500]}..." if res.stderr else "Stderr: None")
