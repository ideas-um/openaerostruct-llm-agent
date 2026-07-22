
from tools import executor


class DummyCompletedProcess:
    returncode = 0
    stdout = ""
    stderr = ""


def test_docker_runs_staged_script_by_original_basename(monkeypatch, tmp_path):
    script = tmp_path / "benchmark_run.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    staged_root = tmp_path / "stage"
    staged_script = staged_root / "src" / "benchmark_run.py"
    (staged_root / "src" / "generated_run_out").mkdir(parents=True)
    (staged_root / "openaerostruct_out").mkdir()
    staged_script.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")

    seccomp = tmp_path / "seccomp.json"
    seccomp.write_text("{}", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return DummyCompletedProcess()

    monkeypatch.setattr(executor, "_stage_container_workspace", lambda _: (str(staged_root), str(staged_script)))
    monkeypatch.setattr(executor, "_docker_image_name", lambda: "openaerostruct-sandbox:latest")
    monkeypatch.setattr(executor, "_docker_seccomp_profile", lambda: str(seccomp))
    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    monkeypatch.setattr(executor.shutil, "rmtree", lambda *args, **kwargs: None)

    result = executor._execute_in_docker(str(script), timeout=120)

    assert result.exit_code == 0
    cmd = captured["cmd"]
    assert any(f"target=/workspace/src/{script.name},readonly" in arg for arg in cmd)
    assert cmd[-2:] == ["python", f"/workspace/src/{script.name}"]
