"""Run the recipe driver against real local child processes and a neutral CLI fixture."""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys

import pytest

from lagent.serving.sandbox import runtime


@pytest.fixture
def driver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Let the neutral CLI fixture precede the checkout.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    for name in ("LOG", "PID", "SOCK", "RESPONSE", "MESSAGES"):
        monkeypatch.setattr(runtime, name, str(tmp_path / name.lower()))
    package = tmp_path / "lagent/serving/sandbox"
    package.mkdir(parents=True)
    for parent in (package, package.parent, package.parent.parent):
        (parent / "__init__.py").write_text("")
    (package / "client_cli.py").write_text("""
import json, os, sys, time
from pathlib import Path
args = sys.argv
def value(key): return args[args.index(key) + 1]
action = args[1]
if action == "chat":
    if os.environ.get("FIXTURE_CHAT_FAIL"):
        sys.exit(7)
    time.sleep(float(os.environ.get("FIXTURE_CHAT_SLEEP", "0")))
    Path(value("--response-out")).write_text(json.dumps({"content": "done"}))
elif action == "get-messages":
    data = [{"messages": [{"role": "assistant", "content": os.environ.get("LITERAL", "done")}]}]
    if os.environ.get("FIXTURE_BAD_TRACE"):
        data = {"error": "trace unavailable"}
    Path(value("--message-out")).write_text(json.dumps(data))
elif action == "shutdown" and os.environ.get("FIXTURE_SHUTDOWN_FAIL"):
    sys.exit(9)
""")
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (tmp_path / "pid").write_text(str(child.pid))
    args = argparse.Namespace(
        python=sys.executable,
        path_prefix="",
        instruction=str(tmp_path / "instruction.md"),
        agent_timeout=1,
        shutdown_timeout=1,
    )
    yield args, child, tmp_path
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=3)


def test_driver_retains_real_messages_and_confirms_process_exit(driver, monkeypatch):
    args, child, root = driver
    literal = "quotes '\"; $(false) `false`\nchild value"
    monkeypatch.setenv("LITERAL", literal)
    runtime.run(args)
    child.wait(timeout=2)
    assert not runtime.live_members(child.pid)
    assert json.loads((root / "response").read_text())["content"] == "done"
    assert (
        json.loads((root / "messages").read_text())[0]["messages"][0]["content"]
        == literal
    )


def test_budget_exhaustion_retains_trace_and_stops_before_success(driver, monkeypatch):
    args, child, root = driver
    monkeypatch.setenv("FIXTURE_CHAT_SLEEP", "30")
    args.agent_timeout = 0.15
    runtime.run(args)
    child.wait(timeout=2)
    assert not runtime.live_members(child.pid)
    assert json.loads((root / "messages").read_text())
    response = json.loads((root / "response").read_text())
    assert response["content"] == ""
    assert response["finish_reason"] == "timeout"
    assert response["finish_info"] == {
        "termination_reason": "task_budget_exhausted",
        "agent_stopped": True,
    }


@pytest.mark.parametrize(
    "failure", ["FIXTURE_CHAT_FAIL", "FIXTURE_BAD_TRACE", "FIXTURE_SHUTDOWN_FAIL"]
)
def test_driver_fails_closed_and_still_stops_process(driver, monkeypatch, failure):
    args, child, _ = driver
    monkeypatch.setenv(failure, "1")
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        runtime.run(args)
    child.wait(timeout=2)
    assert not runtime.live_members(child.pid)


def test_driver_preserves_primary_failure_if_shutdown_also_fails(driver, monkeypatch):
    args, child, _ = driver
    monkeypatch.setenv("FIXTURE_CHAT_FAIL", "1")
    monkeypatch.setenv("FIXTURE_SHUTDOWN_FAIL", "1")
    with pytest.raises(subprocess.CalledProcessError) as error:
        runtime.run(args)
    assert error.value.returncode == 7
    assert any("shutdown also failed" in note for note in error.value.__notes__)
    child.wait(timeout=2)


@pytest.mark.parametrize("fail", [False, True])
def test_gate_cleanup_even_on_agent_failure(driver, monkeypatch, fail):
    args, _, root = driver
    gate = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    gate_pid = root / "gate.pid"
    gate_pid.write_text(str(gate.pid))
    args.gate_pid_file = str(gate_pid)
    if fail:
        monkeypatch.setenv("FIXTURE_CHAT_FAIL", "1")
    try:
        if fail:
            with pytest.raises(subprocess.CalledProcessError):
                runtime.run(args)
        else:
            runtime.run(args)
        gate.wait(timeout=2)
        assert not runtime.live_members(gate.pid) and not gate_pid.exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(gate.pid, signal.SIGKILL)
        gate.wait(timeout=2)


def test_custom_paths_and_stale_response_not_reused(driver, monkeypatch):
    args, _, root = driver
    args.response_out = str(root / "custom-response.json")
    args.message_out = str(root / "custom-messages.json")
    args.sock, args.log, args.pid_file = (
        str(root / "sock"),
        str(root / "log"),
        str(root / "pid"),
    )
    (root / "custom-response.json").write_text('{"content": "stale previous attempt"}')
    args.agent_timeout = 0.15
    monkeypatch.setenv("FIXTURE_CHAT_SLEEP", "30")
    runtime.run(args)
    response = json.loads((root / "custom-response.json").read_text())
    assert response["content"] == "" and response["finish_reason"] == "timeout"
    assert json.loads((root / "custom-messages.json").read_text())


@pytest.mark.parametrize("option", ["--config-dir", "--lagent-root"])
def test_removed_legacy_root_options_are_rejected(monkeypatch, option):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runtime",
            "--instruction",
            "task.txt",
            "--agent-timeout",
            "1",
            option,
            "/old",
        ],
    )
    with pytest.raises(SystemExit) as error:
        runtime.main()
    assert error.value.code == 2
