"""Real daemon/client/driver subprocess integration; no SDK, endpoint, or trainer."""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from lagent.serving.sandbox import runtime


@pytest.mark.parametrize("delay", [0, 30])
def test_real_daemon_completion_and_budget(tmp_path, monkeypatch, delay):
    (tmp_path / "fixture_agent.py").write_text("""
import asyncio
from lagent.schema import AgentMessage
class FixtureAgent:
    def __init__(self, delay):
        self.delay = delay
        self.messages = []
    async def __call__(self, *messages):
        self.messages.append({'role': 'user', 'content': messages[0]})
        await asyncio.sleep(self.delay)
        self.messages.append({'role': 'assistant', 'content': 'done'})
        return AgentMessage(sender='fixture', content='done')
    def get_messages(self):
        return self.messages
""")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"type": "fixture_agent.FixtureAgent", "delay": delay})
    )
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("literal instruction")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path) + os.pathsep + str(Path(runtime.__file__).parents[3]),
    )
    # AF_UNIX paths have a small platform-specific length limit.
    sock = str(tmp_path / "s")
    pid_file, log = tmp_path / "pid", tmp_path / "log"
    cli = [sys.executable, "-m", "lagent.serving.sandbox.client_cli"]
    subprocess.run(
        [
            *cli,
            "start-agent-daemon",
            "--config",
            str(config),
            "--sock",
            sock,
            "--pid-file",
            str(pid_file),
            "--log",
            str(log),
        ],
        check=True,
        timeout=10,
    )
    pid = int(pid_file.read_text())
    try:
        subprocess.run(
            [
                *cli,
                "wait-ready",
                "--sock",
                sock,
                "--pid-file",
                str(pid_file),
                "--log",
                str(log),
                "--timeout",
                "10",
                "--poll-interval",
                ".05",
            ],
            check=True,
            timeout=12,
        )
        args = argparse.Namespace(
            python=sys.executable,
            path_prefix="",
            sock=sock,
            pid_file=str(pid_file),
            log=str(log),
            instruction=str(instruction),
            agent_timeout=1,
            shutdown_timeout=3,
            response_out=str(tmp_path / "response"),
            message_out=str(tmp_path / "messages"),
        )
        runtime.run(args)
        assert not runtime.live_members(pid)
        messages = json.loads((tmp_path / "messages").read_text())
        assert messages[0] == {"role": "user", "content": "literal instruction"}
        response = json.loads((tmp_path / "response").read_text())
        if delay:
            assert response["finish_reason"] == "timeout"
            assert response["finish_info"]["agent_stopped"]
        else:
            assert response["content"] == "done"
            assert messages[-1]["content"] == "done"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
