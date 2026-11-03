"""In-environment daemon driver, using only Python's standard library.

Task timeout stops the agent, preserves available messages, then emits a task
budget result. Infrastructure or shutdown failures remain nonzero exits.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

LOG = "/tmp/agent_daemon.log"
PID = "/tmp/agent_daemon.pid"
SOCK = "/tmp/lagent_agent.sock"
RESPONSE = "/tmp/agent_response.json"
MESSAGES = "/tmp/agent_messages.json"


def live_members(pgid: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(path.parent.name))
    return members


def confirm_stopped(pid_file: Path, timeout: float) -> None:
    pgid = int(pid_file.read_text().strip())
    if pgid <= 1 or pgid == os.getpgrp():
        raise RuntimeError("unsafe agent process group")
    if live_members(pgid):
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while live_members(pgid):
        if time.monotonic() >= deadline:
            raise RuntimeError("agent process group is still running after shutdown")
        time.sleep(0.1)


def run(args: argparse.Namespace) -> None:
    env = dict(os.environ)
    sock = getattr(args, "sock", SOCK)
    log = getattr(args, "log", LOG)
    pid = getattr(args, "pid_file", PID)
    response_path = Path(getattr(args, "response_out", RESPONSE))
    messages_path = Path(getattr(args, "message_out", MESSAGES))
    if args.path_prefix:
        env["PATH"] = args.path_prefix + ":" + env.get("PATH", "/usr/bin:/bin")
    cli = [args.python, "-m", "lagent.serving.sandbox.client_cli"]

    def invoke(action: str, *extra: str, timeout: float = 30, check: bool = True):
        return subprocess.run(
            [*cli, action, "--sock", sock, "--log", log, *extra],
            env=env,
            timeout=timeout,
            check=check,
        )

    timed_out = False
    primary_error: BaseException | None = None
    try:
        # Outputs belong to this attempt, never reuse a previous response on timeout.
        response_path.unlink(missing_ok=True)
        messages_path.unlink(missing_ok=True)
        try:
            invoke(
                "chat",
                "--instruction-file",
                args.instruction,
                "--response-out",
                str(response_path),
                timeout=args.agent_timeout,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        # Retrieve real messages on both completion and budget exhaustion. Never
        # invent an empty trajectory to make a timed-out attempt look complete.
        invoke("get-messages", "--message-out", str(messages_path))
        messages = json.loads(messages_path.read_text())
        if not isinstance(messages, list):
            raise ValueError("agent returned a malformed trajectory")  # noqa: TRY004 - malformed wire payload
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        # Shutdown signals are not proof of exit; verify the process group too.
        try:
            try:
                invoke("shutdown", "--pid-file", pid, timeout=args.shutdown_timeout)
            finally:
                confirm_stopped(Path(pid), args.shutdown_timeout)
        except BaseException as cleanup:
            if primary_error is None:
                primary_error = cleanup
                raise
            primary_error.add_note(
                f"Agent shutdown also failed: {type(cleanup).__name__}: {cleanup}"
            )
        finally:
            gate_pid = getattr(args, "gate_pid_file", None)
            if gate_pid and Path(gate_pid).exists():
                try:
                    confirm_stopped(Path(gate_pid), args.shutdown_timeout)
                    Path(gate_pid).unlink(missing_ok=True)
                except BaseException as cleanup:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        f"Gate shutdown also failed: {type(cleanup).__name__}: {cleanup}"
                    )
    if timed_out:
        response = (
            json.loads(response_path.read_text())
            if response_path.exists()
            else {"content": ""}
        )
        response.update(
            finish_reason="timeout",
            finish_info={
                "termination_reason": "task_budget_exhausted",
                "agent_stopped": True,
            },
        )
        response_path.write_text(json.dumps(response) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sock", default=SOCK)
    parser.add_argument("--pid-file", default=PID)
    parser.add_argument(
        "--gate-pid-file", help="Also stop this owned gate on completion or failure"
    )
    parser.add_argument("--log", default=LOG)
    parser.add_argument("--response-out", default=RESPONSE)
    parser.add_argument("--message-out", default=MESSAGES)
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--agent-timeout", type=float, required=True)
    parser.add_argument("--shutdown-timeout", type=float, default=30)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
