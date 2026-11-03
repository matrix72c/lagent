"""Request-boundary tests migrated from the consumer, without scheduling policy."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from lagent.serving.sandbox.request_gate import Gate
from lagent.serving.sandbox.runtime import live_members


@pytest.mark.asyncio
async def test_pause_drains_inflight_and_keeps_next_request_pending():
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(request):
        calls.append(await request.json())
        entered.set()
        await finish.wait()
        return web.json_response({"content": "complete response"})

    app = web.Application()
    app.router.add_post("/v1/messages", model)
    async with TestServer(app) as upstream:
        gate = Gate(
            str(upstream.make_url("")),
            request_timeout=2,
            connect_timeout=1,
            read_timeout=1,
        )
        async with TestServer(gate.app()) as server, ClientSession() as client:

            async def control(action):
                async with client.post(
                    server.make_url("/__agent/" + action)
                ) as response:
                    return await response.json()

            async def request(value):
                async with client.post(
                    server.make_url("/v1/messages"), json={"turn": value}
                ) as response:
                    return await response.json()

            first = asyncio.create_task(request(1))
            await asyncio.sleep(0.03)
            assert not calls and not first.done()  # closed by default
            assert not (await control("status"))["open"]
            await control("resume")
            await entered.wait()
            assert (await control("pause"))["active"] == 1
            second = asyncio.create_task(request(2))
            finish.set()
            assert (await first)["content"] == "complete response"
            await asyncio.sleep(0.03)
            status = await control("status")
            assert status["active"] == 0 and status["completed"] == 1
            assert len(calls) == 1 and not second.done()
            await control("resume")
            assert (await second)["content"] == "complete response"
            assert len(calls) == 2


@pytest.mark.asyncio
async def test_pause_does_not_truncate_stream_or_apply_nonstream_budget():
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(request):
        calls.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        entered.set()
        await finish.wait()
        for _ in range(8):
            await response.write(b"data: chunk\n\n")
            await asyncio.sleep(0.02)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/messages", model)
    async with TestServer(app) as upstream:
        gate = Gate(
            str(upstream.make_url("")),
            request_timeout=0.05,
            connect_timeout=1,
            read_timeout=1,
        )
        gate.ready.set()
        async with TestServer(gate.app()) as server, ClientSession() as client:
            async with client.post(
                server.make_url("/v1/messages"), json={"stream": True}
            ) as response:
                await entered.wait()
                async with client.post(server.make_url("/__agent/pause")) as paused:
                    assert (await paused.json())["active"] == 1
                finish.set()
                body = await response.read()
                assert body.count(b"data: chunk") == 8
                assert body.startswith(b"data: first")
            assert gate.completed == 1 and gate.active == 0 and len(calls) == 1


def test_gate_process_lifecycle(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    pid_file = tmp_path / "gate.pid"
    cli = [sys.executable, "-m", "lagent.serving.sandbox.request_gate"]
    common = ["--port", str(port), "--pid-file", str(pid_file), "--timeout", "5"]

    def run(action, *extra):
        return subprocess.run(
            [*cli, action, *common, *extra],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

    pid = None
    try:
        started = json.loads(
            run(
                "start",
                "--upstream",
                "http://127.0.0.1:1",
                "--log",
                str(tmp_path / "gate.log"),
            ).stdout
        )
        pid = started["pid"]
        assert not started["open"] and live_members(pid)
        assert json.loads(run("resume").stdout)["open"]
        assert not json.loads(run("pause").stdout)["open"]
        assert json.loads(run("status").stdout)["pid"] == pid
        run("stop")
        assert not pid_file.exists() and not live_members(pid)
        run("stop")
    finally:
        if pid and live_members(pid):
            os.killpg(pid, signal.SIGKILL)
