"""Sandbox-local request gate, independent of the caller's scheduling policy.

Pause stops admission, but drains complete responses without truncating SSE or
replaying a model/tool turn. The daemon and client connections remain alive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import Request, urlopen

from aiohttp import ClientSession, ClientTimeout, web

CLIENT = web.AppKey("client", ClientSession)


class Gate:
    def __init__(
        self,
        upstream: str,
        *,
        request_timeout: float,
        connect_timeout: float,
        read_timeout: float,
    ):
        self.upstream = upstream.rstrip("/")
        self.request_timeout = ClientTimeout(
            total=request_timeout, connect=connect_timeout, sock_read=read_timeout
        )
        self.stream_timeout = ClientTimeout(
            total=None, connect=connect_timeout, sock_read=read_timeout
        )
        self.ready = asyncio.Event()  # Start closed until the owner registers us.
        self.active = 0
        self.completed = 0
        self.waiting = 0

    async def control(self, request):
        action = request.match_info["action"]
        if action == "pause":
            self.ready.clear()
        elif action == "resume":
            self.ready.set()
        elif action != "status":
            raise web.HTTPNotFound()
        return web.json_response(
            {
                "open": self.ready.is_set(),
                "active": self.active,
                "completed": self.completed,
                "waiting": self.waiting,
                "pid": os.getpid(),
            }
        )

    async def forward(self, request):
        body = await request.read()
        try:
            payload = json.loads(body)
            stream = isinstance(payload, dict) and payload.get("stream") is True
        except (ValueError, UnicodeDecodeError):
            stream = False
        self.waiting += 1
        try:
            while True:
                await self.ready.wait()
                if self.ready.is_set():
                    break
        finally:
            self.waiting -= 1
        # No await between admission and accounting: pause sees every admitted request.
        self.active += 1
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection"}
        }
        try:
            async with request.app[CLIENT].request(
                request.method,
                self.upstream + request.rel_url.path_qs,
                headers=headers,
                data=body,
                allow_redirects=False,  # Never replay a submitted model/tool turn.
                timeout=self.stream_timeout if stream else self.request_timeout,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower()
                        not in {
                            "transfer-encoding",
                            "content-length",
                            "content-encoding",
                            "connection",
                        }
                    },
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                if (
                    request.method == "POST"
                    and request.path.endswith("/messages")
                    and upstream.status < 300
                ):
                    self.completed += 1
                return response
        finally:
            self.active -= 1

    def app(self):
        app = web.Application(client_max_size=64 * 1024**2)

        @asynccontextmanager
        async def client_context(app):
            async with ClientSession(trust_env=False) as client:
                app[CLIENT] = client
                yield

        async def cleanup(app):
            async with client_context(app):
                yield

        app.cleanup_ctx.append(cleanup)
        app.router.add_post("/__agent/{action}", self.control)
        app.router.add_route("*", "/{path:.*}", self.forward)
        return app


def main() -> None:
    """Serve in foreground, or start/control/stop an owned background gate.

    ``pause`` only closes admission; ``stop`` terminates the gate and is for
    final cleanup after the agent has exited, not for pausing active streams.
    """
    from .client_cli import _read_pid
    from .runtime import confirm_stopped, live_members

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=["serve", "start", "status", "pause", "resume", "stop"],
    )
    parser.add_argument("--upstream")
    parser.add_argument("--port", type=int, default=17891)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--read-timeout", type=float, default=600)
    parser.add_argument("--pid-file", default="/tmp/lagent_gate.pid")
    parser.add_argument("--log", default="/tmp/lagent_gate.log")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    def control(action):
        request = Request(
            f"http://127.0.0.1:{args.port}/__agent/{action}", data=b"", method="POST"
        )
        with urlopen(request, timeout=min(args.timeout, 2)) as response:
            return json.load(response)

    if args.action in {"status", "pause", "resume"}:
        print(json.dumps(control(args.action)))
    elif args.action == "stop":
        if Path(args.pid_file).exists():
            confirm_stopped(Path(args.pid_file), args.timeout)
            Path(args.pid_file).unlink(missing_ok=True)
        print(json.dumps({"stopped": True}))
    elif args.action == "start":
        if not args.upstream:
            parser.error("--upstream is required")
        pid = _read_pid(args.pid_file)
        if pid and live_members(pid):
            parser.error("gate process group already running")
        cmd = [
            sys.executable,
            "-m",
            __spec__.name,
            "serve",
            "--upstream",
            args.upstream,
            "--port",
            str(args.port),
            "--request-timeout",
            str(args.request_timeout),
            "--connect-timeout",
            str(args.connect_timeout),
            "--read-timeout",
            str(args.read_timeout),
        ]
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a") as log:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        try:
            Path(args.pid_file).write_text(str(proc.pid))
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("gate exited before ready; inspect gate log")
                try:
                    status = control("status")
                    if status.get("pid") == proc.pid:
                        print(json.dumps(status))
                        return
                except (OSError, ValueError):
                    pass
                time.sleep(0.1)
            raise TimeoutError("gate did not become ready")
        except BaseException:
            proc.terminate()
            proc.wait(timeout=args.timeout)
            Path(args.pid_file).unlink(missing_ok=True)
            raise
    else:
        if not args.upstream:
            parser.error("--upstream is required")
        gate = Gate(
            args.upstream,
            request_timeout=args.request_timeout,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
        )
        web.run_app(
            gate.app(),
            host="127.0.0.1",
            port=args.port,
            access_log=None,
            shutdown_timeout=1,
            handler_cancellation=True,
        )


if __name__ == "__main__":
    main()
