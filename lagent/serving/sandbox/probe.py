"""Cheap, fail-closed runtime preflight, run as the actual sandbox agent user.

The runtime is three read-only layers mounted side by side under one root:
node/ (Node and CLI tools), python/ (relocatable Python with locked
dependencies) and lagent/ (the lagent package). Checks each layer's manifest
identity and compatibility, not every payload hash. The
publisher is responsible for full artifact verification before publication.
Diagnostics intentionally omit exception text, environment and credentials.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import re
import tempfile
from pathlib import Path

from .templates import TEMPLATES


class ProbeError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ProbeError(message)


def readonly_mount(root: Path, mountinfo: str) -> bool:
    """Check the most-specific mount, including a nested rw bind under a ro root."""
    matches = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        if root == mount or mount in root.parents:
            matches.append((len(mount.parts), "ro" in fields[5].split(",")))
    return bool(matches) and max(matches, key=lambda item: item[0])[1]


def version(value: str) -> tuple[int, ...]:
    require(bool(re.fullmatch(r"\d+(\.\d+)*", value)), "invalid libc version")
    return tuple(int(part) for part in value.split("."))


LAYERS = ("node", "python", "lagent")
NODE_KINDS = {"claude_code", "opencode", "openclaw"}


def layer_manifest(root: Path, name: str, digest: str, mountinfo: str) -> dict:
    """One read-only layer at `<root>/<name>` whose canonical manifest hashes to `digest`."""
    layer = root / name
    raw = (layer / "manifest.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, f"{name} manifest digest mismatch")
    manifest = json.loads(raw)
    require(
        manifest.get("schema_version") == 2 and manifest.get("layer") == name,
        f"unsupported {name} manifest schema",
    )
    require(readonly_mount(layer, mountinfo), f"{name} layer must be mounted read-only")
    target = manifest.get("platform")
    require(target is not None or name == "lagent", f"{name} manifest lacks a platform")
    if target is not None:
        require(target["system"] == platform.system(), "platform system mismatch")
        require(target["machine"] == platform.machine(), "platform machine mismatch")
        libc_name, libc_version = platform.libc_ver()
        require(target["libc"]["name"] == libc_name, "libc family mismatch")
        require(
            version(libc_version) >= version(target["libc"]["minimum"]),
            "libc version below minimum",
        )
    files = manifest["files"]
    require(isinstance(files, dict), "invalid manifest files")
    for relative, entry in files.items():
        path = Path(relative)
        require(
            not path.is_absolute() and ".." not in path.parts, "unsafe manifest path"
        )
        require(
            isinstance(entry, dict)
            and {"type", "sha256", "size", "mode"} <= entry.keys(),
            "invalid manifest file entry",
        )
    return manifest


def check(root: Path, digests: dict, kind: str, state_dir: Path) -> dict:
    """`root` holds the sibling layers node/, python/ and lagent/, digests keyed by layer."""
    root, state_dir = root.resolve(), state_dir.resolve()
    require(
        root != state_dir
        and root not in state_dir.parents
        and state_dir not in root.parents,
        "runtime root and writable state directory must be separate",
    )
    require(set(digests) == set(LAYERS), "a digest is required for every layer")
    mountinfo = Path("/proc/self/mountinfo").read_text()
    manifests = {
        name: layer_manifest(root, name, digests[name], mountinfo) for name in LAYERS
    }
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryFile(dir=state_dir) as output:
        output.write(b"probe")
        output.flush()

    executables = [("python", "bin/python3")]
    if kind in NODE_KINDS:
        executables.append(("node", "bin/node"))
    if kind == "claude_code":
        executables.append(
            ("node", "lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe")
        )
    elif kind in {"opencode", "openclaw"}:
        executables.append(("node", f"bin/{kind}"))
    for name, relative in executables:
        layer = root / name
        path = layer / relative
        require(
            relative in manifests[name]["files"],
            f"manifest missing executable: {name}/{relative}",
        )
        require(
            path.is_file() and os.access(path, os.X_OK),
            f"missing executable: {name}/{relative}",
        )
        require(
            path.resolve().is_relative_to(layer),
            f"executable escapes its layer: {name}/{relative}",
        )
    require((root / "lagent/lagent").is_dir(), "lagent source is missing")
    dependencies = {
        "claude_code": ["claude_agent_sdk"],
        "mini_swe_agent": ["minisweagent"],
        "openhands": ["openhands.sdk", "openhands.tools"],
        "terminus": ["harbor.agents.terminus_2.terminus_2"],
        "interndp": ["lagent.actions.tmux_action"],
        "openclaw": [],
        "opencode": [],
    }
    imports = [
        "aiohttp",
        "lagent.serving.sandbox.daemon",
        "lagent.adapters.proxy",
        *dependencies[kind],
    ]
    # Never import strings supplied by a manifest. Probe the required imports
    # for this role, suppressing third-party diagnostics.
    for name in imports:
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                importlib.import_module(name)
        except Exception:  # noqa: BLE001 - redact arbitrary dependency initialization failures
            raise ProbeError(f"required import failed: {name}") from None
    return {
        "ok": True,
        "kind": kind,
        "layers": dict(digests),
        "lagent_sha": manifests["lagent"].get("lagent_sha"),
        "schema_version": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    for name in LAYERS:
        parser.add_argument(f"--{name}", required=True, help=f"{name} layer digest")
    parser.add_argument("--kind", choices=TEMPLATES, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        digests = {name: getattr(args, name) for name in LAYERS}
        result = check(args.root, digests, args.kind, args.state_dir)
    except ProbeError as exc:
        result = {"ok": False, "error": str(exc)[:240]}
    except Exception as exc:  # noqa: BLE001 - CLI emits only bounded, non-sensitive diagnostics
        result = {
            "ok": False,
            "error": f"runtime preflight failed ({type(exc).__name__})",
        }
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
