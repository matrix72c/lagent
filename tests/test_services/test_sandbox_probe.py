import hashlib
import json
import platform
from pathlib import Path

import pytest

from lagent.serving.sandbox import probe


def test_mountinfo_uses_deepest_mount_and_unescapes():
    info = "1 0 0:1 / / rw - ext4 /dev/root rw\n2 1 0:2 / /runtime ro - ext4 disk rw\n"
    assert probe.readonly_mount(Path("/runtime/v1"), info)
    assert not probe.readonly_mount(Path("/other"), info)
    info += "3 2 0:2 / /runtime/v1 rw - ext4 disk rw\n"
    assert not probe.readonly_mount(Path("/runtime/v1"), info)
    assert probe.readonly_mount(Path("/a b"), "1 0 0:1 / /a\\040b ro - ext4 disk rw")


def entry():
    return {"type": "file", "sha256": "unused", "size": 7, "mode": 493}


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture")
    path.chmod(0o755)


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    root, state = tmp_path / "runtime", tmp_path / "state"
    root.mkdir()
    state.mkdir()
    executable(root / "python/bin/python3")
    executable(root / "node/bin/node")
    executable(root / "node/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe")
    (root / "lagent/lagent").mkdir(parents=True)
    libc_name, libc_version = platform.libc_ver()
    target = {
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": {"name": libc_name, "minimum": libc_version},
    }
    manifests = {
        "python": {"platform": target, "files": {"bin/python3": entry()}},
        "node": {
            "platform": dict(target),
            "files": {
                "bin/node": entry(),
                "lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe": entry(),
            },
        },
        "lagent": {"lagent_sha": "a" * 40, "files": {}},
    }
    for name, manifest in manifests.items():
        manifest.update(schema_version=2, layer=name)
    monkeypatch.setattr(probe, "readonly_mount", lambda root, info: True)
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: None)
    return root, state, manifests


def run(artifact, kind="interndp"):
    root, state, manifests = artifact
    digests = {}
    for name, manifest in manifests.items():
        raw = json.dumps(manifest).encode()
        (root / name / "manifest.json").write_bytes(raw)
        digests[name] = hashlib.sha256(raw).hexdigest()
    return probe.check(root, digests, kind, state)


@pytest.mark.parametrize("kind", ["interndp", "claude_code"])
def test_probe_success_without_full_payload_hash(artifact, kind):
    result = run(artifact, kind)
    assert result["ok"] and result["lagent_sha"] == "a" * 40
    assert set(result["layers"]) == {"node", "python", "lagent"}


def test_probe_creates_missing_state_directory(artifact):
    root, state, manifests = artifact
    state.rmdir()
    state = state / "nested"
    assert run((root, state, manifests))["ok"]
    assert state.is_dir() and state.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("operation", ["create", "write"])
def test_state_permission_failure_remains_fatal(artifact, monkeypatch, operation):
    root, state, manifests = artifact

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    if operation == "create":
        state.rmdir()
        monkeypatch.setattr(Path, "mkdir", denied)
    else:
        monkeypatch.setattr(probe.tempfile, "TemporaryFile", denied)
    with pytest.raises(PermissionError):
        run((root, state, manifests))


def test_probe_does_not_create_state_before_readonly_validation(artifact, monkeypatch):
    root, state, manifests = artifact
    state.rmdir()
    monkeypatch.setattr(probe, "readonly_mount", lambda root, info: False)
    with pytest.raises(probe.ProbeError, match="read-only"):
        run((root, state, manifests))
    assert not state.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "schema",
        "layer",
        "machine",
        "node-libc",
        "no-platform",
        "missing",
        "claude",
        "mode",
        "rw",
        "import",
        "state",
        "digest",
    ],
)
def test_probe_fail_closed(artifact, monkeypatch, failure):
    root, state, manifests = artifact
    kind = "interndp"
    if failure == "schema":
        manifests["lagent"]["schema_version"] = 1
    elif failure == "layer":
        manifests["python"]["layer"] = "node"
    elif failure == "machine":
        manifests["python"]["platform"]["machine"] = "other"
    elif failure == "node-libc":
        manifests["node"]["platform"]["libc"] = {"name": platform.libc_ver()[0], "minimum": "999.0"}
    elif failure == "no-platform":
        del manifests["node"]["platform"]
    elif failure == "claude":
        kind = "claude_code"
        (root / "node/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe").unlink()
    elif failure == "missing":
        (root / "python/bin/python3").unlink()
    elif failure == "mode":
        (root / "python/bin/python3").chmod(0o644)
    elif failure == "rw":
        monkeypatch.setattr(probe, "readonly_mount", lambda root, info: False)
    elif failure == "import":

        def fail(name):
            raise ImportError("sensitive diagnostic not emitted")

        monkeypatch.setattr(probe.importlib, "import_module", fail)
    elif failure == "state":
        state = root / "state"
        state.mkdir()
    elif failure == "digest":
        for name in probe.LAYERS:
            (root / name / "manifest.json").write_text("{}")
        with pytest.raises(probe.ProbeError, match="digest"):
            probe.check(root, dict.fromkeys(probe.LAYERS, "0" * 64), "interndp", state)
        return
    with pytest.raises(probe.ProbeError):
        run((root, state, manifests), kind)
