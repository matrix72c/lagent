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


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    root, state = tmp_path / "runtime", tmp_path / "state"
    root.mkdir()
    state.mkdir()
    executable = root / "python/bin/python3"
    executable.parent.mkdir(parents=True)
    executable.write_text("fixture")
    executable.chmod(0o755)
    (root / "lagent/lagent").mkdir(parents=True)
    libc_name, libc_version = platform.libc_ver()
    manifest = {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": {"name": libc_name, "minimum": libc_version},
        },
        "validated_imports": [],
        "files": {
            "python/bin/python3": {
                "type": "file",
                "sha256": "unused",
                "size": 7,
                "mode": 493,
            }
        },
    }
    monkeypatch.setattr(probe, "readonly_mount", lambda root, info: True)
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: None)
    return root, state, manifest


def run(artifact):
    root, state, manifest = artifact
    raw = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(raw)
    return probe.check(root, hashlib.sha256(raw).hexdigest(), "interndp", state)


def test_probe_success_without_full_payload_hash(artifact):
    assert run(artifact)["ok"]


def test_probe_creates_missing_state_directory(artifact):
    root, state, manifest = artifact
    state.rmdir()
    state = state / "nested"
    assert run((root, state, manifest))["ok"]
    assert state.is_dir() and state.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("operation", ["create", "write"])
def test_state_permission_failure_remains_fatal(artifact, monkeypatch, operation):
    root, state, manifest = artifact

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    if operation == "create":
        state.rmdir()
        monkeypatch.setattr(Path, "mkdir", denied)
    else:
        monkeypatch.setattr(probe.tempfile, "TemporaryFile", denied)
    with pytest.raises(PermissionError):
        run((root, state, manifest))


def test_probe_does_not_create_state_before_readonly_validation(artifact, monkeypatch):
    root, state, manifest = artifact
    state.rmdir()
    monkeypatch.setattr(probe, "readonly_mount", lambda root, info: False)
    with pytest.raises(probe.ProbeError, match="read-only"):
        run((root, state, manifest))
    assert not state.exists()


@pytest.mark.parametrize(
    "failure",
    ["schema", "machine", "libc", "missing", "mode", "rw", "import", "state", "digest"],
)
def test_probe_fail_closed(artifact, monkeypatch, failure):
    root, state, manifest = artifact
    if failure == "schema":
        manifest["schema_version"] = 2
    elif failure == "machine":
        manifest["platform"]["machine"] = "other"
    elif failure == "libc":
        manifest["platform"]["libc"]["minimum"] = "999.0"
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
        (root / "manifest.json").write_text("{}")
        with pytest.raises(probe.ProbeError, match="digest"):
            probe.check(root, "0" * 64, "interndp", state)
        return
    with pytest.raises(probe.ProbeError):
        run((root, state, manifest))
