import copy
import json

import pytest

from lagent.serving.sandbox.client_cli import build_parser
from lagent.serving.sandbox.daemon import _load_config
from lagent.serving.sandbox.templates import TEMPLATES, build_config


@pytest.fixture
def runtime():
    return {
        "workspace": "/task",
        "model": "test-model",
        "base_url": "http://model/v1",
        "api_key": "test-key",
        "session_id": "opaque-session",
        "context_length": 131072,
        "root": "/runtime",
        "request_timeout": 40,
        "connect_timeout": 2,
        "read_timeout": 50,
    }


@pytest.mark.parametrize("kind", TEMPLATES)
def test_templates_accept_json_and_preserve_input(runtime, tmp_path, kind):
    original = copy.deepcopy(runtime)
    config = build_config(kind, runtime)
    assert config["type"].startswith("lagent.")
    assert runtime == original
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime))
    assert _load_config(str(path), "agent", kind) == config
    args = build_parser().parse_args(
        [
            "start-agent-daemon",
            "--template",
            kind,
            "--config",
            str(path),
            "--pid-file",
            "/tmp/pid",
            "--log",
            "/tmp/log",
        ]
    )
    assert args.template == kind
    if kind != "interndp":
        assert config["proxy"]["session_id"] == "opaque-session"
        assert config["proxy"]["stream_timeout"].total is None
        assert config["proxy"]["forward_timeout"].total == 40


def test_nested_options_and_unlimited_turns(runtime, monkeypatch):
    monkeypatch.setenv("HOME", "/role-home")
    runtime["options"] = {
        "max_turns": None,
        "extra_options": {"env": {"CUSTOM": "yes"}},
    }
    config = build_config("claude_code", runtime)
    assert config["max_turns"] is None
    assert config["extra_options"]["env"]["HOME"] == "/role-home"
    assert config["extra_options"]["env"]["CUSTOM"] == "yes"
    assert config["extra_options"]["cli_path"].startswith("/runtime/node/")
    assert (
        build_config("opencode", {**runtime, "options": {}})["binary"]
        == "/runtime/node/node-v22.14.0/bin/opencode"
    )


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("interndp", {"max_turn": 9}),
        ("terminus", {"max_episodes": None, "reasoning_effort": "high"}),
        ("mini_swe_agent", {"step_limit": 6, "command_timeout": 7}),
        ("openhands", {"max_iterations": 8}),
        (
            "opencode",
            {
                "agent": "build",
                "provider": "custom",
                "dangerously_skip_permissions": False,
            },
        ),
        ("openclaw", {"thinking": "high", "agent_id": "custom", "json_output": False}),
    ],
)
def test_adapter_options(runtime, kind, options):
    config = build_config(kind, {**runtime, "options": options})
    assert all(config[key] == value for key, value in options.items())


def test_validation(runtime, tmp_path):
    with pytest.raises(ValueError):
        build_config("unknown", runtime)
    with pytest.raises(ValueError):
        build_config("claude_code", {**runtime, "max_tokens": 131072})
    with pytest.raises(ValueError):
        _load_config("/not/executed.py", "agent", "opencode")
