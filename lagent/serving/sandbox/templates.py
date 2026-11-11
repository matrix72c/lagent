"""Versioned sandbox agent templates, selected with JSON rather than uploaded code.

Required runtime keys: workspace, model, base_url, api_key. Optional keys:
session_id, agent_timeout, request_timeout, connect_timeout, read_timeout,
max_tokens, context_length, max_turns, extra_body, http_proxy, options and
proxy_options. ``options`` recursively overrides adapter defaults; paths and
vendor-specific settings (e.g. binary, extra_options.cli_path) belong there.
No trainer environment variables are read and no optional SDK is imported here.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from aiohttp import ClientTimeout

TEMPLATES = (
    "claude_code",
    "interndp",
    "mini_swe_agent",
    "openclaw",
    "opencode",
    "openhands",
    "terminus",
)


def _merge(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def build_config(template: str, runtime: dict) -> dict:
    """Build a daemon agent config without constructing the agent or mutating input."""
    if template not in TEMPLATES:
        raise ValueError(
            f"Unknown sandbox template {template!r}; choose from {TEMPLATES}"
        )
    workspace, model = runtime["workspace"], runtime["model"]
    base_url, api_key = runtime["base_url"], runtime["api_key"]
    timeout = float(runtime.get("agent_timeout", 3600))
    request_timeout = float(runtime.get("request_timeout", 600))
    connect_timeout = float(runtime.get("connect_timeout", 30))
    read_timeout = float(runtime.get("read_timeout", 600))
    max_tokens = int(runtime.get("max_tokens", 65536))
    proxy = {
        "type": "lagent.adapters.proxy.SessionClient",
        "real_base_url": base_url,
        "real_api_key": api_key,
        "session_id": runtime.get("session_id"),
        "forward_timeout": ClientTimeout(
            total=request_timeout, connect=connect_timeout, sock_read=read_timeout
        ),
        "stream_timeout": ClientTimeout(
            total=None, connect=connect_timeout, sock_read=read_timeout
        ),
        "extra_body": _merge(
            {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_tokens": max_tokens},
            runtime.get("extra_body", {}),
        ),
    }
    if template == "claude_code":
        context_length = int(runtime["context_length"])
        if max_tokens >= context_length:
            raise ValueError("max_tokens must be smaller than context_length")
        proxy["extra_body"].pop("top_k", None)
        home = str(Path.home())
        config = {
            "type": "lagent.adapters.claude_code_sdk.ClaudeCodeSDKAdapter",
            "name": "claude_coder",
            "description": "Claude Code SDK sandbox agent",
            "model": model,
            "max_turns": runtime.get("max_turns"),
            "permission_mode": "bypassPermissions",
            "cwd": workspace,
            "timeout": timeout,
            "extra_options": {
                "env": {
                    "HOME": home,
                    "CLAUDE_CONFIG_DIR": str(Path(home) / ".claude"),
                    "IS_SANDBOX": "1",
                    "CLAUDE_CODE_SANDBOXED": "1",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "ANTHROPIC_VERSION": "2023-06-01",
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context_length),
                    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(
                        min(1_000_000, max(100_000, context_length))
                    ),
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_tokens),
                }
            },
        }
    elif template == "interndp":
        config = {
            "type": "lagent.agents.fc_agent.FunctionCallAgent",
            "policy_agent": {
                "type": "lagent.agents.AsyncAgent",
                "llm": {
                    "type": "lagent.llms.model.AsyncAPIClient",
                    "model": {"model": model, "base_url": base_url, "api_key": api_key},
                    "sample_params": {"temperature": 1, "top_p": 1.0, "top_k": 0},
                    "timeout": request_timeout,
                    "max_retry": 2,
                    "sleep_interval": 5,
                    "extra_body": {"spaces_between_special_tokens": False, "top_k": 0},
                    "session_id": runtime.get("session_id"),
                },
            },
            "env_agent": {
                "type": "lagent.agents.fc_agent.EnvAgent",
                "actions": [{"type": "lagent.actions.tmux_action.TerminalExecute"}],
                "max_tool_response_length": 4096,
                "tool_response_truncate_side": "left",
            },
            "max_turn": runtime.get("max_turns", 30),
        }
    elif template == "mini_swe_agent":
        config = {
            "type": "lagent.adapters.mini_swe_agent.MiniSWEAgentAdapter",
            "name": "mini_swe_agent",
            "description": "mini-SWE-agent (sandbox bash tool)",
            "model": model,
            "cwd": workspace,
            "step_limit": 100,
            "command_timeout": 300,
            "timeout": timeout,
            "trajectory_path": "/tmp/mini.traj.json",
        }
    elif template == "openclaw":
        config = {
            "type": "lagent.adapters.openclaw.OpenClawAdapter",
            "model": model,
            "thinking": "medium",
            "agent_id": "main",
            "json_output": True,
            "openclaw_home": "~/.openclaw",
            "working_dir": workspace,
            "timeout": int(timeout),
            "env_vars": {},
        }
    elif template == "opencode":
        config = {
            "type": "lagent.adapters.opencode_cli.OpenCodeCLIAdapter",
            "name": "opencode_coder",
            "description": "OpenCode CLI sandbox agent",
            "model": model,
            "provider": "openai-compatible",
            "agent": "build",
            "cwd": workspace,
            "binary": "opencode",
            "timeout": int(timeout),
            "dangerously_skip_permissions": True,
            "opencode_home": "/tmp/opencode-home",
            "system_prompt": (
                "You are a terminal-bench agent operating inside a Linux sandbox "
                f"with full shell access at {workspace}. The task description is in "
                f"{workspace}/instruction.md and any input data files are already in "
                f"{workspace}. Use shell and file-editing tools to inspect files, "
                "run commands, edit code or data, and produce the requested outputs. When you "
                "believe the task is complete, summarize what you did and stop calling tools."
            ),
        }
    elif template == "openhands":
        config = {
            "type": "lagent.adapters.openhands.OpenHandsAdapter",
            "name": "openhands",
            "description": "OpenHands SDK sandbox agent",
            "model": model if "/" in model else f"openai/{model}",
            "working_dir": workspace,
            "max_iterations": 500,
            "timeout": timeout,
        }
    else:
        proxy.update(port=0, http_proxy=runtime.get("http_proxy"))
        config = {
            "type": "lagent.adapters.terminus2.Terminus2Adapter",
            "model": model,
            "max_episodes": None,
            "parser_name": "json",
            "temperature": 1.0,
            "reasoning_effort": "max",
            "timeout": timeout,
            "working_dir": workspace,
            "enable_summarize": True,
            "record_terminal_session": False,
            "suppress_max_turns_warning": True,
        }
    root = runtime.get("root")
    if root:
        node = Path(root) / "node"
        if template == "claude_code":
            config["extra_options"]["cli_path"] = str(
                node / "lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe"
            )
        elif template in {"opencode", "openclaw"}:
            config["binary"] = str(node / "bin" / template)
    if template != "interndp":
        config["proxy"] = _merge(proxy, runtime.get("proxy_options", {}))
    return _merge(config, runtime.get("options", {}))
