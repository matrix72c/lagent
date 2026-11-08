"""SDK failures must preserve diagnostics without becoming successful responses."""
import asyncio
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from lagent.adapters.claude_code_sdk import ClaudeCodeSDKAdapter
from lagent.schema import AgentMessage


@dataclass
class ResultMessage:
    result: str = 'finished'
    session_id: str = 'sdk-session'
    usage: object = None
    total_cost_usd: float = 0.0
    num_turns: int = 1
    is_error: bool = False
    stop_reason: str = 'end_turn'


@pytest.fixture
def sdk(monkeypatch):
    events = []

    async def query(**kwargs):
        for event in events:
            if isinstance(event, BaseException):
                raise event
            yield event

    module = SimpleNamespace(
        query=query,
        ClaudeAgentOptions=lambda **kw: SimpleNamespace(env={}, **kw),
        ResultMessage=ResultMessage,
        AssistantMessage=type('AssistantMessage', (), {}),
        SystemMessage=type('SystemMessage', (), {}),
        UserMessage=type('UserMessage', (), {}),
    )
    monkeypatch.setitem(sys.modules, 'claude_agent_sdk', module)
    return events


@pytest.mark.asyncio
async def test_success_result_remains_successful(sdk):
    sdk.append(ResultMessage())
    agent = ClaudeCodeSDKAdapter()
    result = await agent.forward(AgentMessage(sender='user', content='task'))
    assert result.content == 'finished'
    assert 'error' not in result.extra_info
    assert agent._session_id == 'sdk-session'
    assert agent._call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['API Error: 504 status code (no body)', ''])
async def test_terminal_sdk_error_is_structured_failure(sdk, text):
    sdk.append(ResultMessage(result=text, is_error=True))
    agent = ClaudeCodeSDKAdapter()
    result = await agent.forward(AgentMessage(sender='user', content='task'))
    assert 'error result' in result.extra_info['error']
    assert agent._sdk_trace[0]['is_error'] is True
    assert agent._sdk_trace[0]['result'] == text
    assert agent._session_id == 'sdk-session'
    assert agent._call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('partial', [False, True])
async def test_sdk_exception_is_not_swallowed(sdk, partial):
    if partial:
        sdk.append(ResultMessage(result='partial response'))
    sdk.append(RuntimeError('transport failed'))
    agent = ClaudeCodeSDKAdapter()
    result = await agent.forward(AgentMessage(sender='user', content='task'))
    assert result.extra_info['error'] == 'transport failed'
    assert len(agent._sdk_trace) == int(partial)
    assert agent._call_count == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_and_preserves_trace(sdk):
    sdk.extend([ResultMessage(), asyncio.CancelledError()])
    agent = ClaudeCodeSDKAdapter()
    with pytest.raises(asyncio.CancelledError):
        await agent.forward(AgentMessage(sender='user', content='task'))
    assert len(agent._sdk_trace) == 1
    assert agent._session_id == 'sdk-session'
    assert agent._call_count == 1
