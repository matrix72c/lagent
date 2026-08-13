import json
import logging
import os

import aiohttp
import pytest

from lagent.adapters.proxy import SessionClient


def _proxy(**kwargs):
    return SessionClient(
        real_api_key='EMPTY',
        real_base_url='http://example.test/v1',
        session_id='timeout-test',
        **kwargs,
    )


async def _capture_forwarded_request(monkeypatch, proxy, payload, path='v1/messages'):
    captured = {}

    class Request:
        headers = {}
        match_info = {'path': path}
        method = 'POST'
        query_string = ''

        async def read(self):
            return json.dumps(payload).encode('utf-8')

    class ClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('stop before sending the request')

    monkeypatch.setattr(aiohttp, 'ClientSession', lambda **kwargs: ClientSession())
    with pytest.raises(RuntimeError, match='stop before sending the request'):
        await proxy._handle_request(Request())
    return json.loads(captured['data'])


def test_client_timeout_accepts_aiohttp_object():
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=3600)
    proxy = _proxy(client_timeout=timeout)

    assert proxy.client_timeout is timeout


def test_client_timeout_builds_from_serializable_dict():
    proxy = _proxy(
        client_timeout={
            'total': None,
            'sock_connect': 30,
            'sock_read': 3600,
        }
    )

    assert isinstance(proxy.client_timeout, aiohttp.ClientTimeout)
    assert proxy.client_timeout.total is None
    assert proxy.client_timeout.sock_connect == 30
    assert proxy.client_timeout.sock_read == 3600


@pytest.mark.asyncio
@pytest.mark.parametrize('client_timeout', [None, {'sock_read': 3600}])
async def test_client_timeout_is_passed_to_client_session(monkeypatch, client_timeout):
    proxy = _proxy(client_timeout=client_timeout)
    client_kwargs = None

    def create_client(**kwargs):
        nonlocal client_kwargs
        client_kwargs = kwargs
        raise RuntimeError('stop before sending the request')

    class Request:
        headers = {}
        match_info = {'path': 'v1/chat/completions'}
        method = 'POST'
        query_string = ''

        async def read(self):
            return b'{}'

    monkeypatch.setattr(aiohttp, 'ClientSession', create_client)
    with pytest.raises(RuntimeError, match='stop before sending the request'):
        await proxy._handle_request(Request())

    expected = {} if client_timeout is None else {'timeout': proxy.client_timeout}
    assert client_kwargs == expected


@pytest.mark.asyncio
async def test_anthropic_system_messages_are_preserved_by_default(monkeypatch):
    proxy = _proxy()
    payload = {
        'model': 'fake-model',
        'max_tokens': 32,
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'system', 'content': 'project instruction'},
        ],
    }

    forwarded = await _capture_forwarded_request(monkeypatch, proxy, payload)

    assert forwarded['messages'] == payload['messages']
    assert 'system' not in forwarded


@pytest.mark.asyncio
async def test_anthropic_request_without_system_messages_is_unchanged(monkeypatch):
    proxy = _proxy(merge_anthropic_system_messages=True)
    payload = {
        'model': 'fake-model',
        'max_tokens': 32,
        'system': 'base instruction',
        'messages': [{'role': 'user', 'content': 'question'}],
    }

    forwarded = await _capture_forwarded_request(monkeypatch, proxy, payload)

    assert forwarded['system'] == payload['system']
    assert forwarded['messages'] == payload['messages']


@pytest.mark.asyncio
async def test_anthropic_system_messages_are_merged_before_forwarding_and_recording(monkeypatch):
    proxy = _proxy(merge_anthropic_system_messages=True)
    payload = {
        'model': 'fake-model',
        'max_tokens': 32,
        'system': [
            {
                'type': 'text',
                'text': 'base instruction',
                'cache_control': {'type': 'ephemeral'},
            }
        ],
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'system', 'content': 'project instruction'},
            {'role': 'assistant', 'content': 'working'},
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'later instruction',
                        'cache_control': {'type': 'ephemeral'},
                    }
                ],
            },
            {'role': 'user', 'content': 'continue'},
        ],
    }

    forwarded = await _capture_forwarded_request(monkeypatch, proxy, payload)

    assert forwarded['system'] == [
        {
            'type': 'text',
            'text': 'base instruction',
            'cache_control': {'type': 'ephemeral'},
        },
        {'type': 'text', 'text': '\n\n'},
        {'type': 'text', 'text': 'project instruction'},
        {'type': 'text', 'text': '\n\n'},
        {
            'type': 'text',
            'text': 'later instruction',
            'cache_control': {'type': 'ephemeral'},
        },
    ]
    assert forwarded['messages'] == [
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': 'working'},
        {'role': 'user', 'content': 'continue'},
    ]

    record = proxy._build_anthropic_record(
        forwarded,
        {'content': [{'type': 'text', 'text': 'answer'}]},
    )
    assert record is not None
    messages, _ = record
    assert messages[0] == {
        'role': 'system',
        'content': 'base instruction\n\nproject instruction\n\nlater instruction',
    }
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'user', 'assistant']


@pytest.mark.asyncio
async def test_anthropic_batch_request_without_top_level_messages_is_unchanged(monkeypatch):
    proxy = _proxy(merge_anthropic_system_messages=True)
    payload = {
        'requests': [
            {
                'custom_id': 'request-1',
                'params': {
                    'model': 'fake-model',
                    'max_tokens': 32,
                    'messages': [{'role': 'user', 'content': 'question'}],
                },
            }
        ]
    }

    forwarded = await _capture_forwarded_request(monkeypatch, proxy, payload, path='v1/messages/batches')

    assert forwarded['requests'] == payload['requests']


def test_get_messages_normalizes_openclaw_tool_call_id_underscore_loss():
    proxy = SessionClient(
        real_api_key="EMPTY",
        real_base_url="http://example.test/v1",
        session_id="openclaw_id_replay",
    )
    short = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a45f453d817e4c0cbc5ec29d", "content": "/app"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": None,
    }
    long = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "calla45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "calla45f453d817e4c0cbc5ec29d", "content": "/app"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ],
        "tools": None,
    }
    proxy._records[proxy.session_id] = [short, long]

    assert proxy.get_messages() == [long]


def test_get_messages_keeps_distinct_tool_call_ids_distinct():
    proxy = SessionClient(
        real_api_key="EMPTY",
        real_base_url="http://example.test/v1",
        session_id="distinct_tool_ids",
    )
    first = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_a45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
        ],
        "tools": None,
    }
    second = {
        "messages": [
            {"role": "user", "content": "write files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_b45f453d817e4c0cbc5ec29d",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    }
                ],
            },
            {"role": "user", "content": "next"},
        ],
        "tools": None,
    }
    proxy._records[proxy.session_id] = [first, second]

    assert proxy.get_messages() == [first, second]


@pytest.mark.asyncio
async def test_session_client_openai():
    # 1. Start the proxy
    proxy = SessionClient(
        real_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),  # Provide a valid OpenAI key if needed
        real_base_url="http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
        session_id="test_session_123",
        port=0,  # Auto-assign port
    )
    await proxy.start()

    # 2. Simulate an Agent sending an OpenAI-format request to the proxy
    dummy_payload = {
        "model": "agentic_rl_qwen35a3b_service",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        try:
            # Send to proxy URL, the proxy forwards it to real_base_url
            async with session.post(
                f"{proxy.url}/chat/completions",
                json=dummy_payload,
                headers={"Authorization": "Bearer sk-proxy-test"},
            ) as resp:
                assert resp.status in (200, 401, 502, 404), f"Unexpected status: {resp.status}"
                # Result won't be evaluated strictly without real endpoints, but we ensure connection proxy works
        except Exception as e:
            logging.warning(f"Request failed (is the target server running?): {e}")

    # 3. Check what the proxy recorded
    messages = proxy.get_messages()
    print(messages)
    assert isinstance(messages, list)

    # Clean up
    proxy.release_trace()
    await proxy.stop()


@pytest.mark.asyncio
async def test_session_client_claude():
    proxy = SessionClient(
        real_api_key=os.getenv("ANTHROPIC_AUTH_TOKEN", "EMPTY"),  # Provide a valid Claude key if needed
        real_base_url=os.getenv('ANTHROPIC_BASE_URL', "https://api.anthropic.com"),  # The Claude v1 proxy target
        session_id="test_claude_123",
        http_proxy=os.getenv("HTTP_PROXY"),
        port=0,
    )
    await proxy.start()

    # Simulate sending Anthropic schema request
    dummy_payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "system": "you're a helpful assistant",
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }

    async with aiohttp.ClientSession() as session:
        try:
            # Assuming the external agent posts to /v1/messages since we trigger Anthropic handling by endpoint now
            async with session.post(
                f"{proxy.url}/v1/messages",
                json=dummy_payload,
                headers={"Authorization": "Bearer sk-proxy-test", "x-api-key": "sk-proxy-test"},
            ) as resp:
                assert resp.status in (200, 401, 502, 404, 400), f"Unexpected status: {resp.status}"

                if dummy_payload["stream"]:
                    async for _ in resp.content.iter_any():
                        pass
                else:
                    await resp.json()
        except Exception as e:
            logging.warning(f"Request failed: {e}")

    # Check proxy records
    messages = proxy.get_messages()
    print(messages)
    assert isinstance(messages, list)

    proxy.release_trace()
    await proxy.stop()
