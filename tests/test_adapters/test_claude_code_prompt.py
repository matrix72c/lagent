import copy
import json

import aiohttp
import pytest

from lagent.adapters.claude_code_prompt import (
    TASK_TOOLS_REMINDER,
    ClaudeCodePromptStabilizer,
    _strip_exact_system_paragraphs,
)
from lagent.adapters.proxy import SessionClient, _canonical_msg, _maximal_prefix_records
from lagent.adapters.request_processor import ProxyRequestContext, ProxyRequestProcessor


def _context(index=0):
    return ProxyRequestContext(
        session_id='session',
        request_index=index,
        provider='anthropic',
        path='v1/messages',
    )


def _request(**overrides):
    request = {
        'model': 'fake-model',
        'max_tokens': 32,
        'system': 'base system',
        'messages': [{'role': 'user', 'content': 'question'}],
        'tools': [{'name': 'Bash', 'input_schema': {'type': 'object'}}],
    }
    request.update(overrides)
    return request


def _tool_response(tool_id='toolu_1', name='Bash', timeout='60000'):
    return {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'fake-model',
        'content': [
            {'type': 'text', 'text': 'I will inspect it.\n\n'},
            {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {'timeout': timeout}},
        ],
        'stop_reason': 'tool_use',
        'usage': {'input_tokens': 10, 'output_tokens': 4},
    }


def test_strip_exact_system_paragraphs_removes_only_complete_matches():
    text = f'base\n\n{TASK_TOOLS_REMINDER}\n\n{TASK_TOOLS_REMINDER}'

    assert _strip_exact_system_paragraphs(text) == ('base', 2)
    assert _strip_exact_system_paragraphs(f'quoted: {TASK_TOOLS_REMINDER}') == (
        f'quoted: {TASK_TOOLS_REMINDER}',
        0,
    )


def test_reminder_is_removed_from_all_system_shapes_and_wrapped_user_blocks():
    wrapped = f'<system-reminder>\n{TASK_TOOLS_REMINDER}\n</system-reminder>'
    request = _request(
        system=[
            {'type': 'text', 'text': 'base', 'cache_control': {'type': 'ephemeral'}},
            {'type': 'text', 'text': '\n\n'},
            {'type': 'text', 'text': TASK_TOOLS_REMINDER},
        ],
        messages=[
            {'role': 'system', 'content': f'project\n\n{TASK_TOOLS_REMINDER}'},
            {'role': 'user', 'content': [{'type': 'text', 'text': wrapped}]},
            {'role': 'user', 'content': TASK_TOOLS_REMINDER},
        ],
    )
    stabilizer = ClaudeCodePromptStabilizer()

    processed = stabilizer.before_forward(request, _context())

    assert processed['system'] == [
        {'type': 'text', 'text': 'base', 'cache_control': {'type': 'ephemeral'}},
    ]
    assert processed['messages'] == [
        {'role': 'system', 'content': 'project'},
        {'role': 'user', 'content': TASK_TOOLS_REMINDER},
    ]
    assert stabilizer.get_stats()['reminders_removed'] == 3


def test_similar_reminder_is_not_removed():
    changed = TASK_TOOLS_REMINDER.replace('gentle reminder', 'helpful reminder')
    stabilizer = ClaudeCodePromptStabilizer()

    processed = stabilizer.before_forward(_request(system=changed), _context())

    assert processed['system'] == changed
    assert stabilizer.get_stats()['reminders_removed'] == 0


def test_tool_use_replay_restores_original_tokens_and_current_cache_control():
    stabilizer = ClaudeCodePromptStabilizer()
    stabilizer.after_response(_request(), _tool_response(), _context())
    replay = [
        {'type': 'text', 'text': 'I will inspect it.\n'},
        {
            'type': 'tool_use',
            'id': 'toolu_1',
            'name': 'Bash',
            'input': {'timeout': 60000},
            'cache_control': {'type': 'ephemeral'},
        },
    ]

    processed = stabilizer.before_forward(
        _request(messages=[{'role': 'assistant', 'content': replay}, {'role': 'user', 'content': 'continue'}]),
        _context(1),
    )

    restored = processed['messages'][0]['content']
    assert restored[0]['text'] == 'I will inspect it.\n\n'
    assert restored[1]['input'] == {'timeout': '60000'}
    assert restored[1]['cache_control'] == {'type': 'ephemeral'}
    assert stabilizer.get_stats()['assistant_replays_restored'] == 1


def test_plain_text_and_changed_tool_shape_are_not_restored():
    stabilizer = ClaudeCodePromptStabilizer()
    stabilizer.after_response(_request(), _tool_response(), _context())
    messages = [
        {'role': 'assistant', 'content': 'plain answer'},
        {
            'role': 'assistant',
            'content': [
                {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'timeout': 60000}},
                {'type': 'text', 'text': 'different order'},
            ],
        },
    ]

    processed = stabilizer.before_forward(_request(messages=copy.deepcopy(messages)), _context(1))

    assert processed['messages'] == messages
    stats = stabilizer.get_stats()
    assert stats['assistant_replays_restored'] == 0
    assert stats['assistant_replays_unanchored'] == 1
    assert stats['assistant_replays_ambiguous'] == 1


def test_reused_tool_id_with_different_name_is_ambiguous():
    stabilizer = ClaudeCodePromptStabilizer()
    stabilizer.after_response(_request(), _tool_response(name='Bash'), _context())
    stabilizer.after_response(_request(), _tool_response(name='Read'), _context(1))
    replay = _tool_response(name='Bash')['content']
    replay[1]['input'] = {'timeout': 1}

    processed = stabilizer.before_forward(
        _request(messages=[{'role': 'assistant', 'content': replay}]),
        _context(2),
    )

    assert processed['messages'][0]['content'][1]['input'] == {'timeout': 1}
    assert stabilizer.get_stats()['assistant_replays_ambiguous'] == 1


def test_material_head_changes_are_counted_but_not_frozen():
    stabilizer = ClaudeCodePromptStabilizer()
    first = stabilizer.before_forward(_request(), _context())
    changed = _request(model='other-model', system='new system', tools=[])

    second = stabilizer.before_forward(changed, _context(1))

    assert first['system'] == 'base system'
    assert second['model'] == 'other-model'
    assert second['system'] == 'new system'
    assert second['tools'] == []
    stats = stabilizer.get_stats()
    assert stats['model_changed'] == 1
    assert stats['system_changed'] == 1
    assert stats['tools_changed'] == 1


class _BrokenProcessor(ProxyRequestProcessor):
    def before_forward(self, request, context):
        del request, context
        raise RuntimeError('broken processor')


def test_session_client_processor_failure_is_fail_open_and_sticky():
    proxy = SessionClient(
        real_api_key='EMPTY',
        real_base_url='http://example.test/v1',
        session_id='failure',
        request_processor=_BrokenProcessor(),
    )
    request = _request(system=f'base\n\n{TASK_TOOLS_REMINDER}')

    processed = proxy._apply_request_processor(request, _context())

    assert processed == request
    assert proxy.get_request_processor_stats()['processor_errors'] == 1
    assert proxy.get_request_processor_stats()['processor_disabled'] is True


@pytest.mark.asyncio
async def test_forwarded_and_recorded_anthropic_request_share_stabilized_system(monkeypatch):
    captured = {}
    recorded_requests = []

    class Request:
        def __init__(self):
            self.headers = {}
            self.match_info = {'path': 'v1/messages'}
            self.method = 'POST'
            self.query_string = ''

        async def read(self):
            payload = _request(system=f'base system\n\n{TASK_TOOLS_REMINDER}')
            return json.dumps(payload).encode()

    class Response:
        status = 200

        def __init__(self):
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def read(self):
            return json.dumps(
                {
                    'id': 'msg_1',
                    'type': 'message',
                    'role': 'assistant',
                    'model': 'fake-model',
                    'content': [{'type': 'text', 'text': 'answer'}],
                    'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 10, 'output_tokens': 1},
                }
            ).encode()

    class ClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            captured.update(kwargs)
            return Response()

    monkeypatch.setattr(aiohttp, 'ClientSession', lambda: ClientSession())
    proxy = SessionClient(
        real_api_key='EMPTY',
        real_base_url='http://example.test/v1',
        session_id='recording',
        request_processor=ClaudeCodePromptStabilizer(),
    )

    def build_record(request_data, response_data):
        del response_data
        recorded_requests.append(copy.deepcopy(request_data))
        return ([{'role': 'system', 'content': request_data['system']}], None)

    monkeypatch.setattr(proxy, '_build_anthropic_record', build_record)

    await proxy._handle_request(Request())

    assert json.loads(captured['data'])['system'] == 'base system'
    assert recorded_requests[0]['system'] == 'base system'
    records = proxy.get_messages()
    assert records[0]['messages'][0] == {'role': 'system', 'content': 'base system'}


def _legacy_prefix_filter(records):
    keyed = [(record, tuple(_canonical_msg(message) for message in record.get('messages', []))) for record in records]
    filtered = []
    for i, (record, key) in enumerate(keyed):
        is_prefix = False
        for j, (other_record, other_key) in enumerate(keyed):
            if i == j or record.get('tools') != other_record.get('tools'):
                continue
            if len(key) == len(other_key) and key == other_key and i < j:
                is_prefix = True
                break
            if len(key) < len(other_key) and other_key[: len(key)] == key:
                is_prefix = True
                break
        if not is_prefix:
            filtered.append(record)
    return filtered


def test_prefix_trie_matches_legacy_semantics_for_branches_duplicates_and_tools():
    records = [
        {'name': 'short', 'messages': [{'role': 'user', 'content': 'a'}], 'tools': None},
        {
            'name': 'long',
            'messages': [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}],
            'tools': None,
        },
        {
            'name': 'duplicate-long',
            'messages': [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}],
            'tools': None,
        },
        {
            'name': 'branch',
            'messages': [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'c'}],
            'tools': None,
        },
        {'name': 'different-tools', 'messages': [{'role': 'user', 'content': 'a'}], 'tools': [{'name': 'Bash'}]},
    ]

    assert _maximal_prefix_records(records) == _legacy_prefix_filter(records)
    assert [record['name'] for record in _maximal_prefix_records(records)] == [
        'duplicate-long',
        'branch',
        'different-tools',
    ]


def test_prefix_trie_handles_histories_deeper_than_python_recursion_limit():
    short = {'messages': [{'role': 'user', 'content': '0'}], 'tools': None}
    long = {
        'messages': [{'role': 'user', 'content': str(index)} for index in range(1200)],
        'tools': None,
    }

    assert _maximal_prefix_records([short, long]) == [long]
