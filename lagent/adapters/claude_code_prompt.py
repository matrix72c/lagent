"""Cache-safe prompt stabilization for Claude Code Anthropic requests."""

import copy
import hashlib
import json
import re
from typing import Any

from .request_processor import ProxyRequestContext, ProxyRequestProcessor

TASK_TOOLS_REMINDER = (
    "The task tools haven't been used recently. If you're working on tasks that would benefit from tracking "
    "progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress "
    "when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use "
    "these if relevant to the current work. This is just a gentle reminder - ignore if not applicable."
)
_WRAPPED_TASK_TOOLS_REMINDERS = {
    f'<system-reminder>{TASK_TOOLS_REMINDER}</system-reminder>',
    f'<system-reminder>\n{TASK_TOOLS_REMINDER}\n</system-reminder>',
}
_SYSTEM_TASK_TOOLS_REMINDERS = {
    TASK_TOOLS_REMINDER,
    f'{TASK_TOOLS_REMINDER}\n',
    *_WRAPPED_TASK_TOOLS_REMINDERS,
}


def _stable_json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _strip_exact_system_paragraphs(text: str) -> tuple[str, int]:
    """Remove only complete, allowlisted reminder paragraphs from system text."""
    parts = re.split(r'(\n{2,})', text)
    paragraphs = parts[::2]
    separators = parts[1::2]
    kept_indices = [
        index for index, paragraph in enumerate(paragraphs) if paragraph not in _SYSTEM_TASK_TOOLS_REMINDERS
    ]
    removed = len(paragraphs) - len(kept_indices)
    if not removed:
        return text, 0
    if not kept_indices:
        return '', removed

    output = [paragraphs[kept_indices[0]]]
    previous = kept_indices[0]
    for index in kept_indices[1:]:
        # When exact reminder paragraphs disappear from the middle, retain the
        # first original paragraph separator. This preserves neighboring text
        # while preventing Claude Code's repeated "\n\n\n" reminder joins from
        # accumulating blank lines on every request.
        output.extend((separators[previous], paragraphs[index]))
        previous = index
    return ''.join(output), removed


def _strip_system_content(content: Any) -> tuple[Any, int]:
    if isinstance(content, str):
        return _strip_exact_system_paragraphs(content)
    if not isinstance(content, list):
        return content, 0

    output = []
    removed = 0
    drop_next_separator = False
    for block in content:
        if not (isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str)):
            output.append(block)
            drop_next_separator = False
            continue
        if drop_next_separator and block['text'] == '\n\n':
            drop_next_separator = False
            continue
        drop_next_separator = False
        text, count = _strip_exact_system_paragraphs(block['text'])
        removed += count
        if text or not count:
            updated = copy.deepcopy(block)
            updated['text'] = text
            output.append(updated)
        elif (
            output
            and isinstance(output[-1], dict)
            and output[-1].get('type') == 'text'
            and output[-1].get('text') == '\n\n'
        ):
            output.pop()
        else:
            drop_next_separator = True
    return output, removed


def _strip_wrapped_content(content: Any) -> tuple[Any, int]:
    """Remove wrapped reminders outside system only when they occupy a whole block."""
    if isinstance(content, str):
        if content in _WRAPPED_TASK_TOOLS_REMINDERS:
            return '', 1
        return content, 0
    if not isinstance(content, list):
        return content, 0

    output = []
    removed = 0
    for block in content:
        if (
            isinstance(block, dict)
            and block.get('type') == 'text'
            and block.get('text') in _WRAPPED_TASK_TOOLS_REMINDERS
        ):
            removed += 1
        else:
            output.append(block)
    return output, removed


def _assistant_identity(content: Any) -> tuple[tuple[str, str], ...] | None:
    if not isinstance(content, list):
        return None
    identity = []
    seen_ids = set()
    for block in content:
        if not (isinstance(block, dict) and block.get('type') == 'tool_use'):
            continue
        tool_id = block.get('id')
        name = block.get('name')
        if not isinstance(tool_id, str) or not tool_id or not isinstance(name, str) or not name:
            return None
        if tool_id in seen_ids:
            return None
        seen_ids.add(tool_id)
        identity.append((tool_id, name))
    return tuple(identity) or None


def _restore_replay_content(
    original: list[dict[str, Any]],
    replayed: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Restore token-bearing fields while retaining current cache metadata."""
    if len(original) != len(replayed):
        return None
    if any(not isinstance(block, dict) for block in original + replayed):
        return None
    original_types = [block.get('type') for block in original]
    if any(block_type not in {'text', 'thinking', 'tool_use'} for block_type in original_types):
        return None
    if original_types != [block.get('type') for block in replayed]:
        return None

    restored = []
    for original_block, replayed_block in zip(original, replayed):
        if original_block.get('type') == 'tool_use' and (
            original_block.get('id') != replayed_block.get('id')
            or original_block.get('name') != replayed_block.get('name')
        ):
            return None
        block = copy.deepcopy(original_block)
        block.pop('cache_control', None)
        if 'cache_control' in replayed_block:
            block['cache_control'] = copy.deepcopy(replayed_block['cache_control'])
        restored.append(block)
    return restored


class ClaudeCodePromptStabilizer(ProxyRequestProcessor):
    """Remove known prompt noise and restore identified assistant replays."""

    VERSION = 'claude-code-prompt-v1'

    def __init__(
        self,
        drop_task_tools_reminder: bool = True,
        restore_tool_use_replays: bool = True,
    ):
        self.drop_task_tools_reminder = drop_task_tools_reminder
        self.restore_tool_use_replays = restore_tool_use_replays
        self.reset()

    def reset(self) -> None:
        self._assistant_responses: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
        self._tool_id_owners: dict[str, tuple[tuple[str, str], ...]] = {}
        self._ambiguous_identities: set[tuple[tuple[str, str], ...]] = set()
        self._head_digests: dict[str, str] = {}
        self._stats = {
            'requests_seen': 0,
            'reminders_removed': 0,
            'assistant_replays_seen': 0,
            'assistant_replays_restored': 0,
            'assistant_replays_unanchored': 0,
            'assistant_replays_ambiguous': 0,
            'system_changed': 0,
            'tools_changed': 0,
            'model_changed': 0,
        }

    def before_forward(
        self,
        request: dict[str, Any],
        context: ProxyRequestContext,
    ) -> dict[str, Any]:
        if context.provider != 'anthropic' or not isinstance(request.get('messages'), list):
            return request

        self._stats['requests_seen'] += 1
        if self.drop_task_tools_reminder:
            self._stats['reminders_removed'] += self._remove_known_reminders(request)
        if self.restore_tool_use_replays:
            self._stats['assistant_replays_restored'] += self._restore_assistant_replays(request['messages'])
        self._record_head_changes(request)
        return request

    def after_response(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        context: ProxyRequestContext,
    ) -> None:
        del request
        if context.provider == 'anthropic' and self.restore_tool_use_replays:
            self._remember_assistant_response(response)

    def get_stats(self) -> dict[str, Any]:
        return {'processor_version': self.VERSION, **self._stats}

    def _remove_known_reminders(self, request: dict[str, Any]) -> int:
        removed = 0
        if 'system' in request:
            system, count = _strip_system_content(request['system'])
            removed += count
            if count and (system == '' or system == []):
                request.pop('system', None)
            else:
                request['system'] = system

        messages = []
        for message in request.get('messages', []):
            if not isinstance(message, dict):
                messages.append(message)
                continue
            updated = copy.deepcopy(message)
            if updated.get('role') == 'system':
                content, count = _strip_system_content(updated.get('content'))
            else:
                content, count = _strip_wrapped_content(updated.get('content'))
            removed += count
            if count:
                updated['content'] = content
                if content == '' or content == []:
                    continue
            messages.append(updated)
        request['messages'] = messages
        return removed

    def _restore_assistant_replays(self, messages: list[dict[str, Any]]) -> int:
        restored_count = 0
        for message in messages:
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            identity = _assistant_identity(message.get('content'))
            if identity is None:
                self._stats['assistant_replays_unanchored'] += 1
                continue
            self._stats['assistant_replays_seen'] += 1
            if identity in self._ambiguous_identities:
                self._stats['assistant_replays_ambiguous'] += 1
                continue
            original = self._assistant_responses.get(identity)
            replayed = message.get('content')
            if original is None or not isinstance(replayed, list):
                self._stats['assistant_replays_unanchored'] += 1
                continue
            restored = _restore_replay_content(original, replayed)
            if restored is None:
                self._stats['assistant_replays_ambiguous'] += 1
                continue
            if restored != replayed:
                message['content'] = restored
                restored_count += 1
        return restored_count

    def _remember_assistant_response(self, response: dict[str, Any]) -> None:
        content = response.get('content')
        identity = _assistant_identity(content)
        if identity is None or not isinstance(content, list):
            return

        conflicting = False
        for tool_id, _ in identity:
            owner = self._tool_id_owners.get(tool_id)
            if owner is not None and owner != identity:
                self._ambiguous_identities.update((owner, identity))
                conflicting = True
            else:
                self._tool_id_owners[tool_id] = identity

        existing = self._assistant_responses.get(identity)
        if existing is not None and existing != content:
            self._ambiguous_identities.add(identity)
            conflicting = True
        if not conflicting and identity not in self._ambiguous_identities:
            self._assistant_responses[identity] = copy.deepcopy(content)

    def _record_head_changes(self, request: dict[str, Any]) -> None:
        values = {
            'model': request.get('model'),
            'system': request.get('system'),
            'tools': request.get('tools'),
        }
        for name, value in values.items():
            digest = _stable_json_digest(value)
            baseline = self._head_digests.setdefault(name, digest)
            if baseline != digest:
                self._stats[f'{name}_changed'] += 1
