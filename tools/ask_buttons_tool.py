"""ask_user_buttons — agent-callable tool that renders inline buttons on Telegram.

When the agent calls this tool with a question and 2-8 options, the gateway
renders an inline-keyboard prompt and **blocks the agent's turn** until the
user clicks (or until ``timeout`` expires). Only the user who initiated the
current session may click.

Currently Telegram-only. On other platforms the tool returns an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300
MIN_OPTIONS = 2
MAX_OPTIONS = 8


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def _ok(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def ask_user_buttons(
    question: str,
    options: List[str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Synchronous tool handler — bridges into the gateway event loop.

    Reads session identity from contextvars synchronously (BEFORE the
    threadbridge), then schedules ``send + wait`` on the gateway loop via
    ``run_coroutine_threadsafe`` so the python-telegram-bot client (which
    is bound to that loop) can be used safely.
    """
    if not isinstance(question, str) or not question.strip():
        return _err("question is required")
    if not isinstance(options, list):
        return _err("options must be a list of strings")
    options = [str(o).strip() for o in options if str(o).strip()]
    if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
        return _err(f"options must contain between {MIN_OPTIONS} and {MAX_OPTIONS} items")

    try:
        timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS
    timeout = max(15.0, min(timeout, 1800.0))

    # Capture session identity NOW (sync, in caller's contextvars frame).
    try:
        from gateway.session_context import get_session_env
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
        user_id = get_session_env("HERMES_SESSION_USER_ID", "")
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "")
        session_key_env = get_session_env("HERMES_SESSION_KEY", "")
    except Exception as exc:
        return _err(f"failed to read session context: {exc}")

    if platform != "telegram":
        return _err(f"ask_user_buttons currently supports Telegram only (got platform={platform!r})")
    if not chat_id:
        return _err("no chat_id in session context")

    # Locate gateway runner + its event loop.
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref() if callable(_gateway_runner_ref) else None
    except Exception:
        runner = None
    if runner is None:
        return _err("gateway runner not available")

    gateway_loop: Optional[asyncio.AbstractEventLoop] = getattr(runner, "_gateway_loop", None)
    if gateway_loop is None or not gateway_loop.is_running():
        return _err("gateway event loop not running")

    try:
        from gateway.config import Platform
        adapter = runner.adapters.get(Platform.TELEGRAM)
    except Exception as exc:
        return _err(f"failed to locate telegram adapter: {exc}")
    if adapter is None or not hasattr(adapter, "send_multichoice_prompt"):
        return _err("telegram adapter unavailable or not patched")

    # Build a session_key. Prefer the runtime contextvar; fall back to a
    # deterministic synthesis so the resolve path matches the send path.
    session_key = session_key_env or f"telegram:ask_buttons:{chat_id}:{user_id}"

    async def _send_and_wait() -> Optional[str]:
        from tools.multichoice import register, wait, clear
        short_id = register(session_key, options, user_id)
        metadata: Dict[str, Any] = {}
        if thread_id:
            metadata["thread_id"] = thread_id
        result = await adapter.send_multichoice_prompt(
            chat_id=chat_id,
            question=question,
            options=options,
            session_key=session_key,
            short_id=short_id,
            initiator_user_id=user_id,
            metadata=metadata,
        )
        if not getattr(result, "success", False):
            clear(session_key)
            return None
        return await wait(session_key, timeout=timeout)

    fut = asyncio.run_coroutine_threadsafe(_send_and_wait(), gateway_loop)
    try:
        choice = fut.result(timeout=timeout + 30)
    except Exception as exc:
        return _err(f"prompt failed: {exc}")

    if choice is None:
        return _err("user did not respond within the timeout window")
    from tools.multichoice import CANCELLED
    if choice == CANCELLED:
        return _ok({
            "question": question,
            "options": options,
            "cancelled": True,
            "user_response": None,
        })
    return _ok({
        "question": question,
        "options": options,
        "cancelled": False,
        "user_response": choice,
    })


def check_requirements() -> bool:
    """Tool is registered always; runtime check happens inside the handler."""
    return True


# OpenAI function-calling schema
ASK_BUTTONS_SCHEMA = {
    "name": "ask_user_buttons",
    "description": (
        "Ask the user a multiple-choice question rendered as inline buttons "
        "on Telegram. The user picks ONE option by tapping a button. Use "
        "this when you want a quick, unambiguous answer from the user "
        "before continuing.\n\n"
        "Constraints:\n"
        "- Telegram-only. On other platforms this tool returns an error.\n"
        "- Only the user who triggered the current session may click.\n"
        "- Provides between 2 and 8 options. Each option label is shown "
        "verbatim on a button (Telegram truncates labels at ~40 chars).\n"
        "- The agent BLOCKS on this call until the user clicks or until "
        "``timeout`` seconds elapse.\n\n"
        "Prefer this over `clarify` when you have a SHORT closed set of "
        "answers. Use `clarify` (or plain text) when you need free-form "
        "input or open-ended discussion.\n\n"
        "A built-in ❌ Cancel button is always rendered alongside the "
        "supplied options. When the user cancels, the result contains "
        "``cancelled: true`` and ``user_response: null`` — treat this as "
        "an explicit abort and stop the current line of questioning."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question text shown above the buttons. Supports Markdown.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": MIN_OPTIONS,
                "maxItems": MAX_OPTIONS,
                "description": "Between 2 and 8 short option labels.",
            },
            "timeout": {
                "type": "number",
                "description": (
                    f"Seconds to wait for the user's click before giving up. "
                    f"Default {DEFAULT_TIMEOUT_SECONDS}. Clamped to [15, 1800]."
                ),
            },
        },
        "required": ["question", "options"],
    },
}


from tools.registry import registry

registry.register(
    name="ask_user_buttons",
    toolset="clarify",
    schema=ASK_BUTTONS_SCHEMA,
    handler=lambda args, **kw: ask_user_buttons(
        question=args.get("question", ""),
        options=args.get("options") or [],
        timeout=args.get("timeout", DEFAULT_TIMEOUT_SECONDS),
    ),
    check_fn=check_requirements,
    is_async=False,
    emoji="🔘",
)
