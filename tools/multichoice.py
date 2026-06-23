"""Multiple-choice prompt with inline buttons (gateway-side).

Pattern mirrors ``tools/approval.py`` and ``tools/slash_confirm.py``:
the skill calls :func:`ask` and awaits; the active gateway adapter
renders inline buttons; the adapter's callback handler calls
:func:`resolve` which unblocks the waiting skill.

Only the user who initiated the session may click — gating happens in
the adapter's callback handler (see ``gateway/platforms/telegram.py``).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# session_key -> entry dict
_pending: Dict[str, Dict] = {}
# short_id -> session_key (for callback_data → session lookup)
_by_id: Dict[str, str] = {}
_lock = threading.RLock()
_counter = itertools.count(1)

DEFAULT_TIMEOUT_SECONDS = 600

# Sentinel returned by wait() when the user clicked the Cancel button.
# Distinguishable from None (timeout / error) so callers can react explicitly.
CANCELLED = "__multichoice_cancelled__"


def register(session_key: str, options: List[str], initiator_user_id: str) -> str:
    """Register a pending prompt and return the short_id for callback_data."""
    short_id = str(next(_counter))
    with _lock:
        _pending[session_key] = {
            "event": asyncio.Event(),
            "choice": None,
            "options": list(options),
            "id": short_id,
            "initiator": str(initiator_user_id or ""),
            "created_at": time.time(),
        }
        _by_id[short_id] = session_key
    return short_id


def get_by_id(short_id: str) -> Optional[Tuple[str, Dict]]:
    """Look up the session_key + entry snapshot by short_id."""
    with _lock:
        session_key = _by_id.get(short_id)
        if not session_key:
            return None
        entry = _pending.get(session_key)
        return (session_key, dict(entry)) if entry else None


def has_pending(session_key: str) -> bool:
    """Return True iff a prompt is registered AND not yet resolved for this session."""
    with _lock:
        entry = _pending.get(session_key)
        return bool(entry) and entry.get("choice") is None


def get_short_id(session_key: str) -> Optional[str]:
    """Return the short_id for ``session_key`` (or None if no pending prompt)."""
    with _lock:
        entry = _pending.get(session_key)
        return entry.get("id") if entry else None


def cancel_session(session_key: str) -> bool:
    """Resolve the pending prompt with the CANCELLED sentinel. Idempotent."""
    return resolve(session_key, CANCELLED)


def resolve(session_key: str, choice: str) -> bool:
    """Record the user's choice and wake the awaiting task. Idempotent."""
    with _lock:
        entry = _pending.get(session_key)
        if not entry or entry["choice"] is not None:
            return False
        entry["choice"] = choice
        event = entry["event"]
    event.set()
    return True


def clear(session_key: str) -> None:
    """Drop the pending prompt for ``session_key`` without resolving it."""
    with _lock:
        entry = _pending.pop(session_key, None)
        if entry:
            _by_id.pop(entry["id"], None)


async def wait(
    session_key: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Block until :func:`resolve` is called or timeout. Returns the choice."""
    with _lock:
        entry = _pending.get(session_key)
    if not entry:
        return None
    try:
        await asyncio.wait_for(entry["event"].wait(), timeout=timeout)
        with _lock:
            entry = _pending.get(session_key)
            return entry.get("choice") if entry else None
    except asyncio.TimeoutError:
        logger.info("multichoice timeout for session=%s", session_key)
        return None
    finally:
        clear(session_key)


async def ask_user_choice(
    question: str,
    options: List[str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """High-level helper: prompt the active gateway user and await their click.

    Reads session identity from contextvars/env (``HERMES_SESSION_*``), finds
    the live adapter via :data:`gateway.run._gateway_runner_ref`, sends the
    inline-button prompt, and blocks until the user clicks or timeout.

    Returns the selected option string, or ``None`` on timeout or send error.

    Currently supports the Telegram adapter only — other platforms return
    ``None`` with a logged warning.
    """
    if not options:
        return None

    from gateway.session_context import get_session_env
    from gateway.config import Platform

    session_key = get_session_env("HERMES_SESSION_KEY", "")
    platform_str = get_session_env("HERMES_SESSION_PLATFORM", "")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    user_id = get_session_env("HERMES_SESSION_USER_ID", "")
    thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "")

    if not (session_key and chat_id):
        logger.warning("ask_user_choice: missing session_key or chat_id (skill not running inside gateway?)")
        return None

    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is None:
        logger.warning("ask_user_choice: gateway runner not available")
        return None

    try:
        platform = Platform(platform_str)
    except ValueError:
        logger.warning("ask_user_choice: unknown platform %r", platform_str)
        return None

    if platform is not Platform.TELEGRAM:
        logger.warning("ask_user_choice: platform %s not supported yet", platform_str)
        return None

    adapter = runner.adapters.get(platform)
    if adapter is None or not hasattr(adapter, "send_multichoice_prompt"):
        logger.warning("ask_user_choice: telegram adapter unavailable")
        return None

    short_id = register(session_key, options, user_id)
    metadata: Dict = {}
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
        logger.warning("ask_user_choice: send failed: %s", getattr(result, "error", ""))
        clear(session_key)
        return None

    return await wait(session_key, timeout=timeout)
