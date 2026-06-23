"""Tests for the multichoice inline-button prompt (gateway + tools module)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# ── tools.multichoice state machine ──────────────────────────────────

def test_register_returns_short_id(_clean_state):
    from tools.multichoice import register
    sid = register("sess-1", ["A", "B"], initiator_user_id="42")
    assert sid and isinstance(sid, str)


def test_resolve_records_choice_and_clears(_clean_state):
    from tools.multichoice import register, resolve, get_by_id
    sid = register("sess-2", ["A", "B"], initiator_user_id="42")
    assert resolve("sess-2", "A") is True
    lookup = get_by_id(sid)
    assert lookup is not None
    _, entry = lookup
    assert entry["choice"] == "A"


def test_resolve_idempotent(_clean_state):
    from tools.multichoice import register, resolve
    register("sess-3", ["A", "B"], initiator_user_id="42")
    assert resolve("sess-3", "A") is True
    assert resolve("sess-3", "B") is False  # second call ignored


@pytest.mark.asyncio
async def test_wait_unblocks_on_resolve(_clean_state):
    from tools.multichoice import register, resolve, wait
    register("sess-4", ["X", "Y"], initiator_user_id="42")

    async def _delayed_resolve():
        await asyncio.sleep(0.05)
        resolve("sess-4", "Y")

    asyncio.create_task(_delayed_resolve())
    choice = await wait("sess-4", timeout=2.0)
    assert choice == "Y"


@pytest.mark.asyncio
async def test_wait_times_out(_clean_state):
    from tools.multichoice import register, wait
    register("sess-5", ["A"], initiator_user_id="42")
    choice = await wait("sess-5", timeout=0.1)
    assert choice is None


# ── TelegramAdapter.send_multichoice_prompt ──────────────────────────

def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._multichoice_state = {}
    adapter._multichoice_initiator = {}
    adapter.name = "telegram"
    # Stub internal helpers used by send_multichoice_prompt
    adapter._metadata_thread_id = lambda md: None
    adapter._link_preview_kwargs = lambda: {}
    adapter._reply_to_message_id_for_send = lambda *a, **k: None
    adapter._thread_kwargs_for_send = lambda *a, **k: {}
    sent = MagicMock(message_id=999)
    adapter._send_message_with_thread_fallback = AsyncMock(return_value=sent)
    return adapter


@pytest.mark.asyncio
async def test_send_multichoice_stores_state_and_initiator():
    adapter = _make_adapter()
    res = await adapter.send_multichoice_prompt(
        chat_id="123",
        question="Pick one",
        options=["red", "green", "blue"],
        session_key="sess-tg-1",
        short_id="42",
        initiator_user_id="user-7",
    )
    assert res.success is True
    assert adapter._multichoice_state["42"] == "sess-tg-1"
    assert adapter._multichoice_initiator["42"] == "user-7"

    # Verify reply_markup was passed with proper callback_data
    # (option rows + a trailing Cancel row are both present)
    kwargs = adapter._send_message_with_thread_fallback.call_args.kwargs
    keyboard = kwargs["reply_markup"]
    flat = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
    assert flat == ["mc:42:0", "mc:42:1", "mc:42:2", "mc:42:cancel"]
    # Cancel must be on its own row at the bottom
    assert keyboard.inline_keyboard[-1][0].callback_data == "mc:42:cancel"


@pytest.mark.asyncio
async def test_send_multichoice_empty_options_fails():
    adapter = _make_adapter()
    res = await adapter.send_multichoice_prompt(
        chat_id="123", question="?", options=[],
        session_key="sk", short_id="x", initiator_user_id="u",
    )
    assert res.success is False


def test_cancel_sentinel_distinct_from_options(_clean_state):
    """CANCELLED sentinel must not collide with any user-visible option label."""
    from tools.multichoice import CANCELLED
    assert isinstance(CANCELLED, str) and CANCELLED.startswith("__")


@pytest.mark.asyncio
async def test_wait_returns_cancelled_sentinel(_clean_state):
    """When resolve() is called with CANCELLED, wait() should return it."""
    from tools.multichoice import register, resolve, wait, CANCELLED
    register("sess-cancel", ["A", "B"], initiator_user_id="42")

    async def _delayed_cancel():
        await asyncio.sleep(0.05)
        resolve("sess-cancel", CANCELLED)

    asyncio.create_task(_delayed_cancel())
    result = await wait("sess-cancel", timeout=2.0)
    assert result == CANCELLED


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def _clean_state():
    """Reset tools.multichoice module state between tests."""
    from tools import multichoice as mc
    mc._pending.clear()
    mc._by_id.clear()
    yield
    mc._pending.clear()
    mc._by_id.clear()
