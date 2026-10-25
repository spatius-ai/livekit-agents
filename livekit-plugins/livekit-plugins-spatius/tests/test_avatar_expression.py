from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from spatius import AvatarDetail

from livekit import rtc
from livekit.plugins.spatius import AvatarSession
from livekit.plugins.spatius.expressions import AvatarExpression

pytestmark = pytest.mark.unit


@asynccontextmanager
async def _started_avatar() -> AsyncIterator[tuple[Any, MagicMock, asyncio.Event]]:
    sdk = MagicMock()
    for method in ("init", "start", "close", "interrupt"):
        setattr(sdk, method, AsyncMock())
    sdk.get_avatar_detail = AsyncMock(return_value=AvatarDetail(("neutral", "happy"), "neutral"))
    sent_end = asyncio.Event()
    sdk.send_audio = AsyncMock(
        side_effect=lambda **kwargs: sent_end.set() if kwargs["end"] else None
    )
    room = MagicMock()
    room.name = "expression-test-room"
    room.local_participant.identity = "expression-test-agent"
    room.isconnected.return_value = True
    session = AvatarSession(
        api_key="test-key",
        app_id="test-app",
        avatar_id="test-avatar",
        region="us-west",
        sample_rate=24_000,
    )
    with (
        patch("livekit.plugins.spatius.avatar.BaseAvatarSession.start", new=AsyncMock()),
        patch("livekit.plugins.spatius.avatar.new_avatar_session", return_value=sdk),
    ):
        try:
            await session.start(
                MagicMock(),
                room,
                livekit_url="wss://livekit.example.com",
                livekit_api_key="test-key-with-at-least-thirty-two-characters",
                livekit_api_secret="test-secret-with-at-least-thirty-two-characters",
            )
            yield session, sdk, sent_end
        finally:
            await session.aclose()


async def _frames(*frames: rtc.AudioFrame) -> AsyncIterator[rtc.AudioFrame]:
    for frame in frames:
        yield frame


async def test_expression_is_sent_on_the_first_packet_of_the_reply() -> None:
    async with _started_avatar() as (session, sdk, sent_end):
        frame = rtc.AudioFrame(b"\x01\x00" * 480, 24_000, 1, 480)
        audio = session._audio
        async for annotated in session.expression_audio(
            _frames(frame, frame), expression=AvatarExpression("happy", 0)
        ):
            await audio.capture_frame(annotated)
        audio.flush()
        await asyncio.wait_for(sent_end.wait(), timeout=1)

        assert sdk.send_audio.await_args_list == [
            call(audio=bytes(frame.data), end=False, emotion="happy", strength=0),
            call(audio=bytes(frame.data), end=False),
            call(audio=b"", end=True),
        ]


async def test_audio_is_resampled_to_the_session_rate() -> None:
    async with _started_avatar() as (session, sdk, sent_end):
        audio = session._audio
        for _ in range(10):  # 100 ms at 16 kHz
            await audio.capture_frame(rtc.AudioFrame(b"\x01\x00" * 160, 16_000, 1, 160))
        audio.flush()
        await asyncio.wait_for(sent_end.wait(), timeout=1)

        packets = [item.kwargs["audio"] for item in sdk.send_audio.await_args_list]
        assert packets[-1] == b""
        assert sum(len(packet) for packet in packets) == 2400 * 2  # 100 ms at 24 kHz
