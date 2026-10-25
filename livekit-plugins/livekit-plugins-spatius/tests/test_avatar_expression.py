from __future__ import annotations

import asyncio
import math
from array import array
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from spatius import AvatarDetail

from livekit import rtc
from livekit.plugins.spatius import AvatarSession
from livekit.plugins.spatius.expression_output import ExpressionAudioOutput
from livekit.plugins.spatius.expressions import AvatarExpression, ExpressionController

pytestmark = pytest.mark.unit


@asynccontextmanager
async def _started_avatar(**kwargs: Any) -> AsyncIterator[tuple[Any, ...]]:
    sdk = MagicMock()
    for method in ("init", "start", "close", "send_audio", "interrupt"):
        setattr(sdk, method, AsyncMock())
    sdk.get_avatar_detail = AsyncMock(
        return_value=AvatarDetail(("neutral", "happy", "sad"), "neutral")
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
        **kwargs,
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
            yield session, sdk
        finally:
            await session.aclose()


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x01\x00" * 480,
        sample_rate=24_000,
        num_channels=1,
        samples_per_channel=480,
    )


def _rpc(identity: str, payload: str = "") -> rtc.RpcInvocationData:
    return rtc.RpcInvocationData(
        request_id="test-rpc", caller_identity=identity, payload=payload, response_timeout=5.0
    )


async def _frames(*frames: rtc.AudioFrame) -> AsyncIterator[rtc.AudioFrame]:
    for frame in frames:
        yield frame


def _finish_playback(session: Any, *, interrupted: bool = False) -> None:
    payload = f'{{"playback_position": 0, "interrupted": {str(interrupted).lower()}}}'
    assert session._on_playback_finished_rpc(_rpc(session.avatar_identity, payload)) == "ok"


async def test_expression_selection_is_sent_once_for_the_whole_reply() -> None:
    async with _started_avatar() as (session, sdk, *_):
        sent_end = asyncio.Event()

        async def send_audio(**kwargs: Any) -> str:
            if kwargs["end"]:
                sent_end.set()
            return "reply"

        sdk.send_audio.side_effect = send_audio
        frames = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame(), _frame()), expression=AvatarExpression("happy", 0)
            )
        ]
        assert session._audio is not None
        for frame in frames:
            await session._audio.capture_frame(frame)
        session._audio.flush()
        await asyncio.wait_for(sent_end.wait(), timeout=1)

        assert sdk.send_audio.await_args_list == [
            call(audio=bytes(frames[0].data), end=False, emotion="happy", strength=0),
            call(audio=bytes(frames[1].data), end=False),
            call(audio=b"", end=True),
        ]
        assert session._audio._pending_playback_count == 1
        _finish_playback(session)
        await asyncio.wait_for(session._audio.wait_for_playout(), timeout=1)


async def test_unannotated_reply_uses_default_without_leaking_previous_selection() -> None:
    async with _started_avatar() as (session, sdk, *_):
        sent_ends: asyncio.Queue[None] = asyncio.Queue()

        async def send_audio(**kwargs: Any) -> str:
            if kwargs["end"]:
                sent_ends.put_nowait(None)
            return "reply"

        sdk.send_audio.side_effect = send_audio
        annotated = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame()), expression=AvatarExpression("happy", 0.5)
            )
        ]
        assert session._audio is not None
        await session._audio.capture_frame(annotated[0])
        session._audio.flush()
        await asyncio.wait_for(sent_ends.get(), timeout=1)
        _finish_playback(session)
        await session._audio.capture_frame(_frame())
        session._audio.flush()
        await asyncio.wait_for(sent_ends.get(), timeout=1)

        assert sdk.send_audio.await_args_list == [
            call(audio=bytes(annotated[0].data), end=False, emotion="happy", strength=0.5),
            call(audio=b"", end=True),
            call(audio=bytes(_frame().data), end=False, emotion="neutral"),
            call(audio=b"", end=True),
        ]
        sdk.get_avatar_detail.assert_awaited_once()


async def test_configured_default_is_applied_to_each_unannotated_reply() -> None:
    async with _started_avatar(
        default_expression=AvatarExpression("neutral", 0),
    ) as (session, sdk, *_):
        sent_ends: asyncio.Queue[None] = asyncio.Queue()

        async def send_audio(**kwargs: Any) -> str:
            if kwargs["end"]:
                sent_ends.put_nowait(None)
            return "reply"

        sdk.send_audio.side_effect = send_audio
        assert session._audio is not None
        for _ in range(2):
            await session._audio.capture_frame(_frame())
            session._audio.flush()
            await asyncio.wait_for(sent_ends.get(), timeout=1)
            _finish_playback(session)

        starts = [item.kwargs for item in sdk.send_audio.await_args_list if not item.kwargs["end"]]
        assert len(starts) == 2
        assert all(item["emotion"] == "neutral" and item["strength"] == 0 for item in starts)


async def test_overlapping_generators_keep_their_own_expression() -> None:
    async with _started_avatar() as (session, sdk, *_):
        release_first = asyncio.Event()
        first_started = asyncio.Event()
        sent_ends: asyncio.Queue[None] = asyncio.Queue()

        async def first_source() -> AsyncIterator[rtc.AudioFrame]:
            first_started.set()
            await release_first.wait()
            yield _frame()

        async def collect(source: Any, expression: AvatarExpression) -> list[rtc.AudioFrame]:
            return [
                frame async for frame in session.expression_audio(source, expression=expression)
            ]

        async def send_audio(**kwargs: Any) -> str:
            if kwargs["end"]:
                sent_ends.put_nowait(None)
            return "reply"

        sdk.send_audio.side_effect = send_audio
        first = asyncio.create_task(collect(first_source(), AvatarExpression("happy", 0.25)))
        try:
            await asyncio.wait_for(first_started.wait(), timeout=1)
            second_frames = await collect(_frames(_frame()), AvatarExpression("sad", 0.75))
            release_first.set()
            first_frames = await asyncio.wait_for(first, timeout=1)
        finally:
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)

        assert session._audio is not None
        for frames in (first_frames, second_frames):
            await session._audio.capture_frame(frames[0])
            session._audio.flush()
            await asyncio.wait_for(sent_ends.get(), timeout=1)
            _finish_playback(session)
        starts = [item.kwargs for item in sdk.send_audio.await_args_list if not item.kwargs["end"]]
        assert [(item["emotion"], item["strength"]) for item in starts] == [
            ("happy", 0.25),
            ("sad", 0.75),
        ]


async def test_conflicting_generation_is_rejected_before_it_enters_the_queue() -> None:
    async with _started_avatar() as (session, sdk, *_):
        first = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame()), expression=AvatarExpression("happy", 0.5)
            )
        ]
        conflicting = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame()), expression=AvatarExpression("sad", 0.5)
            )
        ]
        assert session._audio is not None
        await session._audio.capture_frame(first[0])
        with pytest.raises(ValueError):
            await session._audio.capture_frame(conflicting[0])
        assert session._audio._pending_playback_count == 1


async def test_output_rejects_multichannel_audio() -> None:
    async with _started_avatar() as (session, sdk, *_):
        frame = rtc.AudioFrame(
            data=b"\0\0" * 480 * 2, sample_rate=24_000, num_channels=2, samples_per_channel=480
        )
        assert session._audio is not None
        with pytest.raises(ValueError, match="one channel"):
            await session._audio.capture_frame(frame)
        sdk.send_audio.assert_not_awaited()
        assert session._audio._pending_playback_count == 0


async def test_output_leaves_resampling_out_of_the_framework() -> None:
    audio = ExpressionAudioOutput(sample_rate=24_000, expressions=ExpressionController())
    # the framework's resampler would drop the expression carried in frame userdata
    assert audio.sample_rate is None
    assert audio.target_sample_rate == 24_000


def _tone(rate: int, samples: int, offset: int = 0) -> rtc.AudioFrame:
    pcm = array(
        "h",
        (int(12_000 * math.sin(2 * math.pi * 440 * (offset + n) / rate)) for n in range(samples)),
    )
    return rtc.AudioFrame(pcm.tobytes(), rate, 1, samples)


@pytest.mark.parametrize("rate", [16_000, 22_050, 48_000])
async def test_output_resamples_continuously_and_sends_the_tail_before_end(rate: int) -> None:
    async with _started_avatar() as (session, sdk, *_):
        sent_end = asyncio.Event()
        sdk.send_audio.side_effect = lambda **kwargs: sent_end.set() if kwargs["end"] else None
        # uneven chunks spanning 100 ms; one resampler must carry state across them
        boundaries = [0, 1, rate // 100, rate // 40, rate // 10]
        frames = [
            _tone(rate, end - start, start)
            for start, end in zip(boundaries, boundaries[1:], strict=False)
        ]
        annotated = [
            frame
            async for frame in session.expression_audio(
                _frames(*frames), expression=AvatarExpression("happy", 0.5)
            )
        ]
        assert session._audio is not None
        for frame in annotated:
            await session._audio.capture_frame(frame)
        session._audio.flush()
        await asyncio.wait_for(sent_end.wait(), timeout=1)

        calls = sdk.send_audio.await_args_list
        assert calls[-1] == call(audio=b"", end=True)
        audio_calls = calls[:-1]
        assert audio_calls[0].kwargs["emotion"] == "happy"
        assert audio_calls[0].kwargs["strength"] == 0.5
        assert all("emotion" not in item.kwargs for item in audio_calls[1:])
        sent = array("h", b"".join(item.kwargs["audio"] for item in audio_calls))
        reference = rtc.AudioResampler(rate, 24_000)
        expected = array(
            "h",
            b"".join(
                bytes(f.data) for f in reference.push(_tone(rate, rate // 10)) + reference.flush()
            ),
        )
        assert len(sent) == len(expected) == 2400
        # independent SoXR instances apply integer-output dither
        assert max(abs(a - b) for a, b in zip(sent, expected, strict=True)) <= 2
        assert session._audio._pending_playback_count == 1


async def test_short_reply_held_by_the_resampler_still_reaches_the_service() -> None:
    async with _started_avatar() as (session, sdk, *_):
        sent_end = asyncio.Event()
        sdk.send_audio.side_effect = lambda **kwargs: sent_end.set() if kwargs["end"] else None
        assert session._audio is not None
        # 10 ms at 16 kHz: the resampler emits nothing until it is flushed
        await session._audio.capture_frame(_tone(16_000, 160))
        assert session._audio._pending_playback_count == 1
        session._audio.flush()
        await asyncio.wait_for(sent_end.wait(), timeout=1)
        audio_calls = sdk.send_audio.await_args_list[:-1]
        assert sum(len(item.kwargs["audio"]) for item in audio_calls) == 240 * 2
        assert audio_calls[0].kwargs["emotion"] == "neutral"
        _finish_playback(session)
        await asyncio.wait_for(session._audio.wait_for_playout(), timeout=1)


async def test_mid_reply_rate_change_flushes_the_old_resampler_first() -> None:
    async with _started_avatar() as (session, sdk, *_):
        sent_end = asyncio.Event()
        sdk.send_audio.side_effect = lambda **kwargs: sent_end.set() if kwargs["end"] else None
        assert session._audio is not None
        await session._audio.capture_frame(_tone(16_000, 160))
        await session._audio.capture_frame(_tone(24_000, 480))
        session._audio.flush()
        await asyncio.wait_for(sent_end.wait(), timeout=1)
        sizes = [len(item.kwargs["audio"]) for item in sdk.send_audio.await_args_list[:-1]]
        assert sizes == [240 * 2, 480 * 2]


async def test_clear_discards_the_resampler_tail() -> None:
    async with _started_avatar() as (session, sdk, *_):
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_tone(16_000, 160))
        audio.clear_buffer()
        assert session._interrupt_atask is not None
        await asyncio.wait_for(session._interrupt_atask, timeout=1)
        sdk.send_audio.assert_not_awaited()
        assert audio._pending_playback_count == 0
        assert audio._resampler is None


async def test_interrupt_serializes_after_blocked_send_and_before_replacement() -> None:
    async with _started_avatar() as (session, sdk, *_):
        old_started = asyncio.Event()
        release_old = asyncio.Event()
        interrupt_sent = asyncio.Event()
        replacement_sent = asyncio.Event()
        operations: list[str] = []

        async def send_audio(**kwargs: Any) -> str:
            if kwargs.get("emotion") == "happy":
                operations.append("old send entered")
                old_started.set()
                await release_old.wait()
                operations.append("old send finished")
                return "old"
            if kwargs.get("emotion") == "sad":
                operations.append("replacement sent")
                replacement_sent.set()
            return "new"

        async def interrupt() -> str:
            operations.append("interrupt sent")
            interrupt_sent.set()
            return "old"

        sdk.send_audio.side_effect = send_audio
        sdk.interrupt.side_effect = interrupt
        old_frames = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame(), _frame()), expression=AvatarExpression("happy", 0.5)
            )
        ]
        new_frames = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame()), expression=AvatarExpression("sad", 1)
            )
        ]
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(old_frames[0])
        await asyncio.wait_for(old_started.wait(), timeout=1)
        await audio.capture_frame(old_frames[1])
        audio.flush()
        audio.clear_buffer()
        await audio.capture_frame(new_frames[0])
        audio.flush()
        assert not interrupt_sent.is_set()
        assert not replacement_sent.is_set()
        release_old.set()
        await asyncio.wait_for(interrupt_sent.wait(), timeout=1)
        assert not replacement_sent.is_set()

        _finish_playback(session, interrupted=True)
        await asyncio.wait_for(replacement_sent.wait(), timeout=1)
        assert operations == [
            "old send entered",
            "old send finished",
            "interrupt sent",
            "replacement sent",
        ]
        assert audio._pending_playback_count == 1
        _finish_playback(session)
        await asyncio.wait_for(audio.wait_for_playout(), timeout=1)
        assert audio._pending_playback_count == 0
        assert len([item for item in sdk.send_audio.await_args_list if not item.kwargs["end"]]) == 2


async def test_clearing_a_reply_fences_its_late_frames_without_poisoning_replacement() -> None:
    async with _started_avatar() as (session, sdk, *_):
        first_sent = asyncio.Event()
        interrupt_sent = asyncio.Event()
        replacement_sent = asyncio.Event()

        async def send_audio(**kwargs: Any) -> str:
            if kwargs.get("emotion") == "happy":
                first_sent.set()
            if kwargs.get("emotion") == "sad":
                replacement_sent.set()
            return "reply"

        async def interrupt() -> str:
            interrupt_sent.set()
            return "reply"

        sdk.send_audio.side_effect = send_audio
        sdk.interrupt.side_effect = interrupt
        old = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame(), _frame()), expression=AvatarExpression("happy", 0.5)
            )
        ]
        new = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame()), expression=AvatarExpression("sad", 0.5)
            )
        ]
        assert session._audio is not None
        await session._audio.capture_frame(old[0])
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        session._audio.clear_buffer()
        await asyncio.wait_for(interrupt_sent.wait(), timeout=1)
        await session._audio.capture_frame(old[1])
        await session._audio.capture_frame(new[0])
        session._audio.flush()
        _finish_playback(session, interrupted=True)
        await asyncio.wait_for(replacement_sent.wait(), timeout=1)

        starts = [item.kwargs for item in sdk.send_audio.await_args_list if not item.kwargs["end"]]
        assert len(starts) == 2
        assert [item["emotion"] for item in starts] == ["happy", "sad"]
        assert session._audio._pending_playback_count == 1


async def test_close_releases_old_and_replacement_segments_while_interrupt_waits() -> None:
    async with _started_avatar() as (session, sdk, *_):
        first_sent = asyncio.Event()
        interrupt_sent = asyncio.Event()

        async def send_audio(**kwargs: Any) -> str:
            first_sent.set()
            return "old"

        async def interrupt() -> str:
            interrupt_sent.set()
            return "old"

        sdk.send_audio.side_effect = send_audio
        sdk.interrupt.side_effect = interrupt
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_frame())
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        audio.flush()
        audio.clear_buffer()
        await asyncio.wait_for(interrupt_sent.wait(), timeout=1)
        await audio.capture_frame(_frame())
        audio.flush()
        assert audio._pending_playback_count == 2
        await asyncio.wait_for(session.aclose(), timeout=1)
        await asyncio.wait_for(audio.wait_for_playout(), timeout=1)
        assert audio._pending_playback_count == 0
        sdk.close.assert_awaited_once()


async def test_clear_before_first_send_completes_locally_and_preserves_session() -> None:
    async with _started_avatar() as (session, sdk, *_):
        sent_end = asyncio.Event()

        async def send_audio(**kwargs: Any) -> str:
            if kwargs["end"]:
                sent_end.set()
            return "replacement"

        sdk.send_audio.side_effect = send_audio
        sdk.interrupt.side_effect = ValueError("interrupt: no request to interrupt")
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_frame())
        audio.flush()
        audio.clear_buffer()
        assert session._interrupt_atask is not None
        await asyncio.wait_for(session._interrupt_atask, timeout=1)
        sdk.interrupt.assert_not_awaited()
        sdk.send_audio.assert_not_awaited()
        assert audio._pending_playback_count == 0
        assert not session._closed

        frames = [
            frame
            async for frame in session.expression_audio(
                _frames(_frame()), expression=AvatarExpression("happy", 0)
            )
        ]
        await audio.capture_frame(frames[0])
        audio.flush()
        await asyncio.wait_for(sent_end.wait(), timeout=1)
        assert sdk.send_audio.await_args_list == [
            call(audio=bytes(frames[0].data), end=False, emotion="happy", strength=0),
            call(audio=b"", end=True),
        ]
        _finish_playback(session)
        await asyncio.wait_for(audio.wait_for_playout(), timeout=1)


async def test_interrupt_timeout_closes_session_before_replacement_can_be_sent() -> None:
    async with _started_avatar() as (session, sdk, *_):
        first_sent = asyncio.Event()
        interrupt_sent = asyncio.Event()
        closed = asyncio.Event()
        sends = 0

        async def send_audio(**kwargs: Any) -> str:
            nonlocal sends
            if not kwargs["end"]:
                sends += 1
                first_sent.set()
            return "old"

        async def interrupt() -> str:
            interrupt_sent.set()
            return "old"

        async def close() -> None:
            closed.set()

        sdk.send_audio.side_effect = send_audio
        sdk.interrupt.side_effect = interrupt
        sdk.close.side_effect = close
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_frame())
        audio.flush()
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        audio.clear_buffer()
        await asyncio.wait_for(interrupt_sent.wait(), timeout=1)
        await audio.capture_frame(_frame())
        audio.flush()
        session._cancel_clear_buffer_timeout()
        session._on_clear_buffer_timeout()
        await asyncio.wait_for(closed.wait(), timeout=1)
        await asyncio.wait_for(audio.wait_for_playout(), timeout=1)
        assert session._closed
        assert sends == 1
        assert audio._pending_playback_count == 0


async def test_second_clear_during_interrupt_ack_does_not_strand_replacement() -> None:
    async with _started_avatar() as (session, sdk, *_):
        first_sent = asyncio.Event()
        interrupt_sent = asyncio.Event()
        replacement_sent = asyncio.Event()
        sends = 0

        async def send_audio(**kwargs: Any) -> str:
            nonlocal sends
            if not kwargs["end"]:
                sends += 1
                (first_sent if sends == 1 else replacement_sent).set()
            return f"request-{sends}"

        async def interrupt() -> str:
            interrupt_sent.set()
            return "request-1"

        sdk.send_audio.side_effect = send_audio
        sdk.interrupt.side_effect = interrupt
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_frame())
        audio.flush()
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        audio.clear_buffer()
        await asyncio.wait_for(interrupt_sent.wait(), timeout=1)
        _finish_playback(session, interrupted=True)
        await audio.capture_frame(_frame())
        audio.flush()
        audio.clear_buffer()
        await audio.capture_frame(_frame())
        audio.flush()
        await asyncio.wait_for(replacement_sent.wait(), timeout=1)
        _finish_playback(session)
        await asyncio.wait_for(audio.wait_for_playout(), timeout=1)
        assert audio._pending_playback_count == 0
        assert sends == 2
        sdk.interrupt.assert_awaited_once()
        assert not session._closed


async def test_unsolicited_interrupted_completion_releases_playout_and_closes_session() -> None:
    async with _started_avatar() as (session, sdk, *_):
        first_sent = asyncio.Event()
        closed = asyncio.Event()

        async def send_audio(**kwargs: Any) -> str:
            first_sent.set()
            return "request"

        async def close() -> None:
            closed.set()

        sdk.send_audio.side_effect = send_audio
        sdk.close.side_effect = close
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_frame())
        audio.flush()
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        _finish_playback(session, interrupted=True)
        await asyncio.wait_for(closed.wait(), timeout=1)
        playback = await asyncio.wait_for(audio.wait_for_playout(), timeout=1)
        assert playback.interrupted
        assert audio._pending_playback_count == 0
        assert session._closed
