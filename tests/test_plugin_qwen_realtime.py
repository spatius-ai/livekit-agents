"""Hermetic Qwen-Omni-Realtime endpoint, protocol, and session lifecycle tests."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.types import APIConnectOptions
from livekit.plugins import qwen
from livekit.plugins.qwen.realtime import realtime_model as _rt
from livekit.plugins.qwen.realtime._transport import (
    ServerEvent,
    TransportClosed,
    TransportMessage,
    WebSocketTransport,
)

pytestmark = pytest.mark.unit

KEY = "sk-test"


# ------------------------------------------------------------------ endpoints


def test_default_endpoint_is_shared_beijing_host() -> None:
    with patch.dict("os.environ", {}, clear=True):
        model = qwen.RealtimeModel(api_key=KEY)
    assert _rt.websocket_url(model._opts.base_url, model.model) == (
        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3.8-omni-flash-realtime"
    )


def test_workspace_endpoint() -> None:
    with patch.dict("os.environ", {}, clear=True):
        model = qwen.RealtimeModel(api_key=KEY, workspace_id="ws-1", region="ap-southeast-1")
    assert model._opts.base_url == "https://ws-1.ap-southeast-1.maas.aliyuncs.com"
    assert _rt.websocket_url(model._opts.base_url, "m") == (
        "wss://ws-1.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=m"
    )


def test_api_key_and_workspace_environment() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            qwen.RealtimeModel()
    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": KEY, "DASHSCOPE_WORKSPACE_ID": "ws-2"}):
        model = qwen.RealtimeModel()
    assert model._opts.base_url == "https://ws-2.cn-beijing.maas.aliyuncs.com"


def test_capabilities_follow_turn_detection() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    assert model.capabilities.turn_detection
    assert model.capabilities.can_disable_turn_detection
    manual = qwen.RealtimeModel(api_key=KEY, turn_detection=None)
    assert not manual.capabilities.turn_detection
    assert not manual.capabilities.can_disable_turn_detection


# ------------------------------------------------------------------ fake transport


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.images: list[bytes] = []
        self.inbox: asyncio.Queue[TransportMessage] = asyncio.Queue()
        self.session_updated = 0
        self.closed = False

    async def connect(self) -> None:
        pass

    async def send_event(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    async def push_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)

    async def push_image(self, jpeg: bytes, width: int, height: int) -> None:
        self.images.append(jpeg)

    async def recv(self) -> TransportMessage:
        return await self.inbox.get()

    def on_session_updated(self) -> None:
        self.session_updated += 1

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.inbox.put_nowait(TransportClosed())

    def server(self, **event: Any) -> None:
        self.inbox.put_nowait(ServerEvent(event))

    def types(self) -> list[str]:
        return [e["type"] for e in self.sent]


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _start(
    model: qwen.RealtimeModel, transport: FakeTransport, **kwargs: Any
) -> _rt.RealtimeSession:
    with patch.object(_rt.RealtimeSession, "_create_transport", lambda self: transport):
        sess = model.session(**kwargs)
        await _settle()
    return sess


def _response_events(transport: FakeTransport, *, audio: bytes = b"") -> None:
    transport.server(type="response.created", response={"id": "resp_1"})
    transport.server(
        type="response.output_item.added",
        response_id="resp_1",
        item={"id": "item_1", "type": "message", "role": "assistant"},
    )
    transport.server(type="response.content_part.added", item_id="item_1", part={"type": "audio"})
    transport.server(type="response.audio_transcript.delta", item_id="item_1", delta="Hel")
    transport.server(type="response.audio_transcript.delta", item_id="item_1", delta="lo")
    if audio:
        transport.server(
            type="response.audio.delta", item_id="item_1", delta=base64.b64encode(audio).decode()
        )
    transport.server(type="response.output_item.done", item={"id": "item_1", "type": "message"})
    transport.server(
        type="response.done",
        response={
            "id": "resp_1",
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "input_tokens_details": {"audio_tokens": 8, "text_tokens": 2},
                "output_tokens_details": {"audio_tokens": 3, "text_tokens": 1},
            },
        },
    )


async def _collect(gen: llm.GenerationCreatedEvent) -> tuple[str, list[rtc.AudioFrame]]:
    text, frames = "", []
    async for msg in gen.message_stream:
        async for delta in msg.text_stream:
            text += delta
        async for frame in msg.audio_stream:
            frames.append(frame)
    return text, frames


# ------------------------------------------------------------------ session


async def test_session_update_on_connect() -> None:
    model = qwen.RealtimeModel(api_key=KEY, voice="Ethan", temperature=0.3)
    transport = FakeTransport()
    sess = await _start(model, transport)
    await sess.update_instructions("be brief")
    await _settle()

    first = transport.sent[0]
    assert first["type"] == "session.update"
    session = first["session"]
    assert session["voice"] == "Ethan"
    assert session["modalities"] == ["text", "audio"]
    assert session["audio"]["input"]["format"] == {"type": "pcm", "sample_rate": 16000}
    assert session["audio"]["output"]["format"] == {"type": "pcm", "sample_rate": 24000}
    assert session["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "silence_duration_ms": 800,
    }
    assert session["input_audio_transcription"] == {"model": "qwen3-asr-flash-realtime"}
    assert session["temperature"] == 0.3
    assert transport.sent[-1]["session"]["instructions"] == "be brief"

    transport.server(type="session.updated", session={})
    await _settle()
    assert transport.session_updated == 1
    await sess.aclose()


async def test_turn_detection_disabled_session() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport, turn_detection_disabled=True)
    assert transport.sent[0]["session"]["turn_detection"] is None
    assert not sess.capabilities.turn_detection
    await sess.aclose()


async def test_response_streams_text_audio_and_metrics() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    generations: list[llm.GenerationCreatedEvent] = []
    metrics: list[Any] = []
    sess.on("generation_created", generations.append)
    sess.on("metrics_collected", metrics.append)

    _response_events(transport, audio=b"\x01\x00" * 240)
    await _settle()

    assert len(generations) == 1 and not generations[0].user_initiated
    text, frames = await _collect(generations[0])
    assert text == "Hello"
    assert sum(f.samples_per_channel for f in frames) == 240
    assert frames[0].sample_rate == 24000

    rt_metrics = [m for m in metrics if isinstance(m, RealtimeModelMetrics) and m.request_id]
    assert rt_metrics[-1].input_tokens == 10
    assert rt_metrics[-1].input_token_details.audio_tokens == 8
    assert rt_metrics[-1].output_token_details.text_tokens == 1

    # the reply is mirrored, so it is not sent back as history
    assert sess.chat_ctx.get_by_id("item_1") is not None
    await sess.aclose()


async def test_generate_reply_with_instructions() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    await sess.update_instructions("base")
    fut = sess.generate_reply(instructions="greet the user")
    await _settle()

    assert transport.types()[-2:] == ["session.update", "response.create"]
    assert transport.sent[-2]["session"]["instructions"] == "base\ngreet the user"
    # without it, the server refuses to reply on an empty conversation
    assert transport.sent[-1]["response"]["instructions"] == "base\ngreet the user"

    _response_events(transport)
    await _settle()
    gen = await asyncio.wait_for(fut, 1)
    assert gen.user_initiated
    await _collect(gen)
    await _settle()

    # restored once the reply is done
    assert transport.sent[-1]["type"] == "session.update"
    assert transport.sent[-1]["session"]["instructions"] == "base"
    await sess.aclose()


async def test_rejected_reply_fails_at_once_and_restores_instructions() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    await sess.update_instructions("base")
    fut = sess.generate_reply(instructions="greet the user")
    await _settle()

    transport.server(
        type="error",
        error={"message": "Cannot create response without input, history, or instructions"},
    )
    with pytest.raises(llm.RealtimeError, match="Cannot create response"):
        await asyncio.wait_for(fut, 1)
    await _settle()
    assert transport.sent[-1]["session"]["instructions"] == "base"
    assert not sess.has_active_generation
    await sess.aclose()


async def test_missing_response_done_does_not_hang_the_reply() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    with (
        patch.object(_rt, "_RESPONSE_DONE_TIMEOUT_S", 0.05),
        patch.object(_rt, "_CANCEL_DONE_TIMEOUT_S", 0.05),
    ):
        sess = await _start(model, transport)
        generations: list[llm.GenerationCreatedEvent] = []
        sess.on("generation_created", generations.append)
        transport.server(type="response.created", response={"id": "resp_1"})
        transport.server(type="response.output_item.added", item={"id": "i", "type": "message"})
        transport.server(type="response.audio_transcript.delta", item_id="i", delta="Hi")
        transport.server(type="response.output_item.done", item={"id": "i", "type": "message"})
        await _settle()

        text, _ = await asyncio.wait_for(_collect(generations[0]), 1)
        assert text == "Hi"
        assert not sess.has_active_generation
        assert sess.chat_ctx.get_by_id("i") is not None

        # interrupting arms it too, and a late response.done is harmless
        transport.server(type="response.created", response={"id": "resp_2"})
        await _settle()
        sess.interrupt()
        await asyncio.wait_for(_collect(generations[1]), 1)
        transport.server(type="response.done", response={"id": "resp_2", "status": "cancelled"})
        await _settle()
        assert "response.cancel" in transport.types()
        await sess.aclose()


async def test_function_call_and_tool_output() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)

    @llm.function_tool
    async def get_weather(location: str) -> str:
        """Get the weather."""
        return "sunny"

    await sess.update_tools([get_weather])
    await _settle()
    tools = transport.sent[-1]["session"]["tools"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"

    generations: list[llm.GenerationCreatedEvent] = []
    sess.on("generation_created", generations.append)
    transport.server(type="response.created", response={"id": "resp_2"})
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather"}
    transport.server(
        type="response.function_call_arguments.done",
        item_id="fc_1",
        call_id="call_1",
        name="get_weather",
        arguments='{"location": "Hangzhou"}',
    )
    transport.server(
        type="response.output_item.done", item={**call, "arguments": '{"location": "Hangzhou"}'}
    )
    transport.server(type="response.done", response={"id": "resp_2", "status": "completed"})
    await _settle()

    calls = [c async for c in generations[0].function_stream]
    assert [(c.call_id, c.arguments) for c in calls] == [("call_1", '{"location": "Hangzhou"}')]

    chat_ctx = sess.chat_ctx
    chat_ctx.items.append(
        llm.FunctionCallOutput(call_id="call_1", name="get_weather", output="sunny", is_error=False)
    )
    chat_ctx.add_message(role="user", content="and tomorrow?")
    await sess.update_chat_ctx(chat_ctx)
    await _settle()

    created = [e["item"] for e in transport.sent if e["type"] == "conversation.item.create"]
    assert created == [
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "and tomorrow?"}],
        },
    ]
    await sess.aclose()


async def test_user_transcription_events() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    events: list[llm.InputTranscriptionCompleted] = []
    started: list[Any] = []
    sess.on("input_audio_transcription_completed", events.append)
    sess.on("input_speech_started", started.append)

    transport.server(type="input_audio_buffer.speech_started", item_id="u1")
    transport.server(
        type="conversation.item.input_audio_transcription.delta",
        item_id="u1",
        text="你好",
        stash="世",
    )
    transport.server(
        type="conversation.item.input_audio_transcription.completed",
        item_id="u1",
        transcript="你好世界",
    )
    await _settle()

    assert len(started) == 1
    assert [(e.transcript, e.is_final) for e in events] == [("你好世", False), ("你好世界", True)]
    assert events[-1].turn_started_at is not None

    # the transcribed turn is already on the server and is not resent as text
    await sess.update_chat_ctx(sess.chat_ctx)
    await _settle()
    assert "conversation.item.create" not in transport.types()
    await sess.aclose()


async def test_speech_start_without_item_id_is_forwarded() -> None:
    transport = FakeTransport()
    sess = await _start(qwen.RealtimeModel(api_key=KEY), transport)
    events: list[llm.InputSpeechStartedEvent] = []
    sess.on("input_speech_started", events.append)
    try:
        transport.server(type="input_audio_buffer.speech_started")
        await _settle()
        assert len(events) == 1
        assert sess._speech_pending_commit
    finally:
        await sess.aclose()


async def test_audio_is_resampled_to_16k_and_committed() -> None:
    model = qwen.RealtimeModel(api_key=KEY, turn_detection=None)
    transport = FakeTransport()
    sess = await _start(model, transport)
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 4800, sample_rate=48000, num_channels=1, samples_per_channel=4800
    )
    sess.push_audio(frame)
    sess.push_audio(frame)
    sess.commit_audio()
    await _settle()
    assert sum(len(a) for a in transport.audio) > 0
    assert all(len(a) <= 3200 for a in transport.audio)  # 100 ms at 16 kHz
    assert transport.types()[-1] == "input_audio_buffer.commit"
    await sess.aclose()


async def test_fatal_error_stops_the_session() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    errors: list[llm.RealtimeModelError] = []
    sess.on("error", errors.append)
    transport.server(type="error", error={"code": "InvalidApiKey", "message": "bad key"})
    with pytest.raises(APIError, match="bad key"):
        await asyncio.wait_for(sess._main_atask, 1)
    assert errors and not errors[-1].recoverable


def _video_frame(width: int = 1920, height: int = 1080) -> rtc.VideoFrame:
    return rtc.VideoFrame(width, height, rtc.VideoBufferType.RGBA, b"\x80" * (width * height * 4))


async def test_websocket_video_waits_for_handshake_and_is_paced_and_scaled() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sizes: list[tuple[int, int]] = []

    async def push_image(jpeg: bytes, width: int, height: int) -> None:
        sizes.append((width, height))
        transport.images.append(jpeg)

    transport.push_image = push_image  # type: ignore[method-assign]
    sess = await _start(model, transport)

    sess.push_video(_video_frame())  # session has not been acknowledged yet
    await _settle()
    assert transport.images == []

    transport.server(type="session.updated")
    await _settle()
    sess.push_video(_video_frame())  # microphone audio is optional

    sess.push_video(_video_frame())  # within the same second
    for _ in range(50):
        if transport.images:
            break
        await asyncio.sleep(0.01)
    await _settle()
    assert len(transport.images) == 1
    assert transport.images[0][:2] == b"\xff\xd8"  # JPEG
    assert sizes == [(1280, 720)]
    await sess.aclose()


@pytest.mark.virtual_time
async def test_video_is_limited_to_one_frame_per_second_without_microphone() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        workspace_id="ws",
    )
    transport = FakeTransport()
    sent_at: list[float] = []
    loop = asyncio.get_running_loop()

    async def encode_video(frame: rtc.VideoFrame) -> _rt._Image:
        return _rt._Image(b"jpeg", frame.width, frame.height)

    async def push_image(jpeg: bytes, width: int, height: int) -> None:
        sent_at.append(loop.time())
        transport.images.append(jpeg)

    transport.push_image = push_image  # type: ignore[method-assign]
    sess = await _start(model, transport)
    with patch.object(sess, "_encode_video", encode_video):
        # Frames cannot get ahead of the session handshake, even without a microphone.
        sess.push_video(_video_frame(640, 360))
        await _settle()
        assert transport.images == []
        transport.server(type="session.updated")
        await _settle()
        sess.push_video(_video_frame(640, 360))
        await _settle()
        assert len(transport.images) == 1
        sess.push_video(_video_frame(640, 360))
        sess.push_video(_video_frame(640, 360))
        await _settle()
        assert len(transport.images) == 1
        await asyncio.sleep(1.0)
        await _settle()
        assert len(transport.images) == 2
        assert sent_at[1] - sent_at[0] == pytest.approx(1.0)
        assert transport.audio == []
    await sess.aclose()


async def test_video_keeps_latest_frame_and_does_not_block_audio() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        workspace_id="ws",
    )
    transport = FakeTransport()
    sess = await _start(model, transport)
    sess._image_interval_s = 0.01
    transport.server(type="session.updated")
    await _settle()
    loop = asyncio.get_running_loop()
    encoding = asyncio.Event()
    sent = asyncio.Event()
    release = threading.Event()
    encoded: list[int] = []
    worker_threads: list[int] = []
    main_thread = threading.get_ident()

    def encode_image(frame: rtc.VideoFrame) -> _rt._Image:
        encoded.append(frame.width)
        worker_threads.append(threading.get_ident())
        if len(encoded) == 1:
            loop.call_soon_threadsafe(encoding.set)
            assert release.wait(2), "test did not release the blocked encoder"
        return _rt._Image(str(frame.width).encode(), frame.width, frame.height)

    async def push_image(jpeg: bytes, width: int, height: int) -> None:
        transport.images.append(jpeg)
        if len(transport.images) == 2:
            sent.set()

    transport.push_image = push_image  # type: ignore[method-assign]
    with patch.object(_rt, "_encode_image", encode_image):
        try:
            sess.push_video(_video_frame(640, 360))
            await asyncio.wait_for(encoding.wait(), 1)
            for width in range(641, 651):
                sess.push_video(_video_frame(width, 360))
            assert sess._pending_video is not None and sess._pending_video.width == 650
            assert encoded == [640]
            samples = 1600
            sess.push_audio(
                rtc.AudioFrame(
                    data=b"\x00\x00" * samples,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=samples,
                )
            )
            sess.send_event({"type": "test.control"})
            await _settle()
            assert len(transport.audio) == 1
            assert "test.control" in transport.types()
            assert transport.images == []
            release.set()
            await asyncio.wait_for(sent.wait(), 1)
            assert encoded == [640, 650]
            assert transport.images == [b"640", b"650"]
            assert all(thread != main_thread for thread in worker_threads)
        finally:
            release.set()
            await sess.aclose()


async def test_close_joins_encoding_and_discards_its_result() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        workspace_id="ws",
    )
    transport = FakeTransport()
    sess = await _start(model, transport)
    transport.server(type="session.updated")
    await _settle()
    loop = asyncio.get_running_loop()
    encoding = asyncio.Event()
    release = threading.Event()

    def encode_image(frame: rtc.VideoFrame) -> _rt._Image:
        loop.call_soon_threadsafe(encoding.set)
        assert release.wait(2), "test did not release the blocked encoder"
        return _rt._Image(b"old-frame", frame.width, frame.height)

    with patch.object(_rt, "_encode_image", encode_image):
        try:
            sess.push_video(_video_frame(640, 360))
            await asyncio.wait_for(encoding.wait(), 1)
            sess.push_video(_video_frame(641, 360))
            close = asyncio.create_task(sess.aclose())
            await _settle()
            assert not close.done()
            assert sess._pending_video is None
            release.set()
            await asyncio.wait_for(close, 1)
            assert transport.images == []
            assert not sess._video_ready
        finally:
            release.set()
            await sess.aclose()


@pytest.mark.parametrize("cancel_target", ["close", "main"])
async def test_cancelled_shutdown_still_joins_encoding(cancel_target: str) -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        workspace_id="ws",
    )
    transport = FakeTransport()
    sess = await _start(model, transport)
    transport.server(type="session.updated")
    await _settle()
    loop = asyncio.get_running_loop()
    encoding = asyncio.Event()
    finished = threading.Event()
    release = threading.Event()

    def encode_image(frame: rtc.VideoFrame) -> _rt._Image:
        loop.call_soon_threadsafe(encoding.set)
        try:
            assert release.wait(2), "test did not release the blocked encoder"
            return _rt._Image(b"old-frame", frame.width, frame.height)
        finally:
            finished.set()

    with patch.object(_rt, "_encode_image", encode_image):
        close: asyncio.Task[None] | None = None
        try:
            sess.push_video(_video_frame(640, 360))
            await asyncio.wait_for(encoding.wait(), 1)
            close = asyncio.create_task(sess.aclose())
            # Let shutdown cancel the worker, which must join its outstanding thread.
            for _ in range(3):
                await _settle()
            target = close if cancel_target == "close" else sess._main_atask
            for _ in range(2):
                target.cancel()
                await _settle()
            assert not close.done()
            assert not sess._main_atask.done()
            assert not finished.is_set()
            assert not transport.closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await close
            assert finished.is_set()
            assert sess._main_atask.done()
            assert transport.closed
            assert transport.images == []
            await _settle()
            assert not [task for task in asyncio.all_tasks() if task.get_name() == "qwen.video"]
        finally:
            release.set()
            if close is not None:
                await asyncio.gather(close, return_exceptions=True)


async def test_reconnect_discards_old_encoding_and_requires_fresh_handshake() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        workspace_id="ws",
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
    )
    first, second = FakeTransport(), FakeTransport()
    transports = iter([first, second])
    reconnected = asyncio.Event()

    def create_transport(_: _rt.RealtimeSession) -> FakeTransport:
        transport = next(transports)
        if transport is second:
            reconnected.set()
        return transport

    loop = asyncio.get_running_loop()
    encoding = asyncio.Event()
    fresh_sent = asyncio.Event()
    release = threading.Event()
    encoded: list[int] = []

    def encode_image(frame: rtc.VideoFrame) -> _rt._Image:
        encoded.append(frame.width)
        if len(encoded) == 1:
            loop.call_soon_threadsafe(encoding.set)
            assert release.wait(2), "test did not release the blocked encoder"
        return _rt._Image(str(frame.width).encode(), frame.width, frame.height)

    async def push_image(jpeg: bytes, width: int, height: int) -> None:
        second.images.append(jpeg)
        fresh_sent.set()

    second.push_image = push_image  # type: ignore[method-assign]
    with (
        patch.object(_rt.RealtimeSession, "_create_transport", create_transport),
        patch.object(_rt, "_encode_image", encode_image),
    ):
        sess = model.session()
        try:
            await _settle()
            first.server(type="session.updated")
            await _settle()
            sess.push_video(_video_frame(640, 360))
            await asyncio.wait_for(encoding.wait(), 1)
            sess.push_video(_video_frame(641, 360))
            first.inbox.put_nowait(TransportClosed(APIConnectionError("retry")))
            await _settle()
            assert sess._pending_video is None and not sess._video_ready
            release.set()
            await asyncio.wait_for(reconnected.wait(), 1)
            await _settle()
            sess.push_video(_video_frame(642, 360))
            await _settle()
            assert encoded == [640]
            second.server(type="session.updated")
            await _settle()
            sess.push_video(_video_frame(650, 360))
            await asyncio.wait_for(fresh_sent.wait(), 1)
            assert encoded == [640, 650]
            assert first.images == [] and second.images == [b"650"]
        finally:
            release.set()
            await sess.aclose()


@pytest.mark.virtual_time
async def test_bad_video_encodes_do_not_stop_later_frames() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        workspace_id="ws",
    )
    transport = FakeTransport()
    sess = await _start(model, transport)
    transport.server(type="session.updated")
    await _settle()
    attempts = 0

    async def encode_video(frame: rtc.VideoFrame) -> _rt._Image | None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("bad frame")
        if attempts == 2:
            return None  # oversized image
        return _rt._Image(b"good-frame", frame.width, frame.height)

    with patch.object(sess, "_encode_video", encode_video):
        for _ in range(3):
            sess.push_video(_video_frame(640, 360))
            await _settle()
            await asyncio.sleep(1.0)
            await _settle()
        assert attempts == 3
        assert transport.images == [b"good-frame"]
    await sess.aclose()


def test_oversized_video_image_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    with patch.object(_rt, "encode", return_value=b"x" * (_rt._IMAGE_MAX_BYTES + 1)):
        with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
            assert _rt._encode_image(_video_frame(640, 360)) is None
    assert "video frame too large" in caplog.text


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1920, 1080), (1280, 720)),
        ((1080, 1920), (720, 1280)),
        ((640, 360), (640, 360)),
    ],
)
def test_video_is_scaled_down_never_up(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert _rt._fit(*size, 1280, 720) == expected


@dataclass
class _CameraProtocol:
    transport: WebSocketTransport
    # Preserve control-only events for existing strict text/update-order assertions.
    events: list[dict[str, Any]] = field(default_factory=list)
    inbox: asyncio.Queue[aiohttp.WSMessage] = field(default_factory=asyncio.Queue)
    closed: bool = False
    auto_update: bool = True
    auto_restore: bool = True
    auto_commit: bool = True
    commit_error: bool = False
    manual: bool = False
    commits: int = 0
    audio_frames: int = 0
    image_frames: int = 0
    has_nonzero_pcm: bool = False
    audio_since_boundary: bool = False
    pending_images: int = 0
    committed_images: int = 0
    manual_requested: asyncio.Event = field(default_factory=asyncio.Event)
    vad_restored: asyncio.Event = field(default_factory=asyncio.Event)
    # Set by a typed item in manual mode, which implicitly commits its media.
    commit_requested: asyncio.Event = field(default_factory=asyncio.Event)

    def server(self, **event: Any) -> None:
        self.inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), ""))

    def ack_update(self, detection: Any) -> None:
        # The real service omits null turn_detection in manual acknowledgments.
        self.server(
            type="session.updated",
            session={} if detection is None else {"turn_detection": detection},
        )

    def ack_commit(self) -> None:
        self.commits += 1
        self.committed_images = self.pending_images
        self.pending_images = 0
        self.audio_since_boundary = False
        self.server(type="input_audio_buffer.committed", item_id=f"camera-{self.commits}")

    async def send_str(self, data: str) -> None:
        event = json.loads(data)
        kind = event["type"]
        if kind == "input_audio_buffer.append":
            pcm = base64.b64decode(event["audio"])
            self.audio_frames += 1
            self.has_nonzero_pcm |= any(pcm)
            self.audio_since_boundary = True
            return
        if kind == "input_image_buffer.append":
            assert self.audio_since_boundary, "image overtook fresh input PCM"
            self.image_frames += 1
            self.pending_images += 1
            return

        self.events.append(event)
        if kind == "session.update":
            detection = event["session"].get("turn_detection")
            self.manual = detection is None
            if self.manual:
                self.manual_requested.set()
                self.vad_restored.clear()
            else:
                self.vad_restored.set()
            if self.auto_update and (self.auto_restore or detection is None):
                self.ack_update(detection)
        elif kind == "input_audio_buffer.clear":
            self.pending_images = 0
            self.audio_since_boundary = False
        elif kind == "input_audio_buffer.commit":
            # A separate explicit commit would regress the typed-item optimization.
            raise AssertionError("camera input must be committed by the typed item")
        elif kind == "conversation.item.create" and event["item"].get("role") == "user":
            if self.manual:
                self.commit_requested.set()
                if self.commit_error:
                    self.server(type="error", error={"message": "synthetic commit failed"})
                elif self.auto_commit:
                    self.ack_commit()
            self.audio_since_boundary = False
        elif kind == "response.create":
            self.server(type="response.created", response={"id": "camera-response"})
            self.server(
                type="response.done", response={"id": "camera-response", "status": "completed"}
            )

    async def receive(self) -> aiohttp.WSMessage:
        return await self.inbox.get()

    async def close(self) -> bool:
        if not self.closed:
            self.closed = True
            self.inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))
        return True

    def types(self) -> list[str]:
        return [event["type"] for event in self.events]


@pytest.fixture
async def camera_session() -> AsyncIterator[tuple[_rt.RealtimeSession, _CameraProtocol]]:
    transport = WebSocketTransport(
        url="wss://synthetic.test",
        api_key=KEY,
        http_session=None,
        timeout=2.0,
    )  # type: ignore[arg-type]
    protocol = _CameraProtocol(transport)

    async def connect_socket() -> None:
        # No HTTP/session/network object is acquired by this fixture.
        transport._ws = protocol  # type: ignore[assignment]

    async def encode_video(frame: rtc.VideoFrame) -> _rt._Image:
        return _rt._Image(b"synthetic-jpeg", 640, 360)

    model = qwen.RealtimeModel(
        api_key=KEY,
        conn_options=APIConnectOptions(max_retry=0, timeout=2.0),
    )
    with (
        patch.object(transport, "connect", connect_socket),
        patch.object(_rt.RealtimeSession, "_create_transport", lambda self: transport),
        patch.object(_rt.RealtimeSession, "_encode_video", lambda self, frame: encode_video(frame)),
    ):
        sess = model.session()
        await _settle()
        frame = _video_frame(640, 360)

        async def camera() -> None:
            while True:
                sess.push_video(frame)
                await asyncio.sleep(1.0)

        pump = asyncio.create_task(camera())
        try:
            await transport._wait_camera_input(2.0)
            yield sess, protocol
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
            try:
                await sess.aclose()
            except APIError:
                # Failure cases deliberately close the sole fixed synthetic socket.
                pass


def _camera_text(sess: _rt.RealtimeSession) -> llm.ChatContext:
    context = sess.chat_ctx
    context.add_message(role="user", content="What color is the synthetic image?")
    return context


@pytest.mark.virtual_time
async def test_typed_camera_sends_text_in_manual_mode_then_restores_vad(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    await sess.update_chat_ctx(_camera_text(sess))
    reply = sess.generate_reply()
    await _settle()
    await reply
    types = protocol.types()
    text = types.index("conversation.item.create")
    response = types.index("response.create")
    assert "input_audio_buffer.commit" not in types
    assert protocol.events[text - 1]["session"]["turn_detection"] is None
    assert protocol.events[text + 1]["session"]["turn_detection"]["type"] == "server_vad"
    assert text < response
    assert protocol.commits == 1
    assert not sess._camera_turn_manual and not protocol.transport._video_paused
    assert sess._opts.turn_detection is not None
    assert sess.capabilities.turn_detection
    assert protocol.audio_frames > 0 and protocol.image_frames > 0
    assert not protocol.has_nonzero_pcm


@pytest.mark.virtual_time
async def test_typed_camera_warns_when_the_item_commits_nothing(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sess, protocol = camera_session
    protocol.auto_commit = False
    with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
        await sess.update_chat_ctx(_camera_text(sess))
        assert "Qwen did not commit camera input with the typed message" not in caplog.messages
        await asyncio.sleep(_rt._CAMERA_COMMIT_CHECK_S + 0.1)
    assert "Qwen did not commit camera input with the typed message" in caplog.messages
    assert protocol.types().count("conversation.item.create") == 1
    assert protocol.manual is False


@pytest.mark.virtual_time
async def test_typed_camera_late_commit_does_not_warn(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sess, protocol = camera_session
    protocol.auto_commit = False
    with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
        await sess.update_chat_ctx(_camera_text(sess))
        protocol.ack_commit()  # after the VAD restore was acknowledged
        await asyncio.sleep(_rt._CAMERA_COMMIT_CHECK_S + 0.1)
    assert "Qwen did not commit camera input with the typed message" not in caplog.messages


@pytest.mark.virtual_time
async def test_typed_camera_waits_for_its_own_session_update_ack(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    protocol.auto_update = False
    await sess.update_instructions("Synthetic safe instructions")
    preparing = asyncio.create_task(sess.update_chat_ctx(_camera_text(sess)))
    await _settle()
    assert len(sess._session_update_acks) == 2
    protocol.ack_update({"type": "server_vad"})
    await _settle()
    assert not protocol.commit_requested.is_set()
    assert not preparing.done()
    protocol.auto_update = True
    protocol.ack_update(None)
    await preparing
    assert protocol.commits == 1


@pytest.mark.virtual_time
async def test_typed_camera_concurrent_instruction_update_keeps_manual_mode(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    protocol.auto_update = False
    preparing = asyncio.create_task(sess.update_chat_ctx(_camera_text(sess)))
    await protocol.manual_requested.wait()
    await sess.update_instructions("Updated synthetic instructions")
    await _settle()
    assert protocol.events[-1]["session"]["turn_detection"] is None
    protocol.auto_update = True
    protocol.ack_update(None)
    protocol.ack_update(None)
    await preparing
    await _settle()
    assert protocol.manual is False
    assert protocol.commits == 1
    assert protocol.types()[-2:] == ["conversation.item.create", "session.update"]


@pytest.mark.virtual_time
async def test_typed_camera_waits_for_vad_commit_not_only_speech_stop(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    protocol.server(type="input_audio_buffer.speech_started", item_id="spoken")
    await _settle()
    before = len(protocol.events)
    preparing = asyncio.create_task(sess.update_chat_ctx(_camera_text(sess)))
    protocol.server(type="input_audio_buffer.speech_stopped", item_id="spoken")
    await _settle()
    assert not preparing.done()
    assert len(protocol.events) == before
    protocol.server(type="input_audio_buffer.committed", item_id="spoken")
    await preparing
    await _settle()
    assert "input_audio_buffer.commit" not in protocol.types()
    assert protocol.types()[-1] == "conversation.item.create"


@pytest.mark.virtual_time
@pytest.mark.parametrize("during_ack", [True, False])
async def test_speech_racing_typed_camera_is_not_manually_committed(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
    during_ack: bool,
) -> None:
    sess, protocol = camera_session
    protocol.auto_update = not during_ack
    preparing = asyncio.create_task(sess.update_chat_ctx(_camera_text(sess)))
    await protocol.manual_requested.wait()
    await _settle()
    protocol.server(type="input_audio_buffer.speech_started", item_id="racing")
    if during_ack:
        protocol.auto_update = True
        protocol.ack_update(None)
    await _settle()
    await asyncio.wait_for(protocol.vad_restored.wait(), 2.0)
    assert not protocol.commit_requested.is_set()
    assert not preparing.done()
    assert not protocol.manual
    protocol.server(type="input_audio_buffer.speech_stopped", item_id="racing")
    await _settle()
    assert not preparing.done()
    protocol.server(type="input_audio_buffer.committed", item_id="racing")
    await preparing
    assert not protocol.commit_requested.is_set()


@pytest.mark.virtual_time
async def test_typed_camera_provider_error_sends_the_text_once(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    protocol.commit_error = True
    await sess.update_chat_ctx(_camera_text(sess))
    await _settle()
    assert protocol.types().count("conversation.item.create") == 1
    assert sess._camera_prepared_transport is None
    assert not protocol.manual
    assert not protocol.transport._video_paused


@pytest.mark.virtual_time
async def test_typed_camera_close_rejects_pending_ack_without_replay(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    protocol.auto_update = False
    preparing = asyncio.create_task(sess.update_chat_ctx(_camera_text(sess)))
    await protocol.manual_requested.wait()
    await sess.aclose()
    await preparing
    assert not sess._camera_waiters
    assert "input_audio_buffer.commit" not in protocol.types()
    assert "conversation.item.create" not in protocol.types()
    assert not protocol.transport._video_paused


@pytest.mark.virtual_time
async def test_cancelled_typed_camera_failed_restore_still_closes_transport(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    protocol.auto_restore = False
    preparing = asyncio.create_task(sess.update_chat_ctx(_camera_text(sess)))
    await protocol.commit_requested.wait()
    await _settle()
    preparing.cancel()
    await _settle()
    preparing.cancel()
    await _settle()
    assert not preparing.done()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert protocol.transport._closing
    assert protocol.closed
    assert protocol.transport._audio_clock_task is not None
    assert protocol.transport._audio_clock_task.done()
    assert not sess._camera_turn_manual
    assert not protocol.transport._video_paused


@pytest.mark.virtual_time
async def test_typed_camera_generation_guards_do_not_send_on_replacement_or_close(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    old = protocol.transport
    replacement = FakeTransport()
    sess._camera_prepared_transport = old
    sess._transport = replacement
    with pytest.raises(llm.RealtimeError):
        await sess.generate_reply()
    await sess._send(
        replacement,
        _rt._ScopedEvent(old, {"type": "conversation.item.create", "item": {"type": "message"}}),
    )
    assert replacement.sent == []
    sess._transport = old
    old._video_dirty = True
    sess._camera_prepared_transport = old
    sess._closing = True
    with pytest.raises(llm.RealtimeError):
        await sess.generate_reply()
    await sess._send(old, _rt._ScopedEvent(old, {"type": "response.create"}))
    assert "response.create" not in protocol.types()
    sess._closing = False


@pytest.mark.virtual_time
async def test_typed_camera_error_during_fresh_media_wait_does_not_commit(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
) -> None:
    sess, protocol = camera_session
    preparing = asyncio.create_task(sess.update_chat_ctx(_camera_text(sess)))
    await protocol.manual_requested.wait()
    await _settle()
    protocol.server(type="error", error={"message": "synthetic media preparation failure"})
    await preparing
    await _settle()
    assert not protocol.commit_requested.is_set()
    assert protocol.types()[-1] == "conversation.item.create"
    assert not protocol.manual
    assert not protocol.transport._video_paused


async def test_greeting_and_text_without_camera_do_not_prepare_media() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        workspace_id="ws",
    )
    transport = FakeTransport()
    sess = await _start(model, transport)
    try:
        transport.server(type="session.updated")
        await _settle()
        greeting = sess.generate_reply()
        _response_events(transport)
        await greeting
        await _settle()
        await sess.update_chat_ctx(_camera_text(sess))
        reply = sess.generate_reply()
        _response_events(transport)
        await reply
        await _settle()
        assert transport.types() == [
            "session.update",
            "response.create",
            "conversation.item.create",
            "response.create",
        ]
    finally:
        await sess.aclose()
