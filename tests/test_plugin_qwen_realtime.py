"""Tests for the Qwen-Omni-Realtime plugin: endpoints, event handling and the AOQ transport."""

from __future__ import annotations

import asyncio
import base64
import enum
import json
import sys
import threading
import time
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.types import APIConnectOptions
from livekit.plugins import qwen
from livekit.plugins.qwen.realtime import _aoq, realtime_model as _rt
from livekit.plugins.qwen.realtime._transport import (
    OutputAudio,
    ServerEvent,
    Transport,
    TransportClosed,
    TransportMessage,
    pcm_frame,
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


def test_workspace_endpoint_and_aoq_credentials_url() -> None:
    with patch.dict("os.environ", {}, clear=True):
        model = qwen.RealtimeModel(
            api_key=KEY, transport="aoq", workspace_id="ws-1", region="ap-southeast-1"
        )
    assert model._opts.base_url == "https://ws-1.ap-southeast-1.maas.aliyuncs.com"
    assert _rt.aoq_credentials_url(model._opts.base_url, "m") == (
        "https://ws-1.ap-southeast-1.maas.aliyuncs.com/api/v1/webrtc/realtime?model=m"
    )
    assert _rt.websocket_url(model._opts.base_url, "m") == (
        "wss://ws-1.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=m"
    )


def test_requires_api_key_and_aoq_workspace() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            qwen.RealtimeModel()
        with pytest.raises(ValueError, match="workspace"):
            qwen.RealtimeModel(api_key=KEY, transport="aoq")
    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": KEY, "DASHSCOPE_WORKSPACE_ID": "ws-2"}):
        model = qwen.RealtimeModel(transport="aoq")
    assert model._opts.base_url == "https://ws-2.cn-beijing.maas.aliyuncs.com"


def test_capabilities_follow_turn_detection() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    assert model.capabilities.turn_detection
    assert model.capabilities.can_disable_turn_detection
    manual = qwen.RealtimeModel(api_key=KEY, turn_detection=None)
    assert not manual.capabilities.turn_detection
    assert not manual.capabilities.can_disable_turn_detection


# ------------------------------------------------------------------ fake transport


class FakeTransport(Transport):
    def __init__(self, *, output_audio_in_events: bool = True) -> None:
        self.output_audio_in_events = output_audio_in_events
        self.sent: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.images: list[bytes] = []
        self.inbox: asyncio.Queue[TransportMessage] = asyncio.Queue()
        self.session_updated = 0
        self.interrupted = 0
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

    def interrupt_output(self) -> None:
        self.interrupted += 1

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


async def test_aoq_response_done_may_follow_the_audio() -> None:
    model = qwen.RealtimeModel(api_key=KEY, transport="aoq", workspace_id="ws")
    transport = FakeTransport(output_audio_in_events=False)
    with patch.object(_rt, "_RESPONSE_DONE_TIMEOUT_S", 0.1):
        sess = await _start(model, transport)
        metrics: list[Any] = []
        sess.on("metrics_collected", metrics.append)
        transport.server(type="response.created", response={"id": "resp_1"})
        transport.server(type="response.output_item.added", item={"id": "i", "type": "message"})
        transport.server(type="response.output_item.done", item={"id": "i", "type": "message"})
        for _ in range(4):  # audio outlasts the timeout, which must not fire while it plays
            await asyncio.sleep(0.05)
            transport.inbox.put_nowait(OutputAudio(pcm_frame(b"\x01\x00" * 240)))
        await _settle()
        assert sess.has_active_generation
        transport.server(type="response.done", response={"id": "resp_1", "status": "completed"})
        await _settle()
        assert not sess.has_active_generation
        assert any(
            isinstance(m, RealtimeModelMetrics) and m.request_id == "resp_1" for m in metrics
        )
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
async def test_aoq_paces_video_to_video_fps_without_microphone() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport="aoq",
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True, video_fps=15),
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
        await asyncio.sleep(1 / 15)
        await _settle()
        assert len(transport.images) == 2
        assert sent_at[1] - sent_at[0] == pytest.approx(1 / 15)
        assert transport.audio == []
    await sess.aclose()


@pytest.mark.parametrize("transport_mode", ["websocket", "aoq"])
async def test_aoq_video_keeps_latest_frame_and_does_not_block_audio(transport_mode: str) -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport=transport_mode,  # type: ignore[arg-type]
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True),
    )
    transport = FakeTransport()
    sess = await _start(model, transport)
    if transport_mode == "websocket":
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
            samples = 1600 if transport_mode == "websocket" else 160
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


@pytest.mark.parametrize("transport_mode", ["websocket", "aoq"])
async def test_aoq_close_joins_encoding_and_discards_its_result(transport_mode: str) -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport=transport_mode,  # type: ignore[arg-type]
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True),
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
@pytest.mark.parametrize("transport_mode", ["websocket", "aoq"])
async def test_aoq_cancelled_shutdown_still_joins_encoding(
    cancel_target: str, transport_mode: str
) -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport=transport_mode,  # type: ignore[arg-type]
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True),
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


@pytest.mark.parametrize("transport_mode", ["websocket", "aoq"])
async def test_aoq_reconnect_discards_old_encoding_and_requires_fresh_handshake(
    transport_mode: str,
) -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport=transport_mode,  # type: ignore[arg-type]
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True),
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


async def test_aoq_unpublished_video_skips_encoding(caplog: pytest.LogCaptureFixture) -> None:
    model = qwen.RealtimeModel(api_key=KEY, transport="aoq", workspace_id="ws")
    sess = await _start(model, FakeTransport())
    with patch.object(_rt, "_encode_image") as encode_image:
        with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
            sess.push_video(_video_frame(640, 360))
            sess.push_video(_video_frame(640, 360))
        encode_image.assert_not_called()
        assert sum("publish_video=True" in record.message for record in caplog.records) == 1
    await sess.aclose()


@pytest.mark.virtual_time
async def test_aoq_bad_video_encodes_do_not_stop_later_frames() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport="aoq",
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True),
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
            await asyncio.sleep(1 / 15)
            await _settle()
        assert attempts == 3
        assert transport.images == [b"good-frame"]
    await sess.aclose()


def test_oversized_video_image_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    with patch.object(_rt, "encode", return_value=b"x" * (_rt._IMAGE_MAX_BYTES + 1)):
        with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
            assert _rt._encode_image(_video_frame(640, 360)) is None
    assert "video frame too large" in caplog.text


def test_aoq_video_fps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="video_fps"):
        qwen.RealtimeModel(
            api_key=KEY,
            transport="aoq",
            workspace_id="ws",
            aoq_options=qwen.AOQOptions(video_fps=0),
        )


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


async def test_aoq_warns_when_video_is_not_published(caplog: pytest.LogCaptureFixture) -> None:
    transport = _aoq.AOQTransport(
        credentials_url="u",
        api_key=KEY,
        http_session=None,  # type: ignore[arg-type]
        timeout=1.0,
        options=_aoq.AOQOptions(),
    )
    with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
        await transport.push_image(b"jpeg", 640, 360)
        await transport.push_image(b"jpeg", 640, 360)
    assert [r.message for r in caplog.records].count(
        "dropping video frames: the AOQ transport needs AOQOptions(publish_video=True) to send them"
    ) == 1


# ------------------------------------------------------------------ AOQ audio routing


async def test_aoq_audio_comes_from_media_track() -> None:
    model = qwen.RealtimeModel(api_key=KEY, transport="aoq", workspace_id="ws")
    transport = FakeTransport(output_audio_in_events=False)
    with patch.object(_rt, "_AOQ_AUDIO_TAIL_S", 0.05):
        sess = await _start(model, transport)
        generations: list[llm.GenerationCreatedEvent] = []
        sess.on("generation_created", generations.append)

        transport.server(type="response.created", response={"id": "resp_1"})
        # media can run ahead of the data track
        transport.inbox.put_nowait(OutputAudio(pcm_frame(b"\x01\x00" * 240)))
        transport.server(
            type="response.output_item.added", item={"id": "item_1", "type": "message"}
        )
        # duplicate audio on the data track is ignored
        transport.server(
            type="response.audio.delta",
            item_id="item_1",
            delta=base64.b64encode(b"\x09\x00" * 100).decode(),
        )
        transport.server(type="response.output_item.done", item={"id": "item_1", "type": "message"})
        transport.server(type="response.done", response={"id": "resp_1", "status": "completed"})
        # the media track still delivers after response.done
        transport.inbox.put_nowait(OutputAudio(pcm_frame(b"\x02\x00" * 240)))
        await _settle()

        _, frames = await asyncio.wait_for(_collect(generations[0]), 1)
        assert [f.samples_per_channel for f in frames] == [240, 240]

        sess.interrupt()
        assert transport.interrupted == 1
        await sess.aclose()


# ------------------------------------------------------------------ AOQ transport


def test_aoq_credentials_from_response() -> None:
    creds = _aoq.AOQCredentials.from_response(
        {
            "sid": "sid-1",
            "aoqTokenForClient": "tok",
            "clientRelayEndpoints": [{"endpoint": "1.2.3.4", "port": 443, "route_index": 0}],
            "clientRelayCertFingerprint": "fp",
            "sidExpiresInSecs": 7200,
            "extraInfo": {"workspaceIdHash": "hash"},
        }
    )
    assert (creds.sid, creds.token, creds.cert_fingerprint) == ("sid-1", "tok", "fp")
    assert creds.workspace_id_hash == "hash" and creds.expires_in_s == 7200


def _fake_sdk() -> types.ModuleType:
    """A stand-in for ``aoq_client_sdk`` with the same names, recording every call."""

    sdk = types.ModuleType("aoq_client_sdk")

    class TrackType(enum.IntEnum):
        AUDIO = 0
        VIDEO = 1
        DATA = 2

    class EncoderType(enum.IntEnum):
        AUDIO_PCM = 1
        AUDIO_OPUS = 2
        VIDEO_JPEG = 4

    class ConnectionStatus(enum.IntEnum):
        DISCONNECTED = 0
        CONNECTING = 1
        CONNECTED = 2
        FAILED = 3

    class ErrorCode(enum.IntEnum):
        OK = 0
        AUDIO_EXTERNAL_BUFFER_FULL = 110

    class AudioSource(enum.IntEnum):
        PUBLISH = 2
        PLAYBACK = 3

    @dataclass
    class Cfg:
        kwargs: dict[str, Any] = field(default_factory=dict)

    def factory(*args: Any, **kwargs: Any) -> Cfg:
        return Cfg({"args": args, **kwargs})

    @dataclass
    class AudioFrameData:
        data: bytes = b""
        num_of_samples: int = 0
        bytes_per_sample: int = 0
        num_of_channels: int = 0
        samples_per_sec: int = 0
        auto_gen_mute: bool = False

    @dataclass
    class DataMsg:
        data: bytes = b""

    class Engine:
        instance: Engine | None = None

        def __init__(self, listener: Any) -> None:
            self.listener = listener
            self.calls: list[tuple[str, tuple[Any, ...]]] = []
            self.observer: Any = None
            self.sent: list[bytes] = []
            self.pushed: list[Any] = []
            self.audio_results: list[int] = []
            self.media: list[tuple[str, float, Any]] = []
            self.published_audio = 0
            self.publish_audio = True
            self.rejected_images = 0
            self.connected = False
            self.external_streams: set[str] = set()

        @classmethod
        def create_engine(cls, cfg: Any, listener: Any) -> Engine:
            cls.instance = cls(listener)
            return cls.instance

        @classmethod
        def destroy(cls) -> int:
            if cls.instance is not None:
                cls.instance.calls.append(("destroy", ()))
            cls.instance = None
            return 0

        def disconnect(self) -> int:
            self.calls.append(("disconnect", ()))

            def disconnected() -> None:
                # like the SDK: disconnect finishes later, on its own thread
                time.sleep(0.05)
                self.calls.append(("disconnected", ()))
                self.listener.on_connection_status_change(ConnectionStatus.DISCONNECTED)

            threading.Thread(target=disconnected).start()
            return 0

        def connect(self, cfg: Any) -> int:
            self.connect_config = cfg
            self.calls.append(("connect", (cfg,)))
            # Native connect rebuilds its input pipeline; pre-connect streams are lost.
            self.external_streams.clear()

            def connected() -> None:
                self.connected = True
                self.calls.append(("connected", ()))
                self.listener.on_connection_status_change(ConnectionStatus.CONNECTED)

            threading.Thread(target=connected).start()
            return 0

        def add_audio_external_stream(self, stream_id: str, cfg: Any) -> int:
            self.calls.append(("add_audio_external_stream", (stream_id, cfg)))
            if not self.connected:
                return 99
            self.external_streams.add(stream_id)
            return 0

        def remove_audio_external_stream(self, stream_id: str) -> int:
            self.calls.append(("remove_audio_external_stream", (stream_id,)))
            self.external_streams.discard(stream_id)
            return 0

        def set_audio_frame_observer(self, observer: Any) -> int:
            self.observer = observer
            return 0

        def send_data_msg(self, msg: DataMsg) -> int:
            self.sent.append(msg.data)
            return 0

        def push_audio_external_stream_data(self, stream_id: str, frame: Any) -> int:
            assert stream_id in self.external_streams
            result = self.audio_results.pop(0) if self.audio_results else 0
            self.pushed.append(frame)
            if result == 0:
                loop = asyncio.get_running_loop()
                self.media.append(("audio", loop.time(), frame.data))
                if self.publish_audio:
                    loop.call_later(0.01, self.publish, frame)
            return result

        def publish(self, frame: Any) -> None:
            self.published_audio += 1
            if self.observer is not None:
                self.observer.on_publish_audio_frame(TrackType.AUDIO, frame)

        def push_external_video_encoded_frame(self, track_type: Any, frame: Any) -> int:
            if not self.published_audio:
                self.rejected_images += 1
                raise AssertionError("Error append image before append audio.")
            self.media.append(("image", asyncio.get_running_loop().time(), frame.kwargs["data"]))
            return 0

        def __getattr__(self, name: str) -> Any:
            def call(*args: Any) -> int:
                self.calls.append((name, args))
                return 0

            return call

    class Listener:
        pass

    class Observer:
        pass

    for name, value in {
        "AoqTrackType": TrackType,
        "AoqEncoderType": EncoderType,
        "AoqConnectionStatus": ConnectionStatus,
        "AoqErrorCode": ErrorCode,
        "AoqAudioSource": AudioSource,
        "AoqAudioFrameData": AudioFrameData,
        "AoqDataMsg": DataMsg,
        "AoqClientEngine": Engine,
        "AoqEngineEventListener": Listener,
        "IAudioFrameObserver": Observer,
    }.items():
        setattr(sdk, name, value)
    for name in (
        "AoqCreateConfig",
        "AoqAudioCodecConfig",
        "AoqAudioCaptureConfig",
        "AoqAudioPlaybackConfig",
        "AoqAudioExternalStreamConfig",
        "AoqAudioObserverConfig",
        "AoqVideoCodecConfig",
        "AoqVideoEncodedFrame",
        "AoqTrackParam",
        "AoqRelayEndpoint",
        "AoqConnectConfig",
    ):
        setattr(sdk, name, factory)
    return sdk


async def test_aoq_transport_with_fake_sdk() -> None:
    sdk = _fake_sdk()
    creds = _aoq.AOQCredentials(
        sid="sid", token="tok", cert_fingerprint="fp", relay_endpoints=[{"endpoint": "h"}]
    )

    async def fake_fetch(**kwargs: Any) -> _aoq.AOQCredentials:
        assert kwargs["url"].endswith("/api/v1/webrtc/realtime?model=m")
        return creds

    with (
        patch.dict(sys.modules, {"aoq_client_sdk": sdk}),
        patch.object(_aoq, "fetch_aoq_credentials", fake_fetch),
    ):
        transport = _aoq.AOQTransport(
            credentials_url="https://ws.cn-beijing.maas.aliyuncs.com/api/v1/webrtc/realtime?model=m",
            api_key=KEY,
            http_session=None,  # type: ignore[arg-type]
            timeout=1.0,
            options=_aoq.AOQOptions(publish_video=True),
        )
        await transport.connect()
        engine = sdk.AoqClientEngine.instance
        encoder = next(args[0] for name, args in engine.calls if name == "set_video_encoder_config")
        assert (encoder.kwargs["width"], encoder.kwargs["height"]) == (1280, 720)
        assert encoder.kwargs["fps"] == 15
        assert engine.connect_config.kwargs["token"] == "tok"
        assert engine.connect_config.kwargs["certificate"] == "fp"
        called = [name for name, _ in engine.calls]
        assert "start_audio_capture" in called and "add_audio_external_stream" in called
        # upstream audio stays off until the model is configured
        assert ("enable_send_media_stream", (sdk.AoqTrackType.AUDIO, False)) in engine.calls

        await transport.push_audio(b"\x00\x00" * 320)
        assert engine.pushed == []
        transport.on_session_updated()
        assert ("enable_send_media_stream", (sdk.AoqTrackType.AUDIO, True)) in engine.calls
        await transport.push_audio(b"\x00\x00" * 320)
        assert engine.pushed[0].samples_per_sec == 16000

        await transport.send_event({"type": "response.create"})
        assert engine.sent == [b'{"type": "response.create"}']

        # callbacks come from SDK threads
        def deliver() -> None:
            engine.listener.on_data_msg(sdk.AoqDataMsg(b'{"type": "session.updated"}'))
            engine.observer.on_playback_audio_frame(
                sdk.AoqAudioFrameData(data=b"", auto_gen_mute=True)
            )
            engine.observer.on_playback_audio_frame(
                sdk.AoqAudioFrameData(data=b"\x01\x00" * 240, samples_per_sec=24000)
            )

        threading.Thread(target=deliver).start()
        first = await asyncio.wait_for(transport.recv(), 1)
        second = await asyncio.wait_for(transport.recv(), 1)
        assert isinstance(first, ServerEvent) and first.event["type"] == "session.updated"
        assert isinstance(second, OutputAudio) and second.frame.samples_per_channel == 240

        # only one AOQ engine per process
        other = _aoq.AOQTransport(
            credentials_url="u",
            api_key=KEY,
            http_session=None,  # type: ignore[arg-type]
            timeout=1.0,
            options=_aoq.AOQOptions(),
        )
        with pytest.raises(Exception, match="single connection"):
            await other.connect()

        await transport.aclose()
        assert sdk.AoqClientEngine.instance is None
        assert isinstance(await transport.recv(), TransportClosed)


@pytest.fixture
async def aoq_sdk_transport() -> AsyncIterator[tuple[_aoq.AOQTransport, Any]]:
    sdk = _fake_sdk()

    async def fake_fetch(**kwargs: Any) -> _aoq.AOQCredentials:
        return _aoq.AOQCredentials(sid="sid", token="tok", cert_fingerprint="fp")

    transport = _aoq.AOQTransport(
        credentials_url="https://ws.test/api/v1/webrtc/realtime?model=m",
        api_key=KEY,
        http_session=None,  # type: ignore[arg-type]
        timeout=1.0,
        options=_aoq.AOQOptions(publish_video=True),
    )

    async def inline_to_thread(fn: Any, *args: Any) -> Any:
        return fn(*args)

    def connect_inline(engine: Any, config: Any) -> int:
        engine.connect_config = config
        engine.calls.append(("connect", (config,)))
        engine.external_streams.clear()
        engine.connected = True
        engine.calls.append(("connected", ()))
        engine.listener.on_connection_status_change(sdk.AoqConnectionStatus.CONNECTED)
        return 0

    with (
        patch.dict(sys.modules, {"aoq_client_sdk": sdk}),
        patch.object(_aoq, "fetch_aoq_credentials", fake_fetch),
        patch.object(_aoq.asyncio, "to_thread", inline_to_thread),
        patch.object(sdk.AoqClientEngine, "connect", connect_inline),
    ):
        await transport.connect()
        engine = sdk.AoqClientEngine.instance
        try:
            yield transport, engine
        finally:
            await transport.aclose()
            assert sdk.AoqClientEngine.instance is None


@pytest.mark.virtual_time
async def test_aoq_camera_primes_published_audio_without_microphone(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any], caplog: pytest.LogCaptureFixture
) -> None:
    transport, engine = aoq_sdk_transport
    await transport.push_image(b"before-handshake", 640, 360)
    await asyncio.sleep(0.1)
    assert engine.media == []

    transport.on_session_updated()
    # Provisioning a video track does not start filler until camera frames arrive.
    await asyncio.sleep(0.1)
    assert engine.media == []
    with caplog.at_level("INFO", logger="livekit.plugins.qwen"):
        await transport.push_image(b"camera", 640, 360)
    assert engine.rejected_images == 0
    assert engine.media[-1][0] == "image"
    audio = [(at, data) for kind, at, data in engine.media if kind == "audio"]
    assert len(audio) >= 6
    assert all(data == bytes(320) for _, data in audio)
    assert engine.media[-1][1] - audio[0][0] >= _aoq._AUDIO_VIDEO_LEAD_S
    assert all(b[0] - a[0] >= 0.01 - 1e-9 for a, b in zip(audio, audio[1:], strict=False))
    assert "input audio timeline established" in caplog.text
    assert "first video frame accepted" in caplog.text
    capture = next(args[0] for name, args in engine.calls if name == "start_audio_capture")
    assert capture.kwargs["is_external"] is True


@pytest.mark.virtual_time
async def test_aoq_camera_silence_yields_to_real_audio_and_resumes_when_muted(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> None:
    transport, engine = aoq_sdk_transport
    transport.on_session_updated()
    await transport.push_image(b"first", 640, 360)
    before_speech = len(engine.media)
    speech = b"\x01\x00" * 160
    for _ in range(10):
        await transport.push_audio(speech)
        await asyncio.sleep(0.01)
    speaking = engine.media[before_speech:]
    assert [data for kind, _, data in speaking if kind == "audio"] == [speech] * 10

    # Muting the browser microphone produces no more real PCM; video keeps a silent timeline.
    await transport.push_image(b"muted", 640, 360)
    muted_at = len(engine.media)
    await asyncio.sleep(0.08)
    assert any(kind == "audio" and data == bytes(320) for kind, _, data in engine.media[muted_at:])
    await transport.push_image(b"still-muted", 640, 360)
    assert [data for kind, _, data in engine.media if kind == "image"] == [
        b"first",
        b"muted",
        b"still-muted",
    ]
    assert engine.rejected_images == 0


@pytest.mark.virtual_time
@pytest.mark.parametrize("failure", ["buffer_full", "rejected", "not_published"])
async def test_aoq_video_requires_accepted_and_published_audio(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
    failure: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport, engine = aoq_sdk_transport
    if failure != "not_published":
        engine.audio_results = [110 if failure == "buffer_full" else 99] * 100
    else:
        engine.publish_audio = False
    transport.on_session_updated()
    start = asyncio.get_running_loop().time()
    with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
        await transport.push_image(b"blocked", 640, 360)
        assert asyncio.get_running_loop().time() - start <= _aoq._VIDEO_IDLE_S + 1e-6
        await transport.push_image(b"blocked-again", 640, 360)
    assert not any(kind == "image" for kind, _, _ in engine.media)
    assert engine.rejected_images == 0
    assert caplog.text.count("input audio has not reached the publish pipeline") == 1
    engine.audio_results = []
    engine.publish_audio = True
    await transport.push_image(b"recovered", 640, 360)
    assert [data for kind, _, data in engine.media if kind == "image"] == [b"recovered"]


@pytest.mark.virtual_time
async def test_aoq_camera_idle_stops_silence_and_requires_fresh_timeline(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> None:
    transport, engine = aoq_sdk_transport
    transport.on_session_updated()
    await transport.push_image(b"first", 640, 360)
    await asyncio.sleep(0.5)
    count = len(engine.media)
    await asyncio.sleep(0.5)
    assert len(engine.media) == count
    restarted_at = asyncio.get_running_loop().time()
    await transport.push_image(b"resumed", 640, 360)
    assert engine.media[-1][1] - restarted_at >= _aoq._AUDIO_VIDEO_LEAD_S
    assert engine.rejected_images == 0
    await transport.aclose()
    closed_count = len(engine.media)
    await asyncio.sleep(0.5)
    assert len(engine.media) == closed_count
    assert not any(task.get_name() == "qwen.aoq.audio" for task in asyncio.all_tasks())
    assert _aoq._engine_owner is None


@pytest.mark.virtual_time
async def test_aoq_audio_clock_failure_does_not_leak_native_engine(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> None:
    transport, engine = aoq_sdk_transport

    def broken_push(*args: Any) -> int:
        raise RuntimeError("native audio push failed")

    engine.push_audio_external_stream_data = broken_push
    transport.on_session_updated()
    pending_image = asyncio.create_task(transport.push_image(b"blocked", 640, 360))
    await _settle()
    closed = await transport.recv()
    assert isinstance(closed, TransportClosed)
    assert isinstance(closed.error, APIConnectionError)
    await transport.aclose()
    await pending_image
    assert _aoq._engine_owner is None
    assert any(name == "disconnect" for name, _ in engine.calls)


@pytest.mark.virtual_time
async def test_aoq_sparse_camera_frames_keep_audio_clock_running(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> None:
    transport, engine = aoq_sdk_transport
    transport._opts.video_fps = 1
    transport._video_idle_s = max(_aoq._VIDEO_IDLE_S, 2 / transport._opts.video_fps)
    transport.on_session_updated()
    await transport.push_image(b"one-fps-1", 640, 360)
    start = transport._audio_started_at
    await asyncio.sleep(1.0)
    request_at = asyncio.get_running_loop().time()
    await transport.push_image(b"one-fps-2", 640, 360)
    assert transport._audio_started_at == start
    assert engine.media[-1][1] == request_at
    accepted = [at for kind, at, _ in engine.media if kind == "audio"]
    assert max(b - a for a, b in zip(accepted, accepted[1:], strict=False)) <= 0.02 + 1e-9
    assert engine.rejected_images == 0


@pytest.mark.virtual_time
async def test_aoq_new_connection_requires_new_audio_publication(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> None:
    transport, engine = aoq_sdk_transport
    transport.on_session_updated()
    await transport.push_image(b"old-connection", 640, 360)
    assert engine.published_audio >= 6
    await transport.aclose()

    fresh = _aoq.AOQTransport(
        credentials_url="https://ws.test/api/v1/webrtc/realtime?model=m",
        api_key=KEY,
        http_session=None,  # type: ignore[arg-type]
        timeout=1.0,
        options=_aoq.AOQOptions(publish_video=True),
    )
    try:
        await fresh.connect()
        fresh_engine = fresh._sdk.AoqClientEngine.instance
        assert fresh_engine is not engine
        assert fresh._audio_started_at is None
        await fresh.push_image(b"before-handshake", 640, 360)
        assert fresh_engine.media == []
        fresh.on_session_updated()
        start = asyncio.get_running_loop().time()
        await fresh.push_image(b"fresh-connection", 640, 360)
        assert fresh_engine.media[-1][1] - start >= _aoq._AUDIO_VIDEO_LEAD_S
        assert fresh_engine.published_audio >= 6
        assert fresh_engine.rejected_images == 0
        assert [data for kind, _, data in fresh_engine.media if kind == "image"] == [
            b"fresh-connection"
        ]
    finally:
        await fresh.aclose()


async def test_aoq_external_stream_is_added_after_connected(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> None:
    transport, engine = aoq_sdk_transport
    names = [name for name, _ in engine.calls]
    assert names.index("start_audio_capture") < names.index("connect")
    assert names.index("connect") < names.index("connected")
    assert names.index("connected") < names.index("add_audio_external_stream")
    assert transport._external_stream_added
    assert _aoq._INPUT_STREAM_ID in engine.external_streams
    transport.on_session_updated()
    await transport.push_audio(b"\x01\x00" * 160)
    assert engine.media[-1][2] == b"\x01\x00" * 160
    await transport.aclose()
    assert not transport._external_stream_added
    names = [name for name, _ in engine.calls]
    assert names.index("remove_audio_external_stream") < names.index("disconnect")


async def test_aoq_teardown_stops_audio_and_waits_for_disconnect_before_destroy(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> None:
    transport, engine = aoq_sdk_transport
    transport.on_session_updated()
    await transport.aclose()
    names = [name for name, _ in engine.calls]
    # the playback thread must be stopped before the connection goes away
    assert names.index("stop_audio_player") < names.index("disconnect")
    assert names.index("stop_audio_capture") < names.index("disconnect")
    assert engine.observer is None
    # destroying during the SDK's asynchronous disconnect crashed its audio thread
    assert names.index("disconnected") < names.index("destroy")


@pytest.mark.virtual_time
@pytest.mark.parametrize("callback_kind", ["generated", "empty"])
async def test_aoq_generated_or_empty_publish_callbacks_do_not_unlock_video(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any], callback_kind: str
) -> None:
    transport, engine = aoq_sdk_transport
    engine.publish_audio = False
    transport.on_session_updated()
    pending_image = asyncio.create_task(transport.push_image(b"blocked", 640, 360))
    frame = transport._sdk.AoqAudioFrameData(
        data=bytes(320) if callback_kind == "generated" else b"",
        num_of_samples=160,
        samples_per_sec=16000,
        auto_gen_mute=callback_kind == "generated",
    )
    for _ in range(8):
        engine.observer.on_publish_audio_frame(transport._sdk.AoqTrackType.AUDIO, frame)
        await asyncio.sleep(0.01)
    await pending_image
    assert transport._accepted_pcm_frames > 0
    assert transport._published_pcm_frames == 0
    assert transport._audio_published_s == 0
    assert not any(kind == "image" for kind, _, _ in engine.media)
    if callback_kind == "generated":
        assert transport._generated_publish_frames == 8
    else:
        assert transport._empty_publish_frames == 8
    engine.publish_audio = True
    await transport.push_audio(b"\x01\x00" * 160)
    await transport.push_image(b"real-external-input", 640, 360)
    assert transport._published_pcm_frames >= 6
    assert transport._accepted_nonzero_pcm
    assert transport._published_nonzero_pcm
    assert engine.media[-1][0] == "image"


@dataclass
class _CameraProtocol:
    transport: _aoq.AOQTransport
    events: list[dict[str, Any]] = field(default_factory=list)
    auto_update: bool = True
    auto_restore: bool = True
    auto_commit: bool = True
    commit_error: bool = False
    manual: bool = False
    commits: int = 0
    manual_requested: asyncio.Event = field(default_factory=asyncio.Event)
    # set when a conversation item arrives in manual mode, which commits the media buffer
    commit_requested: asyncio.Event = field(default_factory=asyncio.Event)

    def server(self, **event: Any) -> None:
        self.transport._on_data(json.dumps(event).encode())

    def ack_update(self, detection: Any) -> None:
        # The real service omits null turn_detection in manual-mode acknowledgments.
        self.server(
            type="session.updated",
            session={} if detection is None else {"turn_detection": detection},
        )

    def ack_commit(self) -> None:
        self.commits += 1
        self.server(type="input_audio_buffer.committed", item_id=f"camera-{self.commits}")

    def send(self, message: Any) -> int:
        event = json.loads(message.data)
        self.events.append(event)
        kind = event["type"]
        if kind == "session.update":
            detection = event["session"].get("turn_detection")
            self.manual = detection is None
            if self.manual:
                self.manual_requested.set()
            if self.auto_update and (self.auto_restore or detection is None):
                self.ack_update(detection)
        elif kind == "input_audio_buffer.commit":
            # like the real service: server VAD keeps no silent audio to commit
            raise AssertionError("camera input must be committed by the typed item")
        elif kind == "conversation.item.create" and self.manual:
            # AOQ commits the pending audio and camera frames with the item
            self.commit_requested.set()
            if self.commit_error:
                self.server(type="error", error={"message": "synthetic commit failed"})
            elif self.auto_commit:
                self.ack_commit()
        elif kind == "response.create":
            self.server(type="response.created", response={"id": "camera-response"})
            self.server(
                type="response.done", response={"id": "camera-response", "status": "completed"}
            )
        return 0

    def types(self) -> list[str]:
        return [event["type"] for event in self.events]


@pytest.fixture
async def camera_session(
    aoq_sdk_transport: tuple[_aoq.AOQTransport, Any],
) -> AsyncIterator[tuple[_rt.RealtimeSession, _CameraProtocol]]:
    transport, engine = aoq_sdk_transport
    protocol = _CameraProtocol(transport)
    engine.send_data_msg = protocol.send

    async def already_connected() -> None:
        pass

    async def encode_video(frame: rtc.VideoFrame) -> _rt._Image:
        return _rt._Image(b"synthetic-jpeg", 640, 360)

    model = qwen.RealtimeModel(
        api_key=KEY,
        transport="aoq",
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )
    with (
        patch.object(transport, "connect", already_connected),
        patch.object(_rt.RealtimeSession, "_create_transport", lambda self: transport),
        patch.object(_rt.RealtimeSession, "_encode_video", lambda self, frame: encode_video(frame)),
    ):
        sess = model.session()
        await _settle()
        frame = _video_frame(640, 360)

        async def camera() -> None:
            while True:
                sess.push_video(frame)
                await asyncio.sleep(1 / 15)

        pump = asyncio.create_task(camera())
        try:
            await transport._wait_camera_input(1.0)
            yield sess, protocol
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
            try:
                await sess.aclose()
            except (APIError, AttributeError):
                # A failure test may deliberately close its fixed fake connection.
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
    assert protocol.transport._accepted_nonzero_pcm is False
    assert protocol.transport._published_nonzero_pcm is False


@pytest.mark.virtual_time
async def test_typed_camera_warns_when_the_item_commits_nothing(
    camera_session: tuple[_rt.RealtimeSession, _CameraProtocol],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sess, protocol = camera_session
    protocol.auto_commit = False
    with caplog.at_level("WARNING", logger="livekit.plugins.qwen"):
        await sess.update_chat_ctx(_camera_text(sess))
        assert "AOQ did not commit camera input with the typed message" not in caplog.messages
        await asyncio.sleep(_rt._CAMERA_COMMIT_CHECK_S + 0.1)
    assert "AOQ did not commit camera input with the typed message" in caplog.messages
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
    assert "AOQ did not commit camera input with the typed message" not in caplog.messages


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
    await asyncio.sleep(0.25)
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
    assert _aoq._engine_owner is None
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


async def test_aoq_greeting_and_text_without_camera_do_not_prepare_media() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport="aoq",
        workspace_id="ws",
        aoq_options=qwen.AOQOptions(publish_video=True),
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
