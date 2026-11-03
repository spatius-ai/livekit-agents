"""Tests for the Qwen-Omni-Realtime plugin: endpoints, event handling and the AOQ transport."""

from __future__ import annotations

import asyncio
import base64
import enum
import sys
import threading
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


async def test_push_video_follows_audio_and_is_paced_and_scaled() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sizes: list[tuple[int, int]] = []

    async def push_image(jpeg: bytes, width: int, height: int) -> None:
        sizes.append((width, height))
        transport.images.append(jpeg)

    transport.push_image = push_image  # type: ignore[method-assign]
    sess = await _start(model, transport)

    sess.push_video(_video_frame())  # no audio yet
    await _settle()
    assert transport.images == []

    sess.push_audio(
        rtc.AudioFrame(
            data=b"\x00\x00" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
        )
    )
    sess.push_video(_video_frame())
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


async def test_aoq_video_keeps_latest_frame_and_does_not_block_audio() -> None:
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
            sess.push_audio(
                rtc.AudioFrame(
                    data=b"\x00\x00" * 160,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=160,
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


async def test_aoq_close_joins_encoding_and_discards_its_result() -> None:
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
async def test_aoq_cancelled_shutdown_still_joins_encoding(cancel_target: str) -> None:
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


async def test_aoq_reconnect_discards_old_encoding_and_requires_fresh_handshake() -> None:
    model = qwen.RealtimeModel(
        api_key=KEY,
        transport="aoq",
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

        @classmethod
        def create_engine(cls, cfg: Any, listener: Any) -> Engine:
            cls.instance = cls(listener)
            return cls.instance

        @classmethod
        def destroy(cls) -> int:
            cls.instance = None
            return 0

        def connect(self, cfg: Any) -> int:
            self.connect_config = cfg
            threading.Thread(
                target=self.listener.on_connection_status_change,
                args=(ConnectionStatus.CONNECTED,),
            ).start()
            return 0

        def set_audio_frame_observer(self, observer: Any) -> int:
            self.observer = observer
            return 0

        def send_data_msg(self, msg: DataMsg) -> int:
            self.sent.append(msg.data)
            return 0

        def push_audio_external_stream_data(self, stream_id: str, frame: Any) -> int:
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
