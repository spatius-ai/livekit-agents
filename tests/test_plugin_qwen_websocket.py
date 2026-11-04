"""Hermetic WebSocket camera input, typed turns, and unpaced output regressions."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins import qwen
from livekit.plugins.qwen.realtime import realtime_model as rt
from livekit.plugins.qwen.realtime._transport import TransportClosed, WebSocketTransport
from tests.test_plugin_qwen_realtime import KEY, _collect, _settle, _video_frame

pytestmark = pytest.mark.unit


class _Socket:
    def __init__(self) -> None:
        self.closed = False
        self.inbox: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.sent: list[tuple[float, dict[str, Any]]] = []
        self.manual = False
        self.audio_since_boundary = False
        self.pending_images = 0
        self.committed_images = 0
        self.commits = 0
        self.auto_update = True
        self.fail_audio = False

    def server(self, **event: Any) -> None:
        self.inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), ""))

    async def send_str(self, data: str) -> None:
        event = json.loads(data)
        kind = event["type"]
        if self.fail_audio and kind == "input_audio_buffer.append":
            raise RuntimeError("synthetic input socket failure")
        self.sent.append((asyncio.get_running_loop().time(), event))
        if kind == "session.update":
            detection = event["session"].get("turn_detection")
            self.manual = detection is None
            if self.auto_update:
                self.server(
                    type="session.updated",
                    session={} if self.manual else {"turn_detection": detection},
                )
        elif kind == "input_audio_buffer.append":
            self.audio_since_boundary = True
        elif kind == "input_image_buffer.append":
            assert self.audio_since_boundary, "image overtook input audio"
            self.pending_images += 1
        elif kind == "input_audio_buffer.clear":
            self.audio_since_boundary = False
            self.pending_images = 0
        elif kind == "input_audio_buffer.commit":
            self.commit()
        elif kind == "conversation.item.create" and event["item"].get("role") == "user":
            if self.manual:
                self.commit()
            self.audio_since_boundary = False
        elif kind == "response.create":
            self.server(type="response.created", response={"id": "ws-response"})
            self.server(
                type="response.output_item.added",
                item={"id": "ws-message", "type": "message", "role": "assistant"},
            )
            self.server(
                type="response.text.delta",
                item_id="ws-message",
                delta="Red" if self.committed_images else "UNKNOWN",
            )
            self.server(
                type="response.output_item.done", item={"id": "ws-message", "type": "message"}
            )
            self.server(type="response.done", response={"id": "ws-response", "status": "completed"})

    def commit(self) -> None:
        self.commits += 1
        self.committed_images = self.pending_images
        self.pending_images = 0
        self.audio_since_boundary = False
        self.server(type="input_audio_buffer.committed", item_id=f"ws-input-{self.commits}")

    async def receive(self) -> aiohttp.WSMessage:
        return await self.inbox.get()

    async def close(self) -> bool:
        if not self.closed:
            self.closed = True
            self.inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))
        return True

    def types(self) -> list[str]:
        return [event["type"] for _, event in self.sent]

    def images(self) -> list[tuple[float, bytes]]:
        return [
            (at, base64.b64decode(event["image"]))
            for at, event in self.sent
            if event["type"] == "input_image_buffer.append"
        ]


class _Http:
    def __init__(self, socket: _Socket) -> None:
        self.socket = socket

    async def ws_connect(self, *args: Any, **kwargs: Any) -> _Socket:
        return self.socket


def _transport(socket: _Socket) -> WebSocketTransport:
    return WebSocketTransport(
        url="wss://synthetic.test", api_key=KEY, http_session=_Http(socket), timeout=2.0
    )  # type: ignore[arg-type]


@pytest.fixture
async def websocket_session() -> AsyncIterator[tuple[rt.RealtimeSession, _Socket]]:
    socket = _Socket()
    transport = _transport(socket)
    model = qwen.RealtimeModel(api_key=KEY, conn_options=APIConnectOptions(max_retry=0, timeout=2))

    async def encode(frame: rtc.VideoFrame) -> rt._Image:
        return rt._Image(str(frame.width).encode(), frame.width, frame.height)

    with (
        patch.object(rt.RealtimeSession, "_create_transport", lambda self: transport),
        patch.object(rt.RealtimeSession, "_encode_video", lambda self, frame: encode(frame)),
    ):
        session = model.session()
        await _settle()
        try:
            yield session, socket
        finally:
            await session.aclose()


@pytest.mark.virtual_time
async def test_websocket_video_accepts_camera_before_microphone_and_samples_latest_at_one_fps(
    websocket_session: tuple[rt.RealtimeSession, _Socket],
) -> None:
    session, socket = websocket_session
    session.push_video(_video_frame(640, 360))
    await _settle()
    assert socket.types()[:3] == [
        "session.update",
        "input_audio_buffer.append",
        "input_image_buffer.append",
    ]
    for width in range(641, 650):
        await asyncio.sleep(0.1)
        session.push_video(_video_frame(width, 360))
        await _settle()
    assert [image for _, image in socket.images()] == [b"640"]
    await asyncio.sleep(0.1)
    await _settle()
    assert [image for _, image in socket.images()] == [b"640", b"649"]
    assert socket.images()[1][0] - socket.images()[0][0] >= 1.0
    assert session._pending_video is None


@pytest.mark.virtual_time
@pytest.mark.parametrize("boundary", ["clear", "commit", "typed"])
async def test_websocket_every_new_input_boundary_precedes_image_with_fresh_pcm(
    boundary: str,
) -> None:
    socket = _Socket()
    transport = _transport(socket)
    await transport.connect()
    try:
        await transport.push_image(b"before-ready", 640, 360)
        assert socket.sent == []
        transport.on_session_updated()
        await transport.push_image(b"one", 640, 360)
        event = (
            {"type": "conversation.item.create", "item": {"role": "user", "type": "message"}}
            if boundary == "typed"
            else {"type": f"input_audio_buffer.{boundary}"}
        )
        await transport.send_event(event)
        index = len(socket.sent)
        await transport.push_image(b"two", 640, 360)
        assert socket.types()[index:] == ["input_audio_buffer.append", "input_image_buffer.append"]
    finally:
        await transport.aclose()


@pytest.mark.virtual_time
async def test_websocket_silence_yields_to_real_audio_resumes_on_mute_and_stops_on_idle() -> None:
    socket = _Socket()
    transport = _transport(socket)
    await transport.connect()
    transport.on_session_updated()
    try:
        await asyncio.sleep(0.5)
        assert socket.sent == []
        await transport.push_image(b"camera", 640, 360)
        speech = b"\x01\x00" * 1600
        before = len(socket.sent)
        for _ in range(5):
            await transport.push_audio(speech)
            await asyncio.sleep(0.1)
        during = [
            base64.b64decode(event["audio"])
            for _, event in socket.sent[before:]
            if event["type"] == "input_audio_buffer.append"
        ]
        assert during == [speech] * 5
        before = len(socket.sent)
        await asyncio.sleep(0.3)
        assert any(
            event["type"] == "input_audio_buffer.append"
            and base64.b64decode(event["audio"]) == bytes(3200)
            for _, event in socket.sent[before:]
        )
        await asyncio.sleep(2.0)
        before = len(socket.sent)
        await asyncio.sleep(1.0)
        assert len(socket.sent) == before
    finally:
        await transport.aclose()
    assert not [task for task in asyncio.all_tasks() if task.get_name() == "qwen.websocket.audio"]


@pytest.mark.virtual_time
async def test_websocket_typed_camera_uses_manual_typed_item_without_duplicate_commit(
    websocket_session: tuple[rt.RealtimeSession, _Socket],
) -> None:
    session, socket = websocket_session

    async def camera() -> None:
        while True:
            session.push_video(_video_frame(640, 360))
            await asyncio.sleep(1 / 15)

    pump = asyncio.create_task(camera())
    try:
        await asyncio.sleep(0.3)
        context = session.chat_ctx
        context.add_message(role="user", content="What color is the synthetic camera?")
        await session.update_chat_ctx(context)
        reply = await session.generate_reply()
        answer, _ = await _collect(reply)
        assert answer == "Red"
        assert socket.commits == 1
        assert "input_audio_buffer.commit" not in socket.types()
        assert socket.manual is False
        text = socket.types().index("conversation.item.create")
        before_text = [
            event for _, event in socket.sent[:text] if event["type"] == "session.update"
        ]
        after_text = [
            event for _, event in socket.sent[text + 1 :] if event["type"] == "session.update"
        ]
        assert before_text[-1]["session"]["turn_detection"] is None
        assert after_text[0]["session"]["turn_detection"]["type"] == "server_vad"
        assert session._opts.turn_detection is not None and session.capabilities.turn_detection
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.virtual_time
async def test_websocket_audio_output_burst_is_delivered_without_playback_pacing(
    websocket_session: tuple[rt.RealtimeSession, _Socket],
) -> None:
    session, socket = websocket_session
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    socket.server(type="response.created", response={"id": "burst"})
    socket.server(
        type="response.output_item.added", item={"id": "burst-message", "type": "message"}
    )
    for _ in range(20):
        socket.server(
            type="response.audio.delta",
            item_id="burst-message",
            delta=base64.b64encode(b"\x01\x00" * 2400).decode(),
        )
    socket.server(type="response.output_item.done", item={"id": "burst-message", "type": "message"})
    socket.server(type="response.done", response={"id": "burst", "status": "completed"})
    before = asyncio.get_running_loop().time()
    await _settle()
    _, frames = await _collect(generations[0])
    assert sum(frame.duration for frame in frames) == pytest.approx(2.0)
    assert asyncio.get_running_loop().time() - before < 0.01
    assert socket.types() == ["session.update"]


@pytest.mark.virtual_time
async def test_websocket_clock_failure_wakes_receive_and_closes_cleanly() -> None:
    socket = _Socket()
    transport = _transport(socket)
    await transport.connect()
    transport.on_session_updated()
    await transport.push_image(b"camera", 640, 360)
    socket.fail_audio = True
    try:
        message = await transport.recv()
        assert isinstance(message, TransportClosed)
        assert isinstance(message.error, APIConnectionError)
    finally:
        await transport.aclose()
    assert socket.closed
    assert not [task for task in asyncio.all_tasks() if task.get_name() == "qwen.websocket.audio"]
