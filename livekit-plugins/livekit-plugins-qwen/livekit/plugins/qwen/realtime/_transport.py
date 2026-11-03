"""Transports carrying a Qwen-Omni-Realtime session.

Both transports speak the same JSON model events. They differ in how media moves:
the WebSocket transport carries audio and images inside events, while AOQ sends
them on media tracks and keeps the data track for the events alone.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError

from ..log import logger

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1


_Result = TypeVar("_Result")


async def _join_task(task: asyncio.Task[_Result]) -> _Result:
    """Join owned work before propagating cancellation, including repeated cancellation."""
    cancelled = False
    try:
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    raise
                cancelled = True
    finally:
        if cancelled:
            raise asyncio.CancelledError


@dataclass
class ServerEvent:
    event: dict[str, Any]


@dataclass
class OutputAudio:
    """Model audio that arrived on a media track (AOQ only)."""

    frame: rtc.AudioFrame


@dataclass
class TransportClosed:
    error: Exception | None = None
    """None when the close was requested by this side."""


TransportMessage = ServerEvent | OutputAudio | TransportClosed


class Transport(ABC):
    """One connection to the model. A reconnection creates a new transport."""

    output_audio_in_events: bool
    """Whether model audio arrives as ``response.audio.delta`` events."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the connection; raises ``APIError`` on failure."""

    @abstractmethod
    async def send_event(self, event: dict[str, Any]) -> None: ...

    @abstractmethod
    async def push_audio(self, pcm: bytes) -> None:
        """Send 16 kHz mono s16le PCM."""

    @abstractmethod
    async def push_image(self, jpeg: bytes, width: int, height: int) -> None: ...

    @abstractmethod
    async def recv(self) -> TransportMessage: ...

    def on_session_updated(self) -> None:  # noqa: B027
        """Called on each ``session.updated``; the model is ready for media."""

    def interrupt_output(self) -> None:  # noqa: B027
        """Drop model audio that is buffered but not delivered yet."""

    @abstractmethod
    async def aclose(self) -> None: ...


class WebSocketTransport(Transport):
    output_audio_in_events = True

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        http_session: aiohttp.ClientSession,
        timeout: float,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._http_session = http_session
        self._timeout = timeout
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._media_enabled = False
        self._media_lock = asyncio.Lock()
        self._audio_progress = asyncio.Event()
        self._video_activity = asyncio.Event()
        self._audio_clock_task: asyncio.Task[None] | None = None
        self._clock_close_task: asyncio.Task[bool] | None = None
        self._clock_error: APIConnectionError | None = None
        self._audio_started_at: float | None = None
        self._audio_duration_s = 0.0
        self._audio_due_at = 0.0
        self._real_audio_pending = 0
        self._last_real_audio_at: float | None = None
        self._video_active_until = 0.0
        self._video_paused = False
        self._video_dirty = False
        self._last_video_accepted_at: float | None = None

    async def connect(self) -> None:
        headers = {"Authorization": f"Bearer {self._api_key}", "User-Agent": "LiveKit Agents"}
        try:
            self._ws = await asyncio.wait_for(
                self._http_session.ws_connect(self._url, headers=headers), self._timeout
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "Qwen-Omni-Realtime WebSocket handshake failed",
                status_code=e.status,
                body=e.message,
                retryable=e.status not in (401, 403),
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("Qwen-Omni-Realtime WebSocket connection error") from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError("Qwen-Omni-Realtime WebSocket connection timed out") from e

    async def _send_event_locked(self, event: dict[str, Any]) -> None:
        assert self._ws is not None
        if self._closing:
            raise APIConnectionError("Qwen-Omni-Realtime WebSocket transport closed")
        await self._ws.send_str(json.dumps(event))

    async def send_event(self, event: dict[str, Any]) -> None:
        async with self._media_lock:
            await self._send_event_locked(event)
            item = event.get("item") or {}
            if event.get("type") in ("input_audio_buffer.clear", "input_audio_buffer.commit") or (
                event.get("type") == "conversation.item.create" and item.get("role") == "user"
            ):
                # Do not let the next image overtake fresh PCM while its ACK is in flight.
                self._begin_input_turn()

    def on_session_updated(self) -> None:
        if not self._media_enabled and not self._closing:
            self._media_enabled = True
            self._audio_clock_task = asyncio.create_task(
                self._audio_clock(), name="qwen.websocket.audio"
            )
            self._audio_clock_task.add_done_callback(self._on_audio_clock_done)

    def _on_audio_clock_done(self, task: asyncio.Task[None]) -> None:
        if (
            not task.cancelled()
            and task.exception() is not None
            and not self._closing
            and self._ws is not None
        ):
            self._clock_error = APIConnectionError("Qwen-Omni-Realtime input audio clock failed")
            # Closing the socket wakes recv, so the session owns the error/reconnect path.
            self._clock_close_task = asyncio.create_task(self._ws.close())

    async def _append_audio_locked(self, pcm: bytes) -> None:
        await self._send_event_locked(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
        )
        now = asyncio.get_running_loop().time()
        if self._audio_started_at is None:
            self._audio_started_at = now
        duration = len(pcm) / (INPUT_SAMPLE_RATE * NUM_CHANNELS * 2)
        self._audio_duration_s += duration
        self._audio_due_at = max(now, self._audio_due_at) + duration
        self._audio_progress.set()

    async def push_audio(self, pcm: bytes) -> None:
        self._real_audio_pending += 1
        self._last_real_audio_at = asyncio.get_running_loop().time()
        try:
            async with self._media_lock:
                await self._append_audio_locked(pcm)
        finally:
            self._real_audio_pending -= 1

    async def _audio_clock(self) -> None:
        # This is an input timeline, not playback pacing. Model output stays unpaced.
        loop = asyncio.get_running_loop()
        while True:
            await self._video_activity.wait()
            now = loop.time()
            if now >= self._video_active_until:
                self._video_activity.clear()
                continue
            if (
                not self._real_audio_pending
                and now + 1e-6 >= self._audio_due_at
                and (self._last_real_audio_at is None or now - self._last_real_audio_at >= 0.2)
            ):
                async with self._media_lock:
                    now = loop.time()
                    if (
                        not self._closing
                        and not self._real_audio_pending
                        and now + 1e-6 >= self._audio_due_at
                        and (
                            self._last_real_audio_at is None
                            or now - self._last_real_audio_at >= 0.2
                        )
                    ):
                        await self._append_audio_locked(bytes(INPUT_SAMPLE_RATE // 10 * 2))
            # Never accumulate silence debt or send a burst after a stalled event loop.
            await asyncio.sleep(0.1)

    def _has_recent_video(self) -> bool:
        return (
            self._video_dirty
            and self._last_video_accepted_at is not None
            and asyncio.get_running_loop().time() - self._last_video_accepted_at < 2.0
        )

    def _begin_input_turn(self) -> None:
        self._audio_started_at = None
        self._audio_duration_s = 0.0
        self._audio_due_at = asyncio.get_running_loop().time()
        self._video_dirty = False
        self._audio_progress.set()

    async def _wait_camera_input(self, timeout: float) -> None:
        async def wait() -> None:
            while not self._closing:
                self._audio_progress.clear()
                if self._video_dirty and self._audio_duration_s + 1e-9 >= 0.2:
                    return
                await self._audio_progress.wait()
            raise APIConnectionError("Qwen camera transport closed during input preparation")

        await asyncio.wait_for(wait(), timeout)

    async def push_image(self, jpeg: bytes, width: int, height: int) -> None:
        if not self._media_enabled or self._closing or self._video_paused:
            return
        loop = asyncio.get_running_loop()
        self._video_active_until = loop.time() + 2.0
        self._video_activity.set()
        async with self._media_lock:
            if self._closing or self._video_paused:
                return
            # TCP ordering makes this audio append precede the image, even with no microphone.
            if self._audio_started_at is None:
                await self._append_audio_locked(bytes(INPUT_SAMPLE_RATE // 10 * 2))
            await self._send_event_locked(
                {"type": "input_image_buffer.append", "image": base64.b64encode(jpeg).decode()}
            )
            self._video_dirty = True
            self._last_video_accepted_at = loop.time()
            self._audio_progress.set()

    async def recv(self) -> TransportMessage:
        assert self._ws is not None
        while True:
            msg = await self._ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                if self._clock_error is not None:
                    return TransportClosed(self._clock_error)
                if self._closing:
                    return TransportClosed()
                return TransportClosed(
                    APIConnectionError("Qwen-Omni-Realtime connection closed unexpectedly")
                )
            if msg.type == aiohttp.WSMsgType.ERROR:
                return TransportClosed(APIConnectionError(f"WebSocket error: {msg.data}"))
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                return ServerEvent(json.loads(msg.data))
            except json.JSONDecodeError:
                logger.warning("invalid JSON from Qwen-Omni-Realtime")

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._aclose(), name="qwen.websocket.close")
        await _join_task(self._close_task)

    async def _aclose(self) -> None:
        self._closing = True
        self._media_enabled = False
        self._video_paused = False
        self._video_dirty = False
        self._video_activity.clear()
        self._audio_progress.set()
        if self._audio_clock_task is not None:
            self._audio_clock_task.cancel()
            try:
                await self._audio_clock_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("error while stopping the WebSocket input audio clock")
        if self._clock_close_task is not None:
            await self._clock_close_task
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()


def pcm_frame(data: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=NUM_CHANNELS,
        samples_per_channel=len(data) // 2,
    )


def now_ms() -> int:
    return int(time.time() * 1000)
