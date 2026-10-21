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
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError

from ..log import logger

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1


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

    async def send_event(self, event: dict[str, Any]) -> None:
        assert self._ws is not None
        await self._ws.send_str(json.dumps(event))

    async def push_audio(self, pcm: bytes) -> None:
        await self.send_event(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
        )

    async def push_image(self, jpeg: bytes, width: int, height: int) -> None:
        await self.send_event(
            {"type": "input_image_buffer.append", "image": base64.b64encode(jpeg).decode()}
        )

    async def recv(self) -> TransportMessage:
        assert self._ws is not None
        while True:
            msg = await self._ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
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
                logger.warning("invalid JSON from Qwen-Omni-Realtime", extra={"data": msg.data})

    async def aclose(self) -> None:
        self._closing = True
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
