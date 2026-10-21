"""AOQ transport, built on Alibaba Cloud's AOQ client SDK (``aoq_client_sdk``).

AOQ is a MoQ-based media transport. Each connection uses one-off credentials that the
agent requests from Model Studio with its API key. Model events travel as JSON on the
data track. User audio is pushed as PCM into an external audio stream that the SDK
encodes and publishes. Model audio is decoded by the SDK and read back as PCM from its
playback observer, so it never appears as ``response.audio.delta`` events.

The SDK is not on PyPI. It ships in the Linux and macOS downloads of the AOQ client SDK
as ``aoq_client_sdk.py`` next to its native library. Point ``AOQOptions.sdk_path`` (or
``AOQ_CLIENT_SDK_PATH``) at that directory, or put it on ``PYTHONPATH``.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError

from ..log import logger
from ._transport import (
    INPUT_SAMPLE_RATE,
    NUM_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    OutputAudio,
    ServerEvent,
    Transport,
    TransportClosed,
    TransportMessage,
    pcm_frame,
)

SDK_PATH_ENV = "AOQ_CLIENT_SDK_PATH"
_INPUT_STREAM_ID = "livekit_input"
_BUFFER_FULL_RETRY_S = 0.02
_BUFFER_FULL_MAX_RETRIES = 10


@dataclass
class AOQOptions:
    """Settings for the AOQ transport."""

    sdk_path: str | None = None
    """Directory holding ``aoq_client_sdk.py`` and its native library. Defaults to the
    ``AOQ_CLIENT_SDK_PATH`` environment variable, then to a normal import."""
    work_dir: str | None = None
    """Directory for the SDK's logs and dumps. Defaults to a temporary directory."""
    client_ip: str | None = None
    """Public IP of this host, used by the gateway to pick the nearest relay. By default
    the gateway uses the IP the credentials request came from."""
    publish_video: bool = False
    """Publish a video track, so frames from ``push_video`` reach the model."""
    audio_bitrate: int = 32000
    """Opus bitrate of the published audio, in bps."""


@dataclass
class AOQCredentials:
    """One-off connection credentials returned by Model Studio."""

    sid: str
    token: str
    cert_fingerprint: str
    relay_endpoints: list[dict[str, Any]] = field(default_factory=list)
    workspace_id_hash: str = ""
    expires_in_s: int | None = None

    @classmethod
    def from_response(cls, body: dict[str, Any]) -> AOQCredentials:
        data = body if "sid" in body else body.get("output") or body.get("data") or {}
        try:
            return cls(
                sid=data["sid"],
                token=data["aoqTokenForClient"],
                cert_fingerprint=data.get("clientRelayCertFingerprint", ""),
                relay_endpoints=list(data.get("clientRelayEndpoints") or []),
                workspace_id_hash=(data.get("extraInfo") or {}).get("workspaceIdHash", ""),
                expires_in_s=data.get("sidExpiresInSecs"),
            )
        except (KeyError, TypeError) as e:
            raise APIConnectionError(
                f"unexpected AOQ credentials response: missing {e}", retryable=False
            ) from e


async def fetch_aoq_credentials(
    *,
    http_session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    client_ip: str | None,
    timeout: float,
) -> AOQCredentials:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-dashscope-rtc-transport": "moq",
        "User-Agent": "LiveKit Agents",
    }
    payload: dict[str, Any] = {"clientIp": client_ip} if client_ip else {}
    try:
        async with http_session.post(
            url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise APIStatusError(
                    "failed to get AOQ credentials",
                    status_code=resp.status,
                    request_id=resp.headers.get("x-request-id"),
                    body=text,
                    retryable=resp.status >= 500 or resp.status == 429,
                )
            body = json.loads(text)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise APIConnectionError("failed to get AOQ credentials") from e
    except json.JSONDecodeError as e:
        raise APIConnectionError("invalid AOQ credentials response", retryable=False) from e
    return AOQCredentials.from_response(body)


def load_sdk(sdk_path: str | None = None) -> ModuleType:
    path = sdk_path or os.environ.get(SDK_PATH_ENV)
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        return importlib.import_module("aoq_client_sdk")
    except ImportError as e:
        raise RuntimeError(
            "the AOQ transport needs Alibaba Cloud's AOQ client SDK (aoq_client_sdk), which "
            "is distributed with the Linux and macOS AOQ SDK downloads. Set "
            f"AOQOptions(sdk_path=...) or {SDK_PATH_ENV} to the directory that holds "
            "aoq_client_sdk.py and its native library"
        ) from e


# The SDK engine is a process-wide singleton, so only one AOQ connection can be live.
_engine_owner: AOQTransport | None = None
_engine_owner_lock = threading.Lock()


class AOQTransport(Transport):
    output_audio_in_events = False

    def __init__(
        self,
        *,
        credentials_url: str,
        api_key: str,
        http_session: aiohttp.ClientSession,
        timeout: float,
        options: AOQOptions,
    ) -> None:
        self._credentials_url = credentials_url
        self._api_key = api_key
        self._http_session = http_session
        self._timeout = timeout
        self._opts = options
        self._queue: asyncio.Queue[TransportMessage] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sdk: Any = None
        self._engine: Any = None
        self._connected_fut: asyncio.Future[None] | None = None
        self._connected = False
        self._closing = False
        self._closed_reported = False
        self._media_enabled = False

    async def connect(self) -> None:
        global _engine_owner

        self._sdk = sdk = load_sdk(self._opts.sdk_path)
        self._loop = asyncio.get_running_loop()

        with _engine_owner_lock:
            if _engine_owner is not None:
                raise APIConnectionError(
                    "the AOQ SDK supports a single connection per process, and another "
                    "Qwen-Omni-Realtime AOQ session is already connected",
                    retryable=False,
                )
            _engine_owner = self

        try:
            creds = await fetch_aoq_credentials(
                http_session=self._http_session,
                url=self._credentials_url,
                api_key=self._api_key,
                client_ip=self._opts.client_ip,
                timeout=self._timeout,
            )
            await self._start_engine(sdk, creds)
        except BaseException:
            await self.aclose()
            raise

    async def _start_engine(self, sdk: Any, creds: AOQCredentials) -> None:
        assert self._loop is not None
        self._connected_fut = self._loop.create_future()
        work_dir = self._opts.work_dir or os.path.join(tempfile.gettempdir(), "livekit-qwen-aoq")
        os.makedirs(work_dir, exist_ok=True)

        self._engine = engine = await asyncio.to_thread(
            sdk.AoqClientEngine.create_engine,
            sdk.AoqCreateConfig(work_dir=work_dir),
            self._make_listener(sdk),
        )

        # media stays off until the model acknowledges the session (see on_session_updated)
        opus = {"codec_type": sdk.AoqEncoderType.AUDIO_OPUS, "channel": NUM_CHANNELS}
        _check(
            "set_audio_encoder_config",
            engine.set_audio_encoder_config(
                sdk.AoqAudioCodecConfig(
                    **opus, sample_rate=INPUT_SAMPLE_RATE, bitrate=self._opts.audio_bitrate
                )
            ),
        )
        _check(
            "set_audio_decoder_config",
            engine.set_audio_decoder_config(
                sdk.AoqAudioCodecConfig(**opus, sample_rate=OUTPUT_SAMPLE_RATE)
            ),
        )
        _check(
            "start_audio_capture",
            engine.start_audio_capture(
                sdk.AoqAudioCaptureConfig(is_external=True, channel=NUM_CHANNELS)
            ),
        )
        _check(
            "add_audio_external_stream",
            engine.add_audio_external_stream(
                _INPUT_STREAM_ID,
                sdk.AoqAudioExternalStreamConfig(
                    codec_type=sdk.AoqEncoderType.AUDIO_PCM,
                    channels=NUM_CHANNELS,
                    sample_rate=INPUT_SAMPLE_RATE,
                    playout_volume=0,
                    publish_volume=100,
                ),
            ),
        )
        _check(
            "start_audio_player",
            engine.start_audio_player(
                sdk.AoqAudioPlaybackConfig(is_external=True, channel=NUM_CHANNELS)
            ),
        )
        _check(
            "set_audio_frame_observer", engine.set_audio_frame_observer(self._make_observer(sdk))
        )
        _check(
            "enable_audio_frame_observer",
            engine.enable_audio_frame_observer(
                True,
                sdk.AoqAudioSource.PLAYBACK,
                sdk.AoqAudioObserverConfig(sample_rate=OUTPUT_SAMPLE_RATE, channels=NUM_CHANNELS),
            ),
        )
        engine.enable_send_media_stream(sdk.AoqTrackType.AUDIO, False)

        publish = [
            sdk.AoqTrackParam(sdk.AoqTrackType.AUDIO),
            sdk.AoqTrackParam(sdk.AoqTrackType.DATA),
        ]
        if self._opts.publish_video:
            _check(
                "set_video_encoder_config",
                engine.set_video_encoder_config(
                    sdk.AoqVideoCodecConfig(
                        is_external=True, codec_type=sdk.AoqEncoderType.VIDEO_JPEG, fps=1
                    )
                ),
            )
            engine.enable_send_media_stream(sdk.AoqTrackType.VIDEO, False)
            publish.append(sdk.AoqTrackParam(sdk.AoqTrackType.VIDEO))

        relays = [
            sdk.AoqRelayEndpoint(
                endpoint=str(ep.get("endpoint", "")),
                port=int(ep.get("port") or 0),
                tcp_port=int(ep.get("tcp_port") or ep.get("tcpPort") or 0),
                route_index=int(ep.get("route_index", ep.get("routeIndex", i))),
            )
            for i, ep in enumerate(creds.relay_endpoints)
        ]
        config = sdk.AoqConnectConfig(
            token=creds.token,
            sid=creds.sid,
            certificate=creds.cert_fingerprint,
            relay_endpoints=relays,
            workspace_id_hash=creds.workspace_id_hash,
            publish_tracks=publish,
            subscribe_tracks=[
                sdk.AoqTrackParam(sdk.AoqTrackType.AUDIO),
                sdk.AoqTrackParam(sdk.AoqTrackType.DATA),
            ],
        )
        if (rc := await asyncio.to_thread(engine.connect, config)) != 0:
            raise APIConnectionError(f"AOQ connect failed with code {rc}")
        try:
            # connect() only starts the connection; the status callback reports the outcome
            await asyncio.wait_for(asyncio.shield(self._connected_fut), self._timeout)
        except asyncio.TimeoutError as e:
            raise APIConnectionError("AOQ connection timed out") from e
        logger.debug("AOQ connected", extra={"sid": creds.sid})

    # --- callbacks: these run on SDK threads and hop onto the event loop ---

    def _make_listener(self, sdk: Any) -> Any:
        transport = self

        class _Listener(sdk.AoqEngineEventListener):  # type: ignore[misc]
            def on_error(self, code: int, message: str) -> None:
                transport._threadsafe(transport._on_error, code, message)

            def on_warning(self, code: int, message: str) -> None:
                logger.debug("AOQ warning", extra={"code": code, "aoq_message": message})

            def on_connection_status_change(self, status: Any) -> None:
                transport._threadsafe(transport._on_status, int(status))

            def on_data_msg(self, msg: Any) -> None:
                transport._threadsafe(transport._on_data, msg.data)

        return _Listener()

    def _make_observer(self, sdk: Any) -> Any:
        transport = self

        class _Observer(sdk.IAudioFrameObserver):  # type: ignore[misc]
            def on_playback_audio_frame(self, data: Any) -> None:
                if data.auto_gen_mute or not data.data:
                    return  # filler the player generates while nothing is playing
                rate = data.samples_per_sec or OUTPUT_SAMPLE_RATE
                transport._threadsafe(transport._on_audio, bytes(data.data), rate)

        return _Observer()

    def _threadsafe(self, fn: Any, *args: Any) -> None:
        if self._loop is not None and not self._loop.is_closed():
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(fn, *args)

    def _on_status(self, status: int) -> None:
        sdk = self._sdk
        fut = self._connected_fut
        if status == sdk.AoqConnectionStatus.CONNECTED:
            self._connected = True
            if fut is not None and not fut.done():
                fut.set_result(None)
        elif status == sdk.AoqConnectionStatus.FAILED:
            self._fail(APIConnectionError("AOQ connection failed"))
        elif status == sdk.AoqConnectionStatus.DISCONNECTED and self._connected:
            self._fail(APIConnectionError("AOQ connection lost"))

    def _on_error(self, code: int, message: str) -> None:
        # the SDK reports only what it could not recover from itself
        self._fail(APIConnectionError(f"AOQ SDK error {code}: {message}"))

    def _on_data(self, data: bytes) -> None:
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("invalid JSON on the AOQ data track")
            return
        if isinstance(event, dict):
            self._queue.put_nowait(ServerEvent(event))

    def _on_audio(self, data: bytes, sample_rate: int) -> None:
        self._queue.put_nowait(OutputAudio(pcm_frame(data, sample_rate)))

    def _fail(self, error: Exception) -> None:
        if self._closing:
            return
        fut = self._connected_fut
        if fut is not None and not fut.done():
            fut.set_exception(error)
            return
        if not self._closed_reported:
            self._closed_reported = True
            self._queue.put_nowait(TransportClosed(error))

    # --- Transport ---

    async def send_event(self, event: dict[str, Any]) -> None:
        msg = self._sdk.AoqDataMsg(data=json.dumps(event).encode())
        if (rc := self._engine.send_data_msg(msg)) != 0:
            raise APIConnectionError(f"AOQ send_data_msg failed with code {rc}")

    def on_session_updated(self) -> None:
        if self._media_enabled or self._engine is None:
            return
        self._media_enabled = True
        self._engine.enable_send_media_stream(self._sdk.AoqTrackType.AUDIO, True)
        if self._opts.publish_video:
            self._engine.enable_send_media_stream(self._sdk.AoqTrackType.VIDEO, True)

    async def push_audio(self, pcm: bytes) -> None:
        if not self._media_enabled:
            return  # the model is not configured yet; this audio has nowhere to go
        sdk = self._sdk
        frame = sdk.AoqAudioFrameData(
            data=pcm,
            num_of_samples=len(pcm) // 2,
            bytes_per_sample=2,
            num_of_channels=NUM_CHANNELS,
            samples_per_sec=INPUT_SAMPLE_RATE,
        )
        for _ in range(_BUFFER_FULL_MAX_RETRIES):
            rc = self._engine.push_audio_external_stream_data(_INPUT_STREAM_ID, frame)
            if rc != sdk.AoqErrorCode.AUDIO_EXTERNAL_BUFFER_FULL:
                if rc != 0:
                    logger.warning("AOQ dropped input audio", extra={"code": rc})
                return
            await asyncio.sleep(_BUFFER_FULL_RETRY_S)
        logger.warning("AOQ input audio buffer full, dropping audio")

    async def push_image(self, jpeg: bytes, width: int, height: int) -> None:
        if not self._opts.publish_video or not self._media_enabled:
            return
        sdk = self._sdk
        rc = self._engine.push_external_video_encoded_frame(
            sdk.AoqTrackType.VIDEO,
            sdk.AoqVideoEncodedFrame(
                codec=sdk.AoqEncoderType.VIDEO_JPEG, data=jpeg, width=width, height=height
            ),
        )
        if rc != 0:
            logger.debug("AOQ dropped a video frame", extra={"code": rc})

    def interrupt_output(self) -> None:
        if self._engine is not None:
            self._engine.interrupt_audio_player(self._sdk.AoqTrackType.AUDIO, 0)
        # audio already handed to the loop belongs to the interrupted reply
        kept: list[TransportMessage] = []
        while not self._queue.empty():
            msg = self._queue.get_nowait()
            if not isinstance(msg, OutputAudio):
                kept.append(msg)
        for msg in kept:
            self._queue.put_nowait(msg)

    async def recv(self) -> TransportMessage:
        return await self._queue.get()

    async def aclose(self) -> None:
        global _engine_owner

        if self._closing:
            return
        self._closing = True
        self._media_enabled = False
        self._queue.put_nowait(TransportClosed())
        engine, sdk = self._engine, self._sdk
        self._engine = None
        try:
            if engine is not None:
                await asyncio.to_thread(_teardown, sdk, engine)
        finally:
            with _engine_owner_lock:
                if _engine_owner is self:
                    _engine_owner = None


def _teardown(sdk: Any, engine: Any) -> None:
    for step in (
        engine.disconnect,
        lambda: engine.set_audio_frame_observer(None),
        engine.stop_audio_player,
        lambda: engine.remove_audio_external_stream(_INPUT_STREAM_ID),
        engine.stop_audio_capture,
    ):
        try:
            step()
        except Exception:
            logger.exception("error while closing the AOQ engine")
    sdk.AoqClientEngine.destroy()


def _check(name: str, rc: int) -> None:
    if rc != 0:
        raise APIConnectionError(f"AOQ {name} failed with code {rc}", retryable=False)
