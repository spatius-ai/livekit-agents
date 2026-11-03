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
    _join_task,
    pcm_frame,
)

SDK_PATH_ENV = "AOQ_CLIENT_SDK_PATH"
_INPUT_STREAM_ID = "livekit_input"
VIDEO_MAX_WIDTH = 1280
VIDEO_MAX_HEIGHT = 720
# The playback observer still delivers OUTPUT_SAMPLE_RATE. A 24 kHz decoder makes SDK 1.3.0 on Linux
# x86_64 segfault in WebRtcOpus_DecoderInit on the first frame of model audio.
_DECODE_SAMPLE_RATE = 48000
_BUFFER_FULL_RETRY_S = 0.02
_BUFFER_FULL_MAX_RETRIES = 10
# The external audio stream requires real-time 10 ms PCM, including silence while video is active.
_AUDIO_FRAME_S = 0.01
_SILENCE = bytes(INPUT_SAMPLE_RATE // 100 * NUM_CHANNELS * 2)
# Queue acceptance is not transmission. Also observe the SDK's publish pipeline and allow several
# native Opus packets to pass before submitting the first image. The SDK has no media delivery ACK.
_AUDIO_VIDEO_LEAD_S = 0.06
_AUDIO_STALE_S = 0.05
_VIDEO_IDLE_S = 0.25
# disconnect() only queues the work; destroying the engine before it finishes frees the audio
# observer while the playback thread still calls it
_DISCONNECT_TIMEOUT_S = 3.0


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
    video_fps: int = 15
    """Frame rate of the published video, as in Alibaba's AOQ real-time call example. The
    session sends at most this many frames per second. ``AgentSession`` samples video at about
    1 fps by default, so pass ``video_sampler=None`` to it to reach this rate."""
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
        # set from the SDK's thread, which the teardown thread waits on
        self._disconnected = threading.Event()
        self._external_stream_added = False
        self._closing = False
        self._closed_reported = False
        self._media_enabled = False
        self._warned_no_video = False
        self._video_paused = False
        self._video_dirty = False
        self._last_video_accepted_at: float | None = None
        self._media_lock = asyncio.Lock()
        self._audio_progress = asyncio.Event()
        self._video_activity = asyncio.Event()
        self._audio_clock_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._video_active_until = 0.0
        self._video_idle_s = max(_VIDEO_IDLE_S, 2 / self._opts.video_fps)
        self._audio_due_at = 0.0
        self._audio_started_at: float | None = None
        self._audio_published_s = 0.0
        self._last_audio_publish_at = 0.0
        self._real_audio_pending = 0
        self._last_real_audio_at: float | None = None
        self._audio_timeline_reported = False
        self._first_video_reported = False
        self._warned_audio_not_ready = False
        self._warned_audio_rejected = False
        self._accepted_pcm_frames = 0
        self._published_pcm_frames = 0
        self._generated_publish_frames = 0
        self._empty_publish_frames = 0
        self._accepted_nonzero_pcm = False
        self._published_nonzero_pcm = False

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
                sdk.AoqAudioCodecConfig(**opus, sample_rate=_DECODE_SAMPLE_RATE)
            ),
        )
        _check(
            "start_audio_capture",
            engine.start_audio_capture(
                sdk.AoqAudioCaptureConfig(is_external=True, channel=NUM_CHANNELS)
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
        if self._opts.publish_video:
            _check(
                "enable_publish_audio_frame_observer",
                engine.enable_audio_frame_observer(
                    True,
                    sdk.AoqAudioSource.PUBLISH,
                    sdk.AoqAudioObserverConfig(
                        sample_rate=INPUT_SAMPLE_RATE, channels=NUM_CHANNELS
                    ),
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
                        is_external=True,
                        codec_type=sdk.AoqEncoderType.VIDEO_JPEG,
                        width=VIDEO_MAX_WIDTH,
                        height=VIDEO_MAX_HEIGHT,
                        fps=self._opts.video_fps,
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
        # AOQ requires registering external streams after the CONNECTED callback,
        # before the caller sends its first session.update.
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
        self._external_stream_added = True
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
                if status in (sdk.AoqConnectionStatus.DISCONNECTED, sdk.AoqConnectionStatus.FAILED):
                    transport._disconnected.set()
                transport._threadsafe(transport._on_status, int(status))

            def on_data_msg(self, msg: Any) -> None:
                transport._threadsafe(transport._on_data, msg.data)

        return _Listener()

    def _make_observer(self, sdk: Any) -> Any:
        transport = self

        class _Observer(sdk.IAudioFrameObserver):  # type: ignore[misc]
            def on_publish_audio_frame(self, track_type: Any, data: Any) -> None:
                if track_type == sdk.AoqTrackType.AUDIO:
                    duration = data.num_of_samples / (data.samples_per_sec or INPUT_SAMPLE_RATE)
                    transport._threadsafe(
                        transport._on_audio_published,
                        duration,
                        bool(data.auto_gen_mute),
                        bool(data.data),
                        any(data.data),
                    )

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

    def _on_audio_published(
        self, duration: float, generated: bool, has_data: bool, nonzero: bool
    ) -> None:
        if not self._media_enabled:
            return
        if generated:
            self._generated_publish_frames += 1
        if not has_data:
            self._empty_publish_frames += 1
        # The SDK can publish auto-generated filler even when external input is not connected.
        # It is not evidence that the external PCM reached the encoder's publication pipeline.
        if generated or not has_data or self._audio_started_at is None:
            return
        self._published_pcm_frames += 1
        self._published_nonzero_pcm |= nonzero
        self._last_audio_publish_at = asyncio.get_running_loop().time()
        self._audio_published_s += min(duration, 2 * _AUDIO_FRAME_S)
        self._audio_progress.set()

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
            self._audio_clock_task = asyncio.create_task(self._audio_clock(), name="qwen.aoq.audio")
            self._audio_clock_task.add_done_callback(self._on_audio_clock_done)

    def _on_audio_clock_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            self._fail(APIConnectionError("AOQ input audio clock failed"))

    def _reset_audio_timeline(self) -> None:
        self._audio_started_at = None
        self._audio_published_s = 0.0
        self._last_audio_publish_at = 0.0
        self._audio_progress.set()

    async def _push_pcm(self, pcm: bytes, *, retry: bool) -> bool:
        sdk = self._sdk
        frame = sdk.AoqAudioFrameData(
            data=pcm,
            num_of_samples=len(pcm) // 2,
            bytes_per_sample=2,
            num_of_channels=NUM_CHANNELS,
            samples_per_sec=INPUT_SAMPLE_RATE,
        )
        attempts = _BUFFER_FULL_MAX_RETRIES if retry else 1
        for attempt in range(attempts):
            # The native calls themselves are synchronous. Do not hold the media lock during
            # retry sleeps: fresh speech, publication callbacks and shutdown must keep progressing.
            async with self._media_lock:
                if not self._media_enabled or self._closing:
                    return False
                rc = self._engine.push_audio_external_stream_data(_INPUT_STREAM_ID, frame)
                if rc == 0:
                    self._accepted_pcm_frames += 1
                    self._accepted_nonzero_pcm |= any(pcm)
                    now = asyncio.get_running_loop().time()
                    if self._audio_started_at is None or now > self._audio_due_at + _AUDIO_STALE_S:
                        self._reset_audio_timeline()
                        self._audio_started_at = now
                    self._audio_due_at = max(now, self._audio_due_at) + len(pcm) / (
                        INPUT_SAMPLE_RATE * NUM_CHANNELS * 2
                    )
                    self._audio_progress.set()
                    self._warned_audio_rejected = False
                    return True
                self._reset_audio_timeline()
            if rc != sdk.AoqErrorCode.AUDIO_EXTERNAL_BUFFER_FULL:
                if retry or not self._warned_audio_rejected:
                    self._warned_audio_rejected = True
                    logger.warning("AOQ dropped input audio", extra={"code": rc})
                return False
            if attempt + 1 < attempts:
                await asyncio.sleep(_BUFFER_FULL_RETRY_S)
        if retry:
            logger.warning("AOQ input audio buffer full, dropping audio")
        return False

    async def push_audio(self, pcm: bytes) -> None:
        if not self._media_enabled or self._closing:
            return  # the model is not configured yet; this audio has nowhere to go
        self._real_audio_pending += 1
        self._last_real_audio_at = asyncio.get_running_loop().time()
        try:
            await self._push_pcm(pcm, retry=True)
        finally:
            self._real_audio_pending -= 1

    async def _audio_clock(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._video_activity.wait()
            now = loop.time()
            if now >= self._video_active_until:
                self._video_activity.clear()
                continue
            # Real microphone PCM owns its slots. A late event loop never causes a catch-up burst.
            if (
                not self._real_audio_pending
                and now + 1e-6 >= self._audio_due_at
                and (
                    self._last_real_audio_at is None
                    or now - self._last_real_audio_at >= 2 * _AUDIO_FRAME_S
                )
            ):
                await self._push_pcm(_SILENCE, retry=False)
            await asyncio.sleep(_AUDIO_FRAME_S)

    def _audio_diagnostics(self) -> dict[str, int | bool]:
        return {
            "accepted_pcm_frames": self._accepted_pcm_frames,
            "published_pcm_frames": self._published_pcm_frames,
            "generated_publish_frames": self._generated_publish_frames,
            "empty_publish_frames": self._empty_publish_frames,
            "accepted_nonzero_pcm": self._accepted_nonzero_pcm,
            "published_nonzero_pcm": self._published_nonzero_pcm,
        }

    def _audio_ready_for_video(self, now: float) -> bool:
        return (
            self._audio_started_at is not None
            and now - self._audio_started_at >= _AUDIO_VIDEO_LEAD_S
            and self._audio_published_s + 1e-9 >= _AUDIO_VIDEO_LEAD_S
            and now - self._last_audio_publish_at <= _AUDIO_STALE_S
            and now <= self._audio_due_at + _AUDIO_FRAME_S
        )

    def _has_recent_video(self) -> bool:
        return (
            self._video_dirty
            and self._last_video_accepted_at is not None
            and asyncio.get_running_loop().time() - self._last_video_accepted_at
            < self._video_idle_s
        )

    def _begin_input_turn(self) -> None:
        self._video_dirty = False
        self._reset_audio_timeline()

    async def _wait_camera_input(self, timeout: float) -> None:
        # A temporary switch out of VAD needs fresh PCM and an image in the new manual turn.
        async def wait() -> None:
            while not self._closing:
                self._audio_progress.clear()
                if self._video_dirty and self._audio_published_s + 1e-9 >= 0.2:
                    return
                await self._audio_progress.wait()
            raise APIConnectionError("AOQ camera transport closed during input preparation")

        await asyncio.wait_for(wait(), timeout)

    async def push_image(self, jpeg: bytes, width: int, height: int) -> None:
        if not self._opts.publish_video:
            if not self._warned_no_video:
                self._warned_no_video = True
                logger.warning(
                    "dropping video frames: the AOQ transport needs "
                    "AOQOptions(publish_video=True) to send them"
                )
            return
        if not self._media_enabled or self._video_paused:
            return
        loop = asyncio.get_running_loop()
        # Frame arrival activates the silent timeline, without capturing any system microphone.
        # Camera unpublication stops new arrivals; expire the filler promptly after the last frame.
        self._video_active_until = loop.time() + self._video_idle_s
        self._video_activity.set()
        deadline = loop.time() + min(self._timeout, _VIDEO_IDLE_S)
        while self._media_enabled and not self._closing and not self._video_paused:
            self._audio_progress.clear()
            async with self._media_lock:
                now = loop.time()
                if self._audio_ready_for_video(now):
                    if not self._audio_timeline_reported:
                        assert self._audio_started_at is not None
                        self._audio_timeline_reported = True
                        logger.info(
                            "AOQ input audio timeline established for video",
                            extra={
                                "audio_lead_ms": round((now - self._audio_started_at) * 1000),
                                "published_audio_ms": round(self._audio_published_s * 1000),
                                **self._audio_diagnostics(),
                            },
                        )
                    sdk = self._sdk
                    rc = self._engine.push_external_video_encoded_frame(
                        sdk.AoqTrackType.VIDEO,
                        sdk.AoqVideoEncodedFrame(
                            codec=sdk.AoqEncoderType.VIDEO_JPEG,
                            data=jpeg,
                            width=width,
                            height=height,
                        ),
                    )
                    if rc != 0:
                        logger.warning("AOQ rejected a video frame", extra={"code": rc})
                    else:
                        self._video_dirty = True
                        self._last_video_accepted_at = now
                        self._audio_progress.set()
                        if not self._first_video_reported:
                            self._first_video_reported = True
                            logger.info("AOQ first video frame accepted by the SDK")
                    return
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(self._audio_progress.wait(), remaining)
            except asyncio.TimeoutError:
                break
        if not self._closing and not self._warned_audio_not_ready:
            self._warned_audio_not_ready = True
            logger.warning(
                "AOQ dropped a video frame: input audio has not reached the publish pipeline",
                extra=self._audio_diagnostics(),
            )

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
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._aclose(), name="qwen.aoq.close")
        await _join_task(self._close_task)

    async def _aclose(self) -> None:
        global _engine_owner

        self._closing = True
        self._media_enabled = False
        self._video_paused = False
        self._video_dirty = False
        self._video_activity.clear()
        self._video_active_until = 0.0
        self._reset_audio_timeline()
        if self._audio_clock_task is not None:
            self._audio_clock_task.cancel()
            try:
                await self._audio_clock_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("error while stopping the AOQ input audio clock")
            self._audio_clock_task = None
        self._queue.put_nowait(TransportClosed())
        engine, sdk = self._engine, self._sdk
        self._engine = None
        try:
            if engine is not None:
                await asyncio.to_thread(
                    _teardown,
                    sdk,
                    engine,
                    self._external_stream_added,
                    self._disconnected if self._connected else None,
                )
        finally:
            self._external_stream_added = False
            with _engine_owner_lock:
                if _engine_owner is self:
                    _engine_owner = None


def _teardown(
    sdk: Any, engine: Any, external_stream_added: bool, disconnected: threading.Event | None
) -> None:
    # Stop the audio threads and their observer before the connection goes, and destroy the
    # engine only once disconnect() has finished: it runs on the SDK's own thread.
    for step in (
        lambda: engine.set_audio_frame_observer(None),
        engine.stop_audio_player,
        lambda: (
            engine.remove_audio_external_stream(_INPUT_STREAM_ID) if external_stream_added else None
        ),
        engine.stop_audio_capture,
        engine.disconnect,
    ):
        try:
            step()
        except Exception:
            logger.exception("error while closing the AOQ engine")
    if disconnected is not None and not disconnected.wait(_DISCONNECT_TIMEOUT_S):
        logger.warning("AOQ disconnect did not finish, destroying the engine anyway")
    sdk.AoqClientEngine.destroy()


def _check(name: str, rc: int) -> None:
    if rc != 0:
        raise APIConnectionError(f"AOQ {name} failed with code {rc}", retryable=False)
