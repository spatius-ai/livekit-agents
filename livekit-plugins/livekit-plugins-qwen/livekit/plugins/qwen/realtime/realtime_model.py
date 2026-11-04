"""Qwen-Omni-Realtime (Alibaba Cloud Model Studio) as a LiveKit realtime model.

The protocol resembles the OpenAI Realtime beta API, with differences that keep this
plugin separate from ``livekit-plugins-openai``:

* audio goes in at 16 kHz and comes out at 24 kHz;
* conversation items can be added but not deleted or truncated, so the history is
  append-only;
* responses carry no metadata, so a reply request is matched to the next
  ``response.created``. Per-reply instructions go on ``response.create``, which is the only
  way to reply on an empty conversation, and in a temporary ``session.update``, since the
  session's instructions win when the two conflict;
* images are sent with ``input_image_buffer.append`` at about one frame per second;
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import time
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.llm.utils import compute_chat_ctx_diff
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.utils.images import EncodeOptions, ResizeOptions, encode

from ..log import logger
from ..models import (
    InputAudioTranscription,
    QwenOmniRealtimeModels,
    QwenRegion,
    TurnDetection,
)
from ._transport import (
    INPUT_SAMPLE_RATE,
    NUM_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    ServerEvent,
    TransportClosed,
    WebSocketTransport,
    _join_task,
    pcm_frame,
)

DEFAULT_MODEL: QwenOmniRealtimeModels = "qwen3.8-omni-flash-realtime"
DEFAULT_VOICE = "Tina"
DEFAULT_REGION: QwenRegion = "cn-beijing"

API_KEY_ENV = "DASHSCOPE_API_KEY"
WORKSPACE_ID_ENV = "DASHSCOPE_WORKSPACE_ID"
REGION_ENV = "DASHSCOPE_REGION"
BASE_URL_ENV = "DASHSCOPE_BASE_URL"

# endpoints that need no workspace id
_SHARED_HOSTS: dict[str, str] = {
    "cn-beijing": "dashscope.aliyuncs.com",
    "ap-southeast-1": "dashscope-intl.aliyuncs.com",
}

_WS_PATH = "/api-ws/v1/realtime"

# codes that keep failing however often the connection is retried
_FATAL_ERROR_CODES = frozenset(
    {
        "InvalidApiKey",
        "invalid_api_key",
        "Arrearage",
        "AccessDenied",
        "AccessDenied.Unpurchased",
        "ModelNotFound",
        "Model.AccessDenied",
    }
)

_IMAGE_INTERVAL_S = 1.0
_IMAGE_MAX_BYTES = 190 * 1024  # base64 must stay under 256 KB
_IMAGE_MAX_SIZE = (1280, 720)  # 720p, the recommended upper bound
_GENERATE_REPLY_TIMEOUT_S = 10.0
# how long a typed message in manual mode may take to commit the camera frames
_CAMERA_COMMIT_CHECK_S = 2.0
# Consumers must not hang if response.done fails to arrive after cancellation.
_RESPONSE_DONE_TIMEOUT_S = 3.0
_CANCEL_DONE_TIMEOUT_S = 1.0

QwenEventTypes = Literal["qwen_server_event_received", "qwen_client_event_queued"]


def build_base_url(*, region: str, workspace_id: str | None) -> str:
    """The HTTPS origin of a Model Studio endpoint."""
    if workspace_id:
        return f"https://{workspace_id}.{region}.maas.aliyuncs.com"
    if (host := _SHARED_HOSTS.get(region)) is None:
        raise ValueError(f"unknown region {region!r}, pass base_url instead")
    return f"https://{host}"


def websocket_url(base_url: str, model: str) -> str:
    parts = urlsplit(base_url if "://" in base_url else f"https://{base_url}")
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, _WS_PATH, urlencode({"model": model}), ""))


@dataclass
class _RealtimeOptions:
    model: str
    voice: str
    modalities: list[Literal["text", "audio"]]
    api_key: str
    base_url: str
    turn_detection: TurnDetection | None
    input_audio_transcription: InputAudioTranscription | None
    session_params: dict[str, Any]
    conn_options: APIConnectOptions


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        model: QwenOmniRealtimeModels | str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        api_key: str | None = None,
        region: NotGivenOr[QwenRegion | str] = NOT_GIVEN,
        workspace_id: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        enable_search: NotGivenOr[bool] = NOT_GIVEN,
        extra_session_params: dict[str, Any] | None = None,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Qwen-Omni-Realtime model.

        Args:
            model: Model name, e.g. ``"qwen3.8-omni-flash-realtime"``.
            voice: Output voice. Defaults to ``"Tina"``.
            modalities: ``["text", "audio"]`` (default) or ``["text"]``.
            api_key: Model Studio API key. Defaults to ``DASHSCOPE_API_KEY``. Keys are
                region-specific.
            region: ``"cn-beijing"`` (default) or ``"ap-southeast-1"``. Defaults to
                ``DASHSCOPE_REGION``.
            workspace_id: Model Studio workspace, which selects the workspace endpoint
                ``{workspace_id}.{region}.maas.aliyuncs.com``. Defaults to
                ``DASHSCOPE_WORKSPACE_ID``. Without one, the WebSocket transport uses the
                shared ``dashscope`` endpoint of the region.
            base_url: Endpoint origin, e.g. ``"https://ws-xxx.cn-beijing.maas.aliyuncs.com"``.
                Overrides ``region`` and ``workspace_id``. Defaults to ``DASHSCOPE_BASE_URL``.
            turn_detection: Server-side VAD. Defaults to ``TurnDetection()`` (``server_vad``);
                ``None`` hands turn taking to the agent.
            input_audio_transcription: User speech transcription. Defaults to
                ``qwen3-asr-flash-realtime``; ``None`` turns it off.
            temperature: Sampling temperature in [0, 2).
            top_p: Nucleus sampling threshold in (0, 1].
            top_k: Top-k sampling size.
            max_tokens: Maximum output tokens per reply.
            repetition_penalty: Repetition penalty.
            presence_penalty: Presence penalty in [-2, 2].
            seed: Sampling seed.
            enable_search: Let the model search the web. Cannot be combined with tools.
            extra_session_params: Further ``session.update`` fields, merged last.
            http_session: aiohttp session to reuse.
            conn_options: Connection timeout and retry policy.
        """
        api_key = api_key or os.environ.get(API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"a Model Studio API key is required: pass api_key or set {API_KEY_ENV}"
            )

        if not is_given(base_url):
            base_url = os.environ.get(BASE_URL_ENV) or NOT_GIVEN
        if not is_given(base_url):
            workspace_id = workspace_id or os.environ.get(WORKSPACE_ID_ENV) or None
            resolved_region = region if is_given(region) else None
            resolved_region = resolved_region or os.environ.get(REGION_ENV) or DEFAULT_REGION
            base_url = build_base_url(region=resolved_region, workspace_id=workspace_id)

        modalities = modalities if is_given(modalities) else ["text", "audio"]
        resolved_turn_detection = turn_detection if is_given(turn_detection) else TurnDetection()
        resolved_transcription = (
            input_audio_transcription
            if is_given(input_audio_transcription)
            else InputAudioTranscription()
        )

        session_params: dict[str, Any] = {
            k: v
            for k, v in {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "enable_search": enable_search,
            }.items()
            if is_given(v)
        }
        session_params.update(extra_session_params or {})

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=resolved_turn_detection is not None,
                can_disable_turn_detection=not is_given(turn_detection),
                user_transcription=resolved_transcription is not None,
                auto_tool_reply_generation=False,
                audio_output="audio" in modalities,
                manual_function_calls=False,
                mutable_chat_context=False,
                mutable_instructions=True,
                mutable_tools=True,
                per_response_tool_choice=False,
            )
        )

        self._opts = _RealtimeOptions(
            model=model,
            voice=voice,
            modalities=list(modalities),
            api_key=api_key,
            base_url=base_url,
            turn_detection=resolved_turn_detection,
            input_audio_transcription=resolved_transcription,
            session_params=session_params,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._http_session_owned = False
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Alibaba Cloud Model Studio"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True
        return self._http_session

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        sess = RealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    transcript: str = ""


@dataclass
class _ResponseGeneration:
    response_id: str
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    created_at: float
    messages: dict[str, _MessageGeneration] = field(default_factory=dict)
    function_call_ids: set[str] = field(default_factory=set)
    first_token_at: float | None = None
    restore_instructions: bool = False


@dataclass
class _AudioChunk:
    pcm: bytes


@dataclass
class _Image:
    jpeg: bytes
    width: int
    height: int


class _SessionUpdate(dict[str, Any]):
    """An internal update whose turn detection follows an active camera transaction."""


@dataclass
class _ScopedEvent:
    transport: WebSocketTransport
    event: dict[str, Any]


_Outgoing = dict[str, Any] | _AudioChunk | _ScopedEvent


def _encode_image(frame: rtc.VideoFrame) -> _Image | None:
    width, height = _fit(frame.width, frame.height, *_IMAGE_MAX_SIZE)
    resize = ResizeOptions(width=width, height=height, strategy="skew")
    jpeg = encode(frame, EncodeOptions(format="JPEG", resize_options=resize))
    if len(jpeg) > _IMAGE_MAX_BYTES:
        logger.warning("video frame too large for Qwen-Omni-Realtime, skipping it")
        return None
    return _Image(jpeg, width, height)


class RealtimeSession(llm.RealtimeSession[QwenEventTypes]):
    """A Qwen-Omni-Realtime session.

    Emits ``qwen_server_event_received`` with every raw server event and
    ``qwen_client_event_queued`` with every client event it sends.
    """

    def __init__(self, realtime_model: RealtimeModel, *, turn_detection_disabled: bool = False):
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = replace(
            realtime_model._opts,
            turn_detection=None if turn_detection_disabled else realtime_model._opts.turn_detection,
        )
        self._capabilities = replace(
            realtime_model.capabilities,
            turn_detection=self._opts.turn_detection is not None,
        )

        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._instructions: str | None = None
        # items the server created itself (transcribed turns, replies, function calls)
        self._server_item_ids: set[str] = set()

        self._msg_ch = utils.aio.Chan[_Outgoing]()
        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            INPUT_SAMPLE_RATE,
            NUM_CHANNELS,
            samples_per_channel=INPUT_SAMPLE_RATE // 10,
        )
        self._pushed_duration_s = 0.0
        self._image_interval_s = _IMAGE_INTERVAL_S
        # Video uses one latest-frame slot, independently of audio/control messages.
        self._pending_video: rtc.VideoFrame | None = None
        self._video_available = asyncio.Event()
        self._video_ready = False
        self._video_received = False
        self._video_generation = 0
        self._camera_turn_lock = asyncio.Lock()
        self._camera_turn_transport: WebSocketTransport | None = None
        self._camera_turn_manual = False
        self._camera_turn_failed = False
        self._camera_prepared_transport: WebSocketTransport | None = None
        self._camera_waiters: set[asyncio.Future[None]] = set()
        self._session_update_acks: deque[
            tuple[WebSocketTransport, asyncio.Future[None] | None, Any]
        ] = deque()
        self._input_commit_count = 0
        self._speech_epoch = 0
        self._committed_items: set[str] = set()
        self._speech_pending_commit = False
        self._input_committed = asyncio.Event()
        self._input_committed.set()

        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._pending_reply_instructions: deque[str | None] = deque()
        self._current_generation: _ResponseGeneration | None = None
        self._done_watchdog: asyncio.TimerHandle | None = None
        self._input_transcripts: dict[str, str] = {}
        self._speech_started_at: dict[str, float] = {}

        self._transport: WebSocketTransport | None = None
        self._closing = False
        self._main_atask = asyncio.create_task(self._main_task(), name="qwen.RealtimeSession")

    # ------------------------------------------------------------------ properties

    @property
    def capabilities(self) -> llm.RealtimeCapabilities:
        return self._capabilities

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    @property
    def has_active_generation(self) -> bool:
        return self._current_generation is not None or bool(self._response_created_futures)

    # ------------------------------------------------------------------ session config

    def _session_config(self, *, instructions: str | None = None) -> dict[str, Any]:
        opts = self._opts
        session: dict[str, Any] = {
            "modalities": opts.modalities,
            "voice": opts.voice,
            "audio": {
                "input": {"format": {"type": "pcm", "sample_rate": INPUT_SAMPLE_RATE}},
                "output": {
                    "format": {"type": "pcm", "sample_rate": OUTPUT_SAMPLE_RATE},
                    "voice": opts.voice,
                },
            },
            "turn_detection": opts.turn_detection.to_dict() if opts.turn_detection else None,
            "input_audio_transcription": (
                opts.input_audio_transcription.to_dict() if opts.input_audio_transcription else None
            ),
        }
        instructions = instructions if instructions is not None else self._instructions
        if instructions:
            session["instructions"] = instructions
        tools = self._qwen_tools()
        if tools:
            session["tools"] = tools
        session.update(opts.session_params)
        if self._camera_turn_manual and self._transport is self._camera_turn_transport:
            session["turn_detection"] = None
        return session

    def _session_update(self, *, instructions: str | None = None) -> dict[str, Any]:
        return _SessionUpdate(
            {
                "event_id": utils.shortuuid("session_update_"),
                "type": "session.update",
                "session": self._session_config(instructions=instructions),
            }
        )

    def _qwen_tools(self) -> list[dict[str, Any]]:
        return self._tools.parse_function_tools("openai", strict=False)

    def send_event(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        self.send_event(self._session_update())

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        function_tools: list[llm.Tool] = []
        for tool in tools:
            if isinstance(tool, (llm.FunctionTool, llm.RawFunctionTool)):
                function_tools.append(tool)
            else:
                logger.warning(
                    "Qwen-Omni-Realtime supports function tools only, ignoring tool",
                    extra={"tool": tool},
                )
        self._tools = llm.ToolContext(function_tools)
        if function_tools and self._opts.session_params.get("enable_search"):
            logger.warning("Qwen-Omni-Realtime cannot use tools and enable_search together")
        self.send_event(self._session_update())

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if is_given(tool_choice):
            logger.debug("Qwen-Omni-Realtime does not support tool_choice, ignoring it")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        diff = compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
        skipped = 0
        has_user_text = any(
            item is not None
            and item.type == "message"
            and item.role == "user"
            and bool(item.text_content)
            and item_id not in self._server_item_ids
            for _, item_id in diff.to_create
            for item in [chat_ctx.get_by_id(item_id)]
        )
        events: list[dict[str, Any]] = []
        for _, item_id in diff.to_create:
            item = chat_ctx.get_by_id(item_id)
            if item is None or item_id in self._server_item_ids:
                continue
            if item.type == "function_call_output":
                events.append(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": item.output,
                        },
                    }
                )
            elif item.type == "message" and item.role == "user" and item.text_content:
                events.append(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": item.text_content}],
                        },
                    }
                )
            elif item.type == "message" and item.role in ("system", "developer"):
                continue  # carried by the session instructions
            elif item.type in ("message", "function_call"):
                skipped += 1
        # the camera path sends what it can itself and leaves the rest in `pending`
        pending = list(events)
        prepared: WebSocketTransport | None = None
        if has_user_text:
            try:
                prepared = await self._send_camera_text(pending)
            except APIError as e:
                # the typed message matters more than the camera context it would carry
                logger.warning(
                    "could not commit camera input with typed input, sending it without",
                    extra={"error": str(e)},
                )
        for event in pending:
            if prepared is not None:
                self._msg_ch.send_nowait(_ScopedEvent(prepared, event))
            else:
                self.send_event(event)
        if skipped:
            logger.debug(
                "Qwen-Omni-Realtime accepts only user text and tool results as history, "
                f"skipped {skipped} item(s)"
            )
        if diff.to_remove:
            logger.debug(
                "Qwen-Omni-Realtime history is append-only, "
                f"{len(diff.to_remove)} removed item(s) stay on the server"
            )
        self._chat_ctx = chat_ctx.copy()

    def _camera_transport_current(self, transport: WebSocketTransport) -> bool:
        return self._transport is transport and not self._closing and not transport._closing

    async def _camera_wait(self, future: asyncio.Future[None]) -> None:
        self._camera_waiters.add(future)
        try:
            await asyncio.wait_for(asyncio.shield(future), self._opts.conn_options.timeout)
        except asyncio.TimeoutError:
            raise APIConnectionError("Qwen camera input preparation timed out") from None
        finally:
            self._camera_waiters.discard(future)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()

    def _fail_camera_waiters(self) -> None:
        if self._camera_turn_transport is not None:
            self._camera_turn_failed = True
        for future in self._camera_waiters:
            if not future.done():
                future.set_exception(
                    APIConnectionError("Qwen camera input preparation interrupted")
                )
        self._input_committed.set()

    async def _camera_update(self, transport: WebSocketTransport) -> None:
        """Send a session.update for the current camera mode and wait for its ack."""
        if not self._camera_transport_current(transport):
            raise APIConnectionError("Qwen camera transport changed during input preparation")
        future = asyncio.get_running_loop().create_future()
        self._camera_waiters.add(future)
        try:
            await self._send(transport, self._session_update(), session_ack=future)
            await self._camera_wait(future)
            if not self._camera_transport_current(transport):
                raise APIConnectionError("Qwen camera transport changed during input preparation")
        finally:
            self._camera_waiters.discard(future)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()

    async def _restore_camera_vad(self, transport: WebSocketTransport) -> None:
        self._camera_turn_manual = False
        await self._camera_update(transport)

    async def _restore_camera_or_close(self, transport: WebSocketTransport) -> None:
        try:
            await self._restore_camera_vad(transport)
        except Exception:
            await transport.aclose()
            raise

    def _check_camera_commit(self, transport: WebSocketTransport, commits: int) -> None:
        if self._input_commit_count == commits and self._camera_transport_current(transport):
            logger.warning(
                "Qwen did not commit camera input with the typed message",
            )

    async def _wait_spoken_turn(self, transport: WebSocketTransport) -> None:
        try:
            await asyncio.wait_for(self._input_committed.wait(), self._opts.conn_options.timeout)
        except asyncio.TimeoutError:
            raise APIConnectionError("Qwen speech turn did not commit before typed input") from None
        if not self._camera_transport_current(transport) or self._speech_pending_commit:
            raise APIConnectionError("Qwen speech turn was interrupted before committing")

    async def _send_camera_text(self, events: list[dict[str, Any]]) -> WebSocketTransport | None:
        """Make typed input carry the camera, sending (and removing) what it can from ``events``.

        Server VAD keeps no silent audio, so a commit fails while it is on. In manual mode Qwen
        commits the pending audio and camera frames when a conversation item is created, so the
        events are sent there, before VAD is restored. When a spoken turn is in flight its VAD
        commit carries the camera instead and ``events`` are left to the caller. Returns the
        transport later events of this turn must go to, or None when there is no camera.
        """
        async with self._camera_turn_lock:
            transport = self._transport
            if not isinstance(transport, WebSocketTransport):
                return None
            if not transport._has_recent_video():
                if not self._video_received or transport._last_video_accepted_at is not None:
                    return None
                try:
                    await transport._wait_camera_input(self._opts.conn_options.timeout)
                except asyncio.TimeoutError:
                    raise APIConnectionError("Qwen first camera frame was not published") from None
            if self._speech_pending_commit:
                # Keep audio and video flowing until VAD commits the complete spoken turn.
                await self._wait_spoken_turn(transport)
                self._camera_prepared_transport = transport
                return transport

            baseline = self._session_config().get("turn_detection")
            had_vad = baseline is not None
            restored = not had_vad
            self._camera_turn_transport = transport
            self._camera_turn_failed = False
            speech_epoch = self._speech_epoch
            try:
                if had_vad:
                    self._camera_turn_manual = True
                    await self._camera_update(transport)
                    if self._speech_epoch != speech_epoch:
                        await self._restore_camera_vad(transport)
                        restored = True
                        await self._wait_spoken_turn(transport)
                        self._camera_prepared_transport = transport
                        return transport
                    transport._begin_input_turn()
                try:
                    await transport._wait_camera_input(self._opts.conn_options.timeout)
                except asyncio.TimeoutError:
                    raise APIConnectionError("Qwen fresh camera input was not published") from None
                if self._camera_turn_failed or not self._camera_transport_current(transport):
                    raise APIConnectionError("Qwen camera input preparation interrupted")
                if self._speech_epoch != speech_epoch:
                    if had_vad:
                        await self._restore_camera_vad(transport)
                        restored = True
                    await self._wait_spoken_turn(transport)
                    self._camera_prepared_transport = transport
                    return transport
                transport._video_paused = True
                commits = self._input_commit_count
                while events:
                    await self._send(transport, events[0])
                    events.pop(0)
                if had_vad:
                    await self._restore_camera_vad(transport)
                    restored = True
                # the commit can arrive after the restore's ack; check without delaying the reply
                asyncio.get_running_loop().call_later(
                    _CAMERA_COMMIT_CHECK_S, self._check_camera_commit, transport, commits
                )
                self._camera_prepared_transport = transport
                return transport
            finally:
                try:
                    if had_vad and not restored and self._camera_transport_current(transport):
                        # Restoration is owned cleanup: repeated caller cancellation cannot abandon it.
                        task = asyncio.create_task(self._restore_camera_or_close(transport))
                        await _join_task(task)
                finally:
                    transport._video_paused = False
                    self._camera_turn_manual = False
                    self._camera_turn_transport = None

    # ------------------------------------------------------------------ input

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                self._send_audio(nf)

    def _send_audio(self, frame: rtc.AudioFrame) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(_AudioChunk(bytes(frame.data)))
        self._pushed_duration_s += frame.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        if self._closing:
            return
        if self._video_ready:
            self._video_received = True
            self._pending_video = frame
            self._video_available.set()

    def _reset_video_input(self) -> None:
        self._video_ready = False
        self._video_received = False
        self._video_generation += 1
        self._pending_video = None
        self._video_available.clear()

    async def _encode_video(self, frame: rtc.VideoFrame) -> _Image | None:
        # Cancelling to_thread does not stop its thread. Join the encode before another
        # transport starts, and never let a completed encode send on its own.
        task = asyncio.create_task(asyncio.to_thread(_encode_image, frame))
        return await _join_task(task)

    async def _video_task(self, transport: WebSocketTransport) -> None:
        loop = asyncio.get_running_loop()
        next_send_at = 0.0
        while True:
            await self._video_available.wait()
            delay = next_send_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            # Sample the newest raw frame at the send boundary, before encoding it off-loop.
            frame = self._pending_video
            self._pending_video = None
            self._video_available.clear()
            if frame is None or not self._video_ready:
                continue
            generation = self._video_generation
            try:
                image = await self._encode_video(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("failed to encode a Qwen video frame, skipping it")
                image = None
            if (
                self._closing
                or not self._video_ready
                or generation != self._video_generation
                or self._transport is not transport
            ):
                continue
            # Pace unsuccessful encodes too, so a stream of large frames cannot monopolize CPU.
            if image is None:
                next_send_at = loop.time() + self._image_interval_s
                continue
            try:
                await transport.push_image(image.jpeg, image.width, image.height)
            except APIError:
                raise
            except Exception:
                logger.exception("failed to send a Qwen video frame, skipping it")
            finally:
                next_send_at = loop.time() + self._image_interval_s

    def commit_audio(self) -> None:
        for nf in self._bstream.flush():
            self._send_audio(nf)
        if self._pushed_duration_s > 0.1:
            self.send_event({"type": "input_audio_buffer.commit"})
            self._pushed_duration_s = 0

    def clear_audio(self) -> None:
        self._bstream.clear()
        self.send_event({"type": "input_audio_buffer.clear"})
        self._pushed_duration_s = 0

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler and frame.sample_rate != self._input_resampler._input_rate:
            self._input_resampler = None
        if self._input_resampler is None and (
            frame.sample_rate != INPUT_SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    # ------------------------------------------------------------------ replies

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        prepared = self._camera_prepared_transport
        self._camera_prepared_transport = None
        if prepared is not None and not self._camera_transport_current(prepared):
            failed = asyncio.Future[llm.GenerationCreatedEvent]()
            failed.set_exception(
                llm.RealtimeError("Qwen camera transport changed before typed reply")
            )
            return failed
        if is_given(tools) or is_given(tool_choice):
            logger.debug("Qwen-Omni-Realtime does not support per-reply tools, ignoring them")

        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut

        reply_instructions: str | None = None
        if is_given(instructions) and instructions:
            # widen the session instructions for this reply, restored once it is done
            reply_instructions = (
                f"{self._instructions}\n{instructions}" if self._instructions else instructions
            )
            self.send_event(self._session_update(instructions=reply_instructions))
        self._pending_reply_instructions.append(reply_instructions)
        create: dict[str, Any] = {"event_id": event_id, "type": "response.create"}
        if reply_instructions:
            create["response"] = {"instructions": reply_instructions}
        if prepared is not None:
            self._msg_ch.send_nowait(_ScopedEvent(prepared, create))
        else:
            self.send_event(create)

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_event_loop().call_later(_GENERATE_REPLY_TIMEOUT_S, _on_timeout)

        def _on_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            handle.cancel()
            self._response_created_futures.pop(event_id, None)
            if f.cancelled():
                self.send_event({"type": "response.cancel"})

        fut.add_done_callback(_on_done)
        return fut

    def interrupt(self) -> None:
        if self._current_generation is not None:
            self.send_event({"type": "response.cancel"})
            self._arm_done_watchdog(_CANCEL_DONE_TIMEOUT_S)

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        # Qwen-Omni-Realtime cannot truncate items; keep the local mirror close to
        # what the user actually heard
        if is_given(audio_transcript) and (item := self._chat_ctx.get_by_id(message_id)):
            if item.type == "message":
                item.content = [audio_transcript]

    async def aclose(self) -> None:
        self._closing = True
        self._fail_camera_waiters()
        self._reset_video_input()
        self._cancel_done_watchdog()
        self._close_current_generation()
        self._msg_ch.close()
        await _join_task(self._main_atask)

    # ------------------------------------------------------------------ connection

    def _create_transport(self) -> WebSocketTransport:
        opts = self._opts
        http_session = self._realtime_model._ensure_http_session()
        return WebSocketTransport(
            url=websocket_url(opts.base_url, opts.model),
            api_key=opts.api_key,
            http_session=http_session,
            timeout=opts.conn_options.timeout,
        )

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        max_retries = self._opts.conn_options.max_retry
        num_retries = 0
        connected_once = False
        try:
            while not self._msg_ch.closed:
                transport = self._create_transport()
                self._transport = transport
                try:
                    t0 = time.perf_counter()
                    await transport.connect()
                    self._report_connection_acquired(time.perf_counter() - t0)
                    await self._send(transport, self._session_update())
                    if connected_once:
                        self._on_reconnected()
                    connected_once = True
                    num_retries = 0
                    await self._run(transport)
                except APIError as e:
                    if max_retries == 0 or not e.retryable:
                        self._emit_error(e, recoverable=False)
                        raise
                    if num_retries == max_retries:
                        self._emit_error(e, recoverable=False)
                        raise APIError(
                            f"Qwen-Omni-Realtime connection failed after {num_retries} attempts",
                            retryable=False,
                        ) from e
                    self._emit_error(e, recoverable=True)
                    interval = self._opts.conn_options._interval_for_retry(num_retries)
                    logger.warning(
                        f"Qwen-Omni-Realtime connection failed, retrying in {interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    num_retries += 1
                    await asyncio.sleep(interval)
                except Exception as e:
                    self._emit_error(e, recoverable=False)
                    raise
                finally:
                    self._fail_camera_waiters()
                    self._session_update_acks.clear()
                    self._speech_pending_commit = False
                    self._reset_video_input()
                    await transport.aclose()
                    self._transport = None
        finally:
            self._cancel_done_watchdog()
            self._close_current_generation()
            for fut in self._response_created_futures.values():
                if not fut.done():
                    fut.set_exception(llm.RealtimeError("realtime session closed"))
            self._response_created_futures.clear()

    def _on_reconnected(self) -> None:
        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(llm.RealtimeError("pending reply discarded by reconnection"))
        self._response_created_futures.clear()
        self._pending_reply_instructions.clear()
        self._close_current_generation()
        self._input_transcripts.clear()
        self._speech_started_at.clear()
        if self._chat_ctx.items:
            logger.warning(
                "Qwen-Omni-Realtime reconnected; the conversation history is not restored"
            )
        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    async def _send(
        self,
        transport: WebSocketTransport,
        msg: _Outgoing,
        *,
        session_ack: asyncio.Future[None] | None = None,
    ) -> None:
        if isinstance(msg, _ScopedEvent):
            if (
                msg.transport is not transport
                or self._transport is not msg.transport
                or self._closing
                or getattr(transport, "_closing", False)
            ):
                event_id = msg.event.get("event_id")
                future = self._response_created_futures.get(event_id or "")
                if future is not None and not future.done():
                    future.set_exception(llm.RealtimeError("Qwen typed reply transport changed"))
                return
            msg = msg.event
        if isinstance(msg, _AudioChunk):
            await transport.push_audio(msg.pcm)
        else:
            if isinstance(msg, _SessionUpdate):
                msg = {**msg, "session": {**msg["session"]}}
                # Internal updates queued during the transaction must use its current mode,
                # including updates to tools, instructions, and per-reply instructions.
                baseline = self._opts.session_params.get(
                    "turn_detection",
                    self._opts.turn_detection.to_dict() if self._opts.turn_detection else None,
                )
                msg["session"]["turn_detection"] = (
                    None
                    if self._camera_turn_manual and transport is self._camera_turn_transport
                    else baseline
                )
            if msg.get("type") == "session.update":
                self._session_update_acks.append(
                    (transport, session_ack, (msg.get("session") or {}).get("turn_detection"))
                )
            self.emit("qwen_client_event_queued", msg)
            await transport.send_event(msg)

    async def _run(self, transport: WebSocketTransport) -> None:
        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            async for msg in self._msg_ch:
                try:
                    await self._send(transport, msg)
                except APIError:
                    raise
                except Exception:
                    logger.exception("failed to send event to Qwen-Omni-Realtime")

        async def _recv_task() -> None:
            while True:
                msg = await transport.recv()
                if isinstance(msg, TransportClosed):
                    if msg.error is not None:
                        raise msg.error
                    return
                if self._closing:
                    continue
                if isinstance(msg, ServerEvent):
                    self._handle_event(msg.event)

        tasks = [
            asyncio.create_task(_send_task(), name="qwen.send"),
            asyncio.create_task(_recv_task(), name="qwen.recv"),
            asyncio.create_task(self._video_task(transport), name="qwen.video"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            self._reset_video_input()
            cleanup = asyncio.create_task(utils.aio.cancel_and_wait(*tasks), name="qwen.cleanup")
            await _join_task(cleanup)

    # ------------------------------------------------------------------ server events

    def _handle_event(self, event: dict[str, Any]) -> None:
        self.emit("qwen_server_event_received", event)
        event_type = event.get("type")
        try:
            if event_type == "session.updated":
                if self._session_update_acks:
                    transport, future, expected = self._session_update_acks.popleft()
                    if future is not None and not future.done():
                        actual = (event.get("session") or {}).get("turn_detection")
                        if transport is not self._transport:
                            future.set_exception(
                                APIConnectionError("Qwen session update transport changed")
                            )
                        elif (expected is None) != (actual is None):
                            future.set_exception(
                                APIConnectionError(
                                    "Qwen turn detection update was not acknowledged"
                                )
                            )
                        else:
                            future.set_result(None)
                if self._transport is not None:
                    self._transport.on_session_updated()
                    self._video_ready = True
            elif event_type == "error":
                self._handle_error(event)
            elif event_type == "input_audio_buffer.speech_started":
                self._handle_speech_started(event)
            elif event_type == "input_audio_buffer.committed":
                item_id = event.get("item_id")
                if item_id and item_id in self._committed_items:
                    return
                if item_id:
                    self._committed_items.add(item_id)
                self._speech_pending_commit = False
                self._input_committed.set()
                self._pushed_duration_s = 0
                if isinstance(self._transport, WebSocketTransport):
                    self._transport._begin_input_turn()
                self._input_commit_count += 1
            elif event_type == "input_audio_buffer.speech_stopped":
                self.emit(
                    "input_speech_stopped",
                    llm.InputSpeechStoppedEvent(
                        user_transcription_enabled=self._opts.input_audio_transcription is not None
                    ),
                )
            elif event_type == "conversation.item.input_audio_transcription.delta":
                self._handle_transcription_delta(event)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self._handle_transcription_completed(event)
            elif event_type == "conversation.item.input_audio_transcription.failed":
                self._handle_transcription_failed(event)
            elif event_type == "response.created":
                self._handle_response_created(event)
            elif event_type == "response.output_item.added":
                item = event.get("item") or {}
                self._cancel_done_watchdog()
                if item.get("type") == "message" and item.get("id"):
                    self._message(item["id"])
            elif event_type == "response.content_part.added":
                self._handle_content_part_added(event)
            elif event_type in ("response.audio_transcript.delta", "response.text.delta"):
                self._handle_text_delta(event)
            elif event_type == "response.audio.delta":
                self._handle_audio_delta(event)
            elif event_type == "response.output_item.done":
                self._handle_output_item_done(event)
            elif event_type == "response.function_call_arguments.done":
                self._handle_function_call(event)
            elif event_type == "response.done":
                self._handle_response_done(event)
        except APIError:
            raise
        except Exception:
            logger.exception(
                "failed to handle Qwen-Omni-Realtime event", extra={"event_type": event_type}
            )

    def _handle_error(self, event: dict[str, Any]) -> None:
        if self._camera_turn_transport is not None:
            self._fail_camera_waiters()
        error = event.get("error") or {}
        code = error.get("code")
        message = error.get("message") or "unknown error"
        fatal = code in _FATAL_ERROR_CODES
        if "create response" in message:
            self._reject_oldest_reply(message, code)
        logger.error(
            f"Qwen-Omni-Realtime returned an error: {message}",
            extra={"error": error},
        )
        err = APIError(f"Qwen-Omni-Realtime error: {message}", body=error, retryable=not fatal)
        if fatal:
            raise err
        self._emit_error(err, recoverable=True)

    def _reject_oldest_reply(self, message: str, code: str | None) -> None:
        # errors name no client event: this one answers the oldest reply request
        if self._pending_reply_instructions and self._pending_reply_instructions.popleft():
            self.send_event(self._session_update())
        for event_id, fut in list(self._response_created_futures.items()):
            self._response_created_futures.pop(event_id, None)
            if not fut.done():
                fut.set_exception(llm.RealtimeError(message, code=code))
                break

    def _handle_speech_started(self, event: dict[str, Any]) -> None:
        self._speech_epoch += 1
        self._speech_pending_commit = True
        self._input_committed.clear()
        if item_id := event.get("item_id"):
            self._speech_started_at[item_id] = time.time()
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_transcription_delta(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        if not item_id:
            return
        # `text` is the settled prefix, `stash` the part still being revised
        transcript = (event.get("text") or "") + (event.get("stash") or "")
        if not transcript and (delta := event.get("delta")):
            transcript = self._input_transcripts.get(item_id, "") + delta
        if not transcript:
            return
        self._input_transcripts[item_id] = transcript
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=transcript, is_final=False),
        )

    def _handle_transcription_completed(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        if not item_id:
            return
        self._input_transcripts.pop(item_id, None)
        transcript = event.get("transcript") or ""
        self._server_item_ids.add(item_id)
        if transcript and self._chat_ctx.get_by_id(item_id) is None:
            self._chat_ctx.add_message(role="user", content=transcript, id=item_id)
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=True,
                turn_started_at=self._speech_started_at.pop(item_id, None),
            ),
        )

    def _handle_transcription_failed(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id") or ""
        logger.error(
            "Qwen-Omni-Realtime failed to transcribe input audio",
            extra={"error": event.get("error")},
        )
        partial = self._input_transcripts.pop(item_id, None)
        started_at = self._speech_started_at.pop(item_id, None)
        if partial is None:
            return
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id, transcript=partial, is_final=True, turn_started_at=started_at
            ),
        )

    def _handle_response_created(self, event: dict[str, Any]) -> None:
        response = event.get("response") or {}
        response_id = response.get("id") or utils.shortuuid("response_")
        self._cancel_done_watchdog()
        self._close_current_generation()
        generation = _ResponseGeneration(
            response_id=response_id,
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            created_at=time.time(),
        )
        self._current_generation = generation
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=generation.message_ch,
            function_stream=generation.function_ch,
            user_initiated=False,
            response_id=response_id,
        )

        # responses carry no client id: credit this one to the oldest reply request
        if self._pending_reply_instructions:
            generation.restore_instructions = self._pending_reply_instructions.popleft() is not None
        for event_id, fut in list(self._response_created_futures.items()):
            self._response_created_futures.pop(event_id, None)
            if not fut.done():
                generation_ev.user_initiated = True
                fut.set_result(generation_ev)
                break

        self.emit("generation_created", generation_ev)

    def _message(self, item_id: str) -> _MessageGeneration | None:
        generation = self._current_generation
        if generation is None:
            return None
        if (msg := generation.messages.get(item_id)) is not None:
            return msg

        msg = _MessageGeneration(
            message_id=item_id,
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
            modalities=asyncio.Future(),
        )
        if not self._capabilities.audio_output:
            msg.audio_ch.close()
            msg.modalities.set_result(["text"])
        generation.messages[item_id] = msg
        self._server_item_ids.add(item_id)
        generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=item_id,
                text_stream=msg.text_ch,
                audio_stream=msg.audio_ch,
                modalities=msg.modalities,
            )
        )
        return msg

    def _handle_content_part_added(self, event: dict[str, Any]) -> None:
        if not (item_id := event.get("item_id")) or not (msg := self._message(item_id)):
            return
        part_type = (event.get("part") or {}).get("type")
        if not msg.modalities.done():
            msg.modalities.set_result(["audio", "text"] if part_type == "audio" else ["text"])

    def _handle_text_delta(self, event: dict[str, Any]) -> None:
        if not (item_id := event.get("item_id")) or not (msg := self._message(item_id)):
            return
        delta = event.get("delta") or ""
        generation = self._current_generation
        if generation and generation.first_token_at is None and msg.audio_ch.closed:
            generation.first_token_at = time.time()
        msg.transcript += delta
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            msg.text_ch.send_nowait(delta)

    def _handle_audio_delta(self, event: dict[str, Any]) -> None:
        if not (item_id := event.get("item_id")) or not (msg := self._message(item_id)):
            return
        self._push_output_audio(msg, pcm_frame(base64.b64decode(event.get("delta") or "")))

    def _push_output_audio(self, msg: _MessageGeneration, frame: rtc.AudioFrame) -> None:
        generation = self._current_generation
        if generation is not None and generation.first_token_at is None:
            generation.first_token_at = time.time()
        if not msg.modalities.done():
            msg.modalities.set_result(["audio", "text"])
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            msg.audio_ch.send_nowait(frame)

    def _arm_done_watchdog(self, timeout: float | None = None) -> None:
        self._cancel_done_watchdog()
        self._done_watchdog = asyncio.get_running_loop().call_later(
            _RESPONSE_DONE_TIMEOUT_S if timeout is None else timeout,
            self._on_response_done_timeout,
        )

    def _cancel_done_watchdog(self) -> None:
        if self._done_watchdog is not None:
            self._done_watchdog.cancel()
            self._done_watchdog = None

    def _on_response_done_timeout(self) -> None:
        self._done_watchdog = None
        if (generation := self._current_generation) is not None:
            logger.warning("no response.done from Qwen-Omni-Realtime, finishing the reply anyway")
            self._complete_generation(generation)

    def _complete_generation(self, generation: _ResponseGeneration) -> None:
        for item_id, msg in generation.messages.items():
            if msg.transcript and self._chat_ctx.get_by_id(item_id) is None:
                self._chat_ctx.add_message(role="assistant", content=msg.transcript, id=item_id)
        self._finish_generation(generation)
        if generation.restore_instructions:
            self.send_event(self._session_update())

    def _handle_output_item_done(self, event: dict[str, Any]) -> None:
        item = event.get("item") or {}
        if self._current_generation is not None:
            self._arm_done_watchdog()
        if item.get("type") == "function_call":
            self._handle_function_call({**item, "item_id": item.get("id")})
        elif item.get("type") == "message" and (generation := self._current_generation):
            if msg := generation.messages.get(item.get("id") or ""):
                msg.text_ch.close()
                msg.audio_ch.close()

    def _handle_function_call(self, event: dict[str, Any]) -> None:
        generation = self._current_generation
        call_id = event.get("call_id")
        name = event.get("name")
        if generation is None or not call_id or not name:
            return
        if call_id in generation.function_call_ids:
            return  # output_item.done and function_call_arguments.done both carry it
        generation.function_call_ids.add(call_id)
        item_id = event.get("item_id") or utils.shortuuid("item_")
        arguments = event.get("arguments") or "{}"
        self._server_item_ids.add(item_id)
        fnc = llm.FunctionCall(id=item_id, call_id=call_id, name=name, arguments=arguments)
        self._chat_ctx.items.append(fnc)
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            generation.function_ch.send_nowait(fnc)

    def _handle_response_done(self, event: dict[str, Any]) -> None:
        generation = self._current_generation
        if generation is None:
            return
        response = event.get("response") or {}
        self._cancel_done_watchdog()
        self._complete_generation(generation)

        status = response.get("status")
        if status == "failed":
            self._emit_error(
                APIError(
                    "Qwen-Omni-Realtime response failed",
                    body=response.get("status_details"),
                    retryable=True,
                ),
                recoverable=True,
            )
        self._emit_metrics(generation, response)

    def _finish_generation(self, generation: _ResponseGeneration) -> None:
        for msg in generation.messages.values():
            if not msg.modalities.done():
                msg.modalities.set_result(list(self._opts.modalities))
            msg.text_ch.close()
            msg.audio_ch.close()
        generation.function_ch.close()
        generation.message_ch.close()
        if self._current_generation is generation:
            self._current_generation = None

    def _close_current_generation(self) -> None:
        if self._current_generation is not None:
            self._finish_generation(self._current_generation)

    def _emit_metrics(self, generation: _ResponseGeneration, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        input_details = usage.get("input_tokens_details") or usage.get("input_token_details") or {}
        output_details = (
            usage.get("output_tokens_details") or usage.get("output_token_details") or {}
        )
        duration = time.time() - generation.created_at
        ttft = (
            generation.first_token_at - generation.created_at if generation.first_token_at else -1
        )
        output_tokens = usage.get("output_tokens", 0)
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=generation.created_at,
                request_id=generation.response_id,
                ttft=ttft,
                duration=duration,
                cancelled=response.get("status") == "cancelled",
                label=self._realtime_model.label,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=output_tokens,
                total_tokens=usage.get("total_tokens", 0),
                tokens_per_second=output_tokens / duration if duration > 0 else 0,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(
                    audio_tokens=input_details.get("audio_tokens", 0),
                    text_tokens=input_details.get("text_tokens", 0),
                    # LiveKit has one visual-input bucket; Qwen reports camera input separately.
                    image_tokens=(
                        input_details.get("image_tokens", 0) + input_details.get("video_tokens", 0)
                    ),
                    cached_tokens=0,
                    cached_tokens_details=None,
                ),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    text_tokens=output_details.get("text_tokens", 0),
                    audio_tokens=output_details.get("audio_tokens", 0),
                    image_tokens=0,
                ),
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )


def _fit(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """Scale down, never up, to fit the bounds in either orientation."""
    if width < height:
        max_width, max_height = max_height, max_width
    scale = min(max_width / width, max_height / height, 1.0) if width and height else 1.0
    return max(1, int(width * scale)), max(1, int(height * scale))
