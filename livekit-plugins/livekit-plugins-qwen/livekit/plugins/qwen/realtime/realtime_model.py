"""Qwen-Omni-Realtime (Alibaba Cloud Model Studio) as a LiveKit realtime model.

The protocol resembles the OpenAI Realtime beta API, with differences that keep this
plugin separate from ``livekit-plugins-openai``:

* audio goes in at 16 kHz and comes out at 24 kHz;
* conversation items can be added but not deleted or truncated, so the history is
  append-only;
* ``response.create`` takes no parameters and responses carry no metadata, so a reply
  request is matched to the next ``response.created``, and per-reply instructions are
  applied with a temporary ``session.update``;
* images are sent with ``input_image_buffer.append`` at about one frame per second;
* there is a second transport, AOQ, where audio and video travel on media tracks.
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
from livekit.agents import APIError, llm, utils
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
    QwenTransport,
    TurnDetection,
)
from ._aoq import AOQOptions, AOQTransport
from ._transport import (
    INPUT_SAMPLE_RATE,
    NUM_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    OutputAudio,
    ServerEvent,
    Transport,
    TransportClosed,
    WebSocketTransport,
    pcm_frame,
)

DEFAULT_MODEL: QwenOmniRealtimeModels = "qwen3.8-omni-flash-realtime"
DEFAULT_VOICE = "Tina"
DEFAULT_REGION: QwenRegion = "cn-beijing"

API_KEY_ENV = "DASHSCOPE_API_KEY"
WORKSPACE_ID_ENV = "DASHSCOPE_WORKSPACE_ID"
REGION_ENV = "DASHSCOPE_REGION"
BASE_URL_ENV = "DASHSCOPE_BASE_URL"

# endpoints that need no workspace id (WebSocket only)
_SHARED_HOSTS: dict[str, str] = {
    "cn-beijing": "dashscope.aliyuncs.com",
    "ap-southeast-1": "dashscope-intl.aliyuncs.com",
}

_WS_PATH = "/api-ws/v1/realtime"
_AOQ_CREDENTIALS_PATH = "/api/v1/webrtc/realtime"

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
# AOQ audio arrives on a media track that lags the data track: a reply's audio keeps
# flowing after response.done, and may start before its message is announced
_AOQ_AUDIO_TAIL_S = 0.5
_AOQ_AUDIO_EARLY_S = 1.0
# the commit travels on the data track, ahead of audio still in the media pipeline
_AOQ_COMMIT_DELAY_S = 0.3

QwenEventTypes = Literal["qwen_server_event_received", "qwen_client_event_queued"]


def build_base_url(*, region: str, workspace_id: str | None) -> str:
    """The HTTPS origin of a Model Studio endpoint."""
    if workspace_id:
        return f"https://{workspace_id}.{region}.maas.aliyuncs.com"
    if (host := _SHARED_HOSTS.get(region)) is None:
        raise ValueError(f"unknown region {region!r}, pass base_url instead")
    return f"https://{host}"


def _endpoint(base_url: str, *, path: str, model: str, websocket: bool) -> str:
    parts = urlsplit(base_url if "://" in base_url else f"https://{base_url}")
    scheme = parts.scheme
    if websocket:
        scheme = {"https": "wss", "http": "ws"}.get(scheme, scheme)
    else:
        scheme = {"wss": "https", "ws": "http"}.get(scheme, scheme)
    return urlunsplit((scheme, parts.netloc, path, urlencode({"model": model}), ""))


def websocket_url(base_url: str, model: str) -> str:
    return _endpoint(base_url, path=_WS_PATH, model=model, websocket=True)


def aoq_credentials_url(base_url: str, model: str) -> str:
    return _endpoint(base_url, path=_AOQ_CREDENTIALS_PATH, model=model, websocket=False)


@dataclass
class _RealtimeOptions:
    model: str
    voice: str
    modalities: list[Literal["text", "audio"]]
    transport: QwenTransport
    api_key: str
    base_url: str
    turn_detection: TurnDetection | None
    input_audio_transcription: InputAudioTranscription | None
    session_params: dict[str, Any]
    aoq: AOQOptions
    conn_options: APIConnectOptions


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        model: QwenOmniRealtimeModels | str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        transport: QwenTransport = "websocket",
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
        aoq_options: AOQOptions | None = None,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Qwen-Omni-Realtime model.

        Args:
            model: Model name, e.g. ``"qwen3.8-omni-flash-realtime"``.
            voice: Output voice. Defaults to ``"Tina"``.
            modalities: ``["text", "audio"]`` (default) or ``["text"]``.
            transport: ``"websocket"`` (default) or ``"aoq"``. AOQ needs Alibaba Cloud's
                AOQ client SDK, see ``AOQOptions``, and a workspace endpoint.
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
            aoq_options: Settings for the AOQ transport.
            http_session: aiohttp session to reuse.
            conn_options: Connection timeout and retry policy.
        """
        api_key = api_key or os.environ.get(API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"a Model Studio API key is required: pass api_key or set {API_KEY_ENV}"
            )
        if transport not in ("websocket", "aoq"):
            raise ValueError(f"unknown transport {transport!r}")

        if not is_given(base_url):
            base_url = os.environ.get(BASE_URL_ENV) or NOT_GIVEN
        if not is_given(base_url):
            workspace_id = workspace_id or os.environ.get(WORKSPACE_ID_ENV) or None
            if transport == "aoq" and not workspace_id:
                raise ValueError(
                    "the AOQ transport needs a workspace endpoint: pass workspace_id "
                    f"(or set {WORKSPACE_ID_ENV}) or base_url"
                )
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
            transport=transport,
            api_key=api_key,
            base_url=base_url,
            turn_detection=resolved_turn_detection,
            input_audio_transcription=resolved_transcription,
            session_params=session_params,
            aoq=aoq_options or AOQOptions(),
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


_Outgoing = dict[str, Any] | _AudioChunk | _Image


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
        self._is_aoq = self._opts.transport == "aoq"

        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._instructions: str | None = None
        # items the server created itself (transcribed turns, replies, function calls)
        self._server_item_ids: set[str] = set()

        self._msg_ch = utils.aio.Chan[_Outgoing]()
        self._input_resampler: rtc.AudioResampler | None = None
        chunk_ms = 20 if self._is_aoq else 100
        self._bstream = utils.audio.AudioByteStream(
            INPUT_SAMPLE_RATE,
            NUM_CHANNELS,
            samples_per_channel=INPUT_SAMPLE_RATE * chunk_ms // 1000,
        )
        self._pushed_duration_s = 0.0
        self._audio_sent = False
        self._last_image_at = 0.0

        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._pending_reply_instructions: deque[str | None] = deque()
        self._current_generation: _ResponseGeneration | None = None
        self._input_transcripts: dict[str, str] = {}
        self._speech_started_at: dict[str, float] = {}

        # AOQ: the message that plays the incoming audio, and audio that came before it
        self._audio_target: _MessageGeneration | None = None
        self._audio_tail_timer: asyncio.TimerHandle | None = None
        self._early_audio: deque[tuple[float, rtc.AudioFrame]] = deque()

        self._transport: Transport | None = None
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
        return session

    def _session_update(self, *, instructions: str | None = None) -> dict[str, Any]:
        return {
            "event_id": utils.shortuuid("session_update_"),
            "type": "session.update",
            "session": self._session_config(instructions=instructions),
        }

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
        for _, item_id in diff.to_create:
            item = chat_ctx.get_by_id(item_id)
            if item is None or item_id in self._server_item_ids:
                continue
            if item.type == "function_call_output":
                self.send_event(
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
                self.send_event(
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

    # ------------------------------------------------------------------ input

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                self._send_audio(nf)

    def _send_audio(self, frame: rtc.AudioFrame) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(_AudioChunk(bytes(frame.data)))
        self._pushed_duration_s += frame.duration
        self._audio_sent = True

    def push_video(self, frame: rtc.VideoFrame) -> None:
        now = time.monotonic()
        # images may only follow audio, at about one per second
        if not self._audio_sent or now - self._last_image_at < _IMAGE_INTERVAL_S:
            return
        self._last_image_at = now
        width, height = _fit(frame.width, frame.height, *_IMAGE_MAX_SIZE)
        resize = ResizeOptions(width=width, height=height, strategy="skew")
        jpeg = encode(frame, EncodeOptions(format="JPEG", resize_options=resize))
        if len(jpeg) > _IMAGE_MAX_BYTES:
            logger.debug("video frame too large for Qwen-Omni-Realtime, skipping it")
            return
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(_Image(jpeg, width, height))

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
        if is_given(tools) or is_given(tool_choice):
            logger.debug("Qwen-Omni-Realtime does not support per-reply tools, ignoring them")

        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut

        reply_instructions: str | None = None
        if is_given(instructions) and instructions:
            # response.create takes no parameters; widen the session instructions for
            # this reply, and restore them once it is done
            reply_instructions = (
                f"{self._instructions}\n{instructions}" if self._instructions else instructions
            )
            self.send_event(self._session_update(instructions=reply_instructions))
        self._pending_reply_instructions.append(reply_instructions)
        self.send_event({"event_id": event_id, "type": "response.create"})

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
        self._drop_output_audio()

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
        self._close_current_generation()
        self._close_audio_target()
        self._msg_ch.close()
        await self._main_atask

    # ------------------------------------------------------------------ connection

    def _create_transport(self) -> Transport:
        opts = self._opts
        http_session = self._realtime_model._ensure_http_session()
        if opts.transport == "aoq":
            return AOQTransport(
                credentials_url=aoq_credentials_url(opts.base_url, opts.model),
                api_key=opts.api_key,
                http_session=http_session,
                timeout=opts.conn_options.timeout,
                options=opts.aoq,
            )
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
                    await transport.aclose()
                    self._transport = None
        finally:
            self._close_current_generation()
            self._close_audio_target()
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
        self._close_audio_target()
        self._input_transcripts.clear()
        self._speech_started_at.clear()
        if self._chat_ctx.items:
            logger.warning(
                "Qwen-Omni-Realtime reconnected; the conversation history is not restored"
            )
        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    async def _send(self, transport: Transport, msg: _Outgoing) -> None:
        if isinstance(msg, _AudioChunk):
            await transport.push_audio(msg.pcm)
        elif isinstance(msg, _Image):
            await transport.push_image(msg.jpeg, msg.width, msg.height)
        else:
            if self._is_aoq and msg.get("type") == "input_audio_buffer.commit":
                await asyncio.sleep(_AOQ_COMMIT_DELAY_S)
            self.emit("qwen_client_event_queued", msg)
            await transport.send_event(msg)

    async def _run(self, transport: Transport) -> None:
        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            async for msg in self._msg_ch:
                try:
                    await self._send(transport, msg)
                except APIError:
                    raise
                except Exception:
                    logger.exception("failed to send event to Qwen-Omni-Realtime")
            await transport.aclose()

        async def _recv_task() -> None:
            while True:
                msg = await transport.recv()
                if isinstance(msg, TransportClosed):
                    if msg.error is not None:
                        raise msg.error
                    return
                if self._closing:
                    continue
                if isinstance(msg, OutputAudio):
                    self._handle_output_audio(msg.frame)
                elif isinstance(msg, ServerEvent):
                    self._handle_event(msg.event)

        tasks = [
            asyncio.create_task(_send_task(), name="qwen.send"),
            asyncio.create_task(_recv_task(), name="qwen.recv"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)

    # ------------------------------------------------------------------ server events

    def _handle_event(self, event: dict[str, Any]) -> None:
        self.emit("qwen_server_event_received", event)
        event_type = event.get("type")
        try:
            if event_type == "session.updated":
                if self._transport is not None:
                    self._transport.on_session_updated()
            elif event_type == "error":
                self._handle_error(event)
            elif event_type == "input_audio_buffer.speech_started":
                self._handle_speech_started(event)
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
        error = event.get("error") or {}
        code = error.get("code")
        message = error.get("message") or "unknown error"
        fatal = code in _FATAL_ERROR_CODES
        logger.error(
            f"Qwen-Omni-Realtime returned an error: {message}",
            extra={"error": error},
        )
        err = APIError(f"Qwen-Omni-Realtime error: {message}", body=error, retryable=not fatal)
        if fatal:
            raise err
        self._emit_error(err, recoverable=True)

    def _handle_speech_started(self, event: dict[str, Any]) -> None:
        if item_id := event.get("item_id"):
            self._speech_started_at[item_id] = time.time()
        if self._opts.turn_detection is not None and self._is_aoq:
            # the server cancels its reply on barge-in; drop what the player still holds
            self._drop_output_audio()
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
        self._close_current_generation()
        if self._is_aoq:
            self._close_audio_target()

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
        if self._is_aoq and self._capabilities.audio_output:
            self._close_audio_target()
            self._audio_target = msg
            horizon = time.monotonic() - _AOQ_AUDIO_EARLY_S
            while self._early_audio:
                received_at, frame = self._early_audio.popleft()
                if received_at >= horizon:
                    self._push_output_audio(msg, frame)
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
        transport = self._transport
        if transport is not None and not transport.output_audio_in_events:
            return  # AOQ plays audio from its media track
        if not (item_id := event.get("item_id")) or not (msg := self._message(item_id)):
            return
        self._push_output_audio(msg, pcm_frame(base64.b64decode(event.get("delta") or "")))

    def _handle_output_audio(self, frame: rtc.AudioFrame) -> None:
        msg = self._audio_target
        if msg is None or msg.audio_ch.closed:
            self._early_audio.append((time.monotonic(), frame))
            horizon = time.monotonic() - _AOQ_AUDIO_EARLY_S
            while self._early_audio and self._early_audio[0][0] < horizon:
                self._early_audio.popleft()
            return
        self._push_output_audio(msg, frame)
        if self._audio_tail_timer is not None:
            # the reply is done; keep the stream open while audio still flows
            self._audio_tail_timer.cancel()
            self._audio_tail_timer = asyncio.get_running_loop().call_later(
                _AOQ_AUDIO_TAIL_S, self._close_audio_target
            )

    def _push_output_audio(self, msg: _MessageGeneration, frame: rtc.AudioFrame) -> None:
        generation = self._current_generation
        if generation is not None and generation.first_token_at is None:
            generation.first_token_at = time.time()
        if not msg.modalities.done():
            msg.modalities.set_result(["audio", "text"])
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            msg.audio_ch.send_nowait(frame)

    def _handle_output_item_done(self, event: dict[str, Any]) -> None:
        item = event.get("item") or {}
        if item.get("type") == "function_call":
            self._handle_function_call({**item, "item_id": item.get("id")})
        elif item.get("type") == "message" and (generation := self._current_generation):
            if msg := generation.messages.get(item.get("id") or ""):
                msg.text_ch.close()
                if msg is not self._audio_target:
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
        for item_id, msg in generation.messages.items():
            if msg.transcript and self._chat_ctx.get_by_id(item_id) is None:
                self._chat_ctx.add_message(role="assistant", content=msg.transcript, id=item_id)
        self._finish_generation(generation)
        if generation.restore_instructions:
            self.send_event(self._session_update())

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
            if msg is self._audio_target and not msg.audio_ch.closed:
                # AOQ: close once the media track goes quiet
                self._audio_tail_timer = asyncio.get_running_loop().call_later(
                    _AOQ_AUDIO_TAIL_S, self._close_audio_target
                )
            else:
                msg.audio_ch.close()
        generation.function_ch.close()
        generation.message_ch.close()
        if self._current_generation is generation:
            self._current_generation = None

    def _close_current_generation(self) -> None:
        if self._current_generation is not None:
            self._finish_generation(self._current_generation)

    def _close_audio_target(self) -> None:
        if self._audio_tail_timer is not None:
            self._audio_tail_timer.cancel()
            self._audio_tail_timer = None
        if self._audio_target is not None:
            self._audio_target.audio_ch.close()
            self._audio_target = None

    def _drop_output_audio(self) -> None:
        self._early_audio.clear()
        self._close_audio_target()
        if self._transport is not None:
            self._transport.interrupt_output()

    def _emit_metrics(self, generation: _ResponseGeneration, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
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
                    image_tokens=input_details.get("image_tokens", 0),
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
