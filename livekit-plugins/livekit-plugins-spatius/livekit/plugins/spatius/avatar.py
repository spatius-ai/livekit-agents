# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    ModelSettings,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.types import TimedString
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarSession as BaseAvatarSession,
)
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from spatius import (
    AudioFormat,
    AvatarDetail,
    AvatarSession as SpatiusSDKSession,
    LiveKitTransportConfig,
    OggOpusEncoderConfig,
    new_avatar_session,
)

from .expression_output import ExpressionAudioOutput
from .expressions import AvatarExpression, ExpressionController, _ExpressionMetadata
from .log import logger

# "auto" resolves the ingress region via the Spatius bootstrap API, falling back
# to us-west if resolution fails; set region/SPATIUS_REGION to pin a concrete region
DEFAULT_REGION = "auto"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_AUDIO_FORMAT = AudioFormat.OGG_OPUS
DEFAULT_OPUS_FRAME_DURATION_MS = 20
DEFAULT_OPUS_APPLICATION = "audio"
SUPPORTED_OPUS_SAMPLE_RATES = {8000, 12000, 16000, 24000, 48000}
DEFAULT_SESSION_TTL = timedelta(hours=1)

_AVATAR_AGENT_IDENTITY = "spatius-avatar-agent"
_AVATAR_AGENT_NAME = "spatius-avatar-agent"
_RPC_PLAYBACK_STARTED = "lk.playback_started"
_RPC_PLAYBACK_FINISHED = "lk.playback_finished"
# fall back to finishing the segment locally if the worker never confirms an
# interrupt, so AgentSession.wait_for_playout() cannot hang
_CLEAR_BUFFER_TIMEOUT = 2.0


class SpatiusException(Exception):
    """Exception raised for Spatius avatar integration errors."""


class AvatarSession(BaseAvatarSession):
    """A Spatius avatar session.

    The LiveKit agent produces speech as usual. This plugin forwards the TTS audio
    to Spatius motion server, it joins the LiveKit room to publish synchronized
    avatar audio and motion data. The worker reports playback started and finished
    back to the agent over LiveKit RPC.

    By default the ingress region is resolved automatically through the Spatius
    bootstrap API (``region="auto"``). Pass ``region`` or set the ``SPATIUS_REGION``
    environment variable to pin a concrete region such as ``us-west``.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
        console_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        ingress_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout_seconds: int = 0,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat | str] = NOT_GIVEN,
        bitrate: int = 0,
        opus_frame_duration_ms: int = DEFAULT_OPUS_FRAME_DURATION_MS,
        opus_application: str = DEFAULT_OPUS_APPLICATION,
        extra_params: dict[str, str] | None = None,
        default_expression: AvatarExpression | None = None,
    ) -> None:
        super().__init__()

        self._api_key = _require(api_key, "SPATIUS_API_KEY", "api_key")
        self._app_id = _require(app_id, "SPATIUS_APP_ID", "app_id")
        self._avatar_id = _require(avatar_id, "SPATIUS_AVATAR_ID", "avatar_id")
        self._region = str(_optional(region, "SPATIUS_REGION", DEFAULT_REGION))
        self._console_endpoint_url = str(
            _optional(console_endpoint_url, "SPATIUS_CONSOLE_ENDPOINT", "")
        )
        self._ingress_endpoint_url = str(
            _optional(ingress_endpoint_url, "SPATIUS_INGRESS_ENDPOINT", "")
        )

        self._avatar_identity = str(
            avatar_participant_identity
            if utils.is_given(avatar_participant_identity)
            else _AVATAR_AGENT_IDENTITY
        )
        self._avatar_name = str(
            avatar_participant_name
            if utils.is_given(avatar_participant_name)
            else _AVATAR_AGENT_NAME
        )

        if idle_timeout_seconds < 0:
            raise SpatiusException("idle_timeout_seconds must be greater than or equal to 0")
        if utils.is_given(sample_rate) and sample_rate <= 0:
            raise SpatiusException("sample_rate must be greater than 0")
        if bitrate < 0:
            raise SpatiusException("bitrate must be greater than or equal to 0")

        audio_format_value = _optional(audio_format, "SPATIUS_AUDIO_FORMAT", DEFAULT_AUDIO_FORMAT)
        try:
            self._audio_format = AudioFormat(audio_format_value)
        except ValueError as e:
            raise SpatiusException(f"unsupported audio_format: {audio_format_value}") from e

        self._idle_timeout_seconds = idle_timeout_seconds
        self._sample_rate = int(sample_rate) if utils.is_given(sample_rate) else None
        self._bitrate = bitrate
        self._extra_params = dict(extra_params or {})
        self._opus_encoder = (
            OggOpusEncoderConfig(
                frame_duration_ms=opus_frame_duration_ms,
                application=opus_application,
            )
            if self._audio_format == AudioFormat.OGG_OPUS
            else None
        )

        if default_expression is not None and not isinstance(default_expression, AvatarExpression):
            raise TypeError("default_expression must be an AvatarExpression")
        self._default_expression = default_expression
        self._avatar_detail: AvatarDetail | None = None
        self._expressions = ExpressionController()
        self._send_lock = asyncio.Lock()
        self._active_expression: _ExpressionMetadata | None = None
        self._interrupt_done = asyncio.Event()
        self._interrupt_done.set()
        self._interrupted_segments = 0
        self._sent_segments = 0
        self._spatius_session: SpatiusSDKSession | None = None
        self._audio: ExpressionAudioOutput | None = None
        self._forward_atask: asyncio.Task[None] | None = None
        self._interrupt_atask: asyncio.Task[None] | None = None
        self._aclose_atask: asyncio.Task[None] | None = None
        self._clear_buffer_timer: asyncio.TimerHandle | None = None
        self._closed = False

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def provider(self) -> str:
        return "spatius"

    @property
    def available_emotions(self) -> tuple[str, ...]:
        """Supported labels discovered when the avatar session starts."""
        return self._require_avatar_detail().available_emotions

    @property
    def default_emotion(self) -> str:
        """The avatar's default label, as returned by the service."""
        return self._require_avatar_detail().default_emotion

    @property
    def expression_instructions(self) -> str:
        """LLM instructions for one avatar expression per spoken reply."""
        self._require_avatar_detail()
        return self._expressions.instructions

    def _require_avatar_detail(self) -> AvatarDetail:
        if self._avatar_detail is None:
            raise SpatiusException("start the avatar session before reading its emotion metadata")
        return self._avatar_detail

    def _expression_sample_rate(self) -> int:
        if self._audio is None or self._audio.sample_rate is None:
            raise SpatiusException("start the avatar session before generating expression audio")
        return self._audio.sample_rate

    def tts_node(
        self,
        agent: Agent,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
        *,
        expression: AvatarExpression | None = None,
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Use from Agent.tts_node to select and retain one expression for the reply."""
        return self._expressions.tts_node(
            agent,
            text,
            model_settings,
            sample_rate=self._expression_sample_rate(),
            expression=expression,
        )

    def transcription_node(
        self, text: AsyncIterable[str | TimedString]
    ) -> AsyncIterator[str | TimedString]:
        """Remove avatar control markers from the displayed transcript."""
        return self._expressions.transcription_node(text)

    def expression_audio(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        *,
        expression: AvatarExpression | None = None,
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Normalize and annotate one scripted or realtime audio reply."""
        return self._expressions.expression_audio(
            audio,
            sample_rate=self._expression_sample_rate(),
            expression=expression,
        )

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        livekit_room_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Start the Spatius avatar session and attach it to the agent audio output."""
        if self._closed:
            raise SpatiusException("Spatius avatar session is closed")
        if self._spatius_session is not None:
            logger.warning("Spatius avatar session already started")
            return

        lk_url = _optional(livekit_url, "LIVEKIT_URL", "")
        lk_api_key = _optional(livekit_api_key, "LIVEKIT_API_KEY", "")
        lk_api_secret = _optional(livekit_api_secret, "LIVEKIT_API_SECRET", "")
        if not lk_url or not lk_api_key or not lk_api_secret:
            raise SpatiusException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        room_name = str(livekit_room_name) if utils.is_given(livekit_room_name) else room.name
        if not room_name:
            raise SpatiusException("livekit_room_name must not be empty")

        sample_rate = self._resolve_sample_rate(agent_session)
        agent_identity = self._local_participant_identity(room)

        logger.debug(
            "starting Spatius avatar session",
            extra={
                "room": room_name,
                "agent_identity": agent_identity,
                "region": self._region,
                "sample_rate": sample_rate,
            },
        )

        # the transport token joins the room as an agent, publishes the avatar tracks,
        # and publishes data so the worker can send the playback lifecycle RPCs
        transport_token = (
            api.AccessToken(api_key=lk_api_key, api_secret=lk_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_identity)
            .with_name(self._avatar_name)
            .with_ttl(DEFAULT_SESSION_TTL)
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: agent_identity})
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_subscribe=False,
                    can_publish_data=True,
                    can_update_own_metadata=True,
                    # the worker publishes exactly one avatar video and one audio track
                    can_publish_sources=["camera", "microphone"],
                )
            )
            .to_jwt()
        )
        transport = LiveKitTransportConfig(
            url=lk_url,
            token=transport_token,
            agent_identity=agent_identity,
            extra_attributes={
                ATTRIBUTE_PUBLISH_ON_BEHALF: agent_identity,
            },
            idle_timeout=self._idle_timeout_seconds,
        )

        try:
            await super().start(agent_session, room)
            self._spatius_session = new_avatar_session(
                api_key=self._api_key,
                app_id=self._app_id,
                avatar_id=self._avatar_id,
                region=self._region,
                console_endpoint_url=self._console_endpoint_url,
                ingress_endpoint_url=self._ingress_endpoint_url,
                expire_at=datetime.now(timezone.utc) + DEFAULT_SESSION_TTL,
                transport=transport,
                sample_rate=sample_rate,
                bitrate=self._bitrate,
                audio_format=self._audio_format,
                ogg_opus_encoder=self._opus_encoder,
                extra_params=self._extra_params,
                on_error=self._on_spatius_error,
                on_close=self._on_spatius_close,
            )
            await self._spatius_session.init()
            detail = await self._spatius_session.get_avatar_detail()
            default = self._default_expression or AvatarExpression()
            self._expressions = ExpressionController(
                supported_emotions=detail.available_emotions,
                default_expression=AvatarExpression(
                    emotion=default.emotion
                    if default.emotion is not None
                    else detail.default_emotion,
                    strength=default.strength,
                ),
            )
            await self._spatius_session.start()
            self._avatar_detail = detail

            self._audio = ExpressionAudioOutput(
                sample_rate=sample_rate, expressions=self._expressions
            )
            await self._audio.start()
            self._audio.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]

            room.local_participant.register_rpc_method(
                _RPC_PLAYBACK_STARTED, self._on_playback_started_rpc
            )
            room.local_participant.register_rpc_method(
                _RPC_PLAYBACK_FINISHED, self._on_playback_finished_rpc
            )
            agent_session.on("close", self._on_session_close)

            agent_session.output.replace_audio_tail(self._audio)
            self._forward_atask = asyncio.create_task(
                self._forward_audio(), name="spatius_avatar_audio_forwarder"
            )
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception as e:
            logger.debug("Spatius avatar session startup failed", exc_info=True)
            await self.aclose()
            raise SpatiusException(
                "Failed to start Spatius avatar session. Check Spatius credentials, LiveKit "
                "room auth, region/endpoint URLs, and outbound network access. "
                f"room={room_name}, avatar_id={self._avatar_id}, region={self._region}, "
                f"sample_rate={sample_rate}, audio_format={self._audio_format.value}"
            ) from e

    async def _forward_audio(self) -> None:
        assert self._audio is not None and self._spatius_session is not None
        audio, spatius = self._audio, self._spatius_session
        try:
            async for item in audio:
                epoch = audio.epoch
                interrupt = self._interrupt_atask
                if interrupt is not None:
                    await asyncio.shield(interrupt)
                async with self._send_lock:
                    if self._closed or self._aclose_atask is not None:
                        return
                    if epoch != audio.epoch:
                        continue
                    if isinstance(item, rtc.AudioFrame):
                        metadata = audio.metadata(item)
                        if audio.is_cancelled(metadata):
                            continue
                        options: dict[str, Any] = {}
                        if self._active_expression is None:
                            self._active_expression = metadata
                            self._sent_segments += 1
                            if metadata.expression.emotion is not None:
                                options["emotion"] = metadata.expression.emotion
                            if metadata.expression.strength is not None:
                                options["strength"] = metadata.expression.strength
                        elif self._active_expression != metadata:
                            raise ValueError("avatar expression changed within an audio request")
                        await spatius.send_audio(audio=bytes(item.data), end=False, **options)
                    elif isinstance(item, AudioSegmentEnd) and self._active_expression is not None:
                        await spatius.send_audio(audio=b"", end=True)
                        self._active_expression = None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Spatius avatar audio forwarding stopped", exc_info=True)
            self._on_spatius_close()

    def _on_playback_started_rpc(self, data: rtc.RpcInvocationData) -> str:
        if not self._is_avatar_caller(data, "playback-started"):
            return "reject"

        logger.debug("Spatius avatar playback started")
        if self._audio is not None:
            self._audio.notify_playback_started()
        return "ok"

    def _on_playback_finished_rpc(self, data: rtc.RpcInvocationData) -> str:
        if not self._is_avatar_caller(data, "playback-finished"):
            return "reject"

        try:
            payload = json.loads(data.payload)
            playback_position = float(payload["playback_position"])
            interrupted = payload["interrupted"]
            if not isinstance(interrupted, bool) or not math.isfinite(playback_position):
                raise ValueError("invalid playback-finished payload")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "invalid Spatius playback-finished RPC payload",
                extra={"payload": data.payload, "error": str(e)},
            )
            return "reject"

        logger.debug(
            "Spatius avatar playback finished",
            extra={"playback_position": playback_position, "interrupted": interrupted},
        )
        if self._audio is not None:
            if self._interrupted_segments:
                self._sent_segments = max(0, self._sent_segments - 1)
                self._audio.notify_playback_finished(
                    playback_position=max(0.0, playback_position), interrupted=True
                )
                self._interrupted_segments -= 1
                if self._interrupted_segments == 0:
                    self._cancel_clear_buffer_timeout()
                    self._interrupt_done.set()
            elif (
                not interrupted
                and self._sent_segments
                and (self._interrupt_atask is None or self._interrupt_atask.done())
            ):
                self._sent_segments -= 1
                self._audio.notify_playback_finished(
                    playback_position=max(0.0, playback_position), interrupted=False
                )
            elif interrupted and self._sent_segments:
                self._sent_segments -= 1
                self._audio.notify_playback_finished(
                    playback_position=max(0.0, playback_position), interrupted=True
                )
                # The server ended playback without our interrupt. Stop forwarding
                # because the current request's remote state is no longer known.
                self._on_spatius_close()
        return "ok"

    def _is_avatar_caller(self, data: rtc.RpcInvocationData, event: str) -> bool:
        if data.caller_identity == self._avatar_identity:
            return True
        logger.warning(
            f"Spatius {event} RPC received from unexpected participant",
            extra={
                "caller_identity": data.caller_identity,
                "expected_identity": self._avatar_identity,
            },
        )
        return False

    def _on_clear_buffer(self) -> None:
        if self._audio is None or self._audio._pending_playback_count == 0:
            return

        self._interrupted_segments = self._audio._pending_playback_count
        self._interrupt_done.clear()
        if self._spatius_session is not None and (
            self._interrupt_atask is None or self._interrupt_atask.done()
        ):
            self._interrupt_atask = asyncio.create_task(self._interrupt())

        self._cancel_clear_buffer_timeout()
        self._clear_buffer_timer = asyncio.get_event_loop().call_later(
            _CLEAR_BUFFER_TIMEOUT, self._on_clear_buffer_timeout
        )

    def _on_clear_buffer_timeout(self) -> None:
        self._clear_buffer_timer = None
        if self._audio is not None and self._interrupted_segments:
            logger.warning("no Spatius playback-finished RPC after interrupt; marking playout done")
            for _ in range(self._interrupted_segments):
                self._audio.notify_playback_finished(playback_position=0.0, interrupted=True)
        self._interrupted_segments = 0
        self._interrupt_done.set()
        # Playback RPCs carry no request ID. A delayed normal completion after
        # this timeout could finish a replacement reply; require a new session.
        self._on_spatius_close()

    def _cancel_clear_buffer_timeout(self) -> None:
        if self._clear_buffer_timer is not None:
            self._clear_buffer_timer.cancel()
            self._clear_buffer_timer = None

    async def _interrupt(self) -> None:
        if self._spatius_session is None:
            return
        try:
            while self._audio is not None and self._aclose_atask is None:
                epoch = self._audio.epoch
                async with self._send_lock:
                    # Queued audio may be cancelled before its first network send.
                    unsent = max(0, self._interrupted_segments - self._sent_segments)
                    for _ in range(unsent):
                        self._audio.notify_playback_finished(
                            playback_position=0.0, interrupted=True
                        )
                    self._interrupted_segments -= unsent
                    if self._sent_segments == 0:
                        self._active_expression = None
                        self._cancel_clear_buffer_timeout()
                        self._interrupt_done.set()
                    else:
                        req_id = await self._spatius_session.interrupt()
                        self._active_expression = None
                        logger.debug(
                            "interrupted Spatius avatar playback", extra={"request_id": req_id}
                        )
                await self._interrupt_done.wait()
                # Another clear can arrive after an acknowledgement wakes this
                # task but before it resumes. Drain that cancellation too before
                # releasing replacement audio through the barrier.
                if epoch == self._audio.epoch:
                    self._sent_segments = 0
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("failed to interrupt Spatius avatar", exc_info=True)
            self._on_spatius_close()
            raise

    def _on_spatius_error(self, error: Exception) -> None:
        logger.warning("Spatius avatar session error", exc_info=error)

    def _on_spatius_close(self) -> None:
        if self._closed or self._aclose_atask is not None:
            return
        logger.warning("Spatius avatar session closed unexpectedly")
        self._aclose_atask = asyncio.create_task(self.aclose())

    def _on_session_close(self, _: Any) -> None:
        if self._closed or self._aclose_atask is not None:
            return
        self._aclose_atask = asyncio.create_task(self.aclose())

    def _resolve_sample_rate(self, agent_session: AgentSession) -> int:
        sample_rate = self._sample_rate
        if sample_rate is None:
            sample_rate = (
                agent_session.tts.sample_rate if agent_session.tts else DEFAULT_SAMPLE_RATE
            )
        if sample_rate <= 0:
            raise SpatiusException("sample_rate must be greater than 0")
        if (
            self._audio_format == AudioFormat.OGG_OPUS
            and sample_rate not in SUPPORTED_OPUS_SAMPLE_RATES
        ):
            supported = ", ".join(str(rate) for rate in sorted(SUPPORTED_OPUS_SAMPLE_RATES))
            raise SpatiusException(
                f"Ogg Opus encoding supports sample rates: {supported} Hz; got {sample_rate} Hz"
            )
        return sample_rate

    @staticmethod
    def _local_participant_identity(room: rtc.Room) -> str:
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            return job_ctx.local_participant_identity
        if room.isconnected():
            return room.local_participant.identity
        raise SpatiusException("failed to get local participant identity")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._agent_session is not None:
            self._agent_session.off("close", self._on_session_close)

        self._cancel_clear_buffer_timeout()
        self._interrupt_done.set()
        if self._interrupt_atask is not None:
            await utils.aio.cancel_and_wait(self._interrupt_atask)
            self._interrupt_atask = None
        if self._forward_atask is not None:
            await utils.aio.cancel_and_wait(self._forward_atask)
            self._forward_atask = None

        # release any segment still awaiting a playback-finished RPC so that
        # AgentSession.wait_for_playout() cannot hang during teardown
        if self._audio is not None:
            for _ in range(self._audio._pending_playback_count):
                self._audio.notify_playback_finished(playback_position=0.0, interrupted=True)
            await self._audio.aclose()
            self._audio = None
        self._active_expression = None
        self._interrupted_segments = 0
        self._sent_segments = 0

        if self._spatius_session is not None:
            try:
                await self._spatius_session.close()
            except Exception:
                logger.warning("error closing Spatius avatar session", exc_info=True)
            self._spatius_session = None

        await super().aclose()


def _require(value: NotGivenOr[str], env_var: str, name: str) -> str:
    resolved = value if utils.is_given(value) else os.getenv(env_var)
    if not resolved:
        raise SpatiusException(
            f"{name} must be set either by passing it to AvatarSession or "
            f"by setting the {env_var} environment variable"
        )
    return str(resolved)


def _optional(value: NotGivenOr[Any], env_var: str, default: Any) -> Any:
    return value if utils.is_given(value) else os.getenv(env_var, default)
