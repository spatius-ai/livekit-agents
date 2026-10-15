from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from livekit import api, rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.voice import room_io
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.plugins.spatius import AvatarSession
from livekit.plugins.spatius.expressions import AvatarExpression

pytestmark = [
    pytest.mark.plugin("spatius"),
    pytest.mark.skipif(
        os.getenv("SPATIUS_RUN_LIVEKIT_TEST") != "1",
        reason="set SPATIUS_RUN_LIVEKIT_TEST=1 to exercise live avatar services",
    ),
]


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"live expression test requires {name}", pytrace=False)
    return value


async def _pcm_frames(pcm: bytes) -> AsyncIterator[rtc.AudioFrame]:
    for offset in range(0, len(pcm), 1600):
        chunk = pcm[offset : offset + 1600]
        yield rtc.AudioFrame(
            data=chunk, sample_rate=16_000, num_channels=1, samples_per_channel=len(chunk) // 2
        )
        await asyncio.sleep(0)


class _PlaybackProbe(AvatarSession):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.finished_callbacks = 0
        self.started_callbacks = 0
        self.started = asyncio.Event()
        self.errors: list[str] = []

    def _on_playback_started_rpc(self, data: rtc.RpcInvocationData) -> str:
        result = super()._on_playback_started_rpc(data)
        if result == "ok" and data.caller_identity == self.avatar_identity:
            self.started_callbacks += 1
            self.started.set()
        return result

    def _on_playback_finished_rpc(self, data: rtc.RpcInvocationData) -> str:
        result = super()._on_playback_finished_rpc(data)
        if result == "ok" and data.caller_identity == self.avatar_identity:
            self.finished_callbacks += 1
        return result

    def _on_spatius_error(self, error: Exception) -> None:
        self.errors.append(type(error).__name__)


async def test_expression_audio_through_real_livekit_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in end-to-end transport test; it does not validate client rendering or motion weights."""
    lk_url = _required("LIVEKIT_URL")
    lk_key = _required("LIVEKIT_API_KEY")
    lk_secret = _required("LIVEKIT_API_SECRET")
    for name in ("SPATIUS_API_KEY", "SPATIUS_APP_ID", "SPATIUS_AVATAR_ID"):
        _required(name)
    pcm_path = Path(_required("SPATIUS_TEST_AUDIO_PCM"))
    if not pcm_path.is_file():
        pytest.fail("SPATIUS_TEST_AUDIO_PCM must name a readable raw PCM file", pytrace=False)
    with pcm_path.open("rb") as source:
        pcm = source.read(16_000 * 2 * 4)
    if len(pcm) != 16_000 * 2 * 4 or not any(pcm):
        pytest.fail(
            "test audio must contain at least four seconds of non-silent 16k mono PCM",
            pytrace=False,
        )

    suffix = uuid4().hex[:12]
    room_name = f"spatius-expression-test-{suffix}"
    agent_identity = f"expression-agent-{suffix}"
    room = rtc.Room()
    avatar = _PlaybackProbe(
        avatar_participant_identity=f"expression-avatar-{suffix}",
        sample_rate=24_000,
        supported_emotions=["happy"],
        idle_timeout_seconds=120,
    )
    session: AgentSession[None] = AgentSession()
    lk_api = api.LiveKitAPI(lk_url, lk_key, lk_secret)
    audio_tasks: list[asyncio.Task[None]] = []
    audio_streams: list[rtc.AudioStream] = []
    audio_received = asyncio.Event()
    audio_nonzero_frames = 0
    playback_events: list[PlaybackFinishedEvent] = []
    sent_requests: list[tuple[dict[str, Any], bool, str]] = []
    created_room = False
    failure: str | None = None
    phase = "initialization"

    async def monitor_audio(track: rtc.Track) -> None:
        nonlocal audio_nonzero_frames
        stream = rtc.AudioStream(track, sample_rate=16_000, num_channels=1)
        audio_streams.append(stream)
        async for event in stream:
            if any(event.frame.data):
                audio_nonzero_frames += 1
                audio_received.set()

    @room.on("track_subscribed")
    def track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            participant.identity == avatar.avatar_identity
            and track.kind == rtc.TrackKind.KIND_AUDIO
        ):
            audio_tasks.append(asyncio.create_task(monitor_audio(track)))

    async def speak(expression: AvatarExpression | None, *, interrupt: bool = False) -> None:
        before_finished = avatar.finished_callbacks
        before_events = len(playback_events)
        before_nonzero = audio_nonzero_frames
        before_sends = len(sent_requests)
        avatar.started.clear()
        audio_received.clear()
        audio = avatar.expression_audio(
            _pcm_frames(pcm * (3 if interrupt else 1)), expression=expression
        )
        handle = session.say("Avatar expression playback test.", audio=audio, add_to_chat_ctx=False)
        try:
            await asyncio.wait_for(avatar.started.wait(), timeout=25)
            await asyncio.wait_for(audio_received.wait(), timeout=10)
            if interrupt:
                handle.interrupt()
            await asyncio.wait_for(handle.wait_for_playout(), timeout=30)
            assert handle.exception() is None, "spoken reply failed"
            assert handle.interrupted is interrupt
            assert len(playback_events) == before_events + 1, "expected one completion per reply"
            assert avatar.finished_callbacks == before_finished + 1, (
                "missing/duplicate playback RPC"
            )
            assert playback_events[-1].interrupted is interrupt
            assert audio_nonzero_frames > before_nonzero, "avatar audio was silent or not delivered"
            request_sends = sent_requests[before_sends:]
            assert request_sends, "reply did not reach the SDK"
            expected_options: dict[str, Any] = {}
            if expression is not None:
                if expression.emotion is not None:
                    expected_options["emotion"] = expression.emotion
                if expression.strength is not None:
                    expected_options["strength"] = expression.strength
            assert request_sends[0][0] == expected_options, "first SDK chunk lost its expression"
            assert all(not options for options, _, _ in request_sends[1:]), (
                "expression was repeated or changed after the first chunk"
            )
            request_ids = {request_id for _, _, request_id in request_sends}
            assert len(request_ids) == 1 and "" not in request_ids, "reply crossed SDK request IDs"
            assert not any(end for _, end, _ in request_sends[:-1]), "reply ended more than once"
            if not interrupt:
                assert request_sends[-1][1], "reply did not send its final SDK chunk"
            assert not avatar.errors, "avatar service reported an error"
            assert avatar._audio is not None and avatar._audio._pending_playback_count == 0
        finally:
            if not handle.done():
                handle.interrupt(force=True)

    async def run() -> None:
        nonlocal created_room, phase
        phase = "temporary room creation"
        created_room = True
        await lk_api.room.create_room(api.CreateRoomRequest(name=room_name, empty_timeout=60))
        token = (
            api.AccessToken(lk_key, lk_secret)
            .with_identity(agent_identity)
            .with_kind("agent")
            .with_ttl(timedelta(minutes=5))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_subscribe=True,
                    can_publish=False,
                    can_publish_data=True,
                    can_update_own_metadata=True,
                )
            )
            .to_jwt()
        )
        phase = "agent room connection"
        await room.connect(lk_url, token)
        phase = "avatar startup"
        await avatar.start(session, room)
        assert avatar._spatius_session is not None
        original_send_audio = avatar._spatius_session.send_audio

        async def observe_send_audio(*, audio: bytes, end: bool, **options: Any) -> str:
            request_id = await original_send_audio(audio=audio, end=end, **options)
            sent_requests.append((dict(options), end, request_id))
            return request_id

        monkeypatch.setattr(avatar._spatius_session, "send_audio", observe_send_audio)
        assert avatar._audio is not None
        avatar._audio.on("playback_finished", playback_events.append)
        phase = "agent session startup"
        await session.start(
            Agent(instructions="Read supplied test speech."),
            room=room,
            room_options=room_io.RoomOptions(
                audio_input=False,
                text_input=False,
                audio_output=False,
                text_output=False,
                close_on_disconnect=False,
            ),
            session_host=False,
            record=False,
        )
        phase = "avatar track publication"
        await avatar.wait_for_join(timeout=25)
        for label, expression in (
            ("default expression", None),
            ("zero strength", AvatarExpression("happy", 0)),
            ("half strength", AvatarExpression("happy", 0.5)),
            ("full strength", AvatarExpression("happy", 1)),
        ):
            phase = label
            await speak(expression)
        phase = "interruption"
        await speak(AvatarExpression("happy", 0.5), interrupt=True)
        phase = "replacement after interruption"
        await speak(AvatarExpression("happy", 0))
        assert avatar.started_callbacks == 6
        assert avatar.finished_callbacks == 6
        assert len(playback_events) == 6

    try:
        await asyncio.wait_for(run(), timeout=145)
    except Exception as error:
        failure = f"live expression playback failed during {phase} ({type(error).__name__})"
    finally:
        for task in audio_tasks:
            task.cancel()
        await asyncio.gather(*audio_tasks, return_exceptions=True)
        for stream in audio_streams:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stream.aclose(), timeout=2)
        for close in (avatar.aclose, session.aclose, room.disconnect):
            try:
                await asyncio.wait_for(close(), timeout=5)
            except Exception as error:
                failure = failure or f"live test cleanup failed ({type(error).__name__})"
        if created_room:
            try:
                await asyncio.wait_for(
                    lk_api.room.delete_room(api.DeleteRoomRequest(room=room_name)), timeout=5
                )
            except api.TwirpError as error:
                if error.code != "not_found":
                    failure = failure or "temporary LiveKit room cleanup failed"
            except Exception:
                failure = failure or "temporary LiveKit room cleanup failed"
        await lk_api.aclose()
    if failure:
        pytest.fail(failure, pytrace=False)
