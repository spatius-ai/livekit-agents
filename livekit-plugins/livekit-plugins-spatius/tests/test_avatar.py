from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from spatius import AudioFormat, LiveKitTransportConfig, new_avatar_session
from spatius.proto.generated import message_pb2

from livekit import api, rtc
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins.spatius import AvatarSession
from livekit.plugins.spatius.avatar import SpatiusException

pytestmark = pytest.mark.unit

_LIVEKIT_API_KEY = "test-key-with-at-least-thirty-two-characters"
_LIVEKIT_API_SECRET = "test-secret-with-at-least-thirty-two-characters"


def _session(**kwargs: Any) -> AvatarSession:
    return AvatarSession(
        api_key="test-key",
        app_id="test-app",
        avatar_id="test-avatar",
        region="us-west",
        sample_rate=24_000,
        **kwargs,
    )


@asynccontextmanager
async def _started_avatar(**kwargs: Any) -> AsyncIterator[tuple[Any, ...]]:
    sdk_session = MagicMock()
    for method in ("init", "start", "close", "send_audio", "interrupt"):
        setattr(sdk_session, method, AsyncMock())
    room = MagicMock()
    room.name = "test-room"
    room.local_participant.identity = "test-agent"
    room.isconnected.return_value = True
    agent_session = MagicMock()
    session = _session(**kwargs)

    with (
        patch("livekit.plugins.spatius.avatar.BaseAvatarSession.start", new=AsyncMock()),
        patch(
            "livekit.plugins.spatius.avatar.new_avatar_session", return_value=sdk_session
        ) as create_sdk_session,
    ):
        try:
            await session.start(
                agent_session,
                room,
                livekit_url="wss://livekit.example.com",
                livekit_api_key=_LIVEKIT_API_KEY,
                livekit_api_secret=_LIVEKIT_API_SECRET,
            )
            yield session, sdk_session, create_sdk_session, room, agent_session
        finally:
            await session.aclose()


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x01\x00" * 480,
        sample_rate=24_000,
        num_channels=1,
        samples_per_channel=480,
    )


def _rpc(identity: str, payload: str = "") -> rtc.RpcInvocationData:
    return rtc.RpcInvocationData(
        request_id="test-rpc", caller_identity=identity, payload=payload, response_timeout=5.0
    )


def test_extra_params_are_copied() -> None:
    params = {"test-option": "original"}

    with patch.dict(
        os.environ,
        {
            "SPATIUS_API_KEY": "test-key",
            "SPATIUS_APP_ID": "test-app",
            "SPATIUS_AVATAR_ID": "test-avatar",
        },
    ):
        session = AvatarSession(extra_params=params)

    params["test-option"] = "changed"

    assert session._extra_params == {"test-option": "original"}


async def test_start_serializes_v3_transport_with_token_auth() -> None:
    async with _started_avatar(
        extra_params={"test-option": "value"}, bitrate=32_000, idle_timeout_seconds=90
    ) as (session, sdk_session, create_sdk_session, room, agent_session):
        kwargs = create_sdk_session.call_args.kwargs
        transport = kwargs["transport"]
        assert isinstance(transport, LiveKitTransportConfig)
        assert transport.agent_identity == "test-agent"
        assert transport.extra_attributes == {ATTRIBUTE_PUBLISH_ON_BEHALF: "test-agent"}
        assert "livekit_egress" not in kwargs
        assert kwargs["audio_format"] == AudioFormat.OGG_OPUS
        assert kwargs["ogg_opus_encoder"].application == "audio"
        assert kwargs["bitrate"] == 32_000
        sdk_session.init.assert_awaited_once()
        sdk_session.start.assert_awaited_once()
        agent_session.output.replace_audio_tail.assert_called_once_with(session._audio)
        assert {
            item.args[0] for item in room.local_participant.register_rpc_method.call_args_list
        } == {
            "lk.playback_started",
            "lk.playback_finished",
        }

        claims = api.TokenVerifier(_LIVEKIT_API_KEY, _LIVEKIT_API_SECRET).verify(transport.token)
        assert claims.identity == session.avatar_identity
        assert claims.name == "spatius-avatar-agent"
        assert claims.video is not None
        assert claims.video.room == "test-room"
        assert claims.video.room_join is True
        assert claims.video.can_subscribe is False
        assert claims.video.can_publish_data is True
        assert claims.video.can_update_own_metadata is True
        assert claims.attributes == transport.extra_attributes

        actual_sdk = new_avatar_session(**kwargs)
        websocket = AsyncMock()
        actual_sdk._connection = websocket
        try:
            assert actual_sdk.config.ingress_endpoint_url.endswith("/v3/driveningress")
            await actual_sdk._send_client_configure_session()
            wire = websocket.send.await_args.args[0]
            message = message_pb2.Message.FromString(wire)
            assert message.type == message_pb2.MESSAGE_CLIENT_CONFIGURE_SESSION
            config = message.client_configure_session
            assert config.audio_sample_rate == 24_000
            assert config.audio_format == message_pb2.AUDIO_FORMAT_OGG_OPUS
            assert dict(config.extra_params) == {"test-option": "value"}
            assert config.transport.WhichOneof("destination") == "livekit"
            assert config.transport.livekit.url == "wss://livekit.example.com"
            assert config.transport.livekit.token == transport.token
            assert config.transport.livekit.agent_identity == "test-agent"
            assert config.transport.livekit.idle_timeout == 90
            assert dict(config.transport.livekit.extra_attributes) == transport.extra_attributes
            assert _LIVEKIT_API_KEY.encode() not in wire
            assert _LIVEKIT_API_SECRET.encode() not in wire
            assert "bitrate" not in config.DESCRIPTOR.fields_by_name
        finally:
            actual_sdk._connection = None


async def test_audio_frames_and_end_are_forwarded_as_one_request() -> None:
    async with _started_avatar() as (session, sdk_session, *_):
        end_sent = asyncio.Event()

        async def send_audio(*, audio: bytes, end: bool) -> None:
            if end:
                end_sent.set()

        sdk_session.send_audio.side_effect = send_audio
        assert session._audio is not None
        frame = _frame()
        await session._audio.capture_frame(frame)
        await session._audio.capture_frame(frame)
        session._audio.flush()
        await asyncio.wait_for(end_sent.wait(), timeout=1.0)

        assert sdk_session.send_audio.await_args_list == [
            call(audio=bytes(frame.data), end=False),
            call(audio=bytes(frame.data), end=False),
            call(audio=b"", end=True),
        ]


async def test_playback_rpcs_complete_the_matching_output_segment() -> None:
    async with _started_avatar() as (session, _, *_):
        assert session._audio is not None
        audio = session._audio
        started = MagicMock()
        audio.on("playback_started", started)
        await audio.capture_frame(_frame())
        audio.flush()

        started.assert_not_called()
        assert session._on_playback_started_rpc(_rpc("other-participant")) == "reject"
        assert session._on_playback_started_rpc(_rpc(session.avatar_identity)) == "ok"
        started.assert_called_once()
        assert (
            session._on_playback_finished_rpc(
                _rpc("other-participant", '{"playback_position": 0.02, "interrupted": false}')
            )
            == "reject"
        )
        assert audio._pending_playback_count == 1
        assert (
            session._on_playback_finished_rpc(
                _rpc(session.avatar_identity, '{"playback_position": 0.02, "interrupted": false}')
            )
            == "ok"
        )
        result = await asyncio.wait_for(audio.wait_for_playout(), timeout=1.0)
        assert result.playback_position == 0.02
        assert result.interrupted is False
        assert audio._pending_playback_count == 0


@pytest.mark.parametrize("confirm_interrupt", [False, True])
async def test_clear_buffer_interrupts_and_releases_playout(confirm_interrupt: bool) -> None:
    async with _started_avatar() as (session, sdk_session, *_):
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_frame())
        audio.flush()
        audio.clear_buffer()
        assert session._interrupt_atask is not None
        await session._interrupt_atask
        sdk_session.interrupt.assert_awaited_once()

        if confirm_interrupt:
            assert (
                session._on_playback_finished_rpc(
                    _rpc(session.avatar_identity, '{"playback_position": 0, "interrupted": true}')
                )
                == "ok"
            )
        else:
            session._cancel_clear_buffer_timeout()
            session._on_clear_buffer_timeout()
        result = await asyncio.wait_for(audio.wait_for_playout(), timeout=1.0)
        assert result.interrupted is True
        assert audio._pending_playback_count == 0
        assert session._clear_buffer_timer is None


async def test_close_releases_pending_playout_and_closes_sdk_once() -> None:
    async with _started_avatar() as (session, sdk_session, *_):
        assert session._audio is not None
        audio = session._audio
        await audio.capture_frame(_frame())
        audio.flush()
        await session.aclose()
        await session.aclose()
        result = await asyncio.wait_for(audio.wait_for_playout(), timeout=1.0)
        assert result.interrupted is True
        sdk_session.close.assert_awaited_once()
        assert session._forward_atask is None


async def test_start_failure_closes_sdk_and_does_not_attach_audio() -> None:
    sdk_session = MagicMock()
    sdk_session.init = AsyncMock()
    sdk_session.start = AsyncMock(side_effect=RuntimeError("handshake rejected"))
    sdk_session.close = AsyncMock()
    room = MagicMock()
    room.name = "test-room"
    room.local_participant.identity = "test-agent"
    agent_session = MagicMock()
    session = _session()

    with (
        patch("livekit.plugins.spatius.avatar.BaseAvatarSession.start", new=AsyncMock()),
        patch("livekit.plugins.spatius.avatar.new_avatar_session", return_value=sdk_session),
        pytest.raises(SpatiusException, match="Failed to start Spatius avatar session"),
    ):
        await session.start(
            agent_session,
            room,
            livekit_url="wss://livekit.example.com",
            livekit_api_key=_LIVEKIT_API_KEY,
            livekit_api_secret=_LIVEKIT_API_SECRET,
        )

    sdk_session.close.assert_awaited_once()
    agent_session.output.replace_audio_tail.assert_not_called()
    assert session._closed is True
