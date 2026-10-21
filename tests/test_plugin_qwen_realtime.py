"""Tests for the Qwen-Omni-Realtime plugin: endpoints, event handling and the AOQ transport."""

from __future__ import annotations

import asyncio
import base64
import enum
import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from livekit import rtc
from livekit.agents import APIError, llm
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.plugins import qwen
from livekit.plugins.qwen.realtime import _aoq, realtime_model as _rt
from livekit.plugins.qwen.realtime._transport import (
    OutputAudio,
    ServerEvent,
    Transport,
    TransportClosed,
    TransportMessage,
    pcm_frame,
)

pytestmark = pytest.mark.unit

KEY = "sk-test"


# ------------------------------------------------------------------ endpoints


def test_default_endpoint_is_shared_beijing_host() -> None:
    with patch.dict("os.environ", {}, clear=True):
        model = qwen.RealtimeModel(api_key=KEY)
    assert _rt.websocket_url(model._opts.base_url, model.model) == (
        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3.8-omni-flash-realtime"
    )


def test_workspace_endpoint_and_aoq_credentials_url() -> None:
    with patch.dict("os.environ", {}, clear=True):
        model = qwen.RealtimeModel(
            api_key=KEY, transport="aoq", workspace_id="ws-1", region="ap-southeast-1"
        )
    assert model._opts.base_url == "https://ws-1.ap-southeast-1.maas.aliyuncs.com"
    assert _rt.aoq_credentials_url(model._opts.base_url, "m") == (
        "https://ws-1.ap-southeast-1.maas.aliyuncs.com/api/v1/webrtc/realtime?model=m"
    )
    assert _rt.websocket_url(model._opts.base_url, "m") == (
        "wss://ws-1.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=m"
    )


def test_requires_api_key_and_aoq_workspace() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            qwen.RealtimeModel()
        with pytest.raises(ValueError, match="workspace"):
            qwen.RealtimeModel(api_key=KEY, transport="aoq")
    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": KEY, "DASHSCOPE_WORKSPACE_ID": "ws-2"}):
        model = qwen.RealtimeModel(transport="aoq")
    assert model._opts.base_url == "https://ws-2.cn-beijing.maas.aliyuncs.com"


def test_capabilities_follow_turn_detection() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    assert model.capabilities.turn_detection
    assert model.capabilities.can_disable_turn_detection
    manual = qwen.RealtimeModel(api_key=KEY, turn_detection=None)
    assert not manual.capabilities.turn_detection
    assert not manual.capabilities.can_disable_turn_detection


# ------------------------------------------------------------------ fake transport


class FakeTransport(Transport):
    def __init__(self, *, output_audio_in_events: bool = True) -> None:
        self.output_audio_in_events = output_audio_in_events
        self.sent: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.images: list[bytes] = []
        self.inbox: asyncio.Queue[TransportMessage] = asyncio.Queue()
        self.session_updated = 0
        self.interrupted = 0
        self.closed = False

    async def connect(self) -> None:
        pass

    async def send_event(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    async def push_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)

    async def push_image(self, jpeg: bytes, width: int, height: int) -> None:
        self.images.append(jpeg)

    async def recv(self) -> TransportMessage:
        return await self.inbox.get()

    def on_session_updated(self) -> None:
        self.session_updated += 1

    def interrupt_output(self) -> None:
        self.interrupted += 1

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.inbox.put_nowait(TransportClosed())

    def server(self, **event: Any) -> None:
        self.inbox.put_nowait(ServerEvent(event))

    def types(self) -> list[str]:
        return [e["type"] for e in self.sent]


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _start(
    model: qwen.RealtimeModel, transport: FakeTransport, **kwargs: Any
) -> _rt.RealtimeSession:
    with patch.object(_rt.RealtimeSession, "_create_transport", lambda self: transport):
        sess = model.session(**kwargs)
        await _settle()
    return sess


def _response_events(transport: FakeTransport, *, audio: bytes = b"") -> None:
    transport.server(type="response.created", response={"id": "resp_1"})
    transport.server(
        type="response.output_item.added",
        response_id="resp_1",
        item={"id": "item_1", "type": "message", "role": "assistant"},
    )
    transport.server(type="response.content_part.added", item_id="item_1", part={"type": "audio"})
    transport.server(type="response.audio_transcript.delta", item_id="item_1", delta="Hel")
    transport.server(type="response.audio_transcript.delta", item_id="item_1", delta="lo")
    if audio:
        transport.server(
            type="response.audio.delta", item_id="item_1", delta=base64.b64encode(audio).decode()
        )
    transport.server(type="response.output_item.done", item={"id": "item_1", "type": "message"})
    transport.server(
        type="response.done",
        response={
            "id": "resp_1",
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "input_tokens_details": {"audio_tokens": 8, "text_tokens": 2},
                "output_tokens_details": {"audio_tokens": 3, "text_tokens": 1},
            },
        },
    )


async def _collect(gen: llm.GenerationCreatedEvent) -> tuple[str, list[rtc.AudioFrame]]:
    text, frames = "", []
    async for msg in gen.message_stream:
        async for delta in msg.text_stream:
            text += delta
        async for frame in msg.audio_stream:
            frames.append(frame)
    return text, frames


# ------------------------------------------------------------------ session


async def test_session_update_on_connect() -> None:
    model = qwen.RealtimeModel(api_key=KEY, voice="Ethan", temperature=0.3)
    transport = FakeTransport()
    sess = await _start(model, transport)
    await sess.update_instructions("be brief")
    await _settle()

    first = transport.sent[0]
    assert first["type"] == "session.update"
    session = first["session"]
    assert session["voice"] == "Ethan"
    assert session["modalities"] == ["text", "audio"]
    assert session["audio"]["input"]["format"] == {"type": "pcm", "sample_rate": 16000}
    assert session["audio"]["output"]["format"] == {"type": "pcm", "sample_rate": 24000}
    assert session["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "silence_duration_ms": 800,
    }
    assert session["input_audio_transcription"] == {"model": "qwen3-asr-flash-realtime"}
    assert session["temperature"] == 0.3
    assert transport.sent[-1]["session"]["instructions"] == "be brief"

    transport.server(type="session.updated", session={})
    await _settle()
    assert transport.session_updated == 1
    await sess.aclose()


async def test_turn_detection_disabled_session() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport, turn_detection_disabled=True)
    assert transport.sent[0]["session"]["turn_detection"] is None
    assert not sess.capabilities.turn_detection
    await sess.aclose()


async def test_response_streams_text_audio_and_metrics() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    generations: list[llm.GenerationCreatedEvent] = []
    metrics: list[Any] = []
    sess.on("generation_created", generations.append)
    sess.on("metrics_collected", metrics.append)

    _response_events(transport, audio=b"\x01\x00" * 240)
    await _settle()

    assert len(generations) == 1 and not generations[0].user_initiated
    text, frames = await _collect(generations[0])
    assert text == "Hello"
    assert sum(f.samples_per_channel for f in frames) == 240
    assert frames[0].sample_rate == 24000

    rt_metrics = [m for m in metrics if isinstance(m, RealtimeModelMetrics) and m.request_id]
    assert rt_metrics[-1].input_tokens == 10
    assert rt_metrics[-1].input_token_details.audio_tokens == 8
    assert rt_metrics[-1].output_token_details.text_tokens == 1

    # the reply is mirrored, so it is not sent back as history
    assert sess.chat_ctx.get_by_id("item_1") is not None
    await sess.aclose()


async def test_generate_reply_with_instructions() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    await sess.update_instructions("base")
    fut = sess.generate_reply(instructions="greet the user")
    await _settle()

    assert transport.types()[-2:] == ["session.update", "response.create"]
    assert transport.sent[-2]["session"]["instructions"] == "base\ngreet the user"

    _response_events(transport)
    await _settle()
    gen = await asyncio.wait_for(fut, 1)
    assert gen.user_initiated
    await _collect(gen)
    await _settle()

    # restored once the reply is done
    assert transport.sent[-1]["type"] == "session.update"
    assert transport.sent[-1]["session"]["instructions"] == "base"
    await sess.aclose()


async def test_function_call_and_tool_output() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)

    @llm.function_tool
    async def get_weather(location: str) -> str:
        """Get the weather."""
        return "sunny"

    await sess.update_tools([get_weather])
    await _settle()
    tools = transport.sent[-1]["session"]["tools"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"

    generations: list[llm.GenerationCreatedEvent] = []
    sess.on("generation_created", generations.append)
    transport.server(type="response.created", response={"id": "resp_2"})
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather"}
    transport.server(
        type="response.function_call_arguments.done",
        item_id="fc_1",
        call_id="call_1",
        name="get_weather",
        arguments='{"location": "Hangzhou"}',
    )
    transport.server(
        type="response.output_item.done", item={**call, "arguments": '{"location": "Hangzhou"}'}
    )
    transport.server(type="response.done", response={"id": "resp_2", "status": "completed"})
    await _settle()

    calls = [c async for c in generations[0].function_stream]
    assert [(c.call_id, c.arguments) for c in calls] == [("call_1", '{"location": "Hangzhou"}')]

    chat_ctx = sess.chat_ctx
    chat_ctx.items.append(
        llm.FunctionCallOutput(call_id="call_1", name="get_weather", output="sunny", is_error=False)
    )
    chat_ctx.add_message(role="user", content="and tomorrow?")
    await sess.update_chat_ctx(chat_ctx)
    await _settle()

    created = [e["item"] for e in transport.sent if e["type"] == "conversation.item.create"]
    assert created == [
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "and tomorrow?"}],
        },
    ]
    await sess.aclose()


async def test_user_transcription_events() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    events: list[llm.InputTranscriptionCompleted] = []
    started: list[Any] = []
    sess.on("input_audio_transcription_completed", events.append)
    sess.on("input_speech_started", started.append)

    transport.server(type="input_audio_buffer.speech_started", item_id="u1")
    transport.server(
        type="conversation.item.input_audio_transcription.delta",
        item_id="u1",
        text="你好",
        stash="世",
    )
    transport.server(
        type="conversation.item.input_audio_transcription.completed",
        item_id="u1",
        transcript="你好世界",
    )
    await _settle()

    assert len(started) == 1
    assert [(e.transcript, e.is_final) for e in events] == [("你好世", False), ("你好世界", True)]
    assert events[-1].turn_started_at is not None

    # the transcribed turn is already on the server and is not resent as text
    await sess.update_chat_ctx(sess.chat_ctx)
    await _settle()
    assert "conversation.item.create" not in transport.types()
    await sess.aclose()


async def test_audio_is_resampled_to_16k_and_committed() -> None:
    model = qwen.RealtimeModel(api_key=KEY, turn_detection=None)
    transport = FakeTransport()
    sess = await _start(model, transport)
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 4800, sample_rate=48000, num_channels=1, samples_per_channel=4800
    )
    sess.push_audio(frame)
    sess.push_audio(frame)
    sess.commit_audio()
    await _settle()
    assert sum(len(a) for a in transport.audio) > 0
    assert all(len(a) <= 3200 for a in transport.audio)  # 100 ms at 16 kHz
    assert transport.types()[-1] == "input_audio_buffer.commit"
    await sess.aclose()


async def test_fatal_error_stops_the_session() -> None:
    model = qwen.RealtimeModel(api_key=KEY)
    transport = FakeTransport()
    sess = await _start(model, transport)
    errors: list[llm.RealtimeModelError] = []
    sess.on("error", errors.append)
    transport.server(type="error", error={"code": "InvalidApiKey", "message": "bad key"})
    with pytest.raises(APIError, match="bad key"):
        await asyncio.wait_for(sess._main_atask, 1)
    assert errors and not errors[-1].recoverable


# ------------------------------------------------------------------ AOQ audio routing


async def test_aoq_audio_comes_from_media_track() -> None:
    model = qwen.RealtimeModel(api_key=KEY, transport="aoq", workspace_id="ws")
    transport = FakeTransport(output_audio_in_events=False)
    with patch.object(_rt, "_AOQ_AUDIO_TAIL_S", 0.05):
        sess = await _start(model, transport)
        generations: list[llm.GenerationCreatedEvent] = []
        sess.on("generation_created", generations.append)

        transport.server(type="response.created", response={"id": "resp_1"})
        # media can run ahead of the data track
        transport.inbox.put_nowait(OutputAudio(pcm_frame(b"\x01\x00" * 240)))
        transport.server(
            type="response.output_item.added", item={"id": "item_1", "type": "message"}
        )
        # duplicate audio on the data track is ignored
        transport.server(
            type="response.audio.delta",
            item_id="item_1",
            delta=base64.b64encode(b"\x09\x00" * 100).decode(),
        )
        transport.server(type="response.output_item.done", item={"id": "item_1", "type": "message"})
        transport.server(type="response.done", response={"id": "resp_1", "status": "completed"})
        # the media track still delivers after response.done
        transport.inbox.put_nowait(OutputAudio(pcm_frame(b"\x02\x00" * 240)))
        await _settle()

        _, frames = await asyncio.wait_for(_collect(generations[0]), 1)
        assert [f.samples_per_channel for f in frames] == [240, 240]

        sess.interrupt()
        assert transport.interrupted == 1
        await sess.aclose()


# ------------------------------------------------------------------ AOQ transport


def test_aoq_credentials_from_response() -> None:
    creds = _aoq.AOQCredentials.from_response(
        {
            "sid": "sid-1",
            "aoqTokenForClient": "tok",
            "clientRelayEndpoints": [{"endpoint": "1.2.3.4", "port": 443, "route_index": 0}],
            "clientRelayCertFingerprint": "fp",
            "sidExpiresInSecs": 7200,
            "extraInfo": {"workspaceIdHash": "hash"},
        }
    )
    assert (creds.sid, creds.token, creds.cert_fingerprint) == ("sid-1", "tok", "fp")
    assert creds.workspace_id_hash == "hash" and creds.expires_in_s == 7200


def _fake_sdk() -> types.ModuleType:
    """A stand-in for ``aoq_client_sdk`` with the same names, recording every call."""

    sdk = types.ModuleType("aoq_client_sdk")

    class TrackType(enum.IntEnum):
        AUDIO = 0
        VIDEO = 1
        DATA = 2

    class EncoderType(enum.IntEnum):
        AUDIO_PCM = 1
        AUDIO_OPUS = 2
        VIDEO_JPEG = 4

    class ConnectionStatus(enum.IntEnum):
        DISCONNECTED = 0
        CONNECTING = 1
        CONNECTED = 2
        FAILED = 3

    class ErrorCode(enum.IntEnum):
        OK = 0
        AUDIO_EXTERNAL_BUFFER_FULL = 110

    class AudioSource(enum.IntEnum):
        PLAYBACK = 3

    @dataclass
    class Cfg:
        kwargs: dict[str, Any] = field(default_factory=dict)

    def factory(*args: Any, **kwargs: Any) -> Cfg:
        return Cfg({"args": args, **kwargs})

    @dataclass
    class AudioFrameData:
        data: bytes = b""
        num_of_samples: int = 0
        bytes_per_sample: int = 0
        num_of_channels: int = 0
        samples_per_sec: int = 0
        auto_gen_mute: bool = False

    @dataclass
    class DataMsg:
        data: bytes = b""

    class Engine:
        instance: Engine | None = None

        def __init__(self, listener: Any) -> None:
            self.listener = listener
            self.calls: list[tuple[str, tuple[Any, ...]]] = []
            self.observer: Any = None
            self.sent: list[bytes] = []
            self.pushed: list[Any] = []

        @classmethod
        def create_engine(cls, cfg: Any, listener: Any) -> Engine:
            cls.instance = cls(listener)
            return cls.instance

        @classmethod
        def destroy(cls) -> int:
            cls.instance = None
            return 0

        def connect(self, cfg: Any) -> int:
            self.connect_config = cfg
            threading.Thread(
                target=self.listener.on_connection_status_change,
                args=(ConnectionStatus.CONNECTED,),
            ).start()
            return 0

        def set_audio_frame_observer(self, observer: Any) -> int:
            self.observer = observer
            return 0

        def send_data_msg(self, msg: DataMsg) -> int:
            self.sent.append(msg.data)
            return 0

        def push_audio_external_stream_data(self, stream_id: str, frame: Any) -> int:
            self.pushed.append(frame)
            return 0

        def __getattr__(self, name: str) -> Any:
            def call(*args: Any) -> int:
                self.calls.append((name, args))
                return 0

            return call

    class Listener:
        pass

    class Observer:
        pass

    for name, value in {
        "AoqTrackType": TrackType,
        "AoqEncoderType": EncoderType,
        "AoqConnectionStatus": ConnectionStatus,
        "AoqErrorCode": ErrorCode,
        "AoqAudioSource": AudioSource,
        "AoqAudioFrameData": AudioFrameData,
        "AoqDataMsg": DataMsg,
        "AoqClientEngine": Engine,
        "AoqEngineEventListener": Listener,
        "IAudioFrameObserver": Observer,
    }.items():
        setattr(sdk, name, value)
    for name in (
        "AoqCreateConfig",
        "AoqAudioCodecConfig",
        "AoqAudioCaptureConfig",
        "AoqAudioPlaybackConfig",
        "AoqAudioExternalStreamConfig",
        "AoqAudioObserverConfig",
        "AoqVideoCodecConfig",
        "AoqVideoEncodedFrame",
        "AoqTrackParam",
        "AoqRelayEndpoint",
        "AoqConnectConfig",
    ):
        setattr(sdk, name, factory)
    return sdk


async def test_aoq_transport_with_fake_sdk() -> None:
    sdk = _fake_sdk()
    creds = _aoq.AOQCredentials(
        sid="sid", token="tok", cert_fingerprint="fp", relay_endpoints=[{"endpoint": "h"}]
    )

    async def fake_fetch(**kwargs: Any) -> _aoq.AOQCredentials:
        assert kwargs["url"].endswith("/api/v1/webrtc/realtime?model=m")
        return creds

    with (
        patch.dict(sys.modules, {"aoq_client_sdk": sdk}),
        patch.object(_aoq, "fetch_aoq_credentials", fake_fetch),
    ):
        transport = _aoq.AOQTransport(
            credentials_url="https://ws.cn-beijing.maas.aliyuncs.com/api/v1/webrtc/realtime?model=m",
            api_key=KEY,
            http_session=None,  # type: ignore[arg-type]
            timeout=1.0,
            options=_aoq.AOQOptions(),
        )
        await transport.connect()
        engine = sdk.AoqClientEngine.instance
        assert engine.connect_config.kwargs["token"] == "tok"
        assert engine.connect_config.kwargs["certificate"] == "fp"
        called = [name for name, _ in engine.calls]
        assert "start_audio_capture" in called and "add_audio_external_stream" in called
        # upstream audio stays off until the model is configured
        assert ("enable_send_media_stream", (sdk.AoqTrackType.AUDIO, False)) in engine.calls

        await transport.push_audio(b"\x00\x00" * 320)
        assert engine.pushed == []
        transport.on_session_updated()
        assert ("enable_send_media_stream", (sdk.AoqTrackType.AUDIO, True)) in engine.calls
        await transport.push_audio(b"\x00\x00" * 320)
        assert engine.pushed[0].samples_per_sec == 16000

        await transport.send_event({"type": "response.create"})
        assert engine.sent == [b'{"type": "response.create"}']

        # callbacks come from SDK threads
        def deliver() -> None:
            engine.listener.on_data_msg(sdk.AoqDataMsg(b'{"type": "session.updated"}'))
            engine.observer.on_playback_audio_frame(
                sdk.AoqAudioFrameData(data=b"", auto_gen_mute=True)
            )
            engine.observer.on_playback_audio_frame(
                sdk.AoqAudioFrameData(data=b"\x01\x00" * 240, samples_per_sec=24000)
            )

        threading.Thread(target=deliver).start()
        first = await asyncio.wait_for(transport.recv(), 1)
        second = await asyncio.wait_for(transport.recv(), 1)
        assert isinstance(first, ServerEvent) and first.event["type"] == "session.updated"
        assert isinstance(second, OutputAudio) and second.frame.samples_per_channel == 240

        # only one AOQ engine per process
        other = _aoq.AOQTransport(
            credentials_url="u",
            api_key=KEY,
            http_session=None,  # type: ignore[arg-type]
            timeout=1.0,
            options=_aoq.AOQOptions(),
        )
        with pytest.raises(Exception, match="single connection"):
            await other.connect()

        await transport.aclose()
        assert sdk.AoqClientEngine.instance is None
        assert isinstance(await transport.recv(), TransportClosed)
