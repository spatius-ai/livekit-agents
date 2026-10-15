from __future__ import annotations

import asyncio
import math
from array import array
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, TimedString
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.spatius import expressions
from livekit.plugins.spatius.expressions import (
    _EXPRESSION_USERDATA_KEY,
    AvatarExpression,
    ExpressionController,
)

pytestmark = pytest.mark.unit


def _frame(
    samples: Sequence[int] | None = None,
    *,
    rate: int = 24_000,
    channels: int = 1,
    userdata: dict[str, Any] | None = None,
) -> rtc.AudioFrame:
    pcm = array("h", samples if samples is not None else [100] * 480)
    return rtc.AudioFrame(pcm.tobytes(), rate, channels, len(pcm) // channels, userdata=userdata)


async def _chunks(*values: Any) -> AsyncIterator[Any]:
    for value in values:
        yield value


async def _render(
    text: AsyncIterable[str],
    *,
    controller: ExpressionController | None = None,
    expression: AvatarExpression | None = None,
) -> tuple[str, list[rtc.AudioFrame]]:
    spoken: list[str] = []

    async def tts(
        agent: Agent, source: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        async for value in source:
            spoken.append(value)
        if spoken:
            yield _frame()
            yield _frame()

    controller = controller or ExpressionController(["neutral", "happy", "sad"])
    with patch.object(Agent.default, "tts_node", new=tts):
        output = [
            frame
            async for frame in controller.tts_node(
                MagicMock(), text, ModelSettings(), sample_rate=24_000, expression=expression
            )
        ]
    return "".join(spoken), output


def _selection(frame: rtc.AudioFrame) -> AvatarExpression:
    return frame.userdata[_EXPRESSION_USERDATA_KEY].expression


def test_expression_validates_types_and_preserves_omitted_and_zero() -> None:
    assert AvatarExpression().strength is None
    assert AvatarExpression("happy", 0).strength == 0.0
    assert AvatarExpression("happy", 1).strength == 1.0
    with pytest.raises(FrozenInstanceError):
        AvatarExpression().strength = 0.5  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, "0.5", object()])
def test_strength_rejects_non_numbers(value: Any) -> None:
    with pytest.raises(TypeError):
        AvatarExpression(strength=value)


@pytest.mark.parametrize("value", [-0.1, 1.01, math.nan, math.inf, -math.inf, 10**400])
def test_strength_rejects_invalid_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        AvatarExpression(strength=value)


@pytest.mark.parametrize("label", ["", "   "])
def test_emotion_rejects_empty_labels(label: str) -> None:
    with pytest.raises(ValueError):
        AvatarExpression(emotion=label)


def test_controller_requires_explicit_supported_labels() -> None:
    with pytest.raises(ValueError, match="supported_emotions"):
        ExpressionController(default_expression=AvatarExpression("happy"))
    with pytest.raises(TypeError):
        ExpressionController("happy")
    with pytest.raises(TypeError):
        ExpressionController([None])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        ExpressionController(default_expression=False)  # type: ignore[arg-type]
    controller = ExpressionController(["neutral", "happy"], AvatarExpression("neutral", 0.5))
    assert controller.default_expression == AvatarExpression("neutral", 0.5)
    assert '"happy"' in controller.instructions
    assert '"neutral"' in controller.instructions
    assert ExpressionController().instructions == ""


async def test_leading_directive_survives_every_token_boundary() -> None:
    marker = '<avatar-expression emotion="happy" strength="0"/>'
    for split in range(len(marker) + 1):
        spoken, frames = await _render(_chunks(marker[:split], marker[split:] + "Hello!"))
        assert spoken == "Hello!"
        assert [_selection(frame) for frame in frames] == [AvatarExpression("happy", 0)] * 2
        assert (
            frames[0].userdata[_EXPRESSION_USERDATA_KEY]
            is frames[1].userdata[_EXPRESSION_USERDATA_KEY]
        )


async def test_only_leading_directive_selects_expression_and_voice_markers_survive() -> None:
    spoken, frames = await _render(
        _chunks(
            '  <avatar-expression emotion="happy" strength="0.5"/>',
            '<expr type="expression" label="sad"/>Hello ',
            "<avatar-exp",
            'ression emotion="sad" strength="1"/>world.',
            "</avatar-expression>",
        )
    )
    assert spoken == '  <expr type="expression" label="sad"/>Hello world.'
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


@pytest.mark.parametrize(
    "marker",
    [
        '<avatar-expression emotion="other" strength="0.5"/>',
        '<avatar-expression emotion="happy" strength="nan"/>',
        '<avatar-expression emotion="happy" strength="inf"/>',
        '<avatar-expression emotion="happy" strength="1.1"/>',
        '<avatar-expression emotion="happy"/>',
        '<avatar-expression emotion="happy" emotion="sad" strength="0.5"/>',
        '<avatar-expression emotion="happy" strength="0.5">',
        "<avatar-expression malformed=yes/>",
        '<AVATAR-EXPRESSION emotion="happy" strength="0.5"/>',
    ],
)
async def test_invalid_marker_is_removed_and_uses_default(marker: str, caplog: Any) -> None:
    controller = ExpressionController(["neutral", "happy"], AvatarExpression("neutral", 0.25))
    spoken, frames = await _render(_chunks(marker + "Hello"), controller=controller)
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression("neutral", 0.25)
    assert "using configured defaults" in caplog.text
    assert marker not in caplog.text


@pytest.mark.parametrize("text", ["Hello!", "1 < 2", '<expr label="happy"/>', "x" * 5000])
async def test_ordinary_text_is_preserved_with_default_selection(text: str) -> None:
    spoken, frames = await _render(_chunks(text))
    assert spoken == text
    assert _selection(frames[0]) == AvatarExpression()


async def test_large_chunk_with_short_valid_leading_marker_selects_expression() -> None:
    spoken, frames = await _render(
        _chunks('<avatar-expression emotion="happy" strength="0.5"/>' + "x" * 5000)
    )
    assert spoken == "x" * 5000
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


async def test_manual_expression_wins_but_all_directives_are_removed() -> None:
    spoken, frames = await _render(
        _chunks('<avatar-expression emotion="happy" strength="1"/>Hello'),
        expression=AvatarExpression("sad", 0),
    )
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression("sad", 0)


async def test_delayed_first_token_does_not_spend_the_prefix_parsing_budget(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(expressions, "_PREFIX_TIMEOUT_SECONDS", 0.01)

    async def source() -> AsyncIterator[str]:
        yield ""
        await asyncio.sleep(0.03)
        yield '<avatar-expression emotion="happy" strength="0.5"/>Hello'

    spoken, frames = await _render(source())
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


async def test_prefix_timeout_does_not_cancel_or_drop_pending_text(monkeypatch: Any) -> None:
    monkeypatch.setattr(expressions, "_PREFIX_TIMEOUT_SECONDS", 0.01)
    started = asyncio.Event()
    resume = asyncio.Event()
    spoken: list[str] = []

    async def source() -> AsyncIterator[str]:
        yield "<avatar-expr"
        await resume.wait()
        yield 'ession emotion="happy" strength="1"/>Still spoken'

    async def tts(
        agent: Agent, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        started.set()
        async for value in text:
            spoken.append(value)
        yield _frame()

    async def collect() -> list[rtc.AudioFrame]:
        return [
            frame
            async for frame in ExpressionController(["happy"]).tts_node(
                MagicMock(), source(), ModelSettings(), sample_rate=24_000
            )
        ]

    with patch.object(Agent.default, "tts_node", new=tts):
        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(started.wait(), 1)
            assert not task.done()
            resume.set()
            frames = await asyncio.wait_for(task, 1)
        finally:
            task.cancel()
    assert "".join(spoken) == "Still spoken"
    assert _selection(frames[0]) == AvatarExpression()


async def test_oversized_directive_and_truncated_markers_do_not_reach_speech() -> None:
    marker = "<avatar-expression "
    spoken, frames = await _render(
        _chunks(
            marker + "x" * (expressions._MAX_PREFIX_CHARS - len(marker)), "Hello", "<avatar-expr"
        )
    )
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression()


async def test_unclosed_marker_recovers_at_newline_for_speech_and_transcript() -> None:
    chunks = ['<avatar-expression emotion="happy"', "\nHello"]
    spoken, frames = await _render(_chunks(*chunks))
    displayed = "".join(
        [value async for value in ExpressionController().transcription_node(_chunks(*chunks))]
    )
    assert spoken == displayed == "\nHello"
    assert _selection(frames[0]) == AvatarExpression()


async def test_cancelled_prefix_closes_upstream_and_does_not_affect_next_reply() -> None:
    waiting = asyncio.Event()
    closed = asyncio.Event()
    controller = ExpressionController(["happy"])

    async def source() -> AsyncIterator[str]:
        try:
            yield "<avatar-expression"
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    task = asyncio.create_task(_render(source(), controller=controller))
    await asyncio.wait_for(waiting.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    spoken, frames = await _render(_chunks("Next"), controller=controller)
    assert spoken == "Next"
    assert _selection(frames[0]) == AvatarExpression()


async def test_transcript_cleanup_preserves_timed_string_metadata() -> None:
    first = TimedString("<avatar-expr", start_time=0, end_time=0.1)
    second = TimedString('ession emotion="happy" strength="0.5"/>', start_time=0.1, end_time=0.2)
    spoken = TimedString(
        'Hello <avatar-expression emotion="sad" strength="1"/>world',
        start_time=1.0,
        end_time=2.0,
        confidence=0.8,
        start_time_offset=10.0,
        speaker_id="speaker",
    )
    output = [
        value
        async for value in ExpressionController().transcription_node(_chunks(first, second, spoken))
    ]
    assert "".join(output) == "Hello world"
    assert all(isinstance(value, TimedString) for value in output)
    assert all(value.__dict__ == spoken.__dict__ for value in output)
    assert str(spoken).endswith('strength="1"/>world')


async def test_transcript_and_tts_cleanup_agree_for_fragmented_and_malformed_tags() -> None:
    raw = (
        '<avatar-expression emotion="happy" strength="0.5"/>'
        'Hello <expr type="expression" label="sad"/>there '
        "<avatar-expression malformed>friend</avatar-expression><avatar-expr"
    )
    chunks = [raw[index : index + 3] for index in range(0, len(raw), 3)]
    spoken, _ = await _render(_chunks(*chunks))
    displayed = "".join(
        [value async for value in ExpressionController().transcription_node(_chunks(*chunks))]
    )
    assert spoken == displayed == 'Hello <expr type="expression" label="sad"/>there friend'


async def test_same_rate_audio_copies_userdata_and_isolated_frozen_metadata() -> None:
    source = _frame(userdata={"custom": "value"})
    controller = ExpressionController(["happy"])
    output = [
        frame
        async for frame in controller.expression_audio(
            _chunks(source, source), sample_rate=24_000, expression=AvatarExpression("happy", 0)
        )
    ]
    assert bytes(output[0].data) == bytes(source.data)
    assert source.userdata == {"custom": "value"}
    assert output[0].userdata["custom"] == "value"
    assert (
        output[0].userdata[_EXPRESSION_USERDATA_KEY] is output[1].userdata[_EXPRESSION_USERDATA_KEY]
    )
    with pytest.raises(FrozenInstanceError):
        output[0].userdata[_EXPRESSION_USERDATA_KEY].generation_id = "changed"


async def test_resampler_is_continuous_flushes_tail_and_keeps_transcripts_once() -> None:
    samples = [int(16000 * math.sin(2 * math.pi * 440 * n / 22050)) for n in range(2205)]
    boundaries = [0, 1, 24, 724, len(samples)]
    source = [
        _frame(
            samples[start:end],
            rate=22_050,
            userdata={USERDATA_TIMED_TRANSCRIPT: [TimedString(str(index))], "custom": "value"},
        )
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False))
    ]
    output = [
        frame
        async for frame in ExpressionController(["happy"]).expression_audio(
            _chunks(*source), sample_rate=24_000, expression=AvatarExpression("happy", 0.5)
        )
    ]
    reference = rtc.AudioResampler(22_050, 24_000)
    expected = reference.push(_frame(samples, rate=22_050)) + reference.flush()
    assert sum(frame.samples_per_channel for frame in output) == 2400
    actual_samples = array("h", b"".join(bytes(frame.data) for frame in output))
    expected_samples = array("h", b"".join(bytes(frame.data) for frame in expected))
    # Independent SoXR instances apply integer-output dither.
    assert (
        max(
            abs(actual - expected)
            for actual, expected in zip(actual_samples, expected_samples, strict=True)
        )
        <= 2
    )
    assert all(frame.sample_rate == 24_000 for frame in output)
    assert all(_selection(frame) == AvatarExpression("happy", 0.5) for frame in output)
    assert len({id(frame.userdata[_EXPRESSION_USERDATA_KEY]) for frame in output}) == 1
    assert [
        item for frame in output for item in frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
    ] == ["0", "1", "2", "3"]
    assert all(frame.userdata.get("custom") == "value" for frame in output[:-1])


async def test_overlapping_audio_generations_keep_independent_selections() -> None:
    release = asyncio.Event()

    async def source() -> AsyncIterator[rtc.AudioFrame]:
        yield _frame()
        await release.wait()
        yield _frame()

    controller = ExpressionController(["happy", "sad"])
    first = controller.expression_audio(
        source(), sample_rate=24_000, expression=AvatarExpression("happy", 0)
    )
    second = controller.expression_audio(
        source(), sample_rate=24_000, expression=AvatarExpression("sad", 1)
    )
    try:
        a, b = await asyncio.gather(anext(first), anext(second))
        assert _selection(a) == AvatarExpression("happy", 0)
        assert _selection(b) == AvatarExpression("sad", 1)
        assert (
            a.userdata[_EXPRESSION_USERDATA_KEY].generation_id
            != b.userdata[_EXPRESSION_USERDATA_KEY].generation_id
        )
        release.set()
        a2, b2 = await asyncio.gather(anext(first), anext(second))
        assert a.userdata[_EXPRESSION_USERDATA_KEY] is a2.userdata[_EXPRESSION_USERDATA_KEY]
        assert b.userdata[_EXPRESSION_USERDATA_KEY] is b2.userdata[_EXPRESSION_USERDATA_KEY]
    finally:
        await first.aclose()
        await second.aclose()


async def test_cancelled_and_empty_audio_do_not_leak_selection_into_next_reply() -> None:
    closed = asyncio.Event()
    controller = ExpressionController(["happy"])

    async def source() -> AsyncIterator[rtc.AudioFrame]:
        try:
            yield _frame()
            await asyncio.Event().wait()
        finally:
            closed.set()

    audio = controller.expression_audio(
        source(), sample_rate=24_000, expression=AvatarExpression("happy", 1)
    )
    previous = await anext(audio)
    waiting = asyncio.create_task(anext(audio))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert closed.is_set()
    assert [
        frame
        async for frame in controller.expression_audio(
            _chunks(), sample_rate=24_000, expression=AvatarExpression("happy", 1)
        )
    ] == []
    next_frame = [
        frame async for frame in controller.expression_audio(_chunks(_frame()), sample_rate=24_000)
    ][0]
    assert _selection(next_frame) == AvatarExpression()
    assert (
        previous.userdata[_EXPRESSION_USERDATA_KEY] != next_frame.userdata[_EXPRESSION_USERDATA_KEY]
    )


@pytest.mark.parametrize("sample_rate", [0, -1, True, 24_000.5])
async def test_invalid_output_rate_is_rejected(sample_rate: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _ = [
            frame
            async for frame in ExpressionController().expression_audio(
                _chunks(_frame()), sample_rate=sample_rate
            )
        ]


async def test_stereo_audio_and_mid_reply_rate_change_are_rejected() -> None:
    with pytest.raises(ValueError, match="mono"):
        _ = [
            frame
            async for frame in ExpressionController().expression_audio(
                _chunks(_frame(channels=2)), sample_rate=24_000
            )
        ]
    with pytest.raises(ValueError, match="remain constant"):
        _ = [
            frame
            async for frame in ExpressionController().expression_audio(
                _chunks(_frame(), _frame(rate=16_000)), sample_rate=24_000
            )
        ]
