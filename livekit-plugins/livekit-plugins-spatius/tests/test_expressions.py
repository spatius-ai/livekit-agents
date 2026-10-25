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
from livekit.agents.tts._provider_format import (
    TranscriptMarkupStripper,
    convert_markup,
    split_expr_markup,
)
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, TimedString
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.spatius import expressions
from livekit.plugins.spatius.expressions import (
    _EXPRESSION_USERDATA_KEY,
    AvatarExpression,
    ExpressionController,
)

pytestmark = pytest.mark.unit

HAPPY = '<expr type="expression" label="happy" strength="0.5"/>'


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


def _agent(*, expressive: bool) -> MagicMock:
    agent = MagicMock()
    activity = agent._get_activity_or_raise.return_value
    activity._resolve_expressive_options.return_value = {} if expressive else None
    return agent


async def _render(
    text: AsyncIterable[str],
    *,
    controller: ExpressionController | None = None,
    expression: AvatarExpression | None = None,
    expressive: bool = False,
) -> tuple[str, list[rtc.AudioFrame]]:
    spoken: list[str] = []

    async def tts(
        agent: Agent, source: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        async for value in source:
            spoken.append(value)
        if spoken:
            yield _frame(userdata={USERDATA_TIMED_TRANSCRIPT: [TimedString("Hello")]})
            yield _frame()

    controller = controller or ExpressionController(["neutral", "happy", "sad"])
    with patch.object(Agent.default, "tts_node", new=tts):
        output = [
            frame
            async for frame in controller.tts_node(
                _agent(expressive=expressive), text, ModelSettings(), expression=expression
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
    assert ExpressionController().instructions == ""


def test_instructions_teach_expr_markup_in_service_order_without_braces() -> None:
    instructions = ExpressionController(["sad", "happy_exaggerated", "anger"]).instructions
    assert '<expr type="expression" label="LABEL" strength="NUMBER"/>' in instructions
    assert '["sad", "happy_exaggerated", "anger"]' in instructions
    # safe inside the expressive template, which formats {placeholders}
    assert "{" not in instructions and "}" not in instructions


@pytest.mark.parametrize(
    "label,expected",
    [
        ("happy", "happy"),
        ("HAPPY", "happy"),
        (" happy ", "happy"),
        ("happy_exaggerated", "happy_exaggerated"),
        ("happy exaggerated", "happy_exaggerated"),
        ("Happy-Exaggerated", "happy_exaggerated"),
        ("warm, happy", "happy"),
        ("happy but a little sad", "happy"),
        ("very happy_exaggerated", "happy_exaggerated"),
        ("unhappy", None),
        ("sadly", None),
        ("surprised", None),
        ("", None),
    ],
)
def test_label_matching_is_lenient_but_whole_word(label: str, expected: str | None) -> None:
    controller = ExpressionController(["neutral", "happy", "happy_exaggerated", "sad"])
    assert controller.match_emotion(label) == expected


async def test_leading_marker_survives_every_token_boundary() -> None:
    for split in range(len(HAPPY) + 1):
        spoken, frames = await _render(_chunks(HAPPY[:split], HAPPY[split:] + "Hello!"))
        assert spoken == "Hello!"
        assert [_selection(frame) for frame in frames] == [AvatarExpression("happy", 0.5)] * 2
        assert (
            frames[0].userdata[_EXPRESSION_USERDATA_KEY]
            is frames[1].userdata[_EXPRESSION_USERDATA_KEY]
        )


@pytest.mark.parametrize(
    "marker,expected",
    [
        ('<expr type="expression" label="happy"/>', AvatarExpression("happy", 0.25)),
        ('<expr type="expression" label="happy" strength="0"/>', AvatarExpression("happy", 0)),
        ('<expr strength="1" label="happy" type="expression"/>', AvatarExpression("happy", 1)),
        ('<expr type="expression" label="happy" strength="1.5"/>', AvatarExpression("happy", 1)),
        ('<expr type="expression" label="happy" strength="-2"/>', AvatarExpression("happy", 0)),
        ('<expr type="expression" label="happy" strength="nan"/>', AvatarExpression("happy", 0.25)),
        (
            '<expr type="expression" label="happy" strength="lots"/>',
            AvatarExpression("happy", 0.25),
        ),
        ("<expr type='expression' label='happy' strength='0.5'/>", AvatarExpression("happy", 0.5)),
        ('<expr type="expression" label="happy" strength="0.5">', AvatarExpression("happy", 0.5)),
        ('<expr label="happy"/>', AvatarExpression("happy", 0.25)),
        ('<expr type="expression" label="warm, happy"/>', AvatarExpression("happy", 0.25)),
        ('<expr type="EXPRESSION" LABEL="Happy"/>', AvatarExpression("happy", 0.25)),
    ],
)
async def test_lenient_marker_parsing(marker: str, expected: AvatarExpression) -> None:
    controller = ExpressionController(["neutral", "happy"], AvatarExpression("neutral", 0.25))
    spoken, frames = await _render(_chunks(marker + "Hello"), controller=controller)
    assert spoken == "Hello"
    assert _selection(frames[0]) == expected


@pytest.mark.parametrize(
    "marker",
    [
        '<expr type="expression" label="other" strength="0.5"/>',
        '<expr type="expression" label="unhappy"/>',
        '<expr type="expression" strength="0.5"/>',
    ],
)
async def test_unresolved_marker_is_removed_and_uses_default(marker: str, caplog: Any) -> None:
    controller = ExpressionController(["neutral", "happy"], AvatarExpression("neutral", 0.25))
    spoken, frames = await _render(_chunks(marker + "Hello"), controller=controller)
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression("neutral", 0.25)
    assert "names no supported emotion" in caplog.text
    assert marker not in caplog.text


async def test_missing_marker_warns_and_uses_default(caplog: Any) -> None:
    spoken, frames = await _render(_chunks("Hello"))
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression()
    assert "no leading avatar expression marker" in caplog.text


async def test_other_leading_markup_is_skipped_to_find_the_expression() -> None:
    spoken, frames = await _render(
        _chunks(' <expr type="sound" label="laugh"/> <expr type="break" label="200ms"/>', HAPPY),
        expressive=True,
    )
    assert spoken.endswith(HAPPY)
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


async def test_only_first_expression_marker_selects_and_later_markup_is_stripped() -> None:
    spoken, frames = await _render(
        _chunks(
            HAPPY,
            'Hello <expr type="expression" label="sad" strength="1"/>world ',
            '<expr type="prosody" label="slow">and</expr> <ex',
            'pr type="sound" label="laugh"/>bye.',
        )
    )
    assert spoken == "Hello world and bye."
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


async def test_expressive_mode_forwards_all_markup_to_the_tts() -> None:
    raw = [HAPPY, 'Hello <expr type="expression" label="sad"/>world <exp', 'r type="sound"/>.']
    spoken, frames = await _render(_chunks(*raw), expressive=True)
    assert spoken == "".join(raw)
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


@pytest.mark.parametrize(
    "text", ["Hello!", "1 < 2", "a <b>c</b>", "<expression>", "x" * 5000, "<e", "<expr"]
)
async def test_ordinary_text_is_preserved_with_default_selection(text: str) -> None:
    spoken, frames = await _render(_chunks(text))
    assert spoken == text
    assert _selection(frames[0]) == AvatarExpression()


async def test_large_chunk_with_short_valid_leading_marker_selects_expression() -> None:
    spoken, frames = await _render(_chunks(HAPPY + "x" * 5000))
    assert spoken == "x" * 5000
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


async def test_manual_expression_wins_and_markup_is_still_stripped() -> None:
    spoken, frames = await _render(_chunks(HAPPY + "Hello"), expression=AvatarExpression("sad", 0))
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression("sad", 0)


async def test_manual_expression_must_be_supported() -> None:
    with pytest.raises(ValueError, match="supported_emotions"):
        await _render(_chunks("Hello"), expression=AvatarExpression("surprise"))


async def test_tts_frames_keep_their_userdata() -> None:
    _, frames = await _render(_chunks(HAPPY + "Hello"))
    assert frames[0].userdata[USERDATA_TIMED_TRANSCRIPT] == ["Hello"]


async def test_delayed_first_token_does_not_spend_the_prefix_parsing_budget(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(expressions, "_PREFIX_TIMEOUT_SECONDS", 0.01)

    async def source() -> AsyncIterator[str]:
        yield ""
        await asyncio.sleep(0.03)
        yield HAPPY + "Hello"

    spoken, frames = await _render(source())
    assert spoken == "Hello"
    assert _selection(frames[0]) == AvatarExpression("happy", 0.5)


async def test_prefix_timeout_does_not_cancel_or_drop_pending_text(monkeypatch: Any) -> None:
    monkeypatch.setattr(expressions, "_PREFIX_TIMEOUT_SECONDS", 0.01)
    started = asyncio.Event()
    resume = asyncio.Event()
    spoken: list[str] = []

    async def source() -> AsyncIterator[str]:
        yield '<expr type="expr'
        await resume.wait()
        yield 'ession" label="happy" strength="1"/>Still spoken'

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
                _agent(expressive=False), source(), ModelSettings()
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


async def test_oversized_marker_is_released_as_text_with_default_selection() -> None:
    marker = '<expr type="expression" label="' + "x" * expressions._MAX_PREFIX_CHARS
    spoken, frames = await _render(_chunks(marker, "Hello"))
    assert spoken == marker + "Hello"
    assert _selection(frames[0]) == AvatarExpression()


async def test_unclosed_marker_is_released_at_newline() -> None:
    spoken, frames = await _render(_chunks('<expr type="expression" label="happy"', "\nHello"))
    assert spoken == '<expr type="expression" label="happy"\nHello'
    assert _selection(frames[0]) == AvatarExpression()


async def test_cancelled_prefix_closes_upstream_and_does_not_affect_next_reply() -> None:
    waiting = asyncio.Event()
    closed = asyncio.Event()
    controller = ExpressionController(["happy"])

    async def source() -> AsyncIterator[str]:
        try:
            yield "<expr"
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


@pytest.mark.parametrize(
    "provider", ["cartesia", "inworld", "xai", "fishaudio", "fishaudio_drama", "gemini"]
)
def test_framework_markup_handling_ignores_strength(provider: str) -> None:
    with_strength = convert_markup(provider, HAPPY + " Hello")
    without = convert_markup(provider, '<expr type="expression" label="happy"/> Hello')
    assert with_strength.replace(' strength="0.5"', "") == without


def test_framework_transcript_stripping_removes_marker_with_strength() -> None:
    stripper = TranscriptMarkupStripper()
    raw = HAPPY + " Hello there."
    shown = "".join(stripper.push(raw[i : i + 5]) for i in range(0, len(raw), 5))
    assert shown + stripper.flush() == "Hello there."
    assert stripper.tags == [{"type": "expression", "value": "happy"}]
    assert split_expr_markup(raw) == ("Hello there.", [{"type": "expression", "value": "happy"}])


async def test_expression_audio_copies_frames_and_shares_frozen_metadata() -> None:
    source = _frame(userdata={"custom": "value"})
    controller = ExpressionController(["happy"])
    output = [
        frame
        async for frame in controller.expression_audio(
            _chunks(source, _frame(samples=[]), source), expression=AvatarExpression("happy", 0)
        )
    ]
    assert len(output) == 2
    assert bytes(output[0].data) == bytes(source.data)
    assert source.userdata == {"custom": "value"}
    assert output[0].userdata["custom"] == "value"
    assert (
        output[0].userdata[_EXPRESSION_USERDATA_KEY] is output[1].userdata[_EXPRESSION_USERDATA_KEY]
    )
    with pytest.raises(FrozenInstanceError):
        output[0].userdata[_EXPRESSION_USERDATA_KEY].generation_id = "changed"


async def test_expression_audio_leaves_the_sample_rate_to_the_output() -> None:
    output = [
        frame
        async for frame in ExpressionController().expression_audio(_chunks(_frame(rate=16_000)))
    ]
    assert output[0].sample_rate == 16_000


async def test_overlapping_audio_generations_keep_independent_selections() -> None:
    release = asyncio.Event()

    async def source() -> AsyncIterator[rtc.AudioFrame]:
        yield _frame()
        await release.wait()
        yield _frame()

    controller = ExpressionController(["happy", "sad"])
    first = controller.expression_audio(source(), expression=AvatarExpression("happy", 0))
    second = controller.expression_audio(source(), expression=AvatarExpression("sad", 1))
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

    audio = controller.expression_audio(source(), expression=AvatarExpression("happy", 1))
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
            _chunks(), expression=AvatarExpression("happy", 1)
        )
    ] == []
    next_frame = [frame async for frame in controller.expression_audio(_chunks(_frame()))][0]
    assert _selection(next_frame) == AvatarExpression()
    assert (
        previous.userdata[_EXPRESSION_USERDATA_KEY] != next_frame.userdata[_EXPRESSION_USERDATA_KEY]
    )
