from __future__ import annotations

import math
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.spatius.expressions import (
    _EXPRESSION_USERDATA_KEY,
    AvatarExpression,
    ExpressionController,
)

pytestmark = pytest.mark.unit

HAPPY = '<expr type="expression" label="happy" strength="0.5"/>'


async def _chunks(*values: str) -> AsyncIterator[str]:
    for value in values:
        yield value


async def _render(*chunks: str, expressive: bool = False) -> tuple[str, list[AvatarExpression]]:
    spoken: list[str] = []

    async def tts(
        agent: Agent, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        async for value in text:
            spoken.append(value)
        yield rtc.AudioFrame(b"\0\0" * 480, 24_000, 1, 480)

    agent = MagicMock()
    activity = agent._get_activity_or_raise.return_value
    activity._resolve_expressive_options.return_value = {} if expressive else None
    controller = ExpressionController(["neutral", "happy"], AvatarExpression("neutral"))
    with patch.object(Agent.default, "tts_node", new=tts):
        frames = [
            frame async for frame in controller.tts_node(agent, _chunks(*chunks), ModelSettings())
        ]
    return "".join(spoken), [f.userdata[_EXPRESSION_USERDATA_KEY].expression for f in frames]


def test_expression_validates_strength() -> None:
    assert AvatarExpression("happy").strength is None
    assert AvatarExpression("happy", 0).strength == 0.0
    for value in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            AvatarExpression(strength=value)
    bad: Any = True
    with pytest.raises(TypeError):
        AvatarExpression(strength=bad)


async def test_leading_marker_selects_the_reply_expression_and_is_not_spoken() -> None:
    raw = HAPPY + "Hello."
    spoken, selections = await _render(*raw)  # one character per chunk
    assert spoken == "Hello."
    assert selections == [AvatarExpression("happy", 0.5)]


async def test_reply_without_marker_uses_the_default() -> None:
    spoken, selections = await _render("Hello.")
    assert spoken == "Hello."
    assert selections == [AvatarExpression("neutral")]


async def test_expressive_mode_forwards_markup_to_the_tts() -> None:
    spoken, selections = await _render(HAPPY, "Hello.", expressive=True)
    assert spoken == HAPPY + "Hello."
    assert selections == [AvatarExpression("happy", 0.5)]
