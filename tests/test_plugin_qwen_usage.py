"""Qwen visual usage remains visible through LiveKit's image-token metric."""

from typing import Any

import pytest

from livekit.agents.metrics import RealtimeModelMetrics
from livekit.plugins import qwen
from tests.test_plugin_qwen_realtime import KEY, FakeTransport, _settle, _start

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("input_key", "output_key"),
    [
        ("input_tokens_details", "output_tokens_details"),
        ("input_token_details", "output_token_details"),
    ],
)
@pytest.mark.parametrize(
    ("visual_usage", "expected_visual_tokens"),
    [
        ({}, 0),
        ({"image_tokens": 12}, 12),
        ({"video_tokens": 594}, 594),
        ({"image_tokens": 12, "video_tokens": 1188}, 1200),
    ],
)
async def test_visual_usage_maps_image_and_video_to_livekit_metric(
    input_key: str,
    output_key: str,
    visual_usage: dict[str, int],
    expected_visual_tokens: int,
) -> None:
    transport = FakeTransport()
    session = await _start(qwen.RealtimeModel(api_key=KEY), transport)
    metrics: list[Any] = []
    session.on("metrics_collected", metrics.append)
    try:
        transport.server(type="response.created", response={"id": "visual_response"})
        transport.server(
            type="response.done",
            response={
                "id": "visual_response",
                "status": "completed",
                "usage": {
                    "input_tokens": 55 + expected_visual_tokens,
                    "output_tokens": 9,
                    "total_tokens": 64 + expected_visual_tokens,
                    input_key: {"audio_tokens": 44, "text_tokens": 11, **visual_usage},
                    output_key: {"audio_tokens": 7, "text_tokens": 2},
                },
            },
        )
        await _settle()
        metric = next(m for m in metrics if isinstance(m, RealtimeModelMetrics))
        assert metric.input_token_details.image_tokens == expected_visual_tokens
        assert metric.input_token_details.audio_tokens == 44
        assert metric.input_token_details.text_tokens == 11
        assert metric.input_tokens == 55 + expected_visual_tokens
        assert metric.output_tokens == 9
        assert metric.total_tokens == 64 + expected_visual_tokens
        assert metric.output_token_details.audio_tokens == 7
        assert metric.output_token_details.text_tokens == 2
    finally:
        await session.aclose()
