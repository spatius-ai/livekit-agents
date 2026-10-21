from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

QwenOmniRealtimeModels = Literal[
    "qwen3.8-omni-flash-realtime",
    "qwen3.5-omni-plus-realtime",
    "qwen3.5-omni-flash-realtime",
    "qwen3-omni-flash-realtime",
    "qwen3-omni-flash-realtime-2025-09-15",
]

QwenRegion = Literal["cn-beijing", "ap-southeast-1"]
"""Model Studio region: ``cn-beijing`` (China mainland) or ``ap-southeast-1`` (Singapore)."""

QwenTransport = Literal["websocket", "aoq"]
"""How the session reaches the model.

* ``websocket``: JSON events over the Realtime WebSocket API, audio as base64 PCM.
* ``aoq``: the AOQ (MoQ) client SDK. Model events travel on its data track, audio and
  video on media tracks.
"""


@dataclass
class TurnDetection:
    """Server-side voice activity detection (``session.turn_detection``).

    Pass ``turn_detection=None`` to the model instead for manual turn taking, where the
    agent commits the audio buffer and asks for each reply.
    """

    type: Literal["server_vad", "semantic_vad"] = "server_vad"
    """``semantic_vad`` also filters out non-speech sounds."""
    threshold: float | None = 0.5
    """Speech probability threshold in [-1.0, 1.0]."""
    silence_duration_ms: int | None = 800
    """Silence that ends a turn, in [200, 6000] ms."""
    prefix_padding_ms: int | None = None
    idle_timeout_ms: int | None = None
    """Qwen3.5 with ``server_vad`` only, in [5000, 30000] ms."""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class InputAudioTranscription:
    """User speech transcription (``session.input_audio_transcription``)."""

    model: str = "qwen3-asr-flash-realtime"

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model}
