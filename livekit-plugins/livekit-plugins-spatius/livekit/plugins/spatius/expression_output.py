from __future__ import annotations

from uuid import uuid4
from weakref import WeakSet

from livekit import rtc
from livekit.agents.voice.avatar import QueueAudioOutput

from .expressions import (
    _EXPRESSION_USERDATA_KEY,
    ExpressionController,
    _ExpressionMetadata,
)


class ExpressionAudioOutput(QueueAudioOutput):
    """Keep each reply's expression with its audio, including during cancellation."""

    def __init__(self, *, sample_rate: int, expressions: ExpressionController) -> None:
        super().__init__(sample_rate=sample_rate, wait_playback_start=True)
        self._expressions = expressions
        self._selection: _ExpressionMetadata | None = None
        self._plain_segment = False
        self._accepted: WeakSet[_ExpressionMetadata] = WeakSet()
        self._cancelled: WeakSet[_ExpressionMetadata] = WeakSet()
        self.epoch = 0

    @staticmethod
    def metadata(frame: rtc.AudioFrame) -> _ExpressionMetadata:
        value = frame.userdata.get(_EXPRESSION_USERDATA_KEY)
        if not isinstance(value, _ExpressionMetadata):
            raise ValueError("avatar audio is missing its expression metadata")
        return value

    def is_cancelled(self, metadata: _ExpressionMetadata) -> bool:
        return metadata in self._cancelled

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if frame.sample_rate != self.sample_rate or frame.num_channels != 1:
            raise ValueError("avatar audio must use the configured sample rate and one channel")
        metadata = frame.userdata.get(_EXPRESSION_USERDATA_KEY)
        if metadata is not None:
            if not isinstance(metadata, _ExpressionMetadata):
                raise ValueError("invalid avatar expression metadata")
            if self.is_cancelled(metadata):
                return
            self._expressions.validate(metadata.expression)
            if self._selection is not None and (self._plain_segment or self._selection != metadata):
                raise ValueError("avatar expression changed within an audio segment")
        else:
            if self._selection is not None and not self._plain_segment:
                raise ValueError("avatar expression metadata disappeared within an audio segment")
            metadata = self._selection or _ExpressionMetadata(
                generation_id=uuid4().hex,
                expression=self._expressions.default_expression,
            )
            # Do not attach a session-local selection to a caller's reusable frame.
            copy = rtc.AudioFrame(
                data=bytes(frame.data),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel,
            )
            copy.userdata.update(frame.userdata)
            copy.userdata[_EXPRESSION_USERDATA_KEY] = metadata
            frame = copy
            self._plain_segment = True
        self._selection = metadata
        self._accepted.add(metadata)
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self._selection = None
        self._plain_segment = False

    def clear_buffer(self) -> None:
        self.epoch += 1
        self._cancelled.update(self._accepted)
        self._accepted.clear()
        # Reset both QueueAudioOutput and AudioOutput capture state. The resulting
        # stale end marker is drained with the discarded audio before interruption.
        self.flush()
        super().clear_buffer()
