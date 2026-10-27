from __future__ import annotations

from uuid import uuid4
from weakref import WeakSet

from livekit import rtc
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.agents.voice.io import AudioOutput

from .expressions import (
    _EXPRESSION_USERDATA_KEY,
    ExpressionController,
    _ExpressionMetadata,
    _with_metadata,
)


class ExpressionAudioOutput(QueueAudioOutput):
    """Keep each reply's expression with its audio, including during cancellation.

    Audio is resampled to ``target_sample_rate`` here rather than by the framework,
    whose resampler drops the frame ``userdata`` that carries the expression. The
    output therefore reports no ``sample_rate``, so frames arrive at the TTS rate.
    """

    def __init__(self, *, sample_rate: int, expressions: ExpressionController) -> None:
        super().__init__(sample_rate=None, wait_playback_start=True)
        self.target_sample_rate = sample_rate
        self._expressions = expressions
        self._selection: _ExpressionMetadata | None = None
        self._plain_segment = False
        self._accepted: WeakSet[_ExpressionMetadata] = WeakSet()
        self._cancelled: WeakSet[_ExpressionMetadata] = WeakSet()
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_rate: int | None = None
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
        if frame.num_channels != 1:
            raise ValueError("avatar audio must have one channel")
        if frame.sample_rate <= 0:
            raise ValueError("avatar audio sample rate must be positive")
        if not frame.samples_per_channel:
            return
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
            self._plain_segment = True
        self._selection = metadata
        self._accepted.add(metadata)

        # open the segment on the input frame: the resampler may hold all of a short
        # reply until flush(). QueueAudioOutput.capture_frame would queue the frame as is.
        await AudioOutput.capture_frame(self, frame)
        self._capturing = True
        for output in self._resample(frame):
            await self._data_ch.send(_with_metadata(output, metadata))

    def _resample(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        out: list[rtc.AudioFrame] = []
        if self._resampler is not None and self._resampler_rate != frame.sample_rate:
            # the input rate changed mid-segment: send the old resampler's tail first
            out.extend(self._resampler.flush())
            self._resampler = None
            self._resampler_rate = None
        if frame.sample_rate == self.target_sample_rate:
            out.append(frame)
            return out
        if self._resampler is None:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self.target_sample_rate, num_channels=1
            )
            self._resampler_rate = frame.sample_rate
        out.extend(self._resampler.push(frame))
        return out

    def flush(self) -> None:
        resampler, self._resampler, self._resampler_rate = self._resampler, None, None
        if resampler is not None and self._capturing and self._selection is not None:
            for output in resampler.flush():
                self._data_ch.send_nowait(_with_metadata(output, self._selection))
        super().flush()
        self._selection = None
        self._plain_segment = False

    def clear_buffer(self) -> None:
        self.epoch += 1
        self._cancelled.update(self._accepted)
        self._accepted.clear()
        # discard the resampler's tail with the rest of the cancelled audio
        self._resampler = None
        self._resampler_rate = None
        # Reset both QueueAudioOutput and AudioOutput capture state. The resulting
        # stale end marker is drained with the discarded audio before interruption.
        self.flush()
        super().clear_buffer()
