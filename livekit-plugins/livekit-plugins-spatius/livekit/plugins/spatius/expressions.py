from __future__ import annotations

import asyncio
import contextlib
import json
import math
import uuid
import xml.etree.ElementTree as ET
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, TimedString
from livekit.agents.voice.agent import ModelSettings

from .log import logger

_EXPRESSION_USERDATA_KEY = "spatius.avatar_expression"
_PREFIX_TIMEOUT_SECONDS = 0.5
_MAX_PREFIX_CHARS = 1024
_OPEN_TAG = "<avatar-expression"
_CLOSE_TAG = "</avatar-expression"


@dataclass(frozen=True)
class AvatarExpression:
    """An avatar's emotion and optional intensity for one spoken reply.

    Omitted values use the avatar/service defaults. An explicit zero strength is
    distinct from an omitted strength.
    """

    emotion: str | None = None
    strength: float | None = None

    def __post_init__(self) -> None:
        if self.emotion is not None:
            if not isinstance(self.emotion, str):
                raise TypeError("emotion must be a string or None")
            if not self.emotion.strip():
                raise ValueError("emotion must not be empty")
        if self.strength is not None:
            if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
                raise TypeError("strength must be a number or None")
            if not 0 <= self.strength <= 1 or not math.isfinite(self.strength):
                raise ValueError("strength must be finite and between 0 and 1")
            object.__setattr__(self, "strength", float(self.strength))


@dataclass(frozen=True)
class _ExpressionMetadata:
    generation_id: str
    expression: AvatarExpression


class ExpressionController:
    def __init__(
        self,
        supported_emotions: Sequence[str] | None = None,
        default_expression: AvatarExpression | None = None,
    ) -> None:
        if isinstance(supported_emotions, str):
            raise TypeError("supported_emotions must be a sequence of labels")
        labels = tuple(supported_emotions or ())
        for label in labels:
            if not isinstance(label, str):
                raise TypeError("supported_emotions must contain string labels")
            AvatarExpression(emotion=label)
        self._supported_emotions = frozenset(labels)
        self._default_expression = self.validate(
            default_expression if default_expression is not None else AvatarExpression()
        )

    def validate(self, expression: AvatarExpression) -> AvatarExpression:
        if not isinstance(expression, AvatarExpression):
            raise TypeError("expression must be an AvatarExpression")
        if expression.emotion is not None and expression.emotion not in self._supported_emotions:
            raise ValueError("emotion must be one of the configured supported_emotions")
        return expression

    @property
    def default_expression(self) -> AvatarExpression:
        return self._default_expression

    @property
    def instructions(self) -> str:
        if not self._supported_emotions:
            return ""
        return (
            "Begin each spoken reply with exactly one control marker: "
            '<avatar-expression emotion="LABEL" strength="NUMBER"/>. '
            "Choose LABEL from "
            + json.dumps(sorted(self._supported_emotions))
            + ". Choose NUMBER from 0 to 1 for the avatar expression's intensity. "
            "Keep that selection for the entire reply. The marker is removed before speech "
            "and display. Other voice delivery markers may follow it."
        )

    async def tts_node(
        self,
        agent: Agent,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
        *,
        sample_rate: int,
        expression: AvatarExpression | None = None,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        reader = _TextReader(text)
        try:
            if expression is None:
                selected, prefix = await self._select_prefix(reader)
            else:
                selected, prefix = self.validate(expression), ""

            async def cleaned_text() -> AsyncGenerator[str, None]:
                stripper = _DirectiveStripper()
                for value in stripper.feed(prefix):
                    yield str(value)
                while True:
                    try:
                        chunk = await reader.read()
                    except StopAsyncIteration:
                        break
                    for value in stripper.feed(chunk):
                        yield str(value)
                for value in stripper.finish():
                    yield str(value)

            async with contextlib.aclosing(cleaned_text()) as clean:
                audio = Agent.default.tts_node(agent, clean, model_settings)
                async with contextlib.aclosing(
                    self.expression_audio(audio, sample_rate=sample_rate, expression=selected)
                ) as output:
                    async for frame in output:
                        yield frame
        finally:
            await reader.aclose()

    async def transcription_node(
        self, text: AsyncIterable[str | TimedString]
    ) -> AsyncIterator[str | TimedString]:
        stripper = _DirectiveStripper()
        iterator = text.__aiter__()
        try:
            async for chunk in iterator:
                for value in stripper.feed(chunk):
                    yield value
            for value in stripper.finish():
                yield value
        finally:
            await _close_iterator(iterator)

    async def expression_audio(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        *,
        sample_rate: int,
        expression: AvatarExpression | None = None,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        selected = self.validate(expression) if expression is not None else self._default_expression
        metadata = _ExpressionMetadata(uuid.uuid4().hex, selected)
        iterator = audio.__aiter__()
        input_rate: int | None = None
        resampler: rtc.AudioResampler | None = None
        pending_userdata: dict[str, Any] = {}
        try:
            async for frame in iterator:
                if frame.num_channels != 1:
                    raise ValueError("avatar expression audio must be mono")
                if frame.sample_rate <= 0:
                    raise ValueError("audio sample rate must be positive")
                if input_rate is None:
                    input_rate = frame.sample_rate
                    if input_rate != sample_rate:
                        logger.debug(
                            "resampling avatar audio from %s to %s", input_rate, sample_rate
                        )
                        resampler = rtc.AudioResampler(input_rate, sample_rate, num_channels=1)
                elif frame.sample_rate != input_rate:
                    raise ValueError("audio sample rate must remain constant within a reply")

                if resampler is None:
                    if frame.samples_per_channel:
                        yield _with_metadata(frame, frame.userdata, metadata)
                    continue

                _merge_userdata(pending_userdata, frame.userdata)
                for output in resampler.push(frame):
                    yield _with_metadata(output, pending_userdata, metadata)
                    pending_userdata.clear()

            if resampler is not None:
                for output in resampler.flush():
                    yield _with_metadata(output, pending_userdata, metadata)
                    pending_userdata.clear()
        finally:
            await _close_iterator(iterator)

    async def _select_prefix(self, reader: _TextReader) -> tuple[AvatarExpression, str]:
        prefix = ""
        deadline: float | None = None
        while True:
            candidate = prefix.lstrip()
            lowered = candidate.lower()
            if candidate:
                if lowered.startswith(_OPEN_TAG):
                    end = candidate.find(">")
                    if end >= 0:
                        try:
                            return self._parse_marker(candidate[: end + 1]), prefix
                        except (ET.ParseError, TypeError, ValueError):
                            break
                elif not _OPEN_TAG.startswith(lowered):
                    break
            if len(prefix) >= _MAX_PREFIX_CHARS:
                break
            remaining = None
            if deadline is not None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
            try:
                chunk = await reader.read(timeout=remaining)
                if chunk and deadline is None:
                    deadline = asyncio.get_running_loop().time() + _PREFIX_TIMEOUT_SECONDS
                prefix += chunk
            except (TimeoutError, StopAsyncIteration):
                break

        if self._supported_emotions or prefix.lstrip().lower().startswith("<avatar"):
            logger.warning("avatar expression selection unavailable; using configured defaults")
        return self._default_expression, prefix

    def _parse_marker(self, marker: str) -> AvatarExpression:
        if (
            len(marker) > _MAX_PREFIX_CHARS
            or not marker.rstrip().endswith("/>")
            or "\r" in marker
            or "\n" in marker
        ):
            raise ValueError("invalid avatar expression marker")
        element = ET.fromstring(marker)
        if element.tag != "avatar-expression" or set(element.attrib) != {"emotion", "strength"}:
            raise ValueError("invalid avatar expression marker")
        return self.validate(
            AvatarExpression(element.attrib["emotion"], float(element.attrib["strength"]))
        )


class _TextReader:
    def __init__(self, text: AsyncIterable[str]) -> None:
        self._iterator = text.__aiter__()
        self._pending: asyncio.Future[str] | None = None

    async def read(self, timeout: float | None = None) -> str:
        if self._pending is None:
            self._pending = asyncio.ensure_future(self._iterator.__anext__())
        if timeout is not None:
            done, _ = await asyncio.wait((self._pending,), timeout=timeout)
            if not done:
                raise TimeoutError
        pending = self._pending
        try:
            return await pending
        finally:
            self._pending = None

    async def aclose(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pending
            self._pending = None
        await _close_iterator(self._iterator)


class _DirectiveStripper:
    def __init__(self) -> None:
        self._candidate: list[tuple[str | TimedString, str]] = []
        self._discarding = False
        self._discarded_chars = 0

    def feed(self, text: str | TimedString) -> list[str | TimedString]:
        output: list[tuple[str | TimedString, list[str]]] = []

        def emit(source: str | TimedString, value: str) -> None:
            if output and output[-1][0] is source:
                output[-1][1].append(value)
            else:
                output.append((source, [value]))

        for char in text:
            if self._discarding:
                self._discarded_chars += 1
                if char == ">":
                    self._discarding = False
                elif char in "\r\n" or self._discarded_chars >= _MAX_PREFIX_CHARS:
                    self._discarding = False
                    logger.warning(
                        "avatar expression marker exceeded parsing bounds; resuming text"
                    )
                    if char in "\r\n":
                        emit(text, char)
                continue
            if not self._candidate and char != "<":
                emit(text, char)
                continue
            self._candidate.append((text, char))
            value = "".join(part for _, part in self._candidate).lower()
            if value in (_OPEN_TAG, _CLOSE_TAG):
                self._candidate.clear()
                self._discarding = True
                self._discarded_chars = len(value)
            elif not (_OPEN_TAG.startswith(value) or _CLOSE_TAG.startswith(value)):
                pending = self._candidate
                self._candidate = []
                if char == "<":
                    self._candidate.append(pending.pop())
                for source, part in pending:
                    emit(source, part)
        return [_replace_text(source, "".join(parts)) for source, parts in output]

    def finish(self) -> list[str | TimedString]:
        value = "".join(part for _, part in self._candidate).lower()
        output = []
        if not (value.startswith("<avatar") or value.startswith("</avatar")):
            for source, char in self._candidate:
                output.append(_replace_text(source, char))
        self._candidate.clear()
        return output


def _replace_text(source: str | TimedString, text: str) -> str | TimedString:
    if isinstance(source, TimedString):
        replaced = TimedString(text)
        replaced.__dict__.update(source.__dict__)
        return replaced
    return text


def _merge_userdata(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key == USERDATA_TIMED_TRANSCRIPT:
            target.setdefault(key, []).extend(value)
        elif key != _EXPRESSION_USERDATA_KEY:
            target[key] = value


def _with_metadata(
    frame: rtc.AudioFrame, userdata: dict[str, Any], metadata: _ExpressionMetadata
) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        frame.data,
        frame.sample_rate,
        frame.num_channels,
        frame.samples_per_channel,
        userdata={**userdata, _EXPRESSION_USERDATA_KEY: metadata},
    )


async def _close_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()
