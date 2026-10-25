"""Avatar expression selection from LiveKit's expressive markup.

The LLM opens a reply with ``<expr type="expression" label="happy" strength="0.5"/>``,
the same marker LiveKit's expressive mode teaches for voice delivery. The first
expression marker selects the avatar's emotion and strength for the whole reply, since
a V3 request carries one selection. ``strength`` is optional; the framework's markup
lowering and transcript stripping ignore it.

With expressive mode active the markup goes on to the TTS, which lowers it to native
voice controls, so one marker steers both voice and avatar. Otherwise it is stripped
before synthesis so the TTS does not read it aloud. Transcripts need no extra hook:
the framework's transcript sinks strip ``<expr>`` markup either way.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.voice.agent import ModelSettings

from .log import logger

_EXPRESSION_USERDATA_KEY = "spatius.avatar_expression"
_PREFIX_TIMEOUT_SECONDS = 0.5
_MAX_PREFIX_CHARS = 1024

# the framework's expr grammar (livekit.agents.tts._provider_format): opening,
# self-closing, or closing; ``attrs`` is None for a closing tag
_EXPR_TAG_RE = re.compile(r"<expr\b(?P<attrs>[^>]*?)/?\s*>|</expr\s*>")
# an expr tag cut off by the end of the text, which the next chunk may complete
_EXPR_PARTIAL_RE = re.compile(rf"</?(?:e|ex|exp|expr(?:\W[^>\n]{{0,{_MAX_PREFIX_CHARS}}})?)?\Z")
_EXPR_ATTR_RE = re.compile(r"""([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_LABEL_SEPARATOR_RE = re.compile(r"[\s_-]+")


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
        # keep the service's order: it is the order the instructions offer the labels in
        self._labels = labels
        self._supported_emotions = frozenset(labels)
        self._labels_by_key: dict[str, str] = {}
        for label in labels:
            self._labels_by_key.setdefault(_label_key(label), label)
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
        """Prompt text teaching the leading expression marker.

        It has no braces, so it is also safe as an expressive ``tts_instructions_append``.
        """
        if not self._labels:
            return ""
        return (
            "Your avatar's face follows an expression marker at the start of each spoken "
            'reply: <expr type="expression" label="LABEL" strength="NUMBER"/>. Choose LABEL '
            "from " + json.dumps(list(self._labels)) + ". strength is optional: a number "
            "from 0 to 1 for how strongly the face shows it. The face holds the first "
            "expression marker for the whole reply; later expression markers do not "
            "change it."
        )

    def match_emotion(self, label: str) -> str | None:
        """The supported emotion an expression ``label`` names, if any.

        An exact match wins, ignoring case and treating spaces, hyphens, and underscores
        alike. Otherwise the earliest supported label that appears as a whole word, the
        longest at the same position: "warm, happy" is happy, "unhappy" names none.
        """
        key = _label_key(label)
        if not key:
            return None
        if key in self._labels_by_key:
            return self._labels_by_key[key]
        best: tuple[int, int, str] | None = None
        for candidate, original in self._labels_by_key.items():
            match = re.search(rf"(?<![^\W_]){re.escape(candidate)}(?![^\W_])", key)
            if match is not None:
                ranked = (match.start(), -len(candidate), original)
                best = ranked if best is None else min(best, ranked)
        return best[2] if best is not None else None

    async def tts_node(
        self,
        agent: Agent,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
        *,
        expression: AvatarExpression | None = None,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        reader = _TextReader(text)
        try:
            # expressive mode lowers expr markup in the TTS, so the marker also steers the
            # voice; without it the TTS would read the markup aloud
            forward_markup = agent._get_activity_or_raise()._resolve_expressive_options()
            if expression is None:
                selected, prefix = await self._select_prefix(reader)
            else:
                selected, prefix = self.validate(expression), ""

            async def spoken_text() -> AsyncGenerator[str, None]:
                stripper = None if forward_markup is not None else _MarkupStripper()
                chunk = prefix
                while True:
                    out = stripper.push(chunk) if stripper is not None else chunk
                    if out:
                        yield out
                    try:
                        chunk = await reader.read()
                    except StopAsyncIteration:
                        break
                if stripper is not None and (out := stripper.flush()):
                    yield out

            async with contextlib.aclosing(spoken_text()) as spoken:
                audio = Agent.default.tts_node(agent, spoken, model_settings)
                async with contextlib.aclosing(
                    self.expression_audio(audio, expression=selected)
                ) as output:
                    async for frame in output:
                        yield frame
        finally:
            await reader.aclose()

    async def expression_audio(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        *,
        expression: AvatarExpression | None = None,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        selected = self.validate(expression) if expression is not None else self._default_expression
        metadata = _ExpressionMetadata(uuid.uuid4().hex, selected)
        iterator = audio.__aiter__()
        try:
            async for frame in iterator:
                if frame.samples_per_channel:
                    yield _with_metadata(frame, metadata)
        finally:
            await _close_iterator(iterator)

    async def _select_prefix(self, reader: _TextReader) -> tuple[AvatarExpression, str]:
        prefix = ""
        deadline: float | None = None
        attrs: dict[str, str] | None = None
        while True:
            complete, attrs = _leading_expression(prefix)
            if complete or len(prefix) >= _MAX_PREFIX_CHARS:
                break
            remaining = None
            if deadline is not None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
            try:
                chunk = await reader.read(timeout=remaining)
            except (TimeoutError, StopAsyncIteration):
                break
            if chunk and deadline is None:
                deadline = asyncio.get_running_loop().time() + _PREFIX_TIMEOUT_SECONDS
            prefix += chunk

        if attrs is not None and (selected := self._resolve(attrs)) is not None:
            return selected, prefix
        if self._labels:
            if attrs is not None:
                logger.warning(
                    "avatar expression marker names no supported emotion; using defaults"
                )
            else:
                logger.warning("reply has no leading avatar expression marker; using defaults")
        return self._default_expression, prefix

    def _resolve(self, attrs: dict[str, str]) -> AvatarExpression | None:
        emotion = self.match_emotion(attrs.get("label", ""))
        if emotion is None:
            return None
        return AvatarExpression(emotion, self._parse_strength(attrs.get("strength")))

    def _parse_strength(self, value: str | None) -> float | None:
        # a missing or unreadable strength keeps the configured one; an out-of-range
        # number is clamped rather than throwing away the emotion it came with
        default = self._default_expression.strength
        if value is None:
            return default
        try:
            number = float(value)
        except ValueError:
            return default
        if not math.isfinite(number):
            return default
        return min(1.0, max(0.0, number))


def _label_key(label: str) -> str:
    return _LABEL_SEPARATOR_RE.sub("_", label.strip().lower())


def _leading_expression(text: str) -> tuple[bool, dict[str, str] | None]:
    """Find the expression marker heading ``text``, skipping other leading expr markup.

    Returns whether the search is complete and the marker's attributes, if one was
    found. It is incomplete while ``text`` is only whitespace and markup, or ends in
    a cut-off tag; complete once spoken text begins or an expression marker is found.
    """
    pos = 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            return False, None
        if (match := _EXPR_TAG_RE.match(text, pos)) is not None:
            if (raw := match.group("attrs")) is not None:
                attrs = {
                    name.lower(): double or single
                    for name, double, single in _EXPR_ATTR_RE.findall(raw)
                }
                if attrs.get("type", "expression").strip().lower() == "expression":
                    return True, attrs
            pos = match.end()
            continue
        if _EXPR_PARTIAL_RE.match(text, pos):
            return False, None
        return True, None


class _MarkupStripper:
    """Removes expr markup from streamed text, holding back a tag split across chunks."""

    def __init__(self) -> None:
        self._buf = ""

    def push(self, text: str) -> str:
        self._buf += text
        return self._consume(final=False)

    def flush(self) -> str:
        return self._consume(final=True)

    def _consume(self, *, final: bool) -> str:
        buf, pos = self._buf, 0
        out: list[str] = []
        while (start := buf.find("<", pos)) != -1:
            out.append(buf[pos:start])
            if (match := _EXPR_TAG_RE.match(buf, start)) is not None:
                pos = match.end()
            elif not final and _EXPR_PARTIAL_RE.match(buf, start):
                self._buf = buf[start:]
                return "".join(out)
            else:
                out.append("<")  # not markup after all
                pos = start + 1
        out.append(buf[pos:])
        self._buf = ""
        return "".join(out)


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


def _with_metadata(frame: rtc.AudioFrame, metadata: _ExpressionMetadata) -> rtc.AudioFrame:
    # a new frame, so a caller's reusable frame never carries a reply's selection
    return rtc.AudioFrame(
        frame.data,
        frame.sample_rate,
        frame.num_channels,
        frame.samples_per_channel,
        userdata={**frame.userdata, _EXPRESSION_USERDATA_KEY: metadata},
    )


async def _close_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()
