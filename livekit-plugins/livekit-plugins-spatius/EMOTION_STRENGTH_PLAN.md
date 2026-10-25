# Avatar emotion and strength: plugin integration

Status: implemented in the plugin through public Agent hooks. Each spoken reply
carries one emotion and strength, held until that reply's audio request finishes or
is interrupted. Changes within a reply are out of scope: a V3 request carries one
selection, and splitting a reply into several requests would break the
one-request/one-playback-completion relationship.

## Selection syntax

The LLM writes LiveKit's expressive markup with an optional strength attribute:

```text
<expr type="expression" label="happy" strength="0.5"/>I'm glad that worked!
```

Reusing the framework's marker, rather than a plugin-specific one, means:

- The framework's transcript sinks (`TranscriptMarkupStripper`) already strip it,
  so no transcription hook is needed. When expressive mode is off, the framework
  also scrubs `<expr>` markup from past assistant turns before each LLM call, so
  the LLM relies on the instructions rather than on its earlier markers.
- With expressive mode active, the same marker is lowered to the TTS provider's
  native voice controls, so one marker steers both voice and avatar.
- The framework's expr grammar parses attributes generically. Provider lowering
  reads only `type` and `label`, and the Gemini plugin's style extraction matches
  extra attributes, so `strength` passes through every path and is dropped from
  voice lowering. `tests/test_expressions.py` pins this behavior.

Without expressive mode the framework does not lower markup, so `tts_node` strips
every `<expr>` tag before synthesis. With it, all markup is forwarded unchanged.

Only the first expression marker of a reply selects the avatar expression. Other
leading markup (sounds, pauses) is skipped while looking for it; later expression
markers steer the voice only. Parsing is lenient: labels match discovered emotions
ignoring case and separator style, a free-form label selects the earliest
discovered emotion it names as a whole word, a missing or unreadable strength keeps
the configured default, and an out-of-range strength is clamped. A marker naming no
discovered emotion, or a missing marker, falls back to the configured or avatar
default. Prefix parsing waits at most 0.5 seconds after the first text token and
reads at most 1024 characters.

## Discovery

`avatar.start()` fetches the avatar detail once through the SDK's
`get_avatar_detail()`. Its `availableEmotions` (in service order) feed validation,
label matching, and `expression_instructions`; `defaultEmotion` is the fallback
emotion. Failed or cancelled discovery stops startup before audio output is attached.

## Carrying the selection with the audio

`tts_node` snapshots the selection with a unique generation ID and attaches it to
every frame under a namespaced `AudioFrame.userdata` key. `ExpressionAudioOutput`
reports no `sample_rate`, because the framework's resampler builds new frames and
drops `userdata`; it resamples to the session rate itself with one continuous
resampler per reply, flushing the tail before the segment end. The output opens a
playback segment on the first input frame, since the resampler may hold back all of
a short reply until flush.

The forwarder sends emotion and strength on the first SDK `send_audio` call of a
request and relies on the SDK to retain them for later packets; `AudioSegmentEnd`
ends the request. Conflicting metadata within a segment is rejected. Audio without
metadata uses the configured or avatar default.

## Interruption and completion

Interruption fences cancelled generation IDs, discards the resampler tail, orders
the SDK interrupt before replacement audio, and waits for the worker's playback
acknowledgement. The playback RPC has no request identifier, so a missing
acknowledgement after two seconds releases pending playback and closes the avatar
session; a new session is required. An unsolicited server interruption also closes
the session.

## Why simpler alternatives are insufficient

- Listening to `lk.expression` is too late for request configuration and has no
  strength.
- A mutable session-wide "next emotion" can apply a speculative or later reply's
  selection to earlier audio.
- A separate data-channel emotion message, as some avatar plugins use, is not
  synchronized with the audio request.

## Remaining work

- Replace the SDK Git pin with the minimum released V3 SDK version before publishing.
- Realtime speech-to-speech models emit no text marker; they use `expression_audio`
  with a manual selection.
