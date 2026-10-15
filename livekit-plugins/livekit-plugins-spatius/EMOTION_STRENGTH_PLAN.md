# Avatar emotion and strength: proposed plugin integration

Status: design only. The local V3 migration is implemented separately. The agreed
first version selects one emotion and strength for each spoken reply, held until
that reply's audio request finishes or is interrupted. Sentence-level changes are
outside this first version.

## Existing behavior

LiveKit expressive mode injects a TTS markup guide into the LLM prompt. The LLM
emits delivery markers such as `<expr type="expression" label="happy"/>`; LiveKit
converts them to the active provider's syntax and removes them from displayed
transcripts. Provider support and expression vocabulary vary. The current fork
enables the lowering path for inference gateway TTS and non-streaming providers
adapted through `StreamAdapter`; arbitrary native-streaming TTS plugins are not
automatically eligible.

The `lk.expression` transcript attribute contains a display-oriented expression
and mood. It has no numeric strength, no sample offset, and represents the first
expression found in its transcript segment. It is unsuitable as the trigger for
avatar inference, which needs its selection before the first audio request.

The default `Agent.tts_node` yields `SynthesizedAudio.frame`, dropping the wrapper's
request and segment IDs. The current Spatius queue consequently receives only
`AudioFrame` values and an `AudioSegmentEnd`. These are the integration boundaries
to extend inside the plugin.

References in this checkout:

- `livekit-agents/livekit/agents/voice/agent_activity.py`: `_resolve_expressive_options`,
  `_inject_expressive_instructions`, raw LLM text forwarding.
- `livekit-agents/livekit/agents/tts/_provider_format.py`: expression instructions,
  normalization, provider lowering, `expression_attribute`.
- `livekit-agents/livekit/agents/voice/agent.py`: public `tts_node` and
  `transcription_node` hooks and their default implementations.
- `livekit-agents/livekit/agents/voice/generation.py`: audio forwarding/resampling.
- `livekit-agents/livekit/agents/voice/avatar/_queue_io.py`: queue and segment lifecycle.
- Official overview: https://docs.livekit.io/agents/models/tts/expressive/
- Public hooks: https://docs.livekit.io/agents/logic/nodes/

## Recommended first implementation

Keep all runtime changes in the Spatius plugin, using public Agent hooks from the
application. No framework modification is needed for this scope.

1. Add an immutable `AvatarExpression` value with `emotion` and optional `strength`.
   Accept only supported avatar labels and finite numeric strength in `[0, 1]`.
   Preserve the SDK distinction between omitted strength (service default `1`)
   and explicit `0`. Obtain available/default emotions once at startup through
   an explicit capability configuration or a public SDK avatar-detail helper.
   The current Python SDK has no such detail helper; adding one would be an SDK
   change, not a LiveKit framework change. Avoid private SDK token attributes.

2. Provide a plugin helper for the application's public `Agent.tts_node` override.
   It chooses the expression before starting synthesis, then delegates to
   `Agent.default.tts_node`. Snapshot the choice in that invocation, with a unique
   generation ID. Use the logical spoken output segment as the request boundary;
   a tool preamble and a later generated answer can be separate spoken replies.
   Do not create new backend requests for each PCM packet or TTS sentence chunk.

3. Normalize audio to the plugin output's configured rate with a continuous
   resampler scoped to the reply, flush its final samples, and only then attach
   the immutable selection and generation ID under a namespaced
   `AudioFrame.userdata` key. This avoids LiveKit's downstream resampler discarding
   metadata. Copy unrelated user metadata and validate the configured channel
   count. A rate mismatch must be observable and covered by tests.

4. Use a plugin-owned queue/output extension to carry the selection with the
   audio itself. On the first audio frame of the segment, call SDK
   `send_audio(..., emotion=..., strength=...)`; later packets omit those options
   and rely on the SDK's retained request selection. End the request on the
   existing `AudioSegmentEnd`. Reject conflicting metadata within that request.
   Audio without metadata uses explicit configured defaults or the avatar default.

5. Preserve the existing one-request/one-playback-completion relationship. On
   interruption, drop queued audio and its metadata together, fence cancelled
   generation IDs, order the SDK interrupt before replacement audio, and retain
   the existing completion timeout. Empty or cancelled speculative generations
   must not leave a pending expression for the next reply. Close releases all
   pending output state.

## How the LLM chooses the values

Use an explicit, leading plugin control marker with exact avatar labels and a
numeric strength, for example:

```text
<avatar-expression emotion="happy" strength="0.5"/>I'm glad that worked!
```

This is a proposed plugin syntax, not an existing LiveKit tag or implemented API.
The application adds the short rule and allowed labels to its Agent instructions.
The plugin helper parses the leading marker incrementally, handles split tokens,
validates it, and removes it before sending text to TTS. A companion helper in
the public transcription hook removes the control marker from displayed text.
Existing TTS delivery markup remains available to LiveKit expressive mode.

Only the leading selection affects a reply. Later avatar markers cannot change
an active backend request. Missing, malformed, unsupported, or nonfinite values
fall back to the configured/avatar default; record a generic diagnostic. Bound
the prefix parser's size and waiting time so malformed model output cannot hold
speech indefinitely. Do not wait for a complete reply or add a second LLM call.

The avatar directive works independently of whether the TTS supports expressive
mode. Voice delivery and facial expression may intentionally differ. An optional
future adapter can map known LiveKit expression labels to configured avatar
labels and strengths, but it must not pretend that a display mood supplies a
precise numeric intensity. Do not infer strength from audio amplitude.

Provide manual/application selection using the same value object for scripted
speech and realtime speech-to-speech agents. Automatic markers require a text
generation path; do not claim equivalent support for realtime audio models.

## Why simpler alternatives are insufficient

- Listening to `lk.expression` is too late for reliable inference configuration
  and loses strength and exact synthesis correlation.
- A mutable global `current_emotion`, a `next_emotion` slot, or an uncorrelated
  FIFO can apply a speculative/future reply's selection to earlier audio.
- `session.current_speech` refers to current playout, which may differ from the
  TTS generation already running ahead of it.
- Starting a new backend request whenever an inline sentence marker arrives
  requires additional audio segmentation and aggregation of playback callbacks.
  It is not part of the selected one-expression-per-reply implementation.

## Verification and implementation order

1. Add value validation and metadata-aware queue behavior. Verify explicit zero,
   omitted defaults, unsupported labels, frozen per-request selection, completion,
   close, and interruption ordering.
2. Add public-hook helpers and the bounded streaming parser. Verify split tags,
   ordinary text, malformed headers, transcript cleanup, and coexistence with TTS
   delivery markup without speaking the avatar directive.
3. Cover resampling, final tails, multiple TTS chunks, two overlapping/speculative
   generations, cancelled/empty replies, handoffs, and a late playback callback.
   Assert that each emitted audio request receives the intended selection once
   and produces one logical playback completion.
4. Add an example with one expression per reply. Exercise real V3 WebSocket and
   LiveKit delivery for default, zero, half, and full strength, plus interruption.
   Validate motion emotion weights and client rendering separately.

The V3 SDK dependency is currently pinned to tested commit
`731ecf164fe9f2dfd09825998c0624eeb9a001d8` for local development. Replace the Git
dependency with the minimum released V3 SDK version before publishing the plugin.
