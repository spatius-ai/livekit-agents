# LiveKit Plugins Spatius

Agent Framework plugin for [Spatius](https://www.spatius.ai/?utm_source=livekit) avatars.

See the [Spatius documentation](https://docs.spatius.ai?utm_source=livekit) for Spatius account setup and
avatar configuration.

## Client-side rendering

Spatius avatars are rendered on the client instead of being sent as conventional
server-rendered video. The avatar's LiveKit video track carries motion data in
otherwise black frames, so a standard LiveKit video renderer will display a black
screen. Your frontend must use the Spatius client SDK and LiveKit adapter to decode
the track and render the avatar.

See the [client integration guide](https://docs.spatius.ai/livekit-agents/client?utm_source=livekit) and
the [reference frontend](https://github.com/spatius-ai/spatius-avatar-demo/tree/main/platform-integrations/livekit-agents-demo/livekit-agents-reference-demo/frontend)
for a working implementation.

## Installation

This checkout targets the V3 WebSocket API and currently pins the Spatius SDK
commit tested with that protocol. Install the local plugin for development:

```bash
uv pip install -e livekit-plugins/livekit-plugins-spatius
```

The SDK pin is temporary until the V3 SDK is released. Before publishing this
plugin, replace it with the minimum released V3 SDK version. The LiveKit room and
avatar publisher identity come from a locally generated room token; the plugin
passes that token to the SDK's `LiveKitTransportConfig`.

Automatically resolved endpoints use V3. If `SPATIUS_INGRESS_ENDPOINT` is set,
update any `/v2/driveningress` path to `/v3/driveningress`; the SDK appends
`/websocket` and preserves explicit endpoint overrides.

## Usage

```python
from livekit.plugins import spatius

avatar = spatius.AvatarSession()
await avatar.start(session, room=ctx.room)
```

The plugin reads `SPATIUS_API_KEY`, `SPATIUS_APP_ID`, and `SPATIUS_AVATAR_ID` from the
environment when constructor arguments are omitted.

## Warm-up

At session start, the Spatius SDK resolves the ingress region via the bootstrap API
(with the default `region="auto"`) and exchanges the API key for a session token —
two sequential HTTPS round trips on the room-join critical path. Register the
plugin's `prewarm` function to move that work into process warm-up, before a job is
dispatched:

```python
from livekit import agents
from livekit.plugins import spatius


def prewarm(proc: agents.JobProcess):
    spatius.prewarm(proc)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
```

The spatius SDK caches the resolved region (5 minutes) and the prefetched session
token (until shortly before expiry) process-wide, so `AvatarSession.start()` in the
dispatched job reuses them and goes straight to the ingress WebSocket connect. It
also opens throwaway TLS connections to the console and ingress endpoints to prime
DNS and the shared TLS session cache. Warm-up is best-effort and never fails
process initialization; without it, everything resolves inline as before.

Session-token prefetch assumes the Spatius backend allows a token to back more than
one session; opt out with `spatius.prewarm(proc, prefetch_session_token=False)`.

## Emotion and strength

The plugin discovers `availableEmotions` and `defaultEmotion` from the selected
avatar's V3 detail API once during `avatar.start()` and retains them for the session.
No `SPATIUS_EMOTIONS` or manual supported-label list is needed. Each spoken reply,
including all of its TTS sentence chunks, carries one emotion and strength. Tool
preambles and subsequent answers can be separate replies. No LiveKit framework
modifications are required.

The LLM selects the expression with LiveKit's expressive markup, opening each reply
with a marker such as `<expr type="expression" label="happy" strength="0.5"/>`.
Override `tts_node` and add `avatar.expression_instructions` to the prompt:

```python
from collections.abc import AsyncIterable, AsyncIterator

from livekit import rtc
from livekit.agents import Agent, ModelSettings
from livekit.plugins import spatius

avatar = spatius.AvatarSession()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. " + avatar.expression_instructions
        )

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        return avatar.tts_node(self, text, model_settings)


# Discover capabilities before building the agent instructions:
await avatar.start(session, room=ctx.room)
await session.start(agent=Assistant(), room=ctx.room)
```

The framework's transcript sinks strip `<expr>` markup, so no transcription hook is
needed. Without expressive mode the TTS helper strips the markup before synthesis.
With [expressive mode](https://docs.livekit.io/agents/models/tts/expressive/), the
markup goes on to the TTS, which lowers it to native voice controls, so one marker
steers both the voice and the avatar. Pass the instructions to the expressive prompt
instead of the agent instructions:

```python
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            expressive={"tts_instructions_append": avatar.expression_instructions},
        )
```

The TTS receives the avatar's label as the voice expression. Providers with a fixed
vocabulary may not recognize every avatar label (for example `happy_exaggerated`);
the `strength` attribute steers the avatar only and is ignored by voice lowering.

Only the first expression marker chooses the avatar expression; later markers steer
the voice (in expressive mode) without changing the active request. Other leading
markup, such as a sound or a pause, is skipped while looking for it. Parsing is
lenient:

- `label` matches a discovered emotion ignoring case, with spaces, hyphens, and
  underscores treated alike. A free-form label selects the earliest discovered
  emotion it names as a whole word (`"warm, happy"` selects `happy`;
  `"unhappy"` selects none).
- `strength` is optional. When it is missing or not a number, the configured
  default strength applies; numbers outside `0`–`1` are clamped.
- A marker naming no discovered emotion, or a reply with no marker, uses
  `default_expression` (or the avatar default when unset).

Parsing waits at most 0.5 seconds for a partial prefix and accepts markers up to
1024 characters. A tag left unclosed at a newline or past that limit is treated as
text.

Explicit `0` strength is preserved; omitting strength uses the service default of
`1`. Omitting emotion uses the avatar default. An optional
`default_expression=AvatarExpression(...)` overrides that policy; a strength-only
override retains the discovered default emotion. Explicit default labels are
validated against the discovered list during startup. `avatar.available_emotions`,
`avatar.default_emotion`, and `avatar.expression_instructions` are available after
`avatar.start()`. Metadata lookup or validation failures stop startup before audio
forwarding begins. Manually selected unsupported labels or invalid strengths raise
an error.

For scripted audio, select the expression on the audio stream itself:

```python
await session.say(
    "I'm glad that worked!",
    audio=avatar.expression_audio(
        pcm_frames,
        expression=spatius.AvatarExpression("happy", 0.5),
    ),
)
```

`expression_audio` retains one immutable selection for that invocation. It can also
wrap a public `realtime_audio_output_node` implementation for manual selection.
Realtime speech-to-speech models do not automatically emit the text marker. To
override the LLM selection for a TTS reply, pass `expression=` to
`avatar.tts_node(...)` instead. Audio with no annotation uses the configured
defaults. These APIs do not store a session-wide "next emotion," so speculative or
overlapping generations cannot change each other's selection.

The avatar's audio output accepts mono PCM at any rate and resamples it continuously
to the session rate, flushing the resampler tail at the end of each reply. It
resamples itself because the framework's resampler drops the frame metadata that
carries each reply's expression.

On normal interruption, replacement audio waits for the worker's acknowledgement.
If that acknowledgement does not arrive within two seconds, pending playback is
released and the avatar session closes. Start a new avatar session to recover;
playback callbacks currently have no request identifier, so reusing a timed-out
session could apply an old completion to a new reply. An unsolicited server
interruption also closes the session.

To run the opt-in transport integration test, configure the usual `LIVEKIT_*` and
`SPATIUS_*` environment variables plus `SPATIUS_TEST_AUDIO_PCM` pointing to at least
four seconds of non-silent raw 16 kHz mono signed 16-bit PCM, then run:

```bash
SPATIUS_RUN_LIVEKIT_TEST=1 uv run pytest \
  livekit-plugins/livekit-plugins-spatius/tests/test_live_expression.py -q
```

The test creates and deletes a temporary LiveKit room. It exercises real SDK,
ingress, audio delivery, playback RPCs, and interruption. Client rendering and
motion-weight decoding are separate checks.

See [the runnable example](../../examples/avatar/spatius_expression.py) and
[the implementation notes](EMOTION_STRENGTH_PLAN.md). The SDK Git pin described
above still needs a released version before publishing this plugin.
