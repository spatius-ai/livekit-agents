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

Configure the selected avatar's exact supported labels once, then use the public
Agent hooks. One emotion and strength applies to each spoken reply, including all
of its TTS sentence chunks. Tool preambles and subsequent answers can be separate
replies. No LiveKit framework modifications are required.

```python
from collections.abc import AsyncIterable, AsyncIterator

from livekit import rtc
from livekit.agents import Agent, ModelSettings
from livekit.agents.types import TimedString
from livekit.plugins import spatius

avatar = spatius.AvatarSession(
    supported_emotions=["neutral", "happy", "sad"],  # Use your avatar's actual labels.
    default_expression=spatius.AvatarExpression("neutral"),
)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. " + avatar.expression_instructions
        )

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        return avatar.tts_node(self, text, model_settings)

    def transcription_node(
        self, text: AsyncIterable[str | TimedString], model_settings: ModelSettings
    ) -> AsyncIterator[str | TimedString]:
        return avatar.transcription_node(text)


# Start the avatar before generating speech:
await avatar.start(session, room=ctx.room)
await session.start(agent=Assistant(), room=ctx.room)
```

The instructions ask the LLM to begin each reply with a control marker such as
`<avatar-expression emotion="happy" strength="0.5"/>`. The TTS helper removes
avatar markers before synthesis, and the transcript helper removes them before
display. Install **both** hooks. Existing LiveKit voice-expression markup remains
available to the TTS provider; voice delivery and avatar expression are independent.
Only the leading marker chooses the expression; later markers are removed without
changing the active request.

Strength must be a finite number from `0` to `1`. Explicit `0` is preserved;
omitting strength uses the service default of `1`. Omitting emotion uses the
avatar default. Manually configured unsupported labels or invalid strengths raise
an error. Missing, malformed, or unsupported LLM selections use
`default_expression` (or the avatar default when unset). Parsing waits at most
0.5 seconds for a partial prefix and accepts markers up to 1024 characters. An
unclosed malformed tag is discarded until `>`, a newline, or 1024 discarded
characters, then text resumes. This recovery cannot distinguish remaining broken
attributes from ordinary speech, so use the generated instructions and valid tags.

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

`expression_audio` accepts mono PCM frames, uses a continuous resampler to match
the avatar session's rate, and retains one immutable selection for that invocation.
It can also wrap a public `realtime_audio_output_node` implementation for manual
selection. Realtime speech-to-speech models do not automatically emit the text
control marker. To override the LLM selection for a TTS reply, pass `expression=`
to `avatar.tts_node(...)` instead. Audio with no annotation uses the configured
defaults. These APIs do not store a session-wide “next emotion,” so speculative
or overlapping generations cannot change each other's selection.

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
