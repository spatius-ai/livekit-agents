# Qwen-Omni-Realtime plugin for LiveKit Agents

Support for [Qwen-Omni-Realtime](https://help.aliyun.com/zh/model-studio/realtime), the speech-to-speech
models of Alibaba Cloud Model Studio (百炼), as a realtime model for LiveKit Agents.

Sessions use JSON events over the Realtime WebSocket API.

## Installation

```bash
pip install livekit-plugins-qwen
```

## Pre-requisites

A Model Studio API key, in `DASHSCOPE_API_KEY`. Keys are region-specific: the plugin defaults to
`cn-beijing`; pass `region="ap-southeast-1"` (or set `DASHSCOPE_REGION`) for Singapore.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import qwen

session = AgentSession(
    llm=qwen.RealtimeModel(
        model="qwen3.8-omni-flash-realtime",
        voice="Tina",
    ),
)
```

Turn detection runs on the server (`server_vad`) by default. Use `semantic_vad` to also ignore
non-speech sounds, or `turn_detection=None` to let the agent drive turn taking:

```python
qwen.RealtimeModel(turn_detection=qwen.TurnDetection(type="semantic_vad", silence_duration_ms=600))
```

### WebSocket camera input

WebSocket sends at most one JPEG image per second. Keep the caller's camera sampler at 1 fps too,
so frames are reduced before reaching the plugin. Encoding runs off the event loop and retains only
the latest pending frame. Camera input works before microphone audio and while the microphone is
muted; generated silent PCM maintains the input timeline without opening a system microphone.
Typed user messages briefly use manual mode so the message item includes fresh camera context,
then restore the configured VAD mode. Model audio is delivered as soon as each WebSocket delta
arrives, without playback pacing, for consumers such as avatar audio pipelines.

## Notes

- Conversation history is append-only: items can be added (user text, tool results) but not
  deleted or truncated.
- Per-reply instructions (`generate_reply(instructions=...)`) go on `response.create` and in a temporary
  `session.update`, restored once the reply is done.
- Video frames are sent as JPEG, at most 720p and one frame per second.
