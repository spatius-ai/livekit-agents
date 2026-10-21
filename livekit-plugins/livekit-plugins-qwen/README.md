# Qwen-Omni-Realtime plugin for LiveKit Agents

Support for [Qwen-Omni-Realtime](https://help.aliyun.com/zh/model-studio/realtime), the speech-to-speech
models of Alibaba Cloud Model Studio (百炼), as a realtime model for LiveKit Agents.

Two transports are supported:

- **WebSocket** (default): JSON events over the Realtime WebSocket API.
- **AOQ**: Alibaba Cloud's MoQ-based transport, through the
  [AOQ client SDK](https://help.aliyun.com/zh/model-studio/realtime-aoq-access). Model events travel
  on its data track; audio (Opus) and video travel on media tracks.

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

### AOQ transport

The AOQ client SDK is not on PyPI. Download the Linux or macOS AOQ SDK from the
[SDK download page](https://help.aliyun.com/zh/model-studio/realtime-sdk-download); it holds
`aoq_client_sdk.py` and its native library. Then point the plugin at that directory and give it
your workspace, since AOQ credentials are issued by the workspace endpoint:

```python
qwen.RealtimeModel(
    transport="aoq",
    workspace_id="ws-xxxxxxxx",  # or DASHSCOPE_WORKSPACE_ID
    aoq_options=qwen.AOQOptions(sdk_path="/opt/AoqClientSdk"),  # or AOQ_CLIENT_SDK_PATH
)
```

The SDK engine is a process-wide singleton, so a process can hold one AOQ session at a time. Agent
jobs run in separate processes by default, so this only matters when several sessions share one.
Set `AOQOptions(publish_video=True)` to send video frames on a video track.

## Notes

- Conversation history is append-only: items can be added (user text, tool results) but not
  deleted or truncated.
- Per-reply instructions (`generate_reply(instructions=...)`) are applied with a temporary
  `session.update`, restored once the reply is done.
- Video frames are sent as JPEG at about one frame per second, at most 720p.
