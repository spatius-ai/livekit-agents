"""Spatius avatar expressions using public Agent hooks.

Configure LIVEKIT_*, SPATIUS_*, SPATIUS_EMOTIONS (comma-separated avatar labels),
STT_MODEL, LLM_MODEL, TTS_MODEL, and TTS_VOICE in the environment.
"""

import os
from collections.abc import AsyncIterable, AsyncIterator

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    ModelSettings,
    cli,
    inference,
)
from livekit.agents.types import TimedString
from livekit.plugins import spatius

server = AgentServer()


class ExpressiveAgent(Agent):
    def __init__(self, avatar: spatius.AvatarSession) -> None:
        super().__init__(
            instructions="You are a helpful conversational assistant. "
            + avatar.expression_instructions
        )
        self.avatar = avatar

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        return self.avatar.tts_node(self, text, model_settings)

    def transcription_node(
        self, text: AsyncIterable[str | TimedString], model_settings: ModelSettings
    ) -> AsyncIterator[str | TimedString]:
        return self.avatar.transcription_node(text)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    emotions = [
        label.strip() for label in os.environ["SPATIUS_EMOTIONS"].split(",") if label.strip()
    ]
    if not emotions:
        raise ValueError("SPATIUS_EMOTIONS must list the selected avatar's supported emotions")
    default = spatius.AvatarExpression(emotion=os.getenv("SPATIUS_DEFAULT_EMOTION"))
    avatar = spatius.AvatarSession(supported_emotions=emotions, default_expression=default)
    session: AgentSession[None] = AgentSession(
        stt=inference.STT(os.environ["STT_MODEL"]),
        llm=inference.LLM(os.environ["LLM_MODEL"]),
        tts=inference.TTS(os.environ["TTS_MODEL"], voice=os.environ["TTS_VOICE"]),
    )
    ctx.add_shutdown_callback(avatar.aclose)
    await avatar.start(session, room=ctx.room)
    await session.start(agent=ExpressiveAgent(avatar), room=ctx.room)
    session.generate_reply(instructions="Greet the user briefly.")


if __name__ == "__main__":
    cli.run_app(server)
