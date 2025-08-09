#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

The example connects between client and server using a P2P WebRTC connection.

Run the bot using::

    python bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading AI models (30-40 seconds first run, <2 seconds after)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

import aiohttp
from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.services.heygen.api import NewSessionRequest

logger.info("✅ Pipeline components loaded")

logger.info("Loading WebRTC transport...")
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        options={
            "model": "nova-2",
            "detect_language": False,
            "languages": ["en"],
            "smart_format": True,
            "punctuate": True,
        },
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL", "gpt-5"))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI/ML basics tutor for beginners. "
                "Speak in clear, simple English with short sentences and concrete examples. "
                "One step per turn: ask only one question or give one instruction per turn, then wait. "
                "Use everyday analogies (recipes, maps, flashcards) to explain features, labels, training, validation, test sets, and overfitting. "
                "Prefer present tense and avoid heavy math unless the learner asks. "
                "Keep turns under 2 sentences unless explicitly asked for a longer explanation.\n\n"
                "QUIZ MODE: If the learner says 'start quiz', run a 5-question concept check on AI/ML fundamentals. "
                "Each turn: ask one short multiple-choice or short-answer question (e.g., supervised vs. unsupervised, features vs. labels, train/validation/test split, overfitting vs. underfitting, regression vs. classification). "
                "After the learner answers, reply with 'Correct' or 'Almost' and give a one-line explanation. Track a compact running score like '(2/5)'. After 5 questions, show the total and ask if they want another round."
            ),
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    async with aiohttp.ClientSession() as session:
        # Configure HeyGen service with custom avatar
        heygen = HeyGenVideoService(
            api_key=os.getenv("HEYGEN_API_KEY"),
            session=session,
            session_request=NewSessionRequest(
                avatar_id="June_HR_public"  # Or your custom avatar ID
            ),
        )
        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI processor
                stt,
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                heygen,
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation tailored for AI/ML tutoring.
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Start by greeting the learner and ask their name. Stop and wait for a reply. "
                        "After they reply, ask about their background (new to AI/ML, some experience, or refresher). "
                        "Then offer one topic to start with (e.g., features vs. labels, train/test split, overfitting) and wait for their choice."
                    ),
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,         # Enable video output
            video_out_is_live=True,         # Real-time video streaming
            video_out_width=1280,
            video_out_height=720,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
