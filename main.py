from kokoro import KPipeline
import soundfile as sf
import wave
from pathlib import Path
from typing import Generator

import numpy as np
from loguru import logger

def generate_audio(
    text: str, kokoro_language: str, voice: str, speed=1
) -> Generator["KPipeline.Result", None, None]:
    from kokoro import KPipeline

    if not voice.startswith(kokoro_language):
        logger.warning(f"Voice {voice} is not made for language {kokoro_language}")
    pipeline = KPipeline(lang_code=kokoro_language)
    yield from pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")


def generate_and_save_audio(
    output_file: Path, text: str, kokoro_language: str, voice: str, speed=1
) -> None:
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        wav_file.setframerate(24000)  # Sample rate

        for result in generate_audio(
            text, kokoro_language=kokoro_language, voice=voice, speed=speed
        ):
            logger.debug(result.phonemes)
            if result.audio is None:
                continue
            audio_bytes = (result.audio.numpy() * 32767).astype(np.int16).tobytes()
            wav_file.writeframes(audio_bytes)

def main():
    # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
    # 🇪🇸 'e' => Spanish es
    # 🇫🇷 'f' => French fr-fr
    # 🇮🇳 'h' => Hindi hi
    # 🇮🇹 'i' => Italian it
    # 🇯🇵 'j' => Japanese: pip install misaki[ja]
    # 🇧🇷 'p' => Brazilian Portuguese pt-br
    # 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]

    text = """
    In the world of Rails development, integrating large language models (LLMs) like OpenAI's GPT has become increasingly common. One challenge developers face is streaming these responses efficiently to provide a smooth user experience.

    This post will explore some different techniques for streaming LLM responses in Rails applications. We'll look at using server-sent events (SSE) and Turbo Streams as two different options for delivering streaming interfaces in Rails applications. We'll also provide some code examples for a demo chat application we made — it has three different bot personalities you can interact with through SSE or Turbo Streams.
    """

    generate_and_save_audio(
        Path("output/test.wav"),
        text,
        kokoro_language='a',
        voice='af_heart' # NOTE: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
    )

if __name__ == "__main__":
    main()
