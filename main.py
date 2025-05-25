from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch

def main():
    # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
    # 🇪🇸 'e' => Spanish es
    # 🇫🇷 'f' => French fr-fr
    # 🇮🇳 'h' => Hindi hi
    # 🇮🇹 'i' => Italian it
    # 🇯🇵 'j' => Japanese: pip install misaki[ja]
    # 🇧🇷 'p' => Brazilian Portuguese pt-br
    # 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
    pipeline = KPipeline(lang_code='a')

    text = """
    In the world of Rails development, integrating large language models (LLMs) like OpenAI's GPT has become increasingly common. One challenge developers face is streaming these responses efficiently to provide a smooth user experience.

    This post will explore some different techniques for streaming LLM responses in Rails applications. We'll look at using server-sent events (SSE) and Turbo Streams as two different options for delivering streaming interfaces in Rails applications. We'll also provide some code examples for a demo chat application we made — it has three different bot personalities you can interact with through SSE or Turbo Streams.
    """

    generator = pipeline(
        text, voice='af_heart', # NOTE: <= change voice here
        speed=1
    )
    # Alternatively, load voice tensor directly:
    # voice_tensor = torch.load('path/to/voice.pt', weights_only=True)
    # generator = pipeline(
    #     text, voice=voice_tensor,
    #     speed=1, split_pattern=r'\n+'
    # )

    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        display(Audio(data=audio, rate=24000, autoplay=i==0))
        sf.write(f'output/{i}.wav', audio, 24000) # save each audio file


if __name__ == "__main__":
    main()
