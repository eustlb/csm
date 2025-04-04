from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torch
import torchaudio
from datasets import load_dataset, Audio

model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, "cuda:1")

speakers = [1, 0] * 6
transcripts = [
    "What are you working on?",
    "I'm figuring out my budget.",
    "Umm…. What budget?",
    "I'm making a shopping budget, so that I don't spend too much money.",
    "How much money can you spend?",
    "I can only spend three hundred dollars a month.",
    "Why only three hundred dollars?",
    "I need to save the rest.",
    "For what?",
    "I need to pay my bills.",
    "Your budget is a good idea.",
    "I know. It's going to save me a lot of money, I hope.",
]

audio_paths = [
    "/home/eustache_lebihan/add-sesame/conv-data/0/0_1_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/1_0_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/2_1_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/3_0_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/4_1_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/5_0_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/6_1_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/7_0_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/8_1_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/9_0_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/10_1_d0.wav",
    "/home/eustache_lebihan/add-sesame/conv-data/0/11_0_d0.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

transcripts = transcripts[:-1]
speakers = speakers[:-1]
audio_paths = audio_paths[:-1]
segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="I know. It's going to save me a lot of money, I hope.",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

