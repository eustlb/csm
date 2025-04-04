from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torch
import torchaudio
from datasets import load_dataset, Audio

model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, "cuda:1")

prompt_tokens = torch.load("prompt_tokens.pt").to(generator.device)
prompt_tokens_mask = torch.load("prompt_tokens_mask.pt").to(generator.device)

# Create CUDA events for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Run 4 times and measure performance
total_time = 0
results = []

for i in range(4):
    # Start timing
    start_event.record()
    
    # Generate tokens
    tokens = generator.generate_from_tokens(
        prompt_tokens=prompt_tokens,
        prompt_tokens_mask=prompt_tokens_mask,
        max_audio_length_ms=10_000,
    )
    
    # End timing
    end_event.record()
    torch.cuda.synchronize()
    
    # Calculate elapsed time in milliseconds
    elapsed_time = start_event.elapsed_time(end_event)
    total_time += elapsed_time
    results.append(tokens)
    
    print(f"Run {i+1}/4: {elapsed_time:.2f} ms, shape: {tokens.shape}")

# Print average time
print(f"Average generation time: {total_time/4:.2f} ms")
print(f"Final output shape: {results[-1].shape}")

