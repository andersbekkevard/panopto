from transformers import pipeline
from pathlib import Path
import torch

# Input audio file path (update this to point to your audio file)
IN_PATH = Path("audio") / "5min.mp3"
# IN_PATH = Path("audio") / "test.mp3"

# Output folder for transcriptions
OUTPUT_FOLDER = Path("text")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Generate output filename from input filename
lecture_name = IN_PATH.stem + ".txt"
OUT_PATH = OUTPUT_FOLDER / lecture_name

# Determine device (use MPS for Apple Silicon GPU, fallback to CPU)
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple GPU (MPS) for acceleration")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU for acceleration")
else:
    device = "cpu"
    print("Using CPU (no GPU acceleration available)")

# Load the NB-Whisper Small model
print(f"Loading NB-Whisper Small model...")
asr = pipeline(
    "automatic-speech-recognition", "NbAiLab/nb-whisper-small", device=device
)

# Transcribe the audio file
print(f"Transcribing {IN_PATH}...")
result = asr(
    str(IN_PATH),
    chunk_length_s=28,
    generate_kwargs={"task": "transcribe", "language": "no", "num_beams": 5},
)

# Write the transcription to a text file
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(result["text"])  # type: ignore

print(f"Transcription saved to {OUT_PATH}")
