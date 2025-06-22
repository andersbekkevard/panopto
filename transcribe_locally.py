from transformers import pipeline

IN_PATH = (
    "C:\\Users\\Anders\\forelesninger\\styring\\audio\\TIØ4105 - 17.03.2025 - Økt 2.mp3"
)
lecture_name = IN_PATH.split("\\")[-1]
OUT_PATH = "C:\\Users\\Anders\\forelesninger\\styring\\text\\" + lecture_name

# Load the NB-Whisper Large model
print(f"Loading NB-Whisper Small model...")
asr = pipeline("automatic-speech-recognition", "NbAiLab/nb-whisper-small")

# Transcribe the audio file
print(f"Transcribing {IN_PATH}...")
result = asr(
    IN_PATH,
    chunk_length_s=28,
    generate_kwargs={"task": "transcribe", "language": "no", "num_beams": 5},
)

# Write the transcription to a text file
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"Transcription saved to {OUT_PATH}")
