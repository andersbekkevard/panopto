from transformers import pipeline
from pathlib import Path
from typing import List, Optional
import torch

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Input/Output directories
AUDIO_FOLDER = Path("audio")
TEXT_FOLDER = Path("text")
TEXT_FOLDER.mkdir(exist_ok=True)

# Model selection
# Options: "NbAiLab/nb-whisper-tiny" (4x faster), "NbAiLab/nb-whisper-base" (2x faster),
#          "NbAiLab/nb-whisper-small" (balanced), "NbAiLab/nb-whisper-medium" (most accurate)
MODEL_NAME = "NbAiLab/nb-whisper-small"

# Speed optimization settings (recommended for 5-6x speedup)
BATCH_SIZE = 16  # Process multiple chunks at once (higher = faster, more memory)
NUM_BEAMS = 1  # Greedy decoding (1) is much faster than beam search (5)
# Note: num_beams=1 is ~3x faster than num_beams=5 with minimal accuracy loss

CHUNK_LENGTH_S = 30  # Audio chunk length in seconds

# ============================================================================


def get_audio_files(audio_folder: Path) -> List[Path]:
    """Get all audio files from the specified folder."""
    audio_extensions = {".mp3", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    # Exclude .part (incomplete downloads) and .mp4 (video files)
    exclude_extensions = {".part", ".mp4"}
    audio_files = []

    if not audio_folder.exists():
        print(f"Warning: Audio folder '{audio_folder}' does not exist.")
        return []

    for file_path in audio_folder.iterdir():
        suffix = file_path.suffix.lower()
        # Skip if it's an excluded extension or not an audio file
        if suffix in exclude_extensions:
            continue
        if file_path.is_file() and suffix in audio_extensions:
            audio_files.append(file_path)

    return sorted(audio_files)


def transcribe_audio_files(
    audio_files: Optional[List[Path]] = None,
    audio_folder: Path = AUDIO_FOLDER,
    output_folder: Path = TEXT_FOLDER,
) -> List[dict]:
    """
    Transcribe audio files using local NB-Whisper model.

    Args:
        audio_files: List of audio file paths (if None, scans audio_folder)
        audio_folder: Folder containing audio files
        output_folder: Folder to save transcriptions

    Returns:
        List of result dictionaries with transcription status
    """
    # Get audio files if not provided
    if audio_files is None:
        audio_files = get_audio_files(audio_folder)

    if not audio_files:
        print("No audio files found to transcribe.")
        return []

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple GPU (MPS)"
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = "CUDA GPU"
    else:
        device = "cpu"
        device_name = "CPU"

    print(f"\n{'='*70}")
    print(f"Starting transcription of {len(audio_files)} file(s)")
    print(f"Device: {device_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Beam search: {NUM_BEAMS} ({'greedy' if NUM_BEAMS == 1 else 'beam search'})")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")

    # Load model once for all files
    print(f"Loading model...")
    asr = pipeline(
        "automatic-speech-recognition",
        MODEL_NAME,
        device=device,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
    )
    print("Model loaded successfully!\n")

    results = []

    # Process each audio file
    for audio_path in audio_files:
        lecture_name = audio_path.stem
        output_path = output_folder / f"{lecture_name}.txt"

        # Check if already transcribed
        if output_path.exists():
            print(f"[SKIP] {lecture_name} - already transcribed")
            results.append(
                {
                    "file": lecture_name,
                    "audio_path": str(audio_path),
                    "output_path": str(output_path),
                    "status": "skipped",
                }
            )
            continue

        print(f"[START] {lecture_name}")

        try:
            # Transcribe with optimized settings
            result = asr(
                str(audio_path),
                chunk_length_s=CHUNK_LENGTH_S,
                batch_size=BATCH_SIZE,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "no",
                    "num_beams": NUM_BEAMS,
                    "do_sample": False,
                },
            )

            # Save transcription
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])  # type: ignore

            print(f"[SUCCESS] {lecture_name}")
            results.append(
                {
                    "file": lecture_name,
                    "audio_path": str(audio_path),
                    "output_path": str(output_path),
                    "status": "success",
                }
            )

        except Exception as e:
            print(f"[ERROR] {lecture_name}: {str(e)}")
            results.append(
                {
                    "file": lecture_name,
                    "audio_path": str(audio_path),
                    "status": "error",
                    "error": str(e),
                }
            )

    # Print summary
    print(f"\n{'='*70}")
    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = len(results) - successful - skipped
    print(f"TRANSCRIPTION SUMMARY:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(results)}")
    print(f"{'='*70}\n")

    # Print failed transcriptions if any
    if failed > 0:
        print("Failed transcriptions:")
        for r in results:
            if r["status"] == "error":
                error_msg = r.get("error", "Unknown error")
                print(f"  - {r['file']}: {error_msg}")
        print()

    return results


if __name__ == "__main__":
    # Transcribe all audio files in the audio folder
    results = transcribe_audio_files()

    # Example: Transcribe specific files
    # specific_files = [
    #     Path("audio") / "TFY4107 - 19.08.2025 - Økt 1.mp3",
    #     Path("audio") / "test.mp3",
    # ]
    # results = transcribe_audio_files(audio_files=specific_files)
