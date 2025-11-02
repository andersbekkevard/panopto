"""
Key optimizations:
- faster-whisper (CTranslate2 backend) - 2-4x faster than transformers
- VAD filtering to skip silence
- Optimized batch processing
- INT8 quantization for even more speed
- Compute type optimization
"""

from faster_whisper import WhisperModel
from pathlib import Path
from typing import List, Optional
import torch
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Input/Output directories
AUDIO_FOLDER = Path("audio")
TEXT_FOLDER = Path("text")
TEXT_FOLDER.mkdir(exist_ok=True)

# Model selection
# Options: "tiny" (fastest, ~10x faster than large), "base" (5x faster),
#          "small" (2-3x faster), "medium", "large-v2"
# Prefix with language: "NbAiLab/nb-whisper-small" for Norwegian
MODEL_SIZE = "small"
MODEL_NAME = f"NbAiLab/nb-whisper-{MODEL_SIZE}"

# Speed optimization settings
COMPUTE_TYPE = "int8"  # Options: "int8" (fastest), "float16", "float32"
# int8 is 2x faster than float16 with minimal accuracy loss on CPU
# On GPU/MPS: use "float16" for best speed/accuracy balance

BEAM_SIZE = 1  # Greedy decoding (1) is much faster than beam search (5)
BEST_OF = 1  # Number of candidates when sampling (1 = greedy)

# VAD (Voice Activity Detection) settings - skip silence for big speedup
VAD_FILTER = True  # Enable VAD to skip silence
VAD_THRESHOLD = 0.5  # Higher = more aggressive silence filtering (0.0-1.0)
MIN_SILENCE_DURATION_MS = 500  # Minimum silence duration to skip

# Audio processing
BATCH_SIZE = 16  # Number of audio chunks to process in parallel

# Advanced settings
CONDITION_ON_PREVIOUS_TEXT = False  # Faster but may reduce accuracy at chunk boundaries
NO_SPEECH_THRESHOLD = 0.6  # Skip chunks with low speech probability
COMPRESSION_RATIO_THRESHOLD = 2.4  # Skip chunks that are too repetitive
LOG_PROB_THRESHOLD = -1.0  # Skip low-confidence chunks

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


def format_time(seconds: float) -> str:
    """Format seconds into readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def transcribe_audio_files(
    audio_files: Optional[List[Path]] = None,
    audio_folder: Path = AUDIO_FOLDER,
    output_folder: Path = TEXT_FOLDER,
) -> List[dict]:
    """
    Transcribe audio files using faster-whisper for maximum speed.

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

    # Determine device and compute type
    if torch.backends.mps.is_available():
        device = "auto"  # faster-whisper will use CPU, which is often faster than MPS
        device_name = "CPU (optimized with CTranslate2)"
        # Note: faster-whisper doesn't support MPS yet, CPU with CTranslate2 is faster anyway
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = "CUDA GPU"
    else:
        device = "cpu"
        device_name = "CPU (optimized with CTranslate2)"

    # Adjust compute type based on device
    compute_type = COMPUTE_TYPE
    if device == "cuda":
        # CUDA supports float16 for better speed/accuracy
        if COMPUTE_TYPE == "int8":
            compute_type = "int8_float16"  # Use int8 with float16 fallback
    elif device == "auto" or device == "cpu":
        # CPU: int8 is fastest
        if COMPUTE_TYPE == "float16":
            print(f"Warning: float16 not well supported on CPU, using int8 instead")
            compute_type = "int8"

    print(f"\n{'='*70}")
    print(f"🚀 FASTEST TRANSCRIPTION MODE")
    print(f"{'='*70}")
    print(f"Files to process: {len(audio_files)}")
    print(f"Device: {device_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Compute type: {compute_type}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Beam size: {BEAM_SIZE} ({'greedy' if BEAM_SIZE == 1 else 'beam search'})")
    print(f"VAD filter: {'✓ Enabled' if VAD_FILTER else '✗ Disabled'}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")

    # Load model once for all files
    print(f"Loading optimized model...")
    start_load = time.time()

    try:
        model = WhisperModel(
            MODEL_NAME,
            device=device,
            compute_type=compute_type,
            cpu_threads=8,  # Optimize CPU thread count
            num_workers=4,  # Parallel workers for preprocessing
        )
        load_time = time.time() - start_load
        print(f"Model loaded in {format_time(load_time)}!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have installed: pip install faster-whisper")
        return []

    results = []
    total_audio_duration = 0
    total_processing_time = 0

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
        start_time = time.time()

        try:
            # Transcribe with all optimizations
            segments, info = model.transcribe(
                str(audio_path),
                language="no",  # Norwegian
                task="transcribe",
                beam_size=BEAM_SIZE,
                best_of=BEST_OF,
                vad_filter=VAD_FILTER,
                vad_parameters=(
                    dict(
                        threshold=VAD_THRESHOLD,
                        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
                    )
                    if VAD_FILTER
                    else None
                ),
                condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                log_prob_threshold=LOG_PROB_THRESHOLD,
                word_timestamps=False,  # Disable for speed
            )

            # Collect all segments
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text)

            transcription = " ".join(transcription_parts).strip()

            # Save transcription
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription)

            processing_time = time.time() - start_time
            audio_duration = info.duration
            speedup = audio_duration / processing_time if processing_time > 0 else 0

            total_audio_duration += audio_duration
            total_processing_time += processing_time

            print(f"[SUCCESS] {lecture_name}")
            print(
                f"  Duration: {format_time(audio_duration)} | "
                f"Processed in: {format_time(processing_time)} | "
                f"Speedup: {speedup:.1f}x realtime"
            )

            results.append(
                {
                    "file": lecture_name,
                    "audio_path": str(audio_path),
                    "output_path": str(output_path),
                    "status": "success",
                    "audio_duration": audio_duration,
                    "processing_time": processing_time,
                    "speedup": speedup,
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
    print(f"📊 TRANSCRIPTION SUMMARY")
    print(f"{'='*70}")

    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = len(results) - successful - skipped

    print(f"  ✓ Successful: {successful}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(results)}")

    if successful > 0 and total_processing_time > 0:
        overall_speedup = total_audio_duration / total_processing_time
        print(f"\n⚡ Performance:")
        print(f"  Total audio duration: {format_time(total_audio_duration)}")
        print(f"  Total processing time: {format_time(total_processing_time)}")
        print(f"  Overall speedup: {overall_speedup:.1f}x realtime")
        print(
            f"  Average: {format_time(total_audio_duration/successful)} audio "
            f"→ {format_time(total_processing_time/successful)} processing"
        )

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
