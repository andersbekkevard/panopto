import os
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Input/Output directories
AUDIO_FOLDER = Path("audio")
TEXT_FOLDER = Path("text")
TEXT_FOLDER.mkdir(exist_ok=True)

# OpenAI API settings
# IMPORTANT: Set your API key as an environment variable:
#   export OPENAI_API_KEY="your-key-here"  (macOS/Linux)
#   set OPENAI_API_KEY=your-key-here       (Windows)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Whisper model to use for transcription
# Available models for /v1/audio/transcriptions endpoint:
#   - whisper-1: Dedicated transcription model (RECOMMENDED - only supported model)
#
# Note: Other audio models exist (gpt-4o-audio-preview, gpt-audio, etc.) but these
#       are for real-time audio chat, NOT for transcription API endpoint.
#       They use a different API and are not compatible with this script.
WHISPER_MODEL = "whisper-1"

# Parallelization settings
ENABLE_PARALLEL = False  # Set to True to enable parallel transcriptions
MAX_WORKERS = 2  # Number of parallel API calls (recommended: 2-4)
# Note: OpenAI has rate limits, don't set this too high

# File size limit (OpenAI has 25MB limit for audio files)
MAX_FILE_SIZE_MB = 25

# ============================================================================


def get_audio_files(audio_folder: Path) -> List[Path]:
    """
    Get all audio files from the specified folder.

    Args:
        audio_folder: Path to folder containing audio files

    Returns:
        List of Path objects for audio files
    """
    audio_extensions = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    audio_files = []

    if not audio_folder.exists():
        print(f"Warning: Audio folder '{audio_folder}' does not exist.")
        return []

    for file_path in audio_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)

    return sorted(audio_files)


def validate_file_size(file_path: Path, max_size_mb: float = MAX_FILE_SIZE_MB) -> bool:
    """
    Check if file size is within OpenAI's limits.

    Args:
        file_path: Path to audio file
        max_size_mb: Maximum file size in MB

    Returns:
        True if file size is acceptable, False otherwise
    """
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    return file_size_mb <= max_size_mb


def _transcribe_single_file(
    audio_path: Path, output_folder: Path, api_key: str, model: str = WHISPER_MODEL
) -> dict:
    """
    Transcribe a single audio file using OpenAI Whisper API.

    Args:
        audio_path: Path to audio file
        output_folder: Path to save transcription
        api_key: OpenAI API key
        model: Whisper model to use

    Returns:
        Dictionary with transcription status information
    """
    lecture_name = audio_path.stem
    output_path = output_folder / f"{lecture_name}.txt"

    print(f"[START] {lecture_name}")

    # Check if already transcribed
    if output_path.exists():
        print(f"[SKIP] {lecture_name} - already transcribed")
        return {
            "file": lecture_name,
            "audio_path": str(audio_path),
            "output_path": str(output_path),
            "status": "skipped",
            "message": "Already transcribed",
        }

    # Validate file size
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if not validate_file_size(audio_path):
        print(
            f"[FAILED] {lecture_name} - file too large ({file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB)"
        )
        return {
            "file": lecture_name,
            "audio_path": str(audio_path),
            "status": "failed",
            "error": f"File size {file_size_mb:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit",
        }

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Transcribe audio file
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model, file=audio_file, language="no"  # Norwegian language
            )

        # Save transcription to text file
        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(transcription.text)

        print(f"[SUCCESS] {lecture_name}")
        return {
            "file": lecture_name,
            "audio_path": str(audio_path),
            "output_path": str(output_path),
            "status": "success",
        }

    except Exception as e:
        print(f"[ERROR] {lecture_name}: {str(e)}")
        return {
            "file": lecture_name,
            "audio_path": str(audio_path),
            "status": "error",
            "error": str(e),
        }


def transcribe_audio_files(
    audio_files: Optional[List[Path]] = None,
    audio_folder: Path = AUDIO_FOLDER,
    output_folder: Path = TEXT_FOLDER,
) -> List[dict]:
    """
    Transcribe audio files using OpenAI Whisper API.

    Uses parallel or sequential mode based on ENABLE_PARALLEL flag.

    Args:
        audio_files: List of audio file paths (if None, scans audio_folder)
        audio_folder: Folder containing audio files
        output_folder: Folder to save transcriptions

    Returns:
        List of result dictionaries with transcription status
    """
    # Validate API key
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it using:")
        print("  export OPENAI_API_KEY='your-key-here'  (macOS/Linux)")
        print("  set OPENAI_API_KEY=your-key-here       (Windows)")
        return []

    # Get audio files if not provided
    if audio_files is None:
        audio_files = get_audio_files(audio_folder)

    if not audio_files:
        print("No audio files found to transcribe.")
        return []

    print(f"\n{'='*70}")
    print(f"Starting transcription of {len(audio_files)} file(s)")
    print(f"Mode: {'PARALLEL' if ENABLE_PARALLEL else 'SEQUENTIAL'}")
    if ENABLE_PARALLEL:
        print(f"Workers: {MAX_WORKERS}")
    print(f"Model: {WHISPER_MODEL}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")

    results = []

    if ENABLE_PARALLEL:
        # Parallel transcription using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all transcription tasks
            future_to_file = {
                executor.submit(
                    _transcribe_single_file,
                    audio_path,
                    output_folder,
                    OPENAI_API_KEY,
                    WHISPER_MODEL,
                ): audio_path
                for audio_path in audio_files
            }

            # Process completed transcriptions as they finish
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

    else:
        # Sequential transcription
        for audio_path in audio_files:
            result = _transcribe_single_file(
                audio_path, output_folder, OPENAI_API_KEY, WHISPER_MODEL
            )
            results.append(result)

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
            if r["status"] in ["failed", "error"]:
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
    #     Path("audio") / "TFY4107 - 19.08.2025 - Økt 2.mp3",
    # ]
    # results = transcribe_audio_files(audio_files=specific_files)
