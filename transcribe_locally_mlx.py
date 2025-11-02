"""
Apple Silicon optimized transcription using MLX (Metal acceleration).

MLX is Apple's machine learning framework that natively uses Metal for GPU acceleration.
Best performance on M1/M2/M3 Macs.

Key features:
- Native Metal GPU acceleration
- Optimized for Apple Silicon
- Low memory footprint
- Efficient memory management
"""

import mlx_whisper
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import platform

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Input/Output directories
AUDIO_FOLDER = Path("audio")
TEXT_FOLDER = Path("text")
TEXT_FOLDER.mkdir(exist_ok=True)

# Model selection - MLX models from Hugging Face
# Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
# For Norwegian, we'll use the standard models and specify language
MODEL_SIZE = "small"
# MLX models are hosted on Hugging Face under mlx-community
MODEL_PATH = f"mlx-community/whisper-{MODEL_SIZE}-mlx"

# For Norwegian Whisper models (if available in MLX format)
# Uncomment to use Norwegian-specific model
# MODEL_PATH = f"mlx-community/nb-whisper-{MODEL_SIZE}"

# Speed optimization settings
LANGUAGE = "no"  # Norwegian
TASK = "transcribe"

# MLX-specific settings
FP16 = True  # Use float16 for faster inference (recommended for Apple Silicon)

# Decoding settings
# Note: MLX currently only supports greedy decoding (beam search not yet implemented)
TEMPERATURE = 0.0  # Deterministic (0.0), or use sampling (0.0-1.0)
CONDITION_ON_PREVIOUS_TEXT = True  # Use previous text for context

# Verbose output
VERBOSE = False  # Set to True for detailed progress

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


def get_system_info() -> Dict[str, Any]:
    """Get system information for Apple Silicon."""
    info: Dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Check if running on Apple Silicon
    if info["machine"] == "arm64" and info["platform"] == "Darwin":
        info["apple_silicon"] = True
        # Try to detect chip type
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            info["chip"] = result.stdout.strip()
        except:
            info["chip"] = "Apple Silicon (Unknown)"
    else:
        info["apple_silicon"] = False

    return info


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds."""
    try:
        import librosa
        import soundfile as sf

        # Try soundfile first (faster)
        try:
            info = sf.info(str(audio_path))
            return info.duration
        except:
            # Fall back to librosa
            duration = librosa.get_duration(path=str(audio_path))
            return duration
    except ImportError:
        # If neither library is available, return None
        return None
    except Exception as e:
        print(f"  Warning: Could not get audio duration: {e}")
        return None


def transcribe_audio_files(
    audio_files: Optional[List[Path]] = None,
    audio_folder: Path = AUDIO_FOLDER,
    output_folder: Path = TEXT_FOLDER,
) -> List[dict]:
    """
    Transcribe audio files using MLX-Whisper for Apple Silicon.

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

    # Get system information
    sys_info = get_system_info()

    print(f"\n{'='*70}")
    print(f"🍎 MLX TRANSCRIPTION (Apple Silicon Optimized)")
    print(f"{'='*70}")

    if not sys_info.get("apple_silicon", False):
        print(f"⚠️  WARNING: Not running on Apple Silicon!")
        print(f"   Machine: {sys_info['machine']}")
        print(f"   MLX is optimized for M1/M2/M3 chips and may not work properly")
        print(f"{'='*70}")
    else:
        print(f"Hardware: {sys_info.get('chip', 'Apple Silicon')}")
        print(f"Acceleration: Metal (Native GPU)")

    print(f"Files to process: {len(audio_files)}")
    print(f"Model: {MODEL_PATH}")
    print(f"Precision: {'FP16' if FP16 else 'FP32'}")
    print(f"Decoding: Greedy (beam search not yet supported in MLX)")
    print(f"Language: {LANGUAGE}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")

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

        # Get audio duration for performance metrics
        audio_duration = get_audio_duration(audio_path)

        try:
            # Transcribe with MLX
            # Note: MLX doesn't support beam search yet, so we omit beam search parameters
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=MODEL_PATH,
                language=LANGUAGE,
                task=TASK,
                fp16=FP16,
                verbose=VERBOSE,
                temperature=TEMPERATURE,
                # Removed: beam_size, best_of, patience, length_penalty (not supported in MLX)
                condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            )

            # Extract text from result
            transcription: str
            if isinstance(result, dict):
                text_result = result.get("text", "")
                transcription = (
                    str(text_result)
                    if not isinstance(text_result, str)
                    else text_result
                )
            else:
                transcription = str(result)

            # Save transcription
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription.strip())

            processing_time = time.time() - start_time

            # Calculate speedup if we have audio duration
            speedup = None
            if audio_duration:
                speedup = audio_duration / processing_time if processing_time > 0 else 0
                total_audio_duration += audio_duration
                total_processing_time += processing_time

            print(f"[SUCCESS] {lecture_name}")
            if audio_duration and speedup:
                print(
                    f"  Duration: {format_time(audio_duration)} | "
                    f"Processed in: {format_time(processing_time)} | "
                    f"Speedup: {speedup:.1f}x realtime"
                )
            else:
                print(f"  Processed in: {format_time(processing_time)}")

            result_dict = {
                "file": lecture_name,
                "audio_path": str(audio_path),
                "output_path": str(output_path),
                "status": "success",
                "processing_time": processing_time,
            }

            if audio_duration:
                result_dict["audio_duration"] = audio_duration
                result_dict["speedup"] = speedup

            results.append(result_dict)

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

    if successful > 0 and total_processing_time > 0 and total_audio_duration > 0:
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
    # Check if running on Apple Silicon
    sys_info = get_system_info()
    if not sys_info.get("apple_silicon", False):
        print("\n⚠️  WARNING: MLX is designed for Apple Silicon (M1/M2/M3)")
        print(f"Your system: {sys_info['machine']} on {sys_info['platform']}")
        print("Consider using transcribe_locally_fastest.py instead\n")

        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != "y":
            print("Exiting...")
            exit(0)

    # Transcribe all audio files in the audio folder
    results = transcribe_audio_files()

    # Example: Transcribe specific files
    # specific_files = [
    #     Path("audio") / "TFY4107 - 19.08.2025 - Økt 1.mp3",
    #     Path("audio") / "test.mp3",
    # ]
    # results = transcribe_audio_files(audio_files=specific_files)
