import subprocess
import json
import re
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Output folder where files should be saved
OUTPUT_FOLDER = Path("audio")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Parallelization settings
ENABLE_PARALLEL = True  # Set to True to enable parallel downloads
MAX_WORKERS = 3  # Number of parallel downloads (recommended: 3-8)

# ============================================================================


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename similar to how yt-dlp does it.
    This helps predict the output filename for existence check.
    """
    # Remove invalid characters for filenames
    filename = re.sub(r'[<>:"/\\|?*]', "", filename)
    # Replace multiple spaces with single space
    filename = re.sub(r"\s+", " ", filename)
    # Strip leading/trailing spaces
    filename = filename.strip()
    return filename


def check_file_exists(title: str, output_folder: Path) -> Optional[Path]:
    """
    Check if audio file already exists based on title.
    Returns the path if found, None otherwise.
    """
    # Sanitize title to match yt-dlp's output
    sanitized_title = sanitize_filename(title)

    # Check for common audio extensions
    for ext in [".mp3", ".m4a", ".opus", ".webm"]:
        potential_path = output_folder / f"{sanitized_title}{ext}"
        if potential_path.exists():
            return potential_path

    return None


def load_urls_from_json(json_path: str) -> list:
    """Load URLs from JSON file structure."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    urls = []
    # Extract URLs from weeks structure
    if "weeks" in data:
        for week in data["weeks"]:
            if "days" in week:
                for day_name, day_data in week["days"].items():
                    for session_key in ["økt1", "økt2"]:
                        if session_key in day_data:
                            urls.append(
                                {
                                    "url": day_data[session_key]["url"],
                                    "title": day_data[session_key]["title"],
                                }
                            )

    # Extract URLs from special section
    if "special" in data:
        for item in data["special"]:
            urls.append({"url": item["url"], "title": item["title"]})

    return urls


def _download_single_lecture(
    item: dict,
    output_folder: Path,
    format_id: str = "7",
    audio_quality: str = "64K",
) -> dict:
    """
    Download a single lecture and return status.

    Args:
        item: Dictionary with 'url' and 'title' keys
        output_folder: Path to save the downloaded file
        format_id: yt-dlp format ID
        audio_quality: Audio quality setting

    Returns:
        Dictionary with download status information
    """
    url = item["url"] if isinstance(item, dict) else item
    title = item.get("title", "Unknown") if isinstance(item, dict) else url

    # Check if file already exists (idempotent behavior)
    existing_file = check_file_exists(title, output_folder)
    if existing_file:
        print(f"[SKIP] {title} - already downloaded")
        return {
            "title": title,
            "url": url,
            "status": "skipped",
            "output_path": str(existing_file),
        }

    command = [
        "yt-dlp",
        "-f",
        format_id,
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        audio_quality,
        "--postprocessor-args",
        "ffmpeg:-ac 1",  # Convert to mono (1 audio channel)
        "--cookies",
        "cookies.txt",
        "-o",
        str(output_folder / "%(title)s.%(ext)s"),
        url,
    ]

    print(f"[START] {title}")
    try:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            print(f"[SUCCESS] {title}")
            return {"title": title, "url": url, "status": "success"}
        else:
            print(f"[FAILED] {title}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return {
                "title": title,
                "url": url,
                "status": "failed",
                "error": result.stderr,
            }
    except Exception as e:
        print(f"[ERROR] {title}: {str(e)}")
        return {"title": title, "url": url, "status": "error", "error": str(e)}


def download_audio(urls: list, output_folder: Optional[Path] = None) -> list:
    """
    Download audio from Panopto lecture URLs.

    Uses parallel or sequential mode based on ENABLE_PARALLEL flag.

    Args:
        urls: List of URL dictionaries with 'url' and 'title' keys
        output_folder: Path to save downloaded files

    Returns:
        List of result dictionaries with download status for each lecture
    """
    if output_folder is None:
        output_folder = OUTPUT_FOLDER

    # Format ID for video with audio (update based on `yt-dlp -F` output)
    format_id = "7"  # PODCAST format - best for audio extraction
    audio_quality = "64K"

    if not urls:
        print("No URLs provided.")
        return []

    print(f"\n{'='*70}")
    print(f"Starting download of {len(urls)} lecture(s)")
    print(f"Mode: {'PARALLEL' if ENABLE_PARALLEL else 'SEQUENTIAL'}")
    if ENABLE_PARALLEL:
        print(f"Workers: {MAX_WORKERS}")
    print(f"{'='*70}\n")

    results = []

    if ENABLE_PARALLEL:
        # Parallel download using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all download tasks
            future_to_item = {
                executor.submit(
                    _download_single_lecture,
                    item,
                    output_folder,
                    format_id,
                    audio_quality,
                ): item
                for item in urls
            }

            # Process completed downloads as they finish with progress bar
            for future in tqdm(
                as_completed(future_to_item),
                total=len(urls),
                desc="Downloading",
                unit="lecture",
            ):
                result = future.result()
                results.append(result)

    else:
        # Sequential download (original behavior)
        for item in tqdm(urls, desc="Downloading", unit="lecture"):
            result = _download_single_lecture(
                item, output_folder, format_id, audio_quality
            )
            results.append(result)

    # Print summary
    print(f"\n{'='*70}")
    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = len(results) - successful - skipped
    print(f"DOWNLOAD SUMMARY:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(results)}")
    print(f"{'='*70}\n")

    # Print failed downloads if any
    if failed > 0:
        print("Failed downloads:")
        for r in results:
            if r["status"] not in ["success", "skipped"]:
                print(f"  - {r['title']}")
        print()

    return results


if __name__ == "__main__":
    # Load URLs from JSON file
    urls = load_urls_from_json("urls/fysikk_urls.json")

    if urls:
        print(f"Found {len(urls)} lectures total.")
        download_audio(urls)
    else:
        print("No URLs found in JSON file.")
