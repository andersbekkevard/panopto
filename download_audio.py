import subprocess

# region styring
"""
with open("urls\\styring_urls.txt", "r") as f:
output_folder = "C:\\Users\\Anders\\forelesninger\\styring\\audio\\"


"""
# endregion


urls = []

with open("urls\\objekt_urls.txt", "r") as f:
    for line in f.readlines():
        if "https" in line:
            urls.append(line.strip())


# List of urls of lectures to be sourced
def download_audio(urls: list) -> None:
    # Output folder where files should be saved
    output_folder = "C:\\Users\\Anders\\forelesninger\\objekt\\audio\\"

    # Format ID for video with audio (update based on `yt-dlp -F` output)
    format_id = "9"

    # Audio quality
    audio_quality = "64K"

    # Loop through URLs and download each lecture
    for url in urls:
        command = [
            "yt-dlp",
            "-f",
            format_id,  # Select correct video+audio format
            "--extract-audio",  # Extract only audio
            "--audio-format",
            "mp3",  # Convert to MP3
            "--audio-quality",
            audio_quality,  # Set bitrate
            "--cookies",
            "cookies.txt",  # Use authentication
            "-o",
            f"{output_folder}%(title)s.%(ext)s",  # Save with title in specified folder
            url,
        ]
        print("fetching: " + " ".join(command))
        # Run yt-dlp command
        subprocess.run(command, text=True)


new_url = []
new_url.append(
    "https://ntnu.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=7a136096-847d-408e-9120-b26400831b50"
)
download_audio(new_url)
